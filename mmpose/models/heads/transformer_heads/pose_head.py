import copy
import math
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist

from mmcv.ops import MultiScaleDeformableAttention
from mmengine.model import BaseModule, ModuleList, constant_init
from mmengine.structures import InstanceData
from torch import Tensor, nn

from mmpose.models.utils import inverse_sigmoid
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, Features, OptConfigType,
                                 OptSampleList, Predictions)
from .transformers.deformable_detr_layers import (DeformableTransformerGroupDecoderLayer,
                                                  DeformableDetrTransformerEncoder)
from .transformers.utils import FFN, PositionEmbeddingSineHW, MLP, NestedTensor


class GroupDecoder(BaseModule):

    def __init__(self, layer_cfg,
                 num_layers,
                 return_intermediate=False,
                 d_model=256,
                 num_body_points=17):
        super().__init__()

        self.layer_cfg = layer_cfg
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_body_points = num_body_points
        self.return_intermediate = return_intermediate
        self.class_embed = None
        self.pose_embed = None
        self.half_pose_ref_point_head = MLP(d_model, d_model, d_model, 2)

        # add
        self.layers = ModuleList([
            DeformableTransformerGroupDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])

    @staticmethod
    def get_proposal_pos_embed(pos_tensor: Tensor,
                               temperature: int = 10000,
                               num_pos_feats: int = 128) -> Tensor:
        """Get the position embedding of the proposal.

        Args:
            pos_tensor (Tensor): Not normalized proposals, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            temperature (int, optional): The temperature used for scaling the
                position embedding. Defaults to 10000.
            num_pos_feats (int, optional): The feature dimension for each
                position along x, y, w, and h-axis. Note the final returned
                dimension for each position is 4 times of num_pos_feats.
                Default to 128.

        Returns:
            Tensor: The position embedding of proposal, has shape
            (bs, num_queries, num_pos_feats * 4), with the last dimension
            arranged as (cx, cy, w, h)
        """

        scale = 2 * math.pi
        dim_t = torch.arange(
            num_pos_feats, dtype=torch.float32, device=pos_tensor.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        x_embed = pos_tensor[:, :, 0] * scale
        y_embed = pos_tensor[:, :, 1] * scale
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()),
                            dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()),
                            dim=3).flatten(2)
        if pos_tensor.size(-1) == 2:
            pos = torch.cat((pos_y, pos_x), dim=2)
        elif pos_tensor.size(-1) == 4:
            w_embed = pos_tensor[:, :, 2] * scale
            pos_w = w_embed[:, :, None] / dim_t
            pos_w = torch.stack(
                (pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()),
                dim=3).flatten(2)

            h_embed = pos_tensor[:, :, 3] * scale
            pos_h = h_embed[:, :, None] / dim_t
            pos_h = torch.stack(
                (pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()),
                dim=3).flatten(2)

            pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
        else:
            raise ValueError('Unknown pos_tensor shape(-1):{}'.format(
                pos_tensor.size(-1)))
        return pos

    def forward(self,
                tgt: Tensor,
                memory: Tensor,
                key_padding_mask: Optional[Tensor] = None,
                refpoints_sigmoid: Optional[Tensor] = None,

                spatial_shapes: Optional[Tensor] = None,
                level_start_index: Optional[Tensor] = None,
                valid_ratios: Optional[Tensor] = None,

                **kwargs) -> Tuple[Tuple[List[Any]], List[Any]]:
        output_pose = tgt.transpose(0, 1)
        refpoint_pose = refpoints_sigmoid
        intermediate_pose = []
        ref_pose_points = [refpoint_pose]

        for layer_id, layer in enumerate(self.layers):
            refpoint_pose_input = refpoint_pose[:, :, None] * torch.cat([valid_ratios] * (refpoint_pose.shape[-1] // 2),
                                                                        -1)[None, :]
            nq, bs, np = refpoint_pose.shape
            refpoint_pose_reshape = refpoint_pose_input[:, :, 0].reshape(nq, bs, np // 2, 2).reshape(nq * bs, np // 2,
                                                                                                     2)
            pose_query_sine_embed = self.get_proposal_pos_embed(refpoint_pose_reshape).reshape(nq, bs, np // 2,
                                                                                               self.d_model)
            pose_query_pos = self.half_pose_ref_point_head(pose_query_sine_embed)

            output_pose = layer(
                tgt_pose=output_pose,
                query_pos=pose_query_pos[:, :, 1:],
                tgt_pose_reference_points=refpoint_pose_input,

                value=memory,
                key_padding_mask=key_padding_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,

                **kwargs)
            intermediate_pose.append(output_pose)

            # iteration
            nq, bs, np = refpoint_pose.shape
            refpoint_pose = refpoint_pose.reshape(nq, bs, np // 2, 2)
            refpoint_pose_unsigmoid = inverse_sigmoid(refpoint_pose[:, :, 1:])
            delta_pose_unsigmoid = self.pose_embed[layer_id](output_pose[:, :, 1:])
            refpoint_pose_without_center = (refpoint_pose_unsigmoid + delta_pose_unsigmoid).sigmoid()
            # center of pose
            refpoint_center_pose = torch.mean(refpoint_pose_without_center, dim=2, keepdim=True)
            refpoint_pose = torch.cat([refpoint_center_pose, refpoint_pose_without_center], dim=2).flatten(-2)
            ref_pose_points.append(refpoint_pose)
            refpoint_pose = refpoint_pose.detach()

        decoder_outputs = [itm_out.transpose(0, 1) for itm_out in intermediate_pose],

        reference_points = [itm_refpoint.transpose(0, 1)
                            for itm_refpoint in ref_pose_points]

        return decoder_outputs, reference_points


@MODELS.register_module()
class GroupHead(BaseModule):
    def __init__(self,
                 num_queries: int = 100,
                 num_feature_levels: int = 4,
                 num_body_points: int = 17,
                 encoder: OptConfigType = None,
                 decoder: OptConfigType = None,
                 out_head: OptConfigType = None,

                 positional_encoding: OptConfigType = None,
                 data_decoder: OptConfigType = None,
                 denosing_cfg: OptConfigType = None,

                 dec_pred_class_embed_share: bool = False,
                 dec_pred_bbox_embed_share: bool = False,

                 ):
        super().__init__()
        self.num_feature_levels = num_feature_levels

        self.positional_encoding_cfg = positional_encoding
        self.positional_encoding = PositionEmbeddingSineHW(
            **self.positional_encoding_cfg)

        self.encoder_cfg = encoder
        self.decoder_cfg = decoder
        self.encoder = DeformableDetrTransformerEncoder(**self.encoder_cfg)
        self.decoder = GroupDecoder(
            num_body_points=num_body_points, **self.decoder_cfg)

        self.embed_dims = self.encoder.embed_dims

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))

    @staticmethod
    def get_valid_ratio(mask: Tensor) -> Tensor:
        """Get the valid radios of feature map in a level.

        .. code:: text

                    |---> valid_W <---|
                 ---+-----------------+-----+---
                  A |                 |     | A
                  | |                 |     | |
                  | |                 |     | |
            valid_H |                 |     | |
                  | |                 |     | H
                  | |                 |     | |
                  V |                 |     | |
                 ---+-----------------+     | |
                    |                       | V
                    +-----------------------+---
                    |---------> W <---------|

          The valid_ratios are defined as:
                r_h = valid_H / H,  r_w = valid_W / W
          They are the factors to re-normalize the relative coordinates of the
          image to the relative coordinates of the current level feature map.

        Args:
            mask (Tensor): Binary mask of a feature map, has shape (bs, H, W).

        Returns:
            Tensor: valid ratios [r_w, r_h] of a feature map, has shape (1, 2).
        """
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def keypoint_xyzxyz_to_xyxyzz(self, keypoints: torch.Tensor):

        """
        _summary_

        Args:
            keypoints (torch.Tensor): ..., 51
        """
        res = torch.zeros_like(keypoints)
        num_points = keypoints.shape[-1] // 3
        res[..., 0:2 * num_points:2] = keypoints[..., 0::3]
        res[..., 1:2 * num_points:2] = keypoints[..., 1::3]
        res[..., 2 * num_points:] = keypoints[..., 2::3]
        return res

    # denosing_cfg{
    # num_denoising=100,
    # label_noise_ratio=0.5,
    # box_noise_scale=1.0
    # dn_labelbook_size = 100}

    def gen_encoder_output_proposals(self, memory: Tensor, memory_mask: Tensor,
                                     spatial_shapes: Tensor
                                     ) -> Tuple[Tensor, Tensor]:
        """Generate proposals from encoded memory. The function will only be
        used when `as_two_stage` is `True`.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).

        Returns:
            tuple: A tuple of transformed memory and proposals.

            - output_memory (Tensor): The transformed memory for obtaining
              top-k proposals, has shape (bs, num_feat_points, dim).
            - output_proposals (Tensor): The inverse-normalized proposal, has
              shape (batch_size, num_keys, 4) with the last dimension arranged
              as (cx, cy, w, h).
        """
        bs = memory.size(0)
        proposals = []
        _cur = 0  # start index in the sequence of the current level
        for lvl, (H, W) in enumerate(spatial_shapes):
            mask_flatten_ = memory_mask[:, _cur:(_cur + H * W)].view(bs, H, W, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1).unsqueeze(-1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1).unsqueeze(-1)

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(
                    0, H - 1, H, dtype=torch.float32, device=memory.device),
                torch.linspace(
                    0, W - 1, W, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W, valid_H], 1).view(bs, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(bs, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(bs, -1, 4)
            proposals.append(proposal)
            _cur += (H * W)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) &
                                  (output_proposals < 0.99)).all(
            -1, keepdim=True)

        output_proposals = inverse_sigmoid(output_proposals)
        output_proposals = output_proposals.masked_fill(
            memory_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(
            ~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(
            memory_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid,
                                                  float(0))
        output_memory = self.memory_trans_fc(output_memory)
        output_memory = self.memory_trans_norm(output_memory)
        # [bs, sum(hw), 2]
        return output_memory, output_proposals

    def forward(self,
                img_feats: Tuple[Tensor],
                batch_data_samples: OptSampleList = None) -> Dict:
        batch_size = img_feats[0].size(0)
        # construct binary masks for the transformer.
        assert batch_data_samples is not None
        batch_input_shape = batch_data_samples[0].batch_input_shape
        img_shape_list = [sample.img_shape for sample in batch_data_samples]
        input_img_h, input_img_w = batch_input_shape
        masks = img_feats[0].new_ones((batch_size, input_img_h, input_img_w))

        for img_id in range(batch_size):
            img_h, img_w = img_shape_list[img_id]
            masks[img_id, :img_h, :img_w] = 0
        # NOTE following the official DETR repo, non-zero values representing
        # ignored positions, while zero values means valid positions.

        mlvl_masks = []
        mlvl_pos_embeds = []
        for feat in img_feats:
            mlvl_masks.append(
                F.interpolate(masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_pos_embeds.append(self.positional_encoding(mlvl_masks[-1]))

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(img_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2)  # bs, hw, c
            mask = mask.flatten(1)  # bs, hw
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            if self.num_feature_levels > 1 and self.level_embed is not None:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

        src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
        mask_flatten = torch.cat(mask_flatten, 1)  # bs, \sum{hxw}
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  # bs, \sum{hxw}, c
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in mlvl_masks], 1)

        print(f"query:{src_flatten.shape},query_pos:{lvl_pos_embed_flatten.shape}")
        memory = self.encoder(
            query=src_flatten,
            query_pos=lvl_pos_embed_flatten,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            key_padding_mask=mask_flatten)

        if self.two_stage_type in ['standard']:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
            output_memory = self.enc_output_norm(self.enc_output(output_memory))
            # top-k select index
            topk = self.num_queries
            enc_outputs_class_unselected = self.enc_out_class_embed(output_memory)
            topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1]
            # calculate K, e.g., 17 for COCO, points for keypoint
            bs, nq = output_memory.shape[:2]
            delta_unsig_keypoint = self.enc_pose_embed(output_memory).reshape(bs, nq, -1, 2)
            enc_outputs_pose_coord_unselected = (
                    delta_unsig_keypoint + output_proposals[..., :2].unsqueeze(-2)).sigmoid()
            enc_outputs_center_coord_unselected = torch.mean(enc_outputs_pose_coord_unselected, dim=2, keepdim=True)
            enc_outputs_pose_coord_unselected = torch.cat(
                [enc_outputs_center_coord_unselected, enc_outputs_pose_coord_unselected], dim=2).flatten(-2)
            # gather pose
            enc_outputs_pose_coord_sigmoid = torch.gather(enc_outputs_pose_coord_unselected, 1,
                                                          topk_proposals.unsqueeze(-1).repeat(1, 1,
                                                                                              enc_outputs_pose_coord_unselected.shape[
                                                                                                  -1]))
            refpoint_pose_sigmoid = enc_outputs_pose_coord_sigmoid.detach()
            # gather tgt
            tgt_undetach = torch.gather(output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model))
            # combine pose embedding
            if self.learnable_tgt_init:
                tgt = self.tgt_embed.expand_as(tgt_undetach).unsqueeze(-2)
            else:
                tgt = tgt_undetach.detach().unsqueeze(-2)
            # query construction
            tgt_pose = self.keypoint_embedding.weight[None, None].repeat(1, topk, 1, 1).expand(bs, -1, -1, -1) + tgt
            tgt_global = self.instance_embedding.weight[None, None].repeat(1, topk, 1, 1).expand(bs, -1, -1, -1)
            tgt_pose = torch.cat([tgt_global, tgt_pose], dim=2)

        hs_pose, refpoint_pose = self.decoder(
            tgt=tgt_pose,
            memory=memory.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            refpoints_sigmoid=refpoint_pose_sigmoid.transpose(0, 1),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios)
        if self.two_stage_type == 'standard':
            mix_refpoint = enc_outputs_pose_coord_sigmoid[:, :, 2:]
            mix_embedding = tgt_undetach
        else:
            mix_refpoint = None
            mix_embedding = None

        outputs_class = []
        outputs_keypoints_list = []

        for dec_lid, (hs_pose_i, refpoint_pose_i, layer_pose_embed, layer_cls_embed) in enumerate(
                zip(hs_pose, refpoint_pose, self.pose_embed, self.class_embed)):
            # pose
            bs, nq, np = refpoint_pose_i.shape
            refpoint_pose_i = refpoint_pose_i.reshape(bs, nq, np // 2, 2)
            delta_pose_unsig = layer_pose_embed(hs_pose_i[:, :, 1:])
            layer_outputs_pose_unsig = inverse_sigmoid(refpoint_pose_i[:, :, 1:]) + delta_pose_unsig
            vis_flag = torch.ones_like(layer_outputs_pose_unsig[..., -1:], device=layer_outputs_pose_unsig.device)
            layer_outputs_pose_unsig = torch.cat([layer_outputs_pose_unsig, vis_flag], dim=-1).flatten(-2)
            layer_outputs_pose_unsig = layer_outputs_pose_unsig.sigmoid()
            outputs_keypoints_list.append(self.keypoint_xyzxyz_to_xyxyzz(layer_outputs_pose_unsig))

            # cls
            layer_cls = layer_cls_embed(hs_pose_i[:, :, 0])
            outputs_class.append(layer_cls)

        out = {'pred_logits': outputs_class[-1], 'pred_keypoints': outputs_keypoints_list[-1]}

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_keypoints_list)

        # for encoder output
        if mix_refpoint is not None and mix_embedding is not None:
            # prepare intermediate outputs
            interm_class = self.transformer.enc_out_class_embed(mix_embedding)
            out['interm_outputs'] = {'pred_logits': interm_class, 'pred_keypoints': mix_refpoint}
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_keypoints):
        return [{'pred_logits': a, 'pred_keypoints': c}
                for a, c in zip(outputs_class[:-1], outputs_keypoints[:-1])]

    def decode(self, input_shapes: np.ndarray, pred_logits: Tensor,
               pred_boxes: Tensor, pred_keypoints: Tensor):
        """Select the final top-k keypoints, and decode the results from
        normalize size to origin input size.

        Args:
            input_shapes (Tensor): The size of input image.
            pred_logits (Tensor): The result of score.
            pred_boxes (Tensor): The result of bbox.
            pred_keypoints (Tensor): The result of keypoints.

        Returns:
        """

        if self.data_decoder is None:
            raise RuntimeError(f'The data decoder has not been set in \
                {self.__class__.__name__}. '
                               'Please set the data decoder configs in \
                    the init parameters to '
                               'enable head methods `head.predict()` and \
                     `head.decode()`')

        preds = []

        pred_logits = pred_logits.sigmoid()
        pred_logits, pred_boxes, pred_keypoints = to_numpy(
            [pred_logits, pred_boxes, pred_keypoints])

        for input_shape, pred_logit, pred_bbox, pred_kpts in zip(
                input_shapes, pred_logits, pred_boxes, pred_keypoints):
            bboxes, keypoints, keypoint_scores = self.data_decoder.decode(
                input_shape, pred_logit, pred_bbox, pred_kpts)

            # pack outputs
            preds.append(
                InstanceData(
                    keypoints=keypoints,
                    keypoint_scores=keypoint_scores,
                    bboxes=bboxes))

        return preds

    def predict(self,
                feats: Features,
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:
        """Predict results from features."""
        input_shapes = np.array(
            [d.metainfo['input_size'] for d in batch_data_samples])

        if test_cfg.get('flip_test', False):
            assert NotImplementedError(
                'flip_test is currently not supported '
                'for EDPose. Please set `model.test_cfg.flip_test=False`')
        else:
            pred_logits, pred_boxes, pred_keypoints = self.forward(
                feats, batch_data_samples)  # (B, K, D)

            pred = self.decode(
                input_shapes,
                pred_logits=pred_logits,
                pred_boxes=pred_boxes,
                pred_keypoints=pred_keypoints)
        return pred

    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: OptConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        outputs = self.forward(feats, batch_data_samples)

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        device = next(iter(outputs.values())).device

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in batch_data_samples)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)

        num_boxes = torch.clamp(num_boxes / dist.get_world_size(), min=1).item()

        # loss for final layer
        indices = self.matcher(outputs_without_aux, batch_data_samples)
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, batch_data_samples, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for idx, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, batch_data_samples)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, batch_data_samples, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # interm_outputs loss
        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']
            indices = self.matcher(interm_outputs, batch_data_samples)
            for loss in self.losses:
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs = {'log': False}
                l_dict = self.get_loss(loss, interm_outputs, batch_data_samples, indices, num_boxes, **kwargs)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses
