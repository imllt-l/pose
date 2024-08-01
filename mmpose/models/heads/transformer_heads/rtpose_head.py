import copy
import math
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.cnn import Linear
from mmcv.ops import MultiScaleDeformableAttention
from mmdet.models import DETRHead, CdnQueryGenerator
from mmdet.utils import OptInstanceList
from mmengine.model import BaseModule, ModuleList, constant_init
from mmengine.structures import InstanceData
from torch import Tensor, nn

from mmpose.models.utils import inverse_sigmoid
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, Features, OptConfigType,
                                 OptSampleList, Predictions, InstanceList)
from .base_transformer_head import TransformerHead
from .transformers.deformable_detr_layers import (
    DeformableDetrTransformerDecoderLayer, DeformableDetrTransformerEncoder, RTDETRTransformerDecoder)
from .transformers.utils import FFN, PositionEmbeddingSineHW


class OutHead(BaseModule):
    """Head of DeformDETR: Deformable DETR: Deformable Transformers for
    End-to-End Object Detection.

    Code is modified from the `official github repo
    <https://github.com/fundamentalvision/Deformable-DETR>`_.

    More details can be found in the `paper
    <https://arxiv.org/abs/2010.04159>`_ .

    Args:
        share_pred_layer (bool): Whether to share parameters for all the
            prediction layers. Defaults to `False`.
        num_pred_layer (int): The number of the prediction layers.
            Defaults to 6.
        as_two_stage (bool, optional): Whether to generate the proposal
            from the outputs of encoder. Defaults to `False`.
    """

    def __init__(self,
                 *args,
                 num_classes: int,
                 embed_dims: int = 256,
                 num_reg_fcs: int = 2,
                 sync_cls_avg_factor: bool = False,
                 share_pred_layer: bool = False,
                 num_pred_layer: int = 6,
                 as_two_stage: bool = False,
                 **kwargs) -> None:
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_reg_fcs = num_reg_fcs
        self.share_pred_layer = share_pred_layer
        self.num_pred_layer = num_pred_layer
        self.as_two_stage = as_two_stage
        super().__init__(*args, **kwargs)
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize classification branch and regression branch of head."""
        # fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        _bbox_embed = FFN(self.embed_dims, self.embed_dims, 4, 3)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 4))
        reg_branch = nn.Sequential(*reg_branch)

        if self.share_pred_layer:
            self.cls_branches = nn.ModuleList(
                [_bbox_embed for _ in range(self.num_pred_layer)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(self.num_pred_layer)])
        else:
            self.cls_branches = nn.ModuleList(
                [copy.deepcopy(_bbox_embed) for _ in range(self.num_pred_layer)])
            self.reg_branches = nn.ModuleList([
                copy.deepcopy(reg_branch) for _ in range(self.num_pred_layer)
            ])

    def init_weights(self) -> None:
        """Initialize weights of the Deformable DETR head."""
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)
        if self.as_two_stage:
            for m in self.reg_branches:
                nn.init.constant_(m[-1].bias.data[2:], 0.0)

    def forward(self, hidden_states: Tensor,
                references: List[Tensor]) -> Tuple[Tensor]:
        """Forward function.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, bs, num_queries, dim).
            references (list[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries, 4) when `as_two_stage` of the detector is `True`,
                otherwise (bs, num_queries, 2). Each `inter_reference` has
                shape (bs, num_queries, 4) when `with_box_refine` of the
                detector is `True`, otherwise (bs, num_queries, 2). The
                coordinates are arranged as (cx, cy) when the last dimension is
                2, and (cx, cy, w, h) when it is 4.

        Returns:
            tuple[Tensor]: results of head containing the following tensor.

            - all_layers_outputs_classes (Tensor): Outputs from the
              classification head, has shape (num_decoder_layers, bs,
              num_queries, cls_out_channels).
            - all_layers_outputs_coords (Tensor): Sigmoid outputs from the
              regression head with normalized coordinate format (cx, cy, w,
              h), has shape (num_decoder_layers, bs, num_queries, 4) with the
              last dimension arranged as (cx, cy, w, h).
        """
        all_layers_outputs_classes = []
        all_layers_outputs_coords = []

        for layer_id in range(hidden_states.shape[0]):
            reference = inverse_sigmoid(references[layer_id])
            # NOTE The last reference will not be used.
            hidden_state = hidden_states[layer_id]
            outputs_class = self.cls_branches[layer_id](hidden_state)
            tmp_reg_preds = self.reg_branches[layer_id](hidden_state)
            if reference.shape[-1] == 4:
                # When `layer` is 0 and `as_two_stage` of the detector
                # is `True`, or when `layer` is greater than 0 and
                # `with_box_refine` of the detector is `True`.
                tmp_reg_preds += reference
            else:
                # When `layer` is 0 and `as_two_stage` of the detector
                # is `False`, or when `layer` is greater than 0 and
                # `with_box_refine` of the detector is `False`.
                assert reference.shape[-1] == 2
                tmp_reg_preds[..., :2] += reference
            outputs_coord = tmp_reg_preds.sigmoid()
            all_layers_outputs_classes.append(outputs_class)
            all_layers_outputs_coords.append(outputs_coord)

        all_layers_outputs_classes = torch.stack(all_layers_outputs_classes)
        all_layers_outputs_coords = torch.stack(all_layers_outputs_coords)

        return all_layers_outputs_classes, all_layers_outputs_coords


@MODELS.register_module()
class RTPoseHead(TransformerHead):
    def __init__(self,
                 num_queries: int = 100,
                 eval_size: Tuple[int, int] = None,
                 feat_strides: List[int] = [8, 16, 32],
                 encoder: OptConfigType = None,
                 decoder: OptConfigType = None,
                 out_head: OptConfigType = None,
                 positional_encoding: OptConfigType = None,
                 data_decoder: OptConfigType = None,
                 dn_cfg: OptConfigType = None,

                 loss_cls: ConfigType = dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_bbox: ConfigType = dict(type='L1Loss', loss_weight=5.0),
                 loss_iou: ConfigType = dict(type='GIoULoss', loss_weight=2.0),
                 **kwargs) -> None:
        self.eval_size = eval_size
        self.eval_idx = -1

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            out_head=out_head,
            positional_encoding=positional_encoding,
            num_queries=num_queries)
        self._init_layers()
        num_levels = self.decoder.layer_cfg.cross_attn_cfg.num_levels
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)
        self.feat_strides = feat_strides

        if self.eval_size:
            self.proposals, self.valid_mask = self.generate_proposals()
        if data_decoder is not None:
            self.data_decoder = KEYPOINT_CODECS.build(data_decoder)
        else:
            self.data_decoder = None

        self.decoder.bbox_embed = self.out_head.reg_branches
        self.decoder.class_embed = self.out_head.cls_branches
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_iou = MODELS.build(loss_iou)

        if dn_cfg is not None:
            assert 'num_classes' not in dn_cfg and \
                   'num_queries' not in dn_cfg and \
                   'hidden_dim' not in dn_cfg, \
                'The three keyword args `num_classes`, `embed_dims`, and ' \
                '`num_matching_queries` are set in `detector.__init__()`, ' \
                'users should not set them in `dn_cfg` config.'
            dn_cfg['num_classes'] = self.out_head.num_classes
            dn_cfg['embed_dims'] = self.embed_dims
            dn_cfg['num_matching_queries'] = self.num_queries
        self.dn_query_generator = CdnQueryGenerator(**dn_cfg)

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = PositionEmbeddingSineHW(
            **self.positional_encoding_cfg)
        self.decoder = RTDETRTransformerDecoder(**self.decoder_cfg)
        self.out_head = OutHead(**self.out_head_cfg)
        self.embed_dims = self.decoder.embed_dims
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

    @property
    def default_init_cfg(self):
        init_cfg = [dict(type='Normal', layer=['Linear'], std=0.01, bias=0)]
        return init_cfg

    def pre_transformer(
            self,
            mlvl_feats: Tuple[Tensor],
            batch_data_samples: OptSampleList = None) -> Tuple[Dict]:
        """Process image features before feeding them to the transformer.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            mlvl_feats (tuple[Tensor]): Multi-level features that may have
                different resolutions, output from neck. Each feature has
                shape (bs, dim, h_lvl, w_lvl), where 'lvl' means 'layer'.
            batch_data_samples (list[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[dict]: The first dict contains the inputs of encoder and the
            second dict contains the inputs of decoder.

            - encoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_encoder()`, which includes 'feat' and
              'spatial_shapes'.
            - decoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_decoder()`, which includes 'spatial_shapes' and
                'level_start_index'.
        """
        batch_size = mlvl_feats[0].size(0)

        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0]
        for lvl, feat in enumerate(mlvl_feats):
            batch_size, c, h, w = feat.shape
            # [bs, c, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl, c]
            feat = feat.view(batch_size, c, -1).permute(0, 2, 1)
            # [num_levels, 2]
            spatial_shape = (h, w)
            # [l], start index of each level

            feat_flatten.append(feat)
            spatial_shapes.append(spatial_shape)
            level_start_index.append(h * w + level_start_index[-1])

        # (bs, num_feat_points, dim)
        feat_flatten = torch.cat(feat_flatten, 1)
        level_start_index.pop()
        level_start_index = torch.as_tensor(
            level_start_index, dtype=torch.long, device=feat_flatten.device)

        # (num_level, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)

        encoder_inputs_dict = dict(
            feat=feat_flatten,
            spatial_shapes=spatial_shapes,
        )
        decoder_inputs_dict = dict(
            spatial_shapes=spatial_shapes, level_start_index=level_start_index)
        return encoder_inputs_dict, decoder_inputs_dict

    def pre_decoder(
            self,
            memory: Tensor,
            spatial_shapes: Tensor,
            batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        """Prepare intermediate variables before entering Transformer decoder,
        such as `query`, `query_pos`, and `reference_points`.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            spatial_shapes (Tensor): Spatial shapes of features in all levels.
                With shape (num_levels, 2), last dimension represents (h, w).
                Will only be used when `as_two_stage` is `True`.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[dict]: The decoder_inputs_dict and head_inputs_dict.

            - decoder_inputs_dict (dict): The keyword dictionary args of
              `self.forward_decoder()`, which includes 'query', 'memory',
              `reference_points`, and `dn_mask`. The reference points of
              decoder input here are 4D boxes, although it has `points`
              in its name.
            - head_inputs_dict (dict): The keyword dictionary args of the
              bbox_head functions, which includes `topk_score`, `topk_coords`,
              and `dn_meta` when `self.training` is `True`, else is empty.
        """
        bs, _, c = memory.shape
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].out_features

        if self.training or self.eval_size is None:
            output_proposals, valid_mask = self.generate_proposals(
                spatial_shapes, device=memory.device)
        else:
            output_proposals = self.proposals.to(memory.device)
            valid_mask = self.valid_mask.to(memory.device)
        original_memory = memory
        memory = torch.where(valid_mask, memory, memory.new_zeros(1))
        output_memory = self.memory_trans_fc(memory)
        output_memory = self.memory_trans_norm(output_memory)

        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](
            output_memory)
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
                                      self.decoder.num_layers](output_memory) + output_proposals

        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]
        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        query = torch.gather(output_memory, 1,
                             topk_indices.unsqueeze(-1).repeat(1, 1, c))
        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query.detach()], dim=1)
            reference_points = torch.cat(
                [dn_bbox_query, topk_coords_unact],
                dim=1).detach()  # DINO does not use detach
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=original_memory,
            reference_points=reference_points,
            dn_mask=dn_mask)

        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta) if self.training else dict()
        return decoder_inputs_dict, head_inputs_dict

    def generate_proposals(self,
                           spatial_shapes: Tensor = None,
                           device: Optional[torch.device] = None,
                           grid_size: float = 0.05) -> Tuple[Tensor, Tensor]:
        """Generate proposals from spatial shapes.

        Args:
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
                Defaults to None.
            device (str | torch.device): The device where the anchors will be
                put on. Defaults to None.
            grid_size (float): The grid size of the anchors. Defaults to 0.05.

        Returns:
            tuple: A tuple of proposals and valid masks.

            - proposals (Tensor): The proposals of the detector, has shape
                (bs, num_proposals, 4).
            - valid_masks (Tensor): The valid masks of the proposals, has shape
                (bs, num_proposals).
        """

        if spatial_shapes is None:
            spatial_shapes = [[
                int(self.eval_size[0] / s),
                int(self.eval_size[1] / s)
            ] for s in self.feat_strides]

        proposals = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(
                    0, H - 1, H, dtype=torch.float32, device=device),
                torch.linspace(
                    0, W - 1, W, dtype=torch.float32, device=device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            valid_wh = torch.tensor([H, W], dtype=torch.float32, device=device)
            grid = (grid.unsqueeze(0) + 0.5) / valid_wh
            wh = torch.ones_like(grid) * grid_size * (2.0 ** lvl)
            proposals.append(torch.cat((grid, wh), -1).view(-1, H * W, 4))

        proposals = torch.cat(proposals, 1)
        valid_masks = ((proposals > 0.01) * (proposals < 0.99)).all(
            -1, keepdim=True)
        proposals = torch.log(proposals / (1 - proposals))
        proposals = proposals.masked_fill(~valid_masks, float('inf'))
        return proposals, valid_masks

    def forward_encoder(self, feat: Tensor, spatial_shapes: Tensor, **kwargs) -> Dict:
        return dict(memory=feat, spatial_shapes=spatial_shapes)

    def forward_decoder(self,
                        query: Tensor,
                        memory: Tensor,
                        reference_points: Tensor,
                        spatial_shapes: Tensor,
                        level_start_index: Tensor,
                        valid_ratios: Tensor = None,
                        memory_mask: Tensor = None,
                        dn_mask: Optional[Tensor] = None) -> Dict:
        """Forward with Transformer decoder.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            query (Tensor): The queries of decoder inputs, has shape
                (bs, num_queries_total, dim), where `num_queries_total` is the
                sum of `num_denoising_queries` and `num_matching_queries` when
                `self.training` is `True`, else `num_matching_queries`.
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries_total, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2). Defaults to None.
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points). Defaults to None.
            dn_mask (Tensor, optional): The attention mask to prevent
                information leakage from different denoising groups and
                matching parts, will be used as `self_attn_mask` of the
                `self.decoder`, has shape (num_queries_total,
                num_queries_total).
                It is `None` when `self.training` is `False`.

        Returns:
            dict: The dictionary of decoder outputs, which includes the
            `out_logits` and `out_bboxes` of the decoder output.
        """
        out_logits, out_bboxes = self.decoder(
            query=query,
            value=memory,
            key_padding_mask=memory_mask,
            self_attn_mask=dn_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=self.bbox_head.reg_branches,
            cls_branches=self.bbox_head.cls_branches)

        decoder_outputs_dict = dict(
            hidden_states=out_logits, references=out_bboxes)
        return decoder_outputs_dict

    def forward_out_head(self, query: Tensor, query_pos: Tensor, memory: Tensor, **kwargs) -> Dict:
        """Forward function."""
        out = self.out_head()
        return out

    def predict(self, feats: Features, batch_data_samples: OptSampleList,
                test_cfg: OptConfigType = {}) -> Predictions:
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        outs = self.forward(feats, batch_data_samples)

        predictions = self.decode(
            *outs, batch_img_metas=batch_img_metas, rescale=rescale)
        return predictions

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

        """
          def add_pred_to_datasample(self, data_samples: SampleList,
                                 results_list: InstanceList) -> SampleList:
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

    def loss_by_feat(
            self,
            all_layers_cls_scores: Tensor,
            all_layers_bbox_preds: Tensor,
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        """"Loss function.

        Only outputs from the last feature level are used for computing
        losses by default.

        Args:
            all_layers_cls_scores (Tensor): Classification outputs
                of each decoder layers. Each is a 4D-tensor, has shape
                (num_decoder_layers, bs, num_queries, cls_out_channels).
            all_layers_bbox_preds (Tensor): Sigmoid regression
                outputs of each decoder layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                (num_decoder_layers, bs, num_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert batch_gt_instances_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            'for batch_gt_instances_ignore setting to None.'

        losses_cls, losses_bbox, losses_iou = multi_apply(
            self.loss_by_feat_single,
            all_layers_cls_scores,
            all_layers_bbox_preds,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i in \
                zip(losses_cls[:-1], losses_bbox[:-1], losses_iou[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            num_dec_layer += 1
        return loss_dict

    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: OptConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        outs = self.forward(hidden_states)
        loss_inputs = outs + (batch_gt_instances, batch_img_metas)
        losses = self.loss_by_feat(*loss_inputs)
        return losses


"""
cls_scores = all_layers_cls_scores[-1]
bbox_preds = all_layers_bbox_preds[-1]

result_list = []
for img_id in range(len(batch_img_metas)):
cls_score = cls_scores[img_id]
bbox_pred = bbox_preds[img_id]
img_meta = batch_img_metas[img_id]
results = self._predict_by_feat_single(cls_score, bbox_pred,
                                       img_meta, rescale)
result_list.append(results)
return result_list
"""
