import copy
import math
import numpy as np
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mmcv.ops import MultiScaleDeformableAttention
from mmengine.model import BaseModule, ModuleList, constant_init
from mmengine.structures import InstanceData

from mmpose.models.utils import inverse_sigmoid
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, Features, OptConfigType,
                                 OptSampleList, Predictions)
from .base_transformer_head import TransformerHead
from .transformers.deformable_detr_layers import (
    DeformableDetrTransformerDecoderLayer, DeformableDetrTransformerEncoder, DeformableDetrTransformerDecoder)
from .transformers.utils import FFN, PositionEmbeddingSineHW
from ..base_head import BaseHead


@MODELS.register_module()
class RTDetrHead(BaseHead):
    def __init__(self,
                 num_queries: int = 300,
                 num_classea : int = 80,
                 encoder: OptConfigType = None,
                 decoder: OptConfigType = None,

                 positional_encoding: OptConfigType = None,  # 位置编码
                 data_decoder: OptConfigType = None,  # 读取gt
                 denosing_cfg: OptConfigType = None,
                 ):

        self.denosing_cfg = denosing_cfg

        self.num_heads = decoder['layer_cfg']['self_attn_cfg']['num_heads']

        super().__init__()

        self.decoder = DeformableDetrTransformerDecoder(**self.decoder_cfg)

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype=torch.float32,
                          device='cpu'):
        if spatial_shapes is None:
            spatial_shapes = [[int(self.eval_spatial_size[0] / s), int(self.eval_spatial_size[1] / s)]
                              for s in self.feat_strides
                              ]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid( \
                torch.arange(end=h, dtype=dtype), \
                torch.arange(end=w, dtype=dtype), indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], -1)
            valid_WH = torch.tensor([w, h]).to(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            anchors.append(torch.concat([grid_xy, wh], -1).reshape(-1, h * w, 4))

        anchors = torch.concat(anchors, 1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        # anchors = torch.where(valid_mask, anchors, float('inf'))
        # anchors[valid_mask] = torch.inf # valid_mask [1, 8400, 1]
        anchors = torch.where(valid_mask, anchors, torch.inf)

        return anchors, valid_mask

    def pre_decoder(self, memory: Tensor,
                    spatial_shapes: Tensor, input_query_bbox: Tensor,
                    input_query_label: Tensor,
                    ) -> Tuple[Dict, Dict]:

        bs, _, _ = memory.shape

        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors, valid_mask = self.anchors.to(memory.device), self.valid_mask.to(memory.device)

        memory = valid_mask.to(memory.dtype)
        # iou-aware query-selection
        output_memory = self.enc_output(memory)
        enc_outputs_class = self.enc_score_head(memory)
        enc_outputs_coord_unact = self.enc_bbox_head(memory) + anchors

        _, topk_idx = torch.topk(enc_outputs_class.max(-1).values, )

        reference_points_unact = enc_outputs_coord_unact.gather(dim=1, index=topk_idx.unsqueeze(-1).repeat(1, 1,
                                                                                                           enc_outputs_coord_unact.shape[
                                                                                                               -1]))

        enc_topk_bboxes = F.sigmoid(reference_points_unact)
        if input_query_bbox is not None:
            reference_points_unact = torch.concat(
                [input_query_bbox, reference_points_unact], 1)

        enc_topk_logits = enc_outputs_class.gather(dim=1, index=topk_idx.unsqueeze(-1).repeat(1, 1,
                                                                                              enc_outputs_class.shape[
                                                                                                  -1]))

        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = output_memory.gather(dim=1, \
                                          index=topk_idx.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
            target = target.detach()

        if input_query_label is not None:
            target = torch.concat([input_query_label, target], 1)

        decoder_inputs_dict = dict(
            query=target,
            reference_points=reference_points_unact.detach()
        )
        head_inputs_dict = dict(
            enc_topk_logits=enc_topk_logits, enc_topk_bboxes=enc_topk_bboxes)
        return decoder_inputs_dict, head_inputs_dict

    def prepare_for_denosing(self, targets: OptSampleList):

        if self.denosing_cfg['num_denoising'] <= 0:
            return None, None, None, None

        if not self.training:
            return None

        # gt_boxes = [t['boxes'] for t in targets]
        # gt_labels = [t['labels'] for t in targets]
        # gt_keypoints = [t['keypoints'] for t in targets]
        device = targets[0]['labels'].device
        refine_queries_num = self.refine_queries_num

        num_class_gts = [len(t['labels']) for t in targets]
        num_labels_gts = [t['labels'] for t in targets]
        num_boxes_gts = [t['boxes'] for t in targets]
        num_keypoints_gts = [t['keypoints'] for t in targets]

        max_class_gts = max(num_class_gts)

        num_group = self.denosing_cfg['num_denoising'] // max_class_gts
        num_group = 1 if num_group == 0 else num_group

        bs = len(num_class_gts)

        input_query_class = torch.full([bs, max_class_gts], num_class_gts, dtype=torch.int32, device=device)
        input_query_bbox = torch.zeros([bs, max_class_gts, 4])

        pad_gt_mask = torch.zeros([bs, max_class_gts], dtype=torch.bool, device=device)

        for i in range(bs):
            input_query_class = input_query_class.title([1, 2 * num_group])
            input_query_bbox = input_query_bbox.title([1, 2 * num_group, 1])
            pad_gt_mask = pad_gt_mask.title([1, 2 * num_group])
        # 生成mask
        negative_gt_mask = torch.zeros([bs, max_class_gts * 2, 1], device=device)
        negative_gt_mask[:, max_class_gts:] = 1
        negative_gt_mask = negative_gt_mask.title([1, num_group, 1])

        positive_gt_mask = 1 - negative_gt_mask
        positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
        # index
        dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
        dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_class_gts])

        # total denoising queries
        num_denoising = int(max_class_gts * 2 * num_group)

        # add nosing
        # 将label随机替换 gt
        if self.denosing_cfg['label_noise_ratio'] > 0:
            mask = torch.rand_like(input_query_class, dtype=float) < (self.denosing_cfg['label_noise_ratio'] * 0.5)
            new_label = torch.randint_like(mask, 0, num_class_gts, dtype=input_query_class.dtype)
            input_query_class = torch.where(mask & pad_gt_mask, new_label, input_query_class)

        if self.denosing_cfg['box_noise_scale'] > 0:
            known_bbox = self.box_cxcywh_to_xyxy(input_query_bbox)
            diff = torch.tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2]) * self.denosing_cfg['box_noise_scale']
            rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0
            rand_part = torch.rand_like(input_query_bbox)
            rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask)
            rand_part *= rand_sign
            known_bbox += rand_part * diff
            known_bbox.clip_(min=0.0, max=1.0)
            input_query_bbox = self.box_xyxy_to_cxcywh(known_bbox)
            input_query_bbox = inverse_sigmoid(input_query_bbox)

        input_query_class = self.denoising_class_embed(input_query_class)

        tgt_size = num_denoising + self.num_queries
        attn_mask = torch.full([tgt_size, tgt_size], False, dtype=torch.bool, device=device)
        # match query cannot see the reconstruction
        attn_mask[num_denoising:, :num_denoising] = True
        # reconstruct cannot see each other
        for i in range(num_group):
            if i == 0:
                attn_mask[max_class_gts * 2 * i: max_class_gts * 2 * (i + 1),
                max_class_gts * 2 * (i + 1): num_denoising] = True
            if i == num_group - 1:
                attn_mask[max_class_gts * 2 * i: max_class_gts * 2 * (i + 1), :max_class_gts * i * 2] = True
            else:
                attn_mask[max_class_gts * 2 * i: max_class_gts * 2 * (i + 1),
                max_class_gts * 2 * (i + 1): num_denoising] = True
                attn_mask[max_class_gts * 2 * i: max_class_gts * 2 * (i + 1), :max_class_gts * 2 * i] = True
        dn_meta = {
            "dn_positive_idx": dn_positive_idx,
            "dn_num_group": num_group,
            "dn_num_split": [num_denoising, self.num_queries]
        }

        return input_query_class, input_query_bbox, attn_mask, dn_meta

    def forward(self, img_feats: Tuple[Tensor]):
        proj_feats = [self.input_proj_layers[i](feat) for i, feat in enumerate(img_feats)]

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])
            # [B, C, H, W] -> [B, N, C], N=HxW
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1).contiguous())

        # [B, N, C], N = N_0 + N_1 + ...
        feat_flatten = torch.cat(feat_flatten, dim=1)
        level_start_index.pop()
        memory = feat_flatten

        # —————————————— decoder ————————————————
        if self.decoder_cfg['num_denoising'] > 0:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                self.prepare_for_denosing(kwargs.get('batch_data_samples'),
                                          self.decoder_cfg)
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        decoder_in, head_in = self.pre_decoder(memory,
                                               spatial_shapes,
                                               denoising_bbox_unact,
                                               denoising_class)

        out_logits, out_bboxes = self.decoder(query=decoder_in['query'].transpose(0, 1),
                                              value=memory,
                                              key_padding_mask=attn_mask,

                                              reference_points=decoder_in['reference_points'],
                                              spatial_shapes=spatial_shapes,
                                              level_start_index=level_start_index,
                                              valid_ratios=valid_ratios, )
        # ———————————————— head ——————————
        dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
        dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)

        out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]}

        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
            out['aux_outputs'].extend(self._set_aux_loss(head_in['enc_topk_logits'], head_in['enc_topk_bboxes']))

            if self.training and dn_meta is not None:
                out['dn_aux_outputs'] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
                out['dn_meta'] = dn_meta

        return out

    def decode(self, input_shapes: np.ndarray, pred_logits: Tensor,
               pred_boxes: Tensor, pred_keypoints: Tensor):
        """Select the final top-k keypoints, and decode the results from
        normalize size to origin input size.

        Args:
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
        pred_logits, pred_boxes = to_numpy(
            [pred_logits, pred_boxes])

        for input_shape, pred_logit, pred_bbox in zip(
                input_shapes, pred_logits, pred_boxes):
            bboxes, keypoints, keypoint_scores = self.data_decoder.decode(
                input_shape, pred_logit, pred_bbox)

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
            pred_logits, pred_boxes = self.forward(
                feats, batch_data_samples)  # (B, K, D)

            pred = self.decode(
                input_shapes,
                pred_logits=pred_logits,
                pred_boxes=pred_boxes,
            )
        return pred

    def loss(self,
             imgfeats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: OptConfigType = {}) -> dict:

        self.forward(imgfeats)
