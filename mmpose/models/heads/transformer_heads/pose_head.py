
import copy
import math
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.ops import MultiScaleDeformableAttention
from mmengine.model import BaseModule, ModuleList, constant_init
from mmengine.structures import InstanceData
from torch import Tensor, nn

from mmpose.models.utils import inverse_sigmoid
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, Features, OptConfigType,
                                 OptSampleList, Predictions)
from .base_transformer_head import TransformerHead
from .transformers.deformable_detr_layers import (
    DeformableDetrTransformerDecoderLayer, DeformableDetrTransformerEncoder)
from .transformers.utils import FFN, PositionEmbeddingSineHW

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class RTDETRDecoder(BaseModule):
    """

    Args:
        layer_cfg (ConfigDict): the config of each encoder
            layer. All the layers will share the same config.
        num_levels (int): Number of feature levels.
        return_intermediate (bool, optional): Whether to return outputs of intermediate layers. Defaults to `True`.
        embed_dims (int): Dims of embed.
        query_dim (int): Dims of queries.
        num_feature_levels (int): Number of feature levels.
        num_box_decoder_layers (int): Number of box decoder layers.
        num_keypoints (int): Number of datasets' body keypoints.
        num_dn (int): Number of denosing points.
        num_group (int): Number of decoder layers.
    """

# layer_cfg=dict(  # DeformableDetrTransformerDecoderLayer
#     self_attn_cfg=dict(  # MultiheadAttention
#         embed_dims=256,
#         num_heads=8,
#         batch_first=True),
#     cross_attn_cfg=dict(  # MultiScaleDeformableAttention
#         embed_dims=256,
#         batch_first=True),
#     ffn_cfg=dict(
#         embed_dims=256, feedforward_channels=2048, ffn_drop=0.1)),

# self.decoder = DeformableTransformerDecoder(
#                             d_model    = hidden_dim,
#                             num_heads  = num_heads,
#                             num_layers = num_layers,
#                             num_levels = num_levels,
#                             num_points = num_points,
#                             ffn_dim  = ffn_dim,
#                             dropout    = dropout,
#                             act_type   = act_type,
#                             return_intermediate = return_intermediate
#                         )
    def __init__(self,
                 layer_cfg,
                 num_layers,
                 return_intermediate,
                 embed_dims: int = 256,
                 query_dim=4,
                 num_feature_levels=1,
                 num_box_decoder_layers=2,
                 num_keypoints=17,
                 num_dn=100,
                 num_group=100):
        super().__init__()
    ## init self cfg
        self.layer_cfg = layer_cfg
        self.num_layers = num_layers
        

        self.embed_dims = embed_dims
        self.query_dim = query_dim
        self.pose_embed = None
        self.pose_hw_embed = None

    #--- keypoint cfg ---
        self.keypoint_embed = nn.Embedding(self.num_keypoints, embed_dims)
        self.kpt_index = [
            x for x in range(self.num_group * (self.num_keypoints + 1))
            if x % (self.num_keypoints + 1) != 0
        ]

        self.num_layers = num_layers
        assert return_intermediate,'support return_intermediate only'
        self.return_intermediate = return_intermediate
        
        self.decoder_layers = ModuleList([
            DeformableDetrTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.norm = nn.LayerNorm(self.embed_dims)
        
        self.ref_point_head = FFN(self.query_dim // 2 * self.embed_dims,
                                  self.embed_dims, self.embed_dims, 2)
        self.query_pos_head = MLP(4, 2 * self.embed_dims, self.embed_dims, num_layers=2)

    def forward(self, query: Tensor, value: Tensor, key_padding_mask: Tensor,
                reference_points: Tensor, spatial_shapes: Tensor,
                level_start_index: Tensor, valid_ratios: Tensor,
                humandet_attn_mask: Tensor, human2pose_attn_mask: Tensor,
                **kwargs) -> Tuple[Tensor]:
        output = query
        dec_out_bboxes = []
        dec_out_logits = []

        reference_points_detach = F.sigmoid(reference_points)

        for layer_id, layer in enumerate(self.decoder_layers):
            reference_points_input = reference_points_detach.unsqueeze(2)
            query_pose_embed = self.query_pose_head(reference_points_detach)

            output = layer(
                output,
                value = value,
                spatial_shapes = spatial_shapes,
                reference_points = reference_points,
                query_pos =  query_pose_embed,
                level_start_index=level_start_index,
                key_padding_mask = key_padding_mask,
                **kwargs
                )
            inter_ref_bbox = F.sigmoid(bbox_head[i])
        
        return decoder_outputs, reference_points


@MODELS.register_module()
class RTDETRHead(TransformerHead):
    def __init__(self,
                 num_queries: int = 100,
                 num_feature_levels: int = 4,
                 num_keypoints: int = 17,
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

        self.denosing_cfg = denosing_cfg

        self.embed_dims = self.encoder.embed_dims
        if data_decoder is not None:
            self.data_decoder = KEYPOINT_CODECS.build(data_decoder)
        else:
            self.data_decoder = None

        if self.denosing_cfg['num_denoising'] > 0: 
            # self.denoising_class_embed = nn.Embedding(num_classes, hidden_dim, padding_idx=num_classes-1) # TODO for load paddle weights
            self.denoising_class_embed = nn.Embedding(self.denosing_cfg['dn_labelbook_size'] + 1, self.embed_dims, padding_idx=self.denosing_cfg['dn_labelbook_size'])
        
        self.enc_output = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims,)
        )
        self.enc_score_head = nn.Linear(self.embed_dims, out_head.num_classes)
        self.enc_bbox_head = MLP(self.embed_dims, self.embed_dims, 4, num_layers=3)

        self.decoder = RTDETRDecoder()
# out_head cfg{
# num_class
# }
    def box_cxcywh_to_xyxy(x):
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)
    def box_xyxy_to_cxcywh(x):
        x0, y0, x1, y1 = x.unbind(-1)
        b = [(x0 + x1) / 2, (y0 + y1) / 2,
            (x1 - x0), (y1 - y0)]
        return torch.stack(b, dim=-1)
    def _get_encoder_input(self,feats):
        proj_feats = [ self.input_proj[i](feat) for i,feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]

        for i ,feat in enumerate(proj_feats):
            _,_,h,w = feat.shape
            feat_flatten.append(feat.flatten[2].permute(0,2,1))
            spatial_shapes.append([h,w])
            level_start_index.append(h * w + level_start_index[-1])

        feat_flatten = torch.concat(feat_flatten,1)
        level_start_index.pop()
        return (feat_flatten,spatial_shapes,level_start_index)
# denosing_cfg{
# num_denoising=100,
# label_noise_ratio=0.5,
# box_noise_scale=1.0
# dn_labelbook_size = 100}

    def prepare_for_denosing(self,targets: OptSampleList):
        
        if self.denosing_cfg['num_denoising'] <= 0:
            return None, None, None, None
    
        if not self.training:
            return None
        
        # gt_boxes = [t['boxes'] for t in targets]
        # gt_labels = [t['labels'] for t in targets]
        # gt_keypoints = [t['keypoints'] for t in targets]
        device = targets[0]['boxes'].device
        refine_queries_num = self.refine_queries_num

        num_class_gts = [len(t['labels']) for t in targets]
        num_labels_gts = [t['labels'] for t in targets]
        num_boxes_gts = [t['boxes'] for t in targets]
        num_keypoints_gts = [t['keypoints'] for t in targets]

        max_class_gts = max(num_class_gts)

        num_group = self.denosing_cfg['num_denoising'] // max_class_gts
        num_group = 1 if num_group == 0 else num_group

        bs = len(num_class_gts)

        input_query_class = torch.full([bs,max_class_gts],num_class_gts,dtype =torch.int32,device =device)
        input_query_bbox = torch.zeros([bs,max_class_gts,4])

        pad_gt_mask = torch.zeros([bs,max_class_gts],dtype = torch.bool,device = device)

        for i in range(bs):
            input_query_class = input_query_class.title([1,2*num_group])
            input_query_bbox = input_query_bbox.title([1,2 * num_group ,1])
            pad_gt_mask = pad_gt_mask.title([1,2*num_group])
    # 生成mask
        negative_gt_mask = torch.zeros([bs,max_class_gts *2 ,1],device = device)
        negative_gt_mask[:, max_class_gts:] = 1 
        negative_gt_mask = negative_gt_mask.title([1,num_group,1])

        positive_gt_mask = 1 - negative_gt_mask
        positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
    # index
        dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
        dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts])
   
    # total denoising queries
        num_denoising = int(max_class_gts * 2 * num_group)
    
    # add nosing
        # 将label随机替换 gt
        if self.denosing_cfg['label_noise_ratio'] >0:
            mask = torch.rand_like(input_query_class,dtype=float) < ( self.denosing_cfg['label_noise_ratio'] * 0.5)
            new_label = torch.randint_like(mask ,0,num_class_gts,dtype= input_query_class.dtype)
            input_query_class = torch.where(mask &  pad_gt_mask,new_label, input_query_class)
        
        if self.denosing_cfg['box_noise_scale'] > 0:
            known_bbox =  self.box_cxcywh_to_xyxy(input_query_bbox)
            diff = torch.tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2]) * box_noise_scale
            rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0
            rand_part = torch.rand_like(input_query_bbox)
            rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask)
            rand_part *= rand_sign
            known_bbox += rand_part * diff
            known_bbox.clip_(min=0.0, max=1.0)
            input_query_bbox = self.box_xyxy_to_cxcywh(known_bbox)
            input_query_bbox = inverse_sigmoid(input_query_bbox)

        input_query_class = self.denoising_class_embed(input_query_class)
    
        tgt_size = num_denoising +self.num_queries
        attn_mask = torch.full([tgt_size, tgt_size], False, dtype=torch.bool, device=device)
        # match query cannot see the reconstruction
        attn_mask[num_denoising:, :num_denoising] = True
            # reconstruct cannot see each other
        for i in range(num_group):
            if i == 0:
                attn_mask[max_class_gts * 2 * i: max_class_gts * 2 * (i + 1), max_class_gts * 2 * (i + 1): num_denoising] = True
            if i == num_group - 1:
                attn_mask[max_class_gts * 2 * i: max_class_gts * 2 * (i + 1), :max_class_gts * i * 2] = True
            else:
                attn_mask[max_class_gts * 2 * i: max_class_gts * 2 * (i + 1), max_class_gts * 2 * (i + 1): num_denoising] = True
                attn_mask[max_class_gts * 2 * i: max_class_gts * 2 * (i + 1), :max_class_gts * 2 * i] = True
        dn_meta = {
                "dn_positive_idx": dn_positive_idx,
                "dn_num_group": num_group,
                "dn_num_split": [num_denoising, self.num_queries]
            }

        return input_query_class, input_query_bbox, attn_mask, dn_meta
    
    
    def forward_encoder(self,
                        img_feats: Tuple[Tensor],
                        batch_data_samples: OptSampleList = None) -> Dict:
        
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
        encoder_outputs_dict = dict(
            memory = memory,
            feat_flatten = feat_flatten,
            spatial_shapes =spatial_shapes,
            level_start_index = level_start_index
        )

        return encoder_outputs_dict
    
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
            grid_y, grid_x = torch.meshgrid(\
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

    def pre_decoder(self, memory: Tensor, memory_mask: Tensor,
                    spatial_shapes: Tensor, input_query_bbox: Tensor,
                    input_query_label: Tensor,
                    ) -> Tuple[Dict, Dict]:
        
        bs,_,_ = memory.shape

        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors, valid_mask = self.anchors.to(memory.device), self.valid_mask.to(memory.device)

        memory = valid_mask.to(memory.dtype)
        # iou-aware query-selection 
        output_memory = self.enc_output(memory)
        enc_outputs_class = self.enc_score_head(memory)
        enc_outputs_coord_unact = self.enc_bbox_head(memory) +anchors

        _,topk_idx = torch.topk(enc_outputs_class.max(-1).values,)
        
        reference_points_unact = enc_outputs_coord_unact.gather(dim=1, \
                index=topk_idx.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_unact.shape[-1]))

        enc_topk_bboxes = F.sigmoid(reference_points_unact)
        if input_query_bbox is not None:
            reference_points_unact = torch.concat(
                [input_query_bbox, reference_points_unact], 1)
        
        enc_topk_logits = enc_outputs_class.gather(dim=1, \
            index=topk_idx.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1]))

        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = output_memory.gather(dim=1, \
                index=topk_idx.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
            target = target.detach()

        if input_query_label is not None:
            target = torch.concat([input_query_label, target], 1)

        return target, reference_points_unact.detach(), enc_topk_bboxes, enc_topk_logits

    def forward_decoder(self, memory: Tensor, memory_mask: Tensor,
                        spatial_shapes: Tensor, level_start_index: Tensor,
                        valid_ratios: Tensor, attn_mask: Tensor, input_query_bbox: Tensor,
                        input_query_label: Tensor, mask_dict: Dict) -> Dict:
        decoder_in, head_in = self.pre_decoder(memory, memory_mask,
                                               spatial_shapes,
                                               input_query_bbox,
                                               input_query_label)

        decoder_outputs_dict = self.decoder(decoder_in, head_in, attn_mask, mask_dict)
        return decoder_outputs_dict