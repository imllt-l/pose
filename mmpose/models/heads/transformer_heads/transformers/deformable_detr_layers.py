# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

import torch
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.ops import MultiScaleDeformableAttention
from mmdet.models import DinoTransformerDecoder
from mmengine.model import ModuleList
from torch import Tensor, nn
import torch.nn.functional as F
from mmpose.models.utils import inverse_sigmoid
from .detr_layers import (DetrTransformerDecoder, DetrTransformerDecoderLayer,
                          DetrTransformerEncoder, DetrTransformerEncoderLayer)
from mmpose.utils.typing import ConfigType, OptConfigType
from .utils import MLP


class DeformableDetrTransformerEncoder(DetrTransformerEncoder):
    """Transformer encoder of Deformable DETR."""

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            DeformableDetrTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims

    def forward(self, query: Tensor, query_pos: Tensor,
                key_padding_mask: Tensor, spatial_shapes: Tensor,
                level_start_index: Tensor, valid_ratios: Tensor,
                **kwargs) -> Tensor:
        """Forward function of Transformer encoder.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            query_pos (Tensor): The positional encoding for query, has shape
                (bs, num_queries, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (bs, num_queries).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).

        Returns:
            Tensor: Output queries of Transformer encoder, which is also
            called 'encoder output embeddings' or 'memory', has shape
            (bs, num_queries, dim)
        """
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        for layer in self.layers:
            query = layer(
                query=query,
                query_pos=query_pos,
                key_padding_mask=key_padding_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points,
                **kwargs)
        return query

    @staticmethod
    def get_encoder_reference_points(spatial_shapes: Tensor,
                                     valid_ratios: Tensor,
                                     device: Union[torch.device,
                                     str]) -> Tensor:
        """Get the reference points used in encoder.

        Args:
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            device (obj:`device` or str): The device acquired by the
                `reference_points`.

        Returns:
            Tensor: Reference points used in decoder, has shape (bs, length,
            num_levels, 2).
        """

        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (
                    valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (
                    valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        # [bs, sum(hw), num_level, 2]
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points


class DeformableDetrTransformerDecoder(DetrTransformerDecoder):
    """Transformer Decoder of Deformable DETR."""

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            DeformableDetrTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                value: Tensor,
                key_padding_mask: Tensor,
                reference_points: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                reg_branches: Optional[nn.Module] = None,
                **kwargs) -> Tuple[Tensor]:
        """Forward function of Transformer decoder.

        Args:
            query (Tensor): The input queries, has shape (bs, num_queries,
                dim).
            query_pos (Tensor): The input positional query, has shape
                (bs, num_queries, dim). It will be added to `query` before
                forward function.
            value (Tensor): The input values, has shape (bs, num_value, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `cross_attn`
                input. ByteTensor, has shape (bs, num_value).
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h) when `as_two_stage` is `True`, otherwise has
                shape (bs, num_queries, 2) with the last dimension arranged
                as (cx, cy).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`, optional): Used for refining
                the regression results. Only would be passed when
                `with_box_refine` is `True`, otherwise would be `None`.

        Returns:
            tuple[Tensor]: Outputs of Deformable Transformer Decoder.

            - output (Tensor): Output embeddings of the last decoder, has
              shape (num_queries, bs, embed_dims) when `return_intermediate`
              is `False`. Otherwise, Intermediate output embeddings of all
              decoder layers, has shape (num_decoder_layers, num_queries, bs,
              embed_dims).
            - reference_points (Tensor): The reference of the last decoder
              layer, has shape (bs, num_queries, 4)  when `return_intermediate`
              is `False`. Otherwise, Intermediate references of all decoder
              layers, has shape (num_decoder_layers, bs, num_queries, 4). The
              coordinates are arranged as (cx, cy, w, h)
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        for layer_id, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * \
                    torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * \
                    valid_ratios[:, None]
            output = layer(
                output,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                **kwargs)

            if reg_branches is not None:
                tmp_reg_preds = reg_branches[layer_id](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp_reg_preds + inverse_sigmoid(
                        reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp_reg_preds
                    new_reference_points[..., :2] = tmp_reg_preds[
                                                    ..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points


class DeformableDetrTransformerEncoderLayer(DetrTransformerEncoderLayer):
    """Encoder layer of Deformable DETR."""

    def _init_layers(self) -> None:
        """Initialize self_attn, ffn, and norms."""
        self.self_attn = MultiScaleDeformableAttention(**self.self_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(2)
        ]
        self.norms = ModuleList(norms_list)


class DeformableDetrTransformerDecoderLayer(DetrTransformerDecoderLayer):
    """Decoder layer of Deformable DETR."""

    def _init_layers(self) -> None:
        """Initialize self_attn, cross-attn, ffn, and norms."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.cross_attn = MultiScaleDeformableAttention(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(3)
        ]
        self.norms = ModuleList(norms_list)


class DeformableTransformerGroupDecoderLayer(nn.Module):
    def __init__(self,
                 self_attn_cfg: dict = dict(
                     embed_dim=256,
                     num_heads=8,
                     dropout=0.0,
                     batch_first=True),
                 cross_attn_cfg: dict = dict(
                     embed_dim=256,
                     num_head=8,
                     dropout=0.0,
                     batch_first=True),
                 ffn_cfg: dict = dict(
                     embed_dim=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.0,
                     act_cfg=dict(type='ReLU', inplace=True)),
                 norm_cfg: dict = dict(type='LN'),
                 init_cfg: dict = None) -> None:
        super().__init__()

    def forward_FFN(self, tgt):
        tgt2 = self.linear2(self.dropout2(self.activation_fn(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm2(tgt)
        return tgt

    def forward(self,
                tgt_pose: Optional[Tensor],
                tgt_pose_query_pos: Optional[Tensor],
                tgt_pose_reference_points: Optional[Tensor],
                memory: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                memory_level_start_index: Optional[Tensor] = None,
                memory_spatial_shapes: Optional[Tensor] = None,
                ):
        # within-instance self-attention
        q = k = with_pos_embed(tgt_pose, tgt_pose_query_pos)
        tgt2 = self.within_attn(q.flatten(0, 1).transpose(0, 1), k.flatten(0, 1).transpose(0, 1),
                                tgt_pose.flatten(0, 1).transpose(0, 1))[0].transpose(0, 1).reshape(q.shape)
        tgt_pose = tgt_pose + self.within_dropout(tgt2)
        tgt_pose = self.within_norm(tgt_pose)
        # across-instance self-attention
        q_pose = k_pose = tgt_pose
        tgt2_pose = self.across_attn(q_pose.flatten(1, 2), k_pose.flatten(1, 2), tgt_pose.flatten(1, 2))[0].reshape(
            q_pose.shape)
        tgt_pose = tgt_pose + self.across_dropout(tgt2_pose)
        tgt_pose = self.across_norm(tgt_pose)
        # deformable cross-attention
        nq, bs, np, d_model = tgt_pose.shape
        tgt2_pose = self.cross_attn(with_pos_embed(tgt_pose, tgt_pose_query_pos).transpose(0, 1).flatten(1, 2),
                                    tgt_pose_reference_points.transpose(0, 1),
                                    memory.transpose(0, 1), memory_spatial_shapes,
                                    memory_level_start_index, memory_key_padding_mask).reshape(bs, nq, np,
                                                                                               d_model).transpose(0,
                                                                                                                  1)
        tgt_pose = tgt_pose + self.dropout1(tgt2_pose)
        tgt_pose = self.norm1(tgt_pose)
        tgt_pose = self.forward_FFN(tgt_pose)
        return tgt_pose


class RTDETRTransformerDecoder(DinoTransformerDecoder):
    """Transformer decoder used in `RT-DETR.

    <https://arxiv.org/abs/2304.08069>`_.

    Args:
        eval_idx (int): The index of the decoder layer to be evaluated. If
            `eval_idx` is negative, the index will be calculated as
            `num_layers + eval_idx`. Default: -1.
    """

    def __init__(self, eval_idx: int = -1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if eval_idx < 0:
            eval_idx = self.num_layers + eval_idx
        self.eval_idx = eval_idx

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        super(DinoTransformerDecoder, self)._init_layers()
        self.ref_point_head = MLP(4, self.embed_dims * 2, self.embed_dims, 2)

    def forward(self, query: Tensor, value: Tensor, key_padding_mask: Tensor,
                self_attn_mask: Tensor, reference_points: Tensor,
                spatial_shapes: Tensor, level_start_index: Tensor,
                valid_ratios: Tensor, reg_branches: nn.ModuleList,
                cls_branches: nn.ModuleList, **kwargs) -> Tuple[Tensor]:
        """Forward function of Transformer decoder.

        Args:
            query (Tensor): The input query, has shape (num_queries, bs, dim).
            value (Tensor): The input values, has shape (num_value, bs, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (num_queries, bs).
            self_attn_mask (Tensor): The attention mask to prevent information
                leakage from different denoising groups and matching parts, has
                shape (num_queries_total, num_queries_total). It is `None` when
                `self.training` is `False`.
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`): Used for refining the
                regression results.

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
        assert reg_branches is not None
        assert cls_branches is not None
        out_bboxes = []
        out_logits = []
        ref_points = None
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points.unsqueeze(2)
            query_pos = self.ref_point_head(reference_points)

            query = layer(
                query,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                self_attn_mask=self_attn_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                **kwargs)

            tmp = reg_branches[lid](query)
            assert reference_points.shape[-1] == 4
            new_reference_bboxes = tmp + inverse_sigmoid(
                reference_points, eps=1e-3)
            new_reference_bboxes = new_reference_bboxes.sigmoid()

            if self.training:
                out_logits.append(cls_branches[lid](query))
                if lid == 0:
                    out_bboxes.append(new_reference_bboxes)
                else:
                    new_bboxes = tmp + inverse_sigmoid(ref_points, eps=1e-3)
                    new_bboxes = new_bboxes.sigmoid()
                    out_bboxes.append(new_bboxes)
            elif lid == self.eval_idx:
                out_logits.append(cls_branches[lid](query))
                out_bboxes.append(new_reference_bboxes)
                break

            ref_points = new_reference_bboxes
            reference_points = new_reference_bboxes.detach(
            ) if self.training else new_reference_bboxes

        return torch.stack(out_logits), torch.stack(out_bboxes)


def with_pos_embed(tensor, pos):
    if pos is not None:
        np = pos.shape[2]
        tensor[:, :, -np:] += pos
    return tensor
