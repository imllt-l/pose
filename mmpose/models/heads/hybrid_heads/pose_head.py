   
import copy
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule, bias_init_with_prob
from mmengine.structures import InstanceData
from torch import Tensor

from mmpose.evaluation.functional import nms_torch
from mmpose.models.utils import filter_scores_and_topk
from mmpose.registry import MODELS, TASK_UTILS
from mmpose.structures import PoseDataSample
from mmpose.utils import reduce_mean
from mmpose.utils.typing import (ConfigType, Features, OptSampleList,
                                 Predictions, SampleList)


class YOLOV8PoseHeadModule(BaseModule):
    def __init__(
            self,
            num_keypoints: int,
            in_channels: Union[int, Sequence],
            num_classes: int = 1,
            widen_factor: float = 1.0,
            feat_channels: int = 256,
            stacked_convs: int = 2,
            featmap_strides: Sequence[int] = [8, 16, 32],
            conv_bias: Union[bool, str] = 'auto',
            conv_cfg: Optional[ConfigType] = None,
            norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: ConfigType = dict(type='SiLU', inplace=True),
            init_cfg: Optional[ConfigType] = None,
        ):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.featmap_strides = featmap_strides

        if isinstance(in_channels, int):
            in_channels = int(in_channels * widen_factor)
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints

        self.cls_convs  = nn.ModuleList()
        self.reg_convs  = nn.ModuleList()

        self.cls_preds  = nn.ModuleList()
        self.reg_preds  = nn.ModuleList()
        self.obj_preds  = nn.ModuleList()
        self.stems      = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        bias=self.conv_bias))