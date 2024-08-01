# Copyright (c) OpenMMLab. All rights reserved.
from .base_head import BaseHead
from .coord_cls_heads import RTMCCHead, RTMWHead, SimCCHead
from .heatmap_heads import (AssociativeEmbeddingHead, CIDHead, CPMHead,
                            HeatmapHead, InternetHead, MSPNHead, ViPNASHead)
from .hybrid_heads import DEKRHead, RTMOHead, VisPredictHead
from .regression_heads import (DSNTHead, IntegralRegressionHead,
                               MotionRegressionHead, RegressionHead, RLEHead,
                               TemporalRegressionHead,
                               TrajectoryRegressionHead)
from .transformer_heads import EDPoseHead
from .transformer_heads.RTDetr_Head import RTDETR
from .transformer_heads.pose_head import GroupHead
from .transformer_heads.rtpose_head import RTPoseHead

__all__ = [
    'BaseHead', 'HeatmapHead', 'CPMHead', 'MSPNHead', 'ViPNASHead',
    'RegressionHead', 'IntegralRegressionHead', 'SimCCHead', 'RLEHead',
    'DSNTHead', 'AssociativeEmbeddingHead', 'DEKRHead', 'VisPredictHead',
    'CIDHead', 'RTMCCHead', 'TemporalRegressionHead',
    'TrajectoryRegressionHead', 'MotionRegressionHead', 'EDPoseHead',
    'InternetHead', 'RTMWHead', 'RTMOHead', 'GroupHead', 'RTDETR', 'RTPoseHead'
]
