from mmpose.registry import DATASETS
from ..base import BaseCocoStyleDataset


@DATASETS.register_module()
class CowPose(BaseCocoStyleDataset):
    METAINFO: dict = dict(from_file='configs/_base_/datasets/cow-pose.py')
