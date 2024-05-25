import math

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.model import BaseModule

from mmpose.registry import MODELS
#from ..utils import CSPLayer


class Bottleneck(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 group = 1,
                 eps = 0.5,
                 shortcut=True,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='SiLU')):
        super().__init__()
        mid_channels = int(out_channels * eps)
        self.conv1 = ConvModule(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            padding = 1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        self.conv2 = ConvModule(
            mid_channels,
            out_channels,
            kernel_size,
            stride,
            groups = group,
            padding = 1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        self.add = shortcut and in_channels == out_channels
        
    def forward(self,x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))     

class C2f(BaseModule):
    def __init__(self,
                in_channels,
                out_channels,
                group = 1,
                kernel_size = 1,
                eps = 0.5,
                n = 1,
                shortcut=False,
                conv_cfg=None,
                norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                act_cfg=dict(type='SiLU')):
        super().__init__()

        self.mid_channels= int(out_channels * eps) 
        self.conv1 = ConvModule(
            in_channels,
            2 * self.mid_channels,
            kernel_size =kernel_size,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        self.conv2 = ConvModule(
            (2 + n) * self.mid_channels,
            out_channels,
            kernel_size =kernel_size,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        ) 

        self.blocks = nn.ModuleList(
            Bottleneck(
                self.mid_channels, 
                self.mid_channels, 
                kernel_size=3,
                shortcut=shortcut, 
                group=group,
                eps=eps) for _ in range(n))
    def forward(self, x):
        # 进行一个卷积，然后划分成两份，每个通道都为c
        y = list(self.conv1(x).split((self.mid_channels, self.mid_channels), 1))
        # 每进行一次残差结构都保留，然后堆叠在一起，密集残差
        y.extend(block(y[-1]) for block in self.blocks)
        return self.conv2(torch.cat(y, 1))


@MODELS.register_module()
class YOLOV8PAFPN(BaseModule):
    def __init__(self,
                in_channels,
                out_channels,
                upsample_cfg=dict(scale_factor=2, mode='nearest'),
                conv_cfg=None,
                norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                act_cfg=dict(type='SiLU'),
                init_cfg=dict(
                    type='Kaiming',
                    layer='Conv2d',
                    a=math.sqrt(5),
                    distribution='uniform',
                    mode='fan_in',
                    nonlinearity='leaky_relu')):
        super(YOLOV8PAFPN, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.upsample = nn.Upsample(**upsample_cfg)
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.reduce_layers.append(
                ConvModule(
                    in_channels[idx],
                    in_channels[idx - 1],
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.top_down_blocks.append(
                C2f(
                    in_channels[idx - 1] * 2,
                    in_channels[idx - 1],
                    shortcut= False,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        # build bottom-up blocks
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(
                ConvModule(
                    in_channels[idx],
                    in_channels[idx],
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.bottom_up_blocks.append(
                C2f(
                    in_channels[idx] * 2,
                    in_channels[idx + 1],
                    shortcut= False,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        
    def forward(self, inputs):
        """
        Args:
            inputs (tuple[Tensor]): input features.

        Returns:
            tuple[Tensor]: YOLOXPAFPN features.
        """
        assert len(inputs) == len(self.in_channels)

        # top-down path
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = inputs[idx - 1]
            feat_heigh = self.reduce_layers[len(self.in_channels) - 1 - idx](
                feat_heigh)
            inner_outs[0] = feat_heigh

            upsample_feat = self.upsample(feat_heigh)

            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], 1))
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](
                torch.cat([downsample_feat, feat_height], 1))
            outs.append(out)

        # # out convs
        # for idx, conv in enumerate(self.out_convs):
        #     outs[idx] = conv(outs[idx])

        return tuple(outs)
    
# if __name__ == '__main__':
#     model = YOLOV8PAFPN([128,256,256],128)
#     # 正确的调用方式
#     xx1 = torch.randn(16, 128, 60, 60)
#     xx2 = torch.randn(16, 256, 30, 30)
#     xx3 = torch.randn(16, 256, 15, 15)

#     xx = [xx1, xx2, xx3]
#     y = model(xx)
#     for yy in y:
#         print(yy.shape)