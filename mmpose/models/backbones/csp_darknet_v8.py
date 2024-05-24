import math

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from mmpose.registry import MODELS

class SPPFBottleneck(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=5,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='SiLU'),
                 init_cfg=None):
        super().__init__(init_cfg)
        mid_channels = in_channels // 2
        self.conv1 = ConvModule(
            in_channels,
            mid_channels,
            1,
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.pooling = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        conv2_channels = mid_channels * 4
        self.conv2 = ConvModule(
            conv2_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        
    def forward(self, x):
        x = self.conv1(x)
        y1 = self.pooling(x)
        y2 = self.pooling(y1)
        with torch.cuda.amp.autocast(enabled=False):
            x = torch.cat((x, y1, y2, self.pooling(y2)), dim=1)
        
        return self.conv2(x)

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
class CSPDarknetV8(BaseModule):
    arch_settings = {
        'P5': [[64, 128, 3, True, False], [128, 256, 9, True, False],
               [256, 512, 9, True, False], [512, 512, 3, False, True]],
        'P6': [[64, 128, 3, True, False], [128, 256, 9, True, False],
               [256, 512, 9, True, False], [512, 768, 3, True, False],
               [768, 1024, 3, False, True]]
    }

    def __init__(self,
                 arch='P5',
                 deepen_factor=1.0,
                 widen_factor=1.0,
                 out_indices=(2, 3, 4),
                 frozen_stages=-1,
                 use_depthwise=False,
                 arch_ovewrite=None,
                 sppf_kernal_size=5,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='SiLU'),
                 norm_eval=False,
                 init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu')):
        super().__init__(init_cfg)

        arch_setting = self.arch_settings[arch]
        if arch_ovewrite:
            arch_setting = arch_ovewrite
        assert set(out_indices).issubset(
            i for i in range(len(arch_setting) + 1))
        if frozen_stages not in range(-1, len(arch_setting) + 1):
            raise ValueError('frozen_stages must be in range(-1, '
                             'len(arch_setting) + 1). But received '
                             f'{frozen_stages}')

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.use_depthwise = use_depthwise
        self.norm_eval = norm_eval
        #conv = ConvModule

        self.stem = ConvModule(
            3,
            int(arch_setting[0][0] * widen_factor),
            3,
            stride=2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.layers = ['stem']

        for i, (in_channels, out_channels, num_blocks, add_identity,
                use_sppf) in enumerate(arch_setting):
            in_channels = int(in_channels * widen_factor)
            out_channels = int(out_channels * widen_factor)
            num_blocks = max(round(num_blocks * deepen_factor), 1)
            stage = []
            conv_layer = ConvModule(
                in_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            stage.append(conv_layer)
            c2f_layer = C2f(
                out_channels,
                out_channels,
                n=num_blocks,
                shortcut= True,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            stage.append(c2f_layer)
            if use_sppf:
                sppf = SPPFBottleneck(
                    out_channels,
                    out_channels,
                    kernel_size=sppf_kernal_size,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
                stage.append(sppf)
            self.add_module(f'stage{i + 1}', nn.Sequential(*stage))
            self.layers.append(f'stage{i + 1}')
    def forward(self, x):
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)


# if __name__ == '__main__':
#     model = CSPDarknetV8()
#     xx = torch.randn(2,3,640,640)
#     y = model(xx)
#     for yy in y:
#         print(yy.shape)