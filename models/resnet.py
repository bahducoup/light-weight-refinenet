"""RefineNet-LightWeight

RefineNet-LigthWeight PyTorch for non-commercial purposes

Copyright (c) 2018, Vladimir Nekrasov (vladimir.nekrasov@adelaide.edu.au)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.helpers import maybe_download
from utils.layer_factory import conv1x1, conv3x3, CRPBlock

data_info = {7: "Person", 21: "VOC", 40: "NYU", 60: "Context"}

models_urls = {
    "50_person": "https://cloudstor.aarnet.edu.au/plus/s/mLA7NxVSPjNL7Oo/download",
    "101_person": "https://cloudstor.aarnet.edu.au/plus/s/f1tGGpwdCnYS3xu/download",
    "152_person": "https://cloudstor.aarnet.edu.au/plus/s/Ql64rWqiTvWGAA0/download",
    "50_voc": "https://cloudstor.aarnet.edu.au/plus/s/xp7GcVKC0GbxhTv/download",
    "101_voc": "https://cloudstor.aarnet.edu.au/plus/s/CPRKWiaCIDRdOwF/download",
    "152_voc": "https://cloudstor.aarnet.edu.au/plus/s/2w8bFOd45JtPqbD/download",
    "50_nyu": "https://cloudstor.aarnet.edu.au/plus/s/gE8dnQmHr9svpfu/download",
    "101_nyu": "https://cloudstor.aarnet.edu.au/plus/s/VnsaSUHNZkuIqeB/download",
    "152_nyu": "https://cloudstor.aarnet.edu.au/plus/s/EkPQzB2KtrrDnKf/download",
    "101_context": "https://cloudstor.aarnet.edu.au/plus/s/hqmplxWOBbOYYjN/download",
    "152_context": "https://cloudstor.aarnet.edu.au/plus/s/O84NszlYlsu00fW/download",
    "50_imagenet": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "101_imagenet": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "152_imagenet": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}

stages_suffixes = {0: "_conv", 1: "_conv_relu_varout_dimred"}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class WiderOrDeeperBlockWithConv2(nn.Module):
    def __init__(
        self,
        conv_skip_opts,
        conv1_opts,
        conv2_opts
    ):
        super().__init__()

        self.conv_skip = nn.Conv2d(**conv_skip_opts)

        conv1 = nn.Conv2d(**conv1_opts)
        bn = nn.BatchNorm2d(conv1_opts['out_channels'])
        relu = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(**conv2_opts)
        self.conv_all = nn.Sequential(
            conv1,
            bn,
            relu,
            conv2
        )
        
    def forward(self, x):
        conv_skip_out = self.conv_skip(x)
        conv_all_out = self.conv_all(x)
        out = conv_skip_out + conv_all_out

        return out


class WiderOrDeeperBlockWithRes2(nn.Module):
    def __init__(
        self,
        conv1_opts,
        conv2_opts
    ):
        super().__init__()

        relu = nn.ReLU(inplace=True)

        bn1 = nn.BatchNorm2d(conv1_opts['in_channels'])
        conv1 = nn.Conv2d(**conv1_opts)

        bn2 = nn.BatchNorm2d(conv2_opts['in_channels'])
        conv2 = nn.Conv2d(**conv2_opts)

        self.conv_all = nn.Sequential(
            bn1,
            relu,
            conv1,
            bn2,
            relu,
            conv2
        )

    def forward(self, x):
        residual = x
        conv_all_out = self.conv_all(x)
        out = residual + conv_all_out

        return out


class WiderOrDeeperBlockWithConv3(nn.Module):
    def __init__(
        self,
        conv_skip_opts,
        conv1_opts,
        conv2_opts,
        conv3_opts
    ):
        super().__init__()

        self.conv_skip = nn.Conv2d(**conv_skip_opts)

        conv1 = nn.Conv2d(**conv1_opts)
        bn1 = nn.BatchNorm2d(conv1_opts['out_channels'])
        relu = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(**conv2_opts)
        bn2 = nn.BatchNorm2d(conv2_opts['out_channels'])
        conv3 = nn.Conv2d(**conv3_opts)
        self.conv_all = nn.Sequential(
            conv1,
            bn1,
            relu,
            conv2,
            bn2,
            relu,
            conv3
        )

    def forward(self, x):
        conv_skip_out = self.conv_skip(x)
        conv_all_out = self.conv_all(x)
        out = conv_skip_out + conv_all_out

        return out


class WiderOrDeeper(nn.Module):
    def __init__(self, numclasses):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=3, bias=False)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        # relu

        self.block1 = WiderOrDeeperBlockWithConv2(
            conv_skip_opts={'in_channels': 64, 'out_channels': 128, 'kernel_size': 1, 'stride': 1, 'bias': False},
            conv1_opts={'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False},
            conv2_opts={'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False}
        )
        self.block2 = WiderOrDeeperBlockWithRes2(
            conv1_opts={'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False},
            conv2_opts={'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False}
        )
        self.block3 = WiderOrDeeperBlockWithRes2(
            conv1_opts={'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False},
            conv2_opts={'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False}
        )

        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        # relu

        self.block4 = WiderOrDeeperBlockWithConv2(
            conv_skip_opts={'in_channels': 128, 'out_channels': 256, 'kernel_size': 1, 'stride': 1, 'bias': False},
            conv1_opts={'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False},
            conv2_opts={'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False}
        )
        self.block5 = WiderOrDeeperBlockWithRes2(
            conv1_opts={'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False},
            conv2_opts={'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False}
        )
        self.block6 = WiderOrDeeperBlockWithRes2(
            conv1_opts={'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False},
            conv2_opts={'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False}
        )

        self.bn3 = nn.BatchNorm2d(256)
        # relu

        self.block7 = WiderOrDeeperBlockWithConv2(
            conv_skip_opts={'in_channels': 256, 'out_channels': 512, 'kernel_size': 1, 'stride': 2, 'bias': False},
            conv1_opts={'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'bias': False},
            conv2_opts={'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False}
        )
        self.block8 = WiderOrDeeperBlockWithRes2(
            conv1_opts={'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False},
            conv2_opts={'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False}
        )
        self.block9 = WiderOrDeeperBlockWithRes2(
            conv1_opts={'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False},
            conv2_opts={'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False}
        )
        self.block10 = WiderOrDeeperBlockWithRes2(
            conv1_opts={'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False},
            conv2_opts={'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False}
        )
        self.block11 = WiderOrDeeperBlockWithRes2(
            conv1_opts={'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False},
            conv2_opts={'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False}
        )
        self.block12 = WiderOrDeeperBlockWithRes2(
            conv1_opts={'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False},
            conv2_opts={'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False}
        )

        self.bn4 = nn.BatchNorm2d(512)
        # relu

        self.block13 = WiderOrDeeperBlockWithConv2(
            conv_skip_opts={'in_channels': 512, 'out_channels': 1024, 'kernel_size': 1, 'stride': 1, 'bias': False},
            conv1_opts={'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False},
            conv2_opts={'in_channels': 512, 'out_channels': 1024, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False}
        )
        self.block14 = WiderOrDeeperBlockWithRes2(
            conv1_opts={'in_channels': 1024, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False},
            conv2_opts={'in_channels': 512, 'out_channels': 1024, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False}
        )
        self.block15 = WiderOrDeeperBlockWithRes2(
            conv1_opts={'in_channels': 1024, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False},
            conv2_opts={'in_channels': 512, 'out_channels': 1024, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False}
        )

        self.bn5 = nn.BatchNorm2d(1024)
        # relu

        self.block16 = WiderOrDeeperBlockWithConv3(
            conv_skip_opts={'in_channels': 1024, 'out_channels': 2048, 'kernel_size': 1, 'stride': 1, 'bias': False},
            conv1_opts={'in_channels': 1024, 'out_channels': 512, 'kernel_size': 1, 'stride': 1, 'bias': False},
            conv2_opts={'in_channels': 512, 'out_channels': 1024, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False},
            conv3_opts={'in_channels': 1024, 'out_channels': 2048, 'kernel_size': 1, 'stride': 1, 'bias': False}
        )

        self.bn6 = nn.BatchNorm2d(2048)
        # relu

        self.block17 = WiderOrDeeperBlockWithConv3(
            conv_skip_opts={'in_channels': 2048, 'out_channels': 4096, 'kernel_size': 1, 'stride': 1, 'bias': False},
            conv1_opts={'in_channels': 2048, 'out_channels': 1024, 'kernel_size': 1, 'stride': 1, 'bias': False},
            conv2_opts={'in_channels': 1024, 'out_channels': 2048, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False},
            conv3_opts={'in_channels': 2048, 'out_channels': 4096, 'kernel_size': 1, 'stride': 1, 'bias': False}
        )

        self.bn7 = nn.BatchNorm2d(4096)
        # relu

        self.conv2 = nn.Conv2d(4096, 512, kernel_size=3, stride=1, padding=1, bias=False)
        # relu
        self.conv3 = nn.Conv2d(512, 21, kernel_size=3, stride=1, padding=1, bias=False)

        self.do = nn.Dropout(p=0.5)

        self.p_ims1d2_outl1_dimred = conv1x1(4096, 512, bias=False)
        self.mflow_conv_g1_pool = self._make_crp(512, 512, 4)
        self.mflow_conv_g1_b3_joint_varout_dimred = conv1x1(512, 256, bias=False)

        self.p_ims1d2_outl2_dimred = conv1x1(1024, 256, bias=False)
        self.adapt_stage2_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self.mflow_conv_g2_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g2_b3_joint_varout_dimred = conv1x1(256, 256, bias=False)

        self.p_ims1d2_outl3_dimred = conv1x1(256, 256, bias=False)
        self.adapt_stage3_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self.mflow_conv_g3_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g3_b3_joint_varout_dimred = conv1x1(256, 128, bias=False)

        self.p_ims1d2_outl4_dimred = conv1x1(128, 128, bias=False)
        self.adapt_stage4_b2_joint_varout_dimred = conv1x1(128, 128, bias=False)
        self.mflow_conv_g4_pool = self._make_crp(128, 128, 4)

        self.clf_conv = nn.Conv2d(
            128, numclasses, kernel_size=3, stride=1, padding=1, bias=True
        )

    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes, stages)]
        return nn.Sequential(*layers)

    def forward(self, x):
        # L1
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.max_pool2(x)
        x = self.bn2(x)
        l1 = self.relu(x)

        # L2
        x = self.block4(l1)
        x = self.block5(x)
        x = self.block6(x)

        x = self.bn3(x)
        l2 = self.relu(x)

        # L3
        x = self.block7(l2)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        
        x = self.bn4(x)
        x = self.relu(x)

        x = self.block13(x)
        x = self.block14(x)
        l3 = self.block15(x)
        
        # L4
        x = self.bn5(l3)
        x = self.relu(x)

        x = self.block16(x)
        
        x = self.bn6(x)
        x = self.relu(x)

        x = self.block17(x)
        
        x = self.bn7(x)
        l4 = self.relu(x)

        # x = self.conv2(x)
        
        # x = self.relu(x)

        # l4 = self.conv3(x)
        print("starting decoder") 

        l4 = self.do(l4)
        l3 = self.do(l3)

        # l2 = self.do(l2)
        # l1 = self.do(l1)
        print("dropout")

        x4 = self.p_ims1d2_outl1_dimred(l4)
        x4 = self.relu(x4)
        x4 = self.mflow_conv_g1_pool(x4)
        x4 = self.mflow_conv_g1_b3_joint_varout_dimred(x4)
        x4 = nn.Upsample(size=l3.size()[2:], mode="bilinear", align_corners=True)(x4)
        print("passed x4")

        x3 = self.p_ims1d2_outl2_dimred(l3)
        x3 = self.adapt_stage2_b2_joint_varout_dimred(x3)
        x3 = x3 + x4
        x3 = F.relu(x3)
        x3 = self.mflow_conv_g2_pool(x3)
        x3 = self.mflow_conv_g2_b3_joint_varout_dimred(x3)
        x3 = nn.Upsample(size=l2.size()[2:], mode="bilinear", align_corners=True)(x3)
        print("passed x3")

        x2 = self.p_ims1d2_outl3_dimred(l2)
        x2 = self.adapt_stage3_b2_joint_varout_dimred(x2)
        x2 = x2 + x3
        x2 = F.relu(x2)
        x2 = self.mflow_conv_g3_pool(x2)
        x2 = self.mflow_conv_g3_b3_joint_varout_dimred(x2)
        x2 = nn.Upsample(size=l1.size()[2:], mode="bilinear", align_corners=True)(x2)
        print("passed x2")

        x1 = self.p_ims1d2_outl4_dimred(l1)
        print(x1.shape)
        print(x2.shape)
        x1 = self.adapt_stage4_b2_joint_varout_dimred(x1)
        x1 = x1 + x2
        x1 = F.relu(x1)
        x1 = self.mflow_conv_g4_pool(x1)

        print("Got there!")
        out = self.clf_conv(x1)
        return out


class ResNetLW(nn.Module):
    def __init__(self, block, layers, num_classes=21):
        self.inplanes = 64
        super(ResNetLW, self).__init__()
        self.do = nn.Dropout(p=0.5)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.p_ims1d2_outl1_dimred = conv1x1(2048, 512, bias=False)
        self.mflow_conv_g1_pool = self._make_crp(512, 512, 4)
        self.mflow_conv_g1_b3_joint_varout_dimred = conv1x1(512, 256, bias=False)
        self.p_ims1d2_outl2_dimred = conv1x1(1024, 256, bias=False)
        self.adapt_stage2_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self.mflow_conv_g2_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g2_b3_joint_varout_dimred = conv1x1(256, 256, bias=False)

        self.p_ims1d2_outl3_dimred = conv1x1(512, 256, bias=False)
        self.adapt_stage3_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self.mflow_conv_g3_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g3_b3_joint_varout_dimred = conv1x1(256, 256, bias=False)

        self.p_ims1d2_outl4_dimred = conv1x1(256, 256, bias=False)
        self.adapt_stage4_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self.mflow_conv_g4_pool = self._make_crp(256, 256, 4)

        self.clf_conv = nn.Conv2d(
            256, num_classes, kernel_size=3, stride=1, padding=1, bias=True
        )

    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes, stages)]
        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        l4 = self.do(l4)
        l3 = self.do(l3)

        x4 = self.p_ims1d2_outl1_dimred(l4)
        x4 = self.relu(x4)
        x4 = self.mflow_conv_g1_pool(x4)
        x4 = self.mflow_conv_g1_b3_joint_varout_dimred(x4)
        x4 = nn.Upsample(size=l3.size()[2:], mode="bilinear", align_corners=True)(x4)

        x3 = self.p_ims1d2_outl2_dimred(l3)
        x3 = self.adapt_stage2_b2_joint_varout_dimred(x3)
        x3 = x3 + x4
        x3 = F.relu(x3)
        x3 = self.mflow_conv_g2_pool(x3)
        x3 = self.mflow_conv_g2_b3_joint_varout_dimred(x3)
        x3 = nn.Upsample(size=l2.size()[2:], mode="bilinear", align_corners=True)(x3)

        x2 = self.p_ims1d2_outl3_dimred(l2)
        x2 = self.adapt_stage3_b2_joint_varout_dimred(x2)
        x2 = x2 + x3
        x2 = F.relu(x2)
        x2 = self.mflow_conv_g3_pool(x2)
        x2 = self.mflow_conv_g3_b3_joint_varout_dimred(x2)
        x2 = nn.Upsample(size=l1.size()[2:], mode="bilinear", align_corners=True)(x2)

        x1 = self.p_ims1d2_outl4_dimred(l1)
        x1 = self.adapt_stage4_b2_joint_varout_dimred(x1)
        x1 = x1 + x2
        x1 = F.relu(x1)
        x1 = self.mflow_conv_g4_pool(x1)

        out = self.clf_conv(x1)
        return out


"""model.py - Model and module class for EfficientNet.
   They are built to mirror those in the official TensorFlow implementation.
"""

# Author: lukemelas (github username)
# Github repo: https://github.com/lukemelas/EfficientNet-PyTorch
# With adjustments and added comments by workingcoder (github username).

from .utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    Swish,
    MemoryEfficientSwish,
    calculate_output_image_size
)


VALID_MODELS = (
    'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
    'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
    'efficientnet-b8',

    # Support the construction of 'efficientnet-l2' without pretrained weights
    'efficientnet-l2'
)


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block.

    Args:
        block_args (namedtuple): BlockArgs, defined in utils.py.
        global_params (namedtuple): GlobalParam, defined in utils.py.
        image_size (tuple or list): [image_height, image_width].

    References:
        [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
        [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
        [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
    """

    def __init__(self, block_args, global_params, image_size=None):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum  # pytorch's difference from tensorflow
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # whether to use skip connection and drop connect

        # Expansion phase (Inverted Bottleneck)
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
            # image_size = calculate_output_image_size(image_size, 1) <-- this wouldn't modify image_size

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        image_size = calculate_output_image_size(image_size, s)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            Conv2d = get_same_padding_conv2d(image_size=(1, 1))
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Pointwise convolution phase
        final_oup = self._block_args.output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """MBConvBlock's forward function.

        Args:
            inputs (tensor): Input tensor.
            drop_connect_rate (bool): Drop connect rate (float, between 0 and 1).

        Returns:
            Output of this block after processing.
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        # Pointwise Convolution
        x = self._project_conv(x)
        x = self._bn2(x)

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            # The combination of skip connection and drop connect brings about stochastic depth.
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class EfficientNet(nn.Module):
    """EfficientNet model.
       Most easily loaded with the .from_name or .from_pretrained methods.

    Args:
        blocks_args (list[namedtuple]): A list of BlockArgs to construct blocks.
        global_params (namedtuple): A set of GlobalParams shared between blocks.

    References:
        [1] https://arxiv.org/abs/1905.11946 (EfficientNet)

    Example:
        >>> import torch
        >>> from efficientnet.model import EfficientNet
        >>> inputs = torch.rand(1, 3, 224, 224)
        >>> model = EfficientNet.from_pretrained('efficientnet-b0')
        >>> model.eval()
        >>> outputs = model(inputs)
    """

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Get stem static or dynamic convolution depending on image size
        image_size = global_params.image_size
        Conv2d = get_same_padding_conv2d(image_size=image_size)

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        image_size = calculate_output_image_size(image_size, 2)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))
            image_size = calculate_output_image_size(image_size, block_args.stride)
            if block_args.num_repeat > 1:  # modify block_args to keep same output size
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))
                # image_size = calculate_output_image_size(image_size, block_args.stride)  # stride = 1

        # Head
        # in_channels = block_args.output_filters  # output of final block
        # out_channels = round_filters(1280, self._global_params)
        in_channels = 256
        out_channels = 21
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        # self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        # self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        self._conv_head_new = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1_new = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        

        # Final linear layer
        # self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        # if self._global_params.include_top:
        #     self._dropout = nn.Dropout(self._global_params.dropout_rate)
        #     self._fc = nn.Linear(out_channels, self._global_params.num_classes)

        # set activation to memory efficient swish by default
        self._swish = MemoryEfficientSwish()

        # refine-net
        self._do = nn.Dropout(p=0.5)
        self._relu = nn.ReLU(inplace=True)

        self._p_ims1d2_outl1_dimred = conv1x1(352, 512, bias=False)
        self._mflow_conv_g1_pool = self._make_crp(512, 512, 4)
        self._mflow_conv_g1_b3_joint_varout_dimred = conv1x1(512, 256, bias=False)

        self._p_ims1d2_outl2_dimred = conv1x1(208, 256, bias=False)
        self._adapt_stage2_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self._mflow_conv_g2_pool = self._make_crp(256, 256, 4)
        self._mflow_conv_g2_b3_joint_varout_dimred = conv1x1(256, 256, bias=False)

        self._p_ims1d2_outl3_dimred = conv1x1(48, 256, bias=False)
        self._adapt_stage3_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self._mflow_conv_g3_pool = self._make_crp(256, 256, 4)
        self._mflow_conv_g3_b3_joint_varout_dimred = conv1x1(256, 256, bias=False)

        self._p_ims1d2_outl4_dimred = conv1x1(24, 256, bias=False)
        self._adapt_stage4_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self._mflow_conv_g4_pool = self._make_crp(256, 256, 4)

        # print(self)

    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes, stages)]
        return nn.Sequential(*layers)

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_endpoints(self, inputs):
        """Use convolution layer to extract features
        from reduction levels i in [1, 2, 3, 4, 5].

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Dictionary of last intermediate features
            with reduction levels i in [1, 2, 3, 4, 5].
            Example:
                >>> import torch
                >>> from efficientnet.model import EfficientNet
                >>> inputs = torch.rand(1, 3, 224, 224)
                >>> model = EfficientNet.from_pretrained('efficientnet-b0')
                >>> endpoints = model.extract_endpoints(inputs)
                >>> print(endpoints['reduction_1'].shape)  # torch.Size([1, 16, 112, 112])
                >>> print(endpoints['reduction_2'].shape)  # torch.Size([1, 24, 56, 56])
                >>> print(endpoints['reduction_3'].shape)  # torch.Size([1, 40, 28, 28])
                >>> print(endpoints['reduction_4'].shape)  # torch.Size([1, 112, 14, 14])
                >>> print(endpoints['reduction_5'].shape)  # torch.Size([1, 320, 7, 7])
                >>> print(endpoints['reduction_6'].shape)  # torch.Size([1, 1280, 7, 7])
        """
        endpoints = dict()

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            elif idx == len(self._blocks) - 1:
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = x
            prev_x = x

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x

        return endpoints

    def extract_features(self, inputs):
        """use convolution layer to extract feature .

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of the final convolution
            layer in the efficientnet model.
        """
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)

            if idx == 2:
                l1 = x
            elif idx == 7:
                l2 = x
            elif idx == 19:
                l3 = x
            elif idx == 22:
                l4 = x

        l4 = self._do(l4)
        l3 = self._do(l3)

        x4 = self._p_ims1d2_outl1_dimred(l4)
        x4 = self._relu(x4)
        x4 = self._mflow_conv_g1_pool(x4)
        x4 = self._mflow_conv_g1_b3_joint_varout_dimred(x4)
        x4 = nn.Upsample(size=l3.size()[2:], mode="bilinear", align_corners=True)(x4)

        x3 = self._p_ims1d2_outl2_dimred(l3)
        x3 = self._adapt_stage2_b2_joint_varout_dimred(x3)
        x3 = x3 + x4
        x3 = self._relu(x3)
        x3 = self._mflow_conv_g2_pool(x3)
        x3 = self._mflow_conv_g2_b3_joint_varout_dimred(x3)
        x3 = nn.Upsample(size=l2.size()[2:], mode="bilinear", align_corners=True)(x3)

        x2 = self._p_ims1d2_outl3_dimred(l2)
        x2 = self._adapt_stage3_b2_joint_varout_dimred(x2)
        x2 = x2 + x3
        x2 = self._relu(x2)
        x2 = self._mflow_conv_g3_pool(x2)
        x2 = self._mflow_conv_g3_b3_joint_varout_dimred(x2)
        x2 = nn.Upsample(size=l1.size()[2:], mode="bilinear", align_corners=True)(x2)

        x1 = self._p_ims1d2_outl4_dimred(l1)
        x1 = self._adapt_stage4_b2_joint_varout_dimred(x1)
        x1 = x1 + x2
        x1 = self._relu(x1)
        x1 = self._mflow_conv_g4_pool(x1)

        # Head
        x = self._swish(self._bn1_new(self._conv_head_new(x1)))
        return x

    def forward(self, inputs):
        """EfficientNet's forward function.
           Calls extract_features to extract features, applies final linear layer, and returns logits.

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of this model after processing.
        """
        # Convolution layers
        x = self.extract_features(inputs)
        # Pooling and final linear layer
        # x = self._avg_pooling(x)
        # if self._global_params.include_top:
        #     x = x.flatten(start_dim=1)
        #     x = self._dropout(x)
        #     x = self._fc(x)
        return x

    @classmethod
    def from_name(cls, model_name, in_channels=3, **override_params):
        """Create an efficientnet model according to name.

        Args:
            model_name (str): Name for efficientnet.
            in_channels (int): Input data's channel number.
            override_params (other key word params):
                Params to override model's global_params.
                Optional key:
                    'width_coefficient', 'depth_coefficient',
                    'image_size', 'dropout_rate',
                    'num_classes', 'batch_norm_momentum',
                    'batch_norm_epsilon', 'drop_connect_rate',
                    'depth_divisor', 'min_depth'

        Returns:
            An efficientnet model.
        """
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        model = cls(blocks_args, global_params)
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def from_pretrained(cls, model_name, weights_path=None, advprop=False,
                        in_channels=3, num_classes=1000, **override_params):
        """Create an efficientnet model according to name.

        Args:
            model_name (str): Name for efficientnet.
            weights_path (None or str):
                str: path to pretrained weights file on the local disk.
                None: use pretrained weights downloaded from the Internet.
            advprop (bool):
                Whether to load pretrained weights
                trained with advprop (valid when weights_path is None).
            in_channels (int): Input data's channel number.
            num_classes (int):
                Number of categories for classification.
                It controls the output size for final linear layer.
            override_params (other key word params):
                Params to override model's global_params.
                Optional key:
                    'width_coefficient', 'depth_coefficient',
                    'image_size', 'dropout_rate',
                    'batch_norm_momentum',
                    'batch_norm_epsilon', 'drop_connect_rate',
                    'depth_divisor', 'min_depth'

        Returns:
            A pretrained efficientnet model.
        """
        model = cls.from_name(model_name, num_classes=num_classes, **override_params)
        load_pretrained_weights(model, model_name, weights_path=weights_path,
                                load_fc=(num_classes == 1000), advprop=advprop)
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        """Get the input image size for a given efficientnet model.

        Args:
            model_name (str): Name for efficientnet.

        Returns:
            Input image size (resolution).
        """
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """Validates model name.

        Args:
            model_name (str): Name for efficientnet.

        Returns:
            bool: Is a valid name or not.
        """
        if model_name not in VALID_MODELS:
            raise ValueError('model_name should be one of: ' + ', '.join(VALID_MODELS))

    def _change_in_channels(self, in_channels):
        """Adjust model's first convolution layer to in_channels, if in_channels not equals 3.

        Args:
            in_channels (int): Input data's channel number.
        """
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size=self._global_params.image_size)
            out_channels = round_filters(32, self._global_params)
            self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)


def rf_lw50(num_classes, imagenet=False, pretrained=True, **kwargs):
    model = ResNetLW(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)
    if imagenet:
        key = "50_imagenet"
        url = models_urls[key]
        model.load_state_dict(maybe_download(key, url), strict=False)
    elif pretrained:
        dataset = data_info.get(num_classes, None)
        if dataset:
            bname = "50_" + dataset.lower()
            key = "rf_lw" + bname
            url = models_urls[bname]
            model.load_state_dict(maybe_download(key, url), strict=False)
    return model


def rf_lw101(num_classes, imagenet=False, pretrained=True, **kwargs):
    model = ResNetLW(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, **kwargs)
    if imagenet:
        key = "101_imagenet"
        url = models_urls[key]
        model.load_state_dict(maybe_download(key, url), strict=False)
    elif pretrained:
        dataset = data_info.get(num_classes, None)
        if dataset:
            bname = "101_" + dataset.lower()
            key = "rf_lw" + bname
            url = models_urls[bname]
            model.load_state_dict(maybe_download(key, url), strict=False)
    return model


def rf_lw152(num_classes, imagenet=False, pretrained=True, **kwargs):
    model = ResNetLW(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, **kwargs)
    if imagenet:
        key = "152_imagenet"
        url = models_urls[key]
        model.load_state_dict(maybe_download(key, url), strict=False)
    elif pretrained:
        dataset = data_info.get(num_classes, None)
        if dataset:
            bname = "152_" + dataset.lower()
            key = "rf_lw" + bname
            url = models_urls[bname]
            model.load_state_dict(maybe_download(key, url), strict=False)
    return model


def rf_wider_or_deeper(num_classes, imagenet=False, pretrained=False, **kwargs):
    model = WiderOrDeeper()

    return model


def rf_efficient(num_classes, imagenet=False, pretrained=True, **kwargs):
    model = EfficientNet.from_pretrained('efficientnet-b2')

    return model
