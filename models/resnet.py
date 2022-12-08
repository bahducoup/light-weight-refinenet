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
