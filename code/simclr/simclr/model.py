import torchvision.models as models

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
# class Encoder(torch.nn.Module):
#     def __init__(self, base_model='resnet18'):
#         super().__init__()
#         self.base_model = models.__dict__[base_model](pretrained=False)  # 不加载预训练权重
#         self.feature_dim = self.base_model.fc.in_features
#         self.base_model.fc = torch.nn.Identity()  # 移除全连接层
#
#     def forward(self, x):
#         return self.base_model(x)
#
#
# class ProjectionHead(torch.nn.Module):
#     def __init__(self, input_dim=2048, hidden_dim=512, output_dim=128):
#         super().__init__()
#         self.head = torch.nn.Sequential(
#             torch.nn.Linear(input_dim, hidden_dim),
#             torch.nn.ReLU(),
#             torch.nn.Linear(hidden_dim, output_dim)
#         )
#
#     def forward(self, x):
#         return self.head(x)
#
# class SimCLR(torch.nn.Module):
#     def __init__(self, base_model='resnet50'):
#         super().__init__()
#         self.encoder = Encoder(base_model=base_model)
#         self.projector = ProjectionHead(input_dim=self.encoder.feature_dim)
#
#     def forward(self, x1, x2):
#         h1 = self.encoder(x1)
#         h2 = self.encoder(x2)
#         z1 = self.projector(h1)  # [batch_size, output_dim]
#         z2 = self.projector(h2)
#         return z1, z2

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, groups=1):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()  # SiLU 激活函数

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class C2f(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True):
        super().__init__()
        hidden_channels = out_channels // 2
        self.conv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.conv2 = Conv(in_channels, hidden_channels, 1, 1)
        self.bottlenecks = nn.Sequential(
            *[Conv(hidden_channels, hidden_channels, 3, 1) for _ in range(n)]
        )
        self.conv3 = Conv(hidden_channels * 2, out_channels, 1, 1)
        self.shortcut = shortcut

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x2 = self.bottlenecks(x2)
        out = torch.cat((x1, x2), dim=1)
        return self.conv3(out)

class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, k=5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.conv2 = Conv(hidden_channels * 4, out_channels, 1, 1)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        out = torch.cat((x, y1, y2, y3), dim=1)
        return self.conv2(out)

class YOLOv8Head(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = Conv(1024, 512, 1, 1)
        self.c2f1 = C2f(1024, 512, n=3)
        self.conv2 = Conv(512, 256, 1, 1)
        self.c2f2 = C2f(512, 256, n=3)
        self.conv3 = Conv(256, 256, 3, 2)
        self.c2f3 = C2f(512, 512, n=3)
        self.conv4 = Conv(512, 512, 3, 2)
        self.c2f4 = C2f(1024, 1024, n=3)
        self.detect = Detect(num_classes)

    def forward(self, x):
        # 假设 x 是 Backbone 的输出
        x1 = self.conv1(x)
        x1 = self.upsample(x1)
        x1 = torch.cat((x1, x), dim=1)  # 拼接
        x1 = self.c2f1(x1)

        x2 = self.conv2(x1)
        x2 = self.upsample(x2)
        x2 = torch.cat((x2, x1), dim=1)  # 拼接
        x2 = self.c2f2(x2)

        x3 = self.conv3(x2)
        x3 = torch.cat((x3, x1), dim=1)  # 拼接
        x3 = self.c2f3(x3)

        x4 = self.conv4(x3)
        x4 = torch.cat((x4, x), dim=1)  # 拼接
        x4 = self.c2f4(x4)

        return self.detect([x2, x3, x4])
class Detect(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.conv = nn.Conv2d(256, (5 + num_classes) * 3, 1, 1)  # 3 个尺度的输出

    def forward(self, x: List[torch.Tensor]):
        return [self.conv(xi) for xi in x]

class YOLOv8(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        self.backbone = YOLOv8Backbone()
        self.head = YOLOv8Head(num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)

class YOLOv8Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            Conv(3, 64, 3, 2),  # 0-P1/2
            Conv(64, 128, 3, 2),  # 1-P2/4
            C2f(128, 128, n=3),  # 2
            Conv(128, 256, 3, 2),  # 3-P3/8
            C2f(256, 256, n=6),  # 4
            Conv(256, 512, 3, 2),  # 5-P4/16
            C2f(512, 512, n=6),  # 6
            Conv(512, 1024, 3, 2),  # 7-P5/32
            C2f(1024, 1024, n=3),  # 8
            SPPF(1024, 1024),  # 9
        )
        # 添加全局平均池化和投影头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Sequential(
            nn.Linear(1024, 256),  # 投影到 256 维
            nn.ReLU(),
            nn.Linear(256, 128),  # 最终输出 128 维
        )

    def forward(self, x):
        x = self.layers(x)
        x = self.avgpool(x)  # 全局平均池化
        x = torch.flatten(x, 1)  # 展平
        return self.projection(x)  # 投影到低维空间

