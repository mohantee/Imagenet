import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # 1x1 convolution for dimension reduction
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3x3 convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 convolution for dimension increase
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                              kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += self.shortcut(identity)
        out = self.relu(out)

        return out

class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet stages
        self.stage1 = self._make_stage(64, 64, 3, stride=1)      # Output: 256 channels
        self.stage2 = self._make_stage(256, 128, 4, stride=2)    # Output: 512 channels
        self.stage3 = self._make_stage(512, 256, 6, stride=2)    # Output: 1024 channels
        self.stage4 = self._make_stage(1024, 512, 3, stride=2)   # Output: 2048 channels

        # Global average pooling and final fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        layers = []

        # First block with specified stride
        layers.append(Bottleneck(in_channels, out_channels, stride))

        # Remaining blocks with stride 1
        for _ in range(1, num_blocks):
            layers.append(Bottleneck(out_channels * 4, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)    # 224x224x3 -> 112x112x64
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 112x112x64 -> 56x56x64

        x = self.stage1(x)   # 56x56x64 -> 56x56x256
        x = self.stage2(x)   # 56x56x256 -> 28x28x512
        x = self.stage3(x)   # 28x28x512 -> 14x14x1024
        x = self.stage4(x)   # 14x14x1024 -> 7x7x2048

        x = self.avgpool(x)  # 7x7x2048 -> 1x1x2048
        x = torch.flatten(x, 1)
        x = self.fc(x)       # 2048 -> 1000

        return x