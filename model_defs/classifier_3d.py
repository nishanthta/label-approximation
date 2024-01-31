from torch import nn
import torch

class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2, bottleneck=False):
        super(Conv3DBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels//2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(num_features=out_channels//2)
        self.conv2 = nn.Conv3d(in_channels=out_channels//2, out_channels=out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout3d(p=dropout_rate)
        self.bottleneck = bottleneck
        if not bottleneck:
            self.pooling = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, input):
        res = self.relu(self.bn1(self.conv1(input)))
        res = self.relu(self.bn2(self.conv2(res)))
        res = self.dropout(res)
        out = self.pooling(res) if not self.bottleneck else res
        return out, res
    

class VolumeClassifier(nn.Module):
    def __init__(self, in_channels, num_classes=2, level_channels=[64, 128, 256]):
        super(VolumeClassifier, self).__init__()
        self.classifier_blocks = nn.ModuleList()
        in_ch = in_channels
        for out_ch in level_channels:
            self.classifier_blocks.append(Conv3DBlock(in_channels=in_ch, out_channels=out_ch))
            in_ch = out_ch

        # Assuming the input volume is reduced to a size of 8x8x8 by the convolutional blocks
        self.classification_layer = nn.Linear(in_features=256*8*8*8, out_features=num_classes)

    def forward(self, input):
        out = input
        for block in self.classifier_blocks:
            out, res = block(out)
        
        out = torch.flatten(out, start_dim=1)
        out = self.classification_layer(out)

        return out
