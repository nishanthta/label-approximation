from torch import nn
import torch

class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2, bottleneck=False):
        super(Conv2DBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels//2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels//2)
        self.conv2 = nn.Conv2d(in_channels=out_channels//2, out_channels=out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.bottleneck = bottleneck
        if not bottleneck:
            self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, input):
        res = self.relu(self.bn1(self.conv1(input)))
        res = self.relu(self.bn2(self.conv2(res)))
        res = self.dropout(res)
        out = self.pooling(res) if not self.bottleneck else res
        return out, res
    

class frameClassifier(nn.Module):
    def __init__(self, in_channels, num_classes = 2, level_channels=[64, 128, 256]):
        super(frameClassifier, self).__init__()
        self.classifier_blocks = nn.ModuleList()
        in_ch = in_channels
        for out_ch in level_channels:
            self.classifier_blocks.append(Conv2DBlock(in_channels=in_ch, out_channels=out_ch))
            in_ch = out_ch

        self.classification_layer = nn.Linear(in_features = 256*16*16, out_features = num_classes)

    def forward(self, input):
        out = input
        for block in self.classifier_blocks:
            out, res = block(out)
        
        out = torch.flatten(out)
        out = self.classification_layer(out)

        return out