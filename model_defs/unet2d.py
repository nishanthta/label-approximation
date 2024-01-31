'''
Author : Nishanth
'''

from torch import nn
import torch

class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2, bottleneck=False):
        super(Conv2DBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels//2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels//2)
        self.conv2 = nn.Conv2d(in_channels=out_channels//2, out_channels=out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=False)
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

class UpConv2DBlock(nn.Module):
    def __init__(self, in_channels, res_channels=0, last_layer=False, dropout_rate=0.2, num_classes=None):
        super(UpConv2DBlock, self).__init__()
        assert (last_layer==False and num_classes==None) or (last_layer==True and num_classes!=None), 'Invalid arguments'
        self.upconv1 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=False)
        self.bn = nn.BatchNorm2d(num_features=in_channels//2)
        self.conv1 = nn.Conv2d(in_channels=in_channels+res_channels, out_channels=in_channels//2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=3, padding=1)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.last_layer = last_layer
        if last_layer:
            self.conv3 = nn.Conv2d(in_channels=in_channels//2, out_channels=num_classes, kernel_size=1)
        
    def forward(self, input, residual=None):
        out = self.upconv1(input)
        if residual is not None:
            out = torch.cat((out, residual), 1)
        out = self.relu(self.bn(self.conv1(out)))
        out = self.relu(self.bn(self.conv2(out)))
        out = self.dropout(out)
        if self.last_layer:
            out = self.conv3(out)
        return out

class UNet2D(nn.Module):
    def __init__(self, in_channels, num_classes, level_channels=[32, 64, 128, 256], bottleneck_channel=512):
        super(UNet2D, self).__init__()
        self.analysis_blocks = nn.ModuleList()
        in_ch = in_channels
        for out_ch in level_channels:
            self.analysis_blocks.append(Conv2DBlock(in_channels=in_ch, out_channels=out_ch))
            in_ch = out_ch
        
        self.bottleNeck = Conv2DBlock(in_channels=in_ch, out_channels=bottleneck_channel, bottleneck=True)
        
        self.synthesis_blocks = nn.ModuleList()
        in_ch = bottleneck_channel
        for idx, res_ch in enumerate(reversed(level_channels)):
            last_layer = idx == len(level_channels) - 1
            self.synthesis_blocks.append(UpConv2DBlock(in_channels=in_ch, res_channels=res_ch, num_classes=num_classes if last_layer else None, last_layer=last_layer))
            in_ch = res_ch
    
    def forward(self, input):
        residuals = []
        out = input
        for block in self.analysis_blocks:
            out, res = block(out)
            residuals.append(res)
        
        out, _ = self.bottleNeck(out)
        
        for block in self.synthesis_blocks:
            out = block(out, residuals.pop())
        return out
