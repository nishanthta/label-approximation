import torch
import torch.nn as nn
import torch.nn.functional as F
from model_defs.unet2d import Conv2DBlock

class CAMModel(nn.Module):
    def __init__(self, in_channels, num_classes, level_channels=[32, 64, 128, 256]):
        super(CAMModel, self).__init__()
        self.analysis_blocks = nn.ModuleList()
        in_ch = in_channels
        for out_ch in level_channels:
            self.analysis_blocks.append(Conv2DBlock(in_channels=in_ch, out_channels=out_ch))
            in_ch = out_ch
        
        # Assuming the bottleneck has the most channels as in the UNet2D architecture
        self.bottleneck = Conv2DBlock(in_channels=in_ch, out_channels=in_ch * 2, bottleneck=True)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_ch * 2, num_classes)  # The bottleneck output channels are in_ch * 2

    def forward(self, x):
        for block in self.analysis_blocks:
            x, _ = block(x)
        
        x, _ = self.bottleneck(x)  # Use bottleneck features for classification
        self.feature_maps = x  # Store for CAM generation

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def get_cam(self, target_class):
        # Generate CAM for the given target class
        weights = self.classifier.weight[target_class]
        cam = torch.matmul(weights, self.feature_maps.view(self.feature_maps.size(1), -1))
        cam = cam.view(self.feature_maps.size(2), self.feature_maps.size(3))
        cam = F.relu(cam)  
        if cam.max() == 0.:
            return torch.zeros_like(cam)
        cam = cam - cam.min()  
        cam = cam / cam.max()
        return cam