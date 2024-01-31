from torch import nn
import torch
from data_defs.utils import create_approximated_spline_volume
from config import *

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + self.smooth)/(inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice
    
class DiscrepancyLoss(nn.Module):
    def __init__(self):
        super(DiscrepancyLoss, self).__init__()

    def forward(self, mc_predictions, annotation_uncertainty):
        """
        Computes the discrepancy loss between model uncertainty (derived from MC Dropout predictions)
        and annotation variability.

        Parameters:
        - mc_predictions: A tensor of shape (N, 1, C, H, W) representing N sets of predictions from the model
                          under different MC Dropout configurations, where B is the batch size, C is the number
                          of classes, H and W are the height and width of the input images, respectively.
        - annotation_variability: A tensor of shape (1, H, W) representing the variability in annotations
                                  across different annotators for each pixel, typically the variance across labels.

        Returns:
        - loss: A scalar tensor representing the discrepancy loss.
        """

        assert mc_predictions.dim() == 4, "MC predictions should have 5 dimensions (N, B, C, H, W)"

        model_uncertainty = mc_predictions.var(dim=1, unbiased=False)

        assert model_uncertainty.shape == annotation_uncertainty.shape, "Model uncertainty and annotation variability must have the same shape"

        discrepancy = torch.abs(model_uncertainty - annotation_uncertainty)
        loss = discrepancy.mean()

        return loss