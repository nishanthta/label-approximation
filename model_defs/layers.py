import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableNonLinearCombinationLayer(nn.Module):
    def __init__(self, feature_dim, prior_dim, combined_dim, output_dim):
        super(LearnableNonLinearCombinationLayer, self).__init__()
        self.feature_weights = nn.Linear(feature_dim, combined_dim)
        self.prior_weights = nn.Linear(prior_dim, combined_dim)
        self.nonlinear_weights = nn.Linear(combined_dim, output_dim)

    def forward(self, feature_maps, prior_maps):
        combined_features = torch.tanh(self.feature_weights(feature_maps) + self.prior_weights(prior_maps))
        combined_nonlinear = self.nonlinear_weights(combined_features)
        output = torch.sigmoid(combined_nonlinear)
        return output