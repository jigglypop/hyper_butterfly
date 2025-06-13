import torch
import torch.nn as nn
from .manifolds import Poincare

class HyperbolicLinear(nn.Module):
    def __init__(self, in_features, out_features, curvature=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.manifold = Poincare(c=curvature)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight, gain=1.414)
        nn.init.zeros_(self.bias)
    def forward(self, x):
        mv = self.manifold.mobius_add(
            torch.matmul(x, self.weight.t()),
            self.bias
        )
        return mv

class HyperbolicActivation(nn.Module):
    def __init__(self, activation_fn=torch.relu):
        super().__init__()
        self.activation = activation_fn
    def forward(self, x):
        # This is a simplified projection and activation.
        # A more rigorous approach would use logmap/expmap.
        return self.activation(x) 