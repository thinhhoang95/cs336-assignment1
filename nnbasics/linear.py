import torch
from torch import nn 
import math

import logging 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, weights: nn.Parameter = None, device: str = None, dtype: str = None) -> None:
        super().__init__()

        if device is None:
            device == torch.device("cpu")
            logger.warning("CPU device was used automatically")

        # Create the weight parameters
        self.W = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))

        if weights is not None:
            # We might need to map the weights to the "W" key so PyTorch could find the parameter "W" defined in self.W
            local_state_dict = {"W": weights}
            self.load_state_dict(local_state_dict) # only works if self.W is a nn.Parameter already
        else:
            # Weight initialization parameters
            init_weight_variance = 2/(in_features + out_features)
            std_weight_variance = math.sqrt(init_weight_variance)
            weight_truncate_bound = 3 * init_weight_variance 
            nn.init.trunc_normal_(self.W, mean = 0, std = std_weight_variance, a = -weight_truncate_bound, b = weight_truncate_bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum('oi, ...i->...o', self.W, x)

