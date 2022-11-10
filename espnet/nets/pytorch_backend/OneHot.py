
import torch
import torch.nn.functional as F
from torch import nn


class OneHot(nn.Module):
    """Convert to one-hot vector.

    Args:
        depth (int): Dimension of one-hot vector.

    """

    def __init__(self, depth):
        super(OneHot, self).__init__()
        self.depth = depth

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (LongTensor): long tensor variable with the shape  (B, T)

        Returns:
            Tensor: float tensor variable with the shape (B, depth, T)

        """
        x = x % self.depth
        x = torch.unsqueeze(x, 2)
        x_onehot = x.new_zeros(x.size(0), x.size(1), self.depth).float()

        return x_onehot.scatter_(2, x, 1)