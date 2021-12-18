import torch
import torch.nn as nn
import torch.nn.functional as F


class UBLoss(nn.Module):
    def __init__(self, n_positive: int, n_negative: int):
        super().__init__()
        self.negative_weight = n_positive / n_negative
        self.w = torch.FloatTensor([self.negative_weight, 1])

    def forward(self, x, y):
        loss = F.cross_entropy(x, y, weight=self.w)
        return loss
