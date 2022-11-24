import torch
import torch.nn as nn
import numpy as np


class ComputeLoss(nn.Module):
    def __init__(self, loss_fcn):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()

    def forward(self, pred, true):
        loss = ((pred - true)**2).sum()
        return loss

