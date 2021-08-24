r"""
data_utils.py
Utilities for processing of Data
"""
from typing import Any
import torch
from torch.utils.data.dataset import Dataset
import os
import torch.nn as nn


class Normalize(nn.Module):
    r"""
    Normalize Module
    ---
    Responsible for normalization and denormalization of inputs
    """

    def __init__(self, mean: float, std: float):
        super(Normalize, self).__init__()

        if not torch.is_tensor(mean):
            mean = torch.tensor(mean)
        if not torch.is_tensor(std):
            std = torch.tensor(std)

        self.register_buffer("mean", mean.unsqueeze(-1).unsqueeze(-1))
        self.register_buffer("std", std.unsqueeze(-1).unsqueeze(-1))

    def forward(self, x):
        r"""
        Takes an input and normalises it
        """
        
        return self._standardize(x)

    def inverse_normalize(self, x):
        r"""
        Takes an input and de-normalises it
        """
        return self._inverse_standardize(x)

    def _standardize(self, x):

        if not torch.is_tensor(x):
            x = torch.tensor(x)

        x = x.sub(self.mean).div(self.std)
        return x

    def _inverse_standardize(self, x):
        r"""
        Takes an input and de-normalises it
        """
        if not torch.is_tensor(x):
            x = torch.tensor([x])

        x = x.mul(self.std).add(self.mean)
        return x
