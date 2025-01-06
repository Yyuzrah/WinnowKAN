# Import
import json
import os,csv,re

import math
from math import floor

import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm

import pickle
import random

import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch import tensor as Tensor

import scipy
from scipy.sparse import issparse

import pandas as pd
import numpy as np
import scanpy as sc
from scipy.sparse import issparse
import matplotlib.pyplot as plt

from skimage import io, color
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from anndata import concat

import random, torch
import warnings
warnings.filterwarnings("ignore")
import pickle
# from sklearn.model_selection import train_test_split
from anndata import AnnData, read_h5ad

from typing import List, Callable, Union, Any, TypeVar, Tuple
Tensor = TypeVar('torch.tensor')

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )



class WKAN5_cord(nn.Module):

    def __init__(self,
                 in_channels: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(WKAN5_cord, self).__init__()

        if hidden_dims is None:
            hidden_dims = [150, 75]  # [200, 100, 50]

        self.fclayer1 = nn.Sequential(
            KANLinear(in_channels, hidden_dims[0]),
            nn.ReLU()
        )
        self.fclayer2 = nn.Sequential(
            KANLinear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )
        self.fclayer3 = nn.Sequential(
            KANLinear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU()
        )
        self.fclayer4 = nn.Sequential(
            KANLinear(hidden_dims[2], hidden_dims[3]),
            nn.ReLU()
        )

        self.fclayer5 = nn.Sequential(
            KANLinear(hidden_dims[3], hidden_dims[4]),
            nn.ReLU()
        )

        self.fclayer6 = nn.Sequential(
            KANLinear(hidden_dims[4], 2),
            nn.Sigmoid())
        self.to(device)

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        if isinstance(input, (tuple, list)) and len(input) > 0 and isinstance(input[0], torch.Tensor):
            input = [x.to(device) for x in input]
        z = self.fclayer1(input[0])
        z = self.fclayer2(z)
        z = self.fclayer3(z)
        z = self.fclayer4(z)
        z = self.fclayer5(z)
        z = self.fclayer6(z)

        return [z, input]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the spatial coordinates loss function
        :param args: results data and input matrix
        :return:
        """
        cord_pred = args[0]
        input = args[1]

        loss = F.mse_loss(cord_pred, input[1])

        return {'loss': loss}


class Wixos5_cord(nn.Module):

    def __init__(self,
                 in_channels: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(Wixos5_cord, self).__init__()

        if hidden_dims is None:
            hidden_dims = [150, 75]  # [200, 100, 50]

        self.fclayer1 = nn.Sequential(
            KANLinear(in_channels, hidden_dims[0]),
            # nn.Linear(in_channels, hidden_dims[0]),
            nn.ReLU()
        )
        self.fclayer2 = nn.Sequential(
            # KANLinear(hidden_dims[0], hidden_dims[1]),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            # nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU()
        )
        self.fclayer3 = nn.Sequential(
            # KANLinear(hidden_dims[1], hidden_dims[2]),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            # nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU()
        )
        self.fclayer4 = nn.Sequential(
            # KANLinear(hidden_dims[2], hidden_dims[3]),
            nn.Linear(hidden_dims[2], hidden_dims[3]),
            # nn.BatchNorm1d(hidden_dims[3]),
            nn.ReLU()
        )

        self.fclayer5 = nn.Sequential(
            # KANLinear(hidden_dims[3], hidden_dims[4]),
            nn.Linear(hidden_dims[3], hidden_dims[4]),
            # nn.BatchNorm1d(hidden_dims[4]),
            nn.ReLU()

        )

        self.fclayer6 = nn.Sequential(
            # KANLinear(hidden_dims[4], 2),
            nn.Linear(hidden_dims[4], 2),
            # nn.BatchNorm1d(2),
            nn.Sigmoid())
        self.to(device)

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        if isinstance(input, (tuple, list)) and len(input) > 0 and isinstance(input[0], torch.Tensor):
            input = [x.to(device) for x in input]
        z = self.fclayer1(input[0])
        z = self.fclayer2(z)
        z = self.fclayer3(z)
        z = self.fclayer4(z)
        z = self.fclayer5(z)
        z = self.fclayer6(z)

        return [z, input]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the spatial coordinates loss function
        :param args: results data and input matrix
        :return:
        """
        cord_pred = args[0]
        input = args[1]

        loss = F.mse_loss(cord_pred, input[1])

        return {'loss': loss}





class DNN3(nn.Module):

    def __init__(self,
                 in_channels: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(DNN3, self).__init__()

        if hidden_dims is None:
            hidden_dims = [200, 100, 50]

        self.fclayer1 = nn.Sequential(
            nn.Linear(in_channels, hidden_dims[0]),
            # nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU())
        self.fclayer2 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            # nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU())
        self.fclayer3 = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            # nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU())
        self.fclayer4 = nn.Sequential(
            nn.Linear(hidden_dims[2], 2),
            # nn.BatchNorm1d(2),
            nn.Sigmoid())


    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        z = self.fclayer1(input[0])
        z = self.fclayer2(z)
        z = self.fclayer3(z)
        z = self.fclayer4(z)
        return [z, input]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the spatial coordinates loss function
        :param args: results data and input matrix
        :return:
        """
        cord_pred = args[0]
        input = args[1]

        loss = F.mse_loss(cord_pred, input[1])

        return {'loss': loss}


class DNN4(nn.Module):

    def __init__(self,
                 in_channels: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(DNN4, self).__init__()

        if hidden_dims is None:
            hidden_dims = [200, 100, 50]

        self.fclayer1 = nn.Sequential(
            nn.Linear(in_channels, hidden_dims[0]),
            # nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU())
        self.fclayer2 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            # nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU())
        self.fclayer3 = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            # nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU())
        self.fclayer4 = nn.Sequential(
            nn.Linear(hidden_dims[2], hidden_dims[3]),
            # nn.BatchNorm1d(2),
            nn.Sigmoid())
        self.fclayer5 = nn.Sequential(
            nn.Linear(hidden_dims[3], 2),
            # nn.BatchNorm1d(2),
            nn.Sigmoid())


    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        z = self.fclayer1(input[0])
        z = self.fclayer2(z)
        z = self.fclayer3(z)
        z = self.fclayer4(z)
        z = self.fclayer5(z)
        return [z, input]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the spatial coordinates loss function
        :param args: results data and input matrix
        :return:
        """
        cord_pred = args[0]
        input = args[1]

        loss = F.mse_loss(cord_pred, input[1])

        return {'loss': loss}


class DNN5_cord(nn.Module):

    def __init__(self,
                 in_channels: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(DNN5_cord, self).__init__()

        if hidden_dims is None:
            hidden_dims = [200, 100, 50]

        self.fclayer1 = nn.Sequential(
            nn.Linear(in_channels, hidden_dims[0]),
            nn.ReLU())
        self.fclayer2 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU())
        self.fclayer3 = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU())
        self.fclayer4 = nn.Sequential(
            nn.Linear(hidden_dims[2], hidden_dims[3]),
            nn.ReLU())
        self.fclayer5 = nn.Sequential(
            nn.Linear(hidden_dims[3], hidden_dims[4]),
            nn.ReLU())
        self.fclayer6 = nn.Sequential(
            nn.Linear(hidden_dims[4], 2),
            nn.Sigmoid())
        self.to(device)


    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        if isinstance(input, (tuple, list)) and len(input) > 0 and isinstance(input[0], torch.Tensor):
            input = [x.to(device) for x in input]
        z = self.fclayer1(input[0])
        z = self.fclayer2(z)
        z = self.fclayer3(z)
        z = self.fclayer4(z)
        z = self.fclayer5(z)
        z = self.fclayer6(z)
        return [z, input]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the spatial coordinates loss function
        :param args: results data and input matrix
        :return:
        """
        cord_pred = args[0]
        input = args[1]

        loss = F.mse_loss(cord_pred, input[1])

        return {'loss': loss}

class DNN4_cord(nn.Module):

    def __init__(self,
                 in_channels: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(DNN4_cord, self).__init__()

        if hidden_dims is None:
            hidden_dims = [200, 100, 50]

        self.fclayer1 = nn.Sequential(
            nn.Linear(in_channels, hidden_dims[0]),
            nn.ReLU())
        self.fclayer2 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU())
        self.fclayer3 = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU())
        self.fclayer4 = nn.Sequential(
            nn.Linear(hidden_dims[2], hidden_dims[3]),
            nn.ReLU())
        self.fclayer5 = nn.Sequential(
            nn.Linear(hidden_dims[3], 2),
            nn.Sigmoid())
        self.to(device)


    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        if isinstance(input, (tuple, list)) and len(input) > 0 and isinstance(input[0], torch.Tensor):
            input = [x.to(device) for x in input]
        z = self.fclayer1(input[0])
        z = self.fclayer2(z)
        z = self.fclayer3(z)
        z = self.fclayer4(z)
        z = self.fclayer5(z)
        return [z, input]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the spatial coordinates loss function
        :param args: results data and input matrix
        :return:
        """
        cord_pred = args[0]
        input = args[1]

        loss = F.mse_loss(cord_pred, input[1])

        return {'loss': loss}


class DNN3_cord(nn.Module):

    def __init__(self,
                 in_channels: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(DNN3_cord, self).__init__()

        if hidden_dims is None:
            hidden_dims = [200, 100, 50]

        self.fclayer1 = nn.Sequential(
            nn.Linear(in_channels, hidden_dims[0]),
            nn.ReLU())
        self.fclayer2 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU())
        self.fclayer3 = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU())
        self.fclayer4 = nn.Sequential(
            nn.Linear(hidden_dims[2], 2),
            nn.Sigmoid())
        self.to(device)


    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        if isinstance(input, (tuple, list)) and len(input) > 0 and isinstance(input[0], torch.Tensor):
            input = [x.to(device) for x in input]
        z = self.fclayer1(input[0])
        z = self.fclayer2(z)
        z = self.fclayer3(z)
        z = self.fclayer4(z)
        return [z, input]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the spatial coordinates loss function
        :param args: results data and input matrix
        :return:
        """
        cord_pred = args[0]
        input = args[1]

        loss = F.mse_loss(cord_pred, input[1])

        return {'loss': loss}


class KAN(nn.Module):

    def __init__(self,
                 in_channels: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(KAN, self).__init__()

        if hidden_dims is None:
            hidden_dims = [150, 75]  # [200, 100, 50]

        self.fclayer1 = nn.Sequential(
            KANLinear(in_channels, hidden_dims[0]),

        )
        self.fclayer2 = nn.Sequential(
            KANLinear(hidden_dims[0], hidden_dims[1]),

        )
        self.fclayer3 = nn.Sequential(
            KANLinear(hidden_dims[1], hidden_dims[2]),

        )
        self.fclayer4 = nn.Sequential(
            KANLinear(hidden_dims[2], hidden_dims[3]),
        )

        self.fclayer5 = nn.Sequential(
            KANLinear(hidden_dims[3], hidden_dims[4]),

        )

        self.fclayer6 = nn.Sequential(
            KANLinear(hidden_dims[4], 2),
            nn.Sigmoid())
        self.to(device)


    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        if isinstance(input, (tuple, list)) and len(input) > 0 and isinstance(input[0], torch.Tensor):
            input = [x.to(device) for x in input]
        z = self.fclayer1(input[0])
        z = self.fclayer2(z)
        z = self.fclayer3(z)
        z = self.fclayer4(z)
        z = self.fclayer5(z)
        z = self.fclayer6(z)

        return [z, input]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the spatial coordinates loss function
        :param args: results data and input matrix
        :return:
        """
        cord_pred = args[0]
        input = args[1]

        loss = F.mse_loss(cord_pred, input[1])

        return {'loss': loss}


class KAN5_cord(nn.Module):

    def __init__(self,
                 in_channels: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(KAN5_cord, self).__init__()

        if hidden_dims is None:
            hidden_dims = [150, 75]  # [200, 100, 50]

        self.fclayer1 = nn.Sequential(
            KANLinear(in_channels, hidden_dims[0]),

        )
        self.fclayer2 = nn.Sequential(
            KANLinear(hidden_dims[0], hidden_dims[1]),

        )
        self.fclayer3 = nn.Sequential(
            KANLinear(hidden_dims[1], hidden_dims[2]),

        )
        self.fclayer4 = nn.Sequential(
            KANLinear(hidden_dims[2], hidden_dims[3]),
        )

        self.fclayer5 = nn.Sequential(
            KANLinear(hidden_dims[3], hidden_dims[4]),

        )

        self.fclayer6 = nn.Sequential(
            KANLinear(hidden_dims[4], 2),
            nn.Sigmoid())
        self.to(device)


    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        if isinstance(input, (tuple, list)) and len(input) > 0 and isinstance(input[0], torch.Tensor):
            input = [x.to(device) for x in input]
        z = self.fclayer1(input[0])
        z = self.fclayer2(z)
        z = self.fclayer3(z)
        z = self.fclayer4(z)
        z = self.fclayer5(z)
        z = self.fclayer6(z)

        return [z, input]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the spatial coordinates loss function
        :param args: results data and input matrix
        :return:
        """
        cord_pred = args[0]
        input = args[1]

        loss = F.mse_loss(cord_pred, input[1])

        return {'loss': loss}





class KAN3_cord(nn.Module):

    def __init__(self,
                 in_channels: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(KAN3_cord, self).__init__()

        if hidden_dims is None:
            hidden_dims = [150, 75]  # [200, 100, 50]

        self.fclayer1 = nn.Sequential(
            KANLinear(in_channels, hidden_dims[0]),

        )
        self.fclayer2 = nn.Sequential(
            KANLinear(hidden_dims[0], hidden_dims[1]),

        )
        self.fclayer3 = nn.Sequential(
            KANLinear(hidden_dims[1], hidden_dims[2]),

        )
        self.fclayer4 = nn.Sequential(
            KANLinear(hidden_dims[2], 2),
            nn.Sigmoid())
        self.to(device)


    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        if isinstance(input, (tuple, list)) and len(input) > 0 and isinstance(input[0], torch.Tensor):
            input = [x.to(device) for x in input]
        z = self.fclayer1(input[0])
        z = self.fclayer2(z)
        z = self.fclayer3(z)
        z = self.fclayer4(z)


        return [z, input]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the spatial coordinates loss function
        :param args: results data and input matrix
        :return:
        """
        cord_pred = args[0]
        input = args[1]

        loss = F.mse_loss(cord_pred, input[1])

        return {'loss': loss}



class WKAN3_cord(nn.Module):

    def __init__(self,
                 in_channels: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(WKAN3_cord, self).__init__()

        if hidden_dims is None:
            hidden_dims = [150, 75]  # [200, 100, 50]

        self.fclayer1 = nn.Sequential(
            KANLinear(in_channels, hidden_dims[0]),
            nn.ReLU()
        )
        self.fclayer2 = nn.Sequential(
            KANLinear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )
        self.fclayer3 = nn.Sequential(
            KANLinear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU()
        )
        self.fclayer4 = nn.Sequential(
            KANLinear(hidden_dims[2], 2),
            nn.Sigmoid())
        self.to(device)


    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        if isinstance(input, (tuple, list)) and len(input) > 0 and isinstance(input[0], torch.Tensor):
            input = [x.to(device) for x in input]
        z = self.fclayer1(input[0])
        z = self.fclayer2(z)
        z = self.fclayer3(z)
        z = self.fclayer4(z)


        return [z, input]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the spatial coordinates loss function
        :param args: results data and input matrix
        :return:
        """
        cord_pred = args[0]
        input = args[1]

        loss = F.mse_loss(cord_pred, input[1])

        return {'loss': loss}






class KAN4_cord(nn.Module):

    def __init__(self,
                 in_channels: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(KAN4_cord, self).__init__()

        if hidden_dims is None:
            hidden_dims = [150, 75]  # [200, 100, 50]

        self.fclayer1 = nn.Sequential(
            KANLinear(in_channels, hidden_dims[0]),

        )
        self.fclayer2 = nn.Sequential(
            KANLinear(hidden_dims[0], hidden_dims[1]),

        )
        self.fclayer3 = nn.Sequential(
            KANLinear(hidden_dims[1], hidden_dims[2]),

        )
        self.fclayer4 = nn.Sequential(

            KANLinear(hidden_dims[2], hidden_dims[3]),

        )


        self.fclayer5 = nn.Sequential(
            KANLinear(hidden_dims[3], 2),
            nn.Sigmoid())
        self.to(device)


    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        if isinstance(input, (tuple, list)) and len(input) > 0 and isinstance(input[0], torch.Tensor):
            input = [x.to(device) for x in input]
        z = self.fclayer1(input[0])
        z = self.fclayer2(z)
        z = self.fclayer3(z)
        z = self.fclayer4(z)
        z = self.fclayer5(z)

        return [z, input]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the spatial coordinates loss function
        :param args: results data and input matrix
        :return:
        """
        cord_pred = args[0]
        input = args[1]

        loss = F.mse_loss(cord_pred, input[1])

        return {'loss': loss}


class WKAN4_cord(nn.Module):

    def __init__(self,
                 in_channels: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(WKAN4_cord, self).__init__()

        if hidden_dims is None:
            hidden_dims = [150, 75]  # [200, 100, 50]

        self.fclayer1 = nn.Sequential(
            KANLinear(in_channels, hidden_dims[0]),
            nn.ReLU()
        )
        self.fclayer2 = nn.Sequential(
            KANLinear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )
        self.fclayer3 = nn.Sequential(
            KANLinear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU()
        )
        self.fclayer4 = nn.Sequential(
            KANLinear(hidden_dims[2], hidden_dims[3]),
            nn.ReLU()
        )


        self.fclayer5 = nn.Sequential(
            KANLinear(hidden_dims[3], 2),
            nn.Sigmoid())
        self.to(device)


    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        if isinstance(input, (tuple, list)) and len(input) > 0 and isinstance(input[0], torch.Tensor):
            input = [x.to(device) for x in input]
        z = self.fclayer1(input[0])
        z = self.fclayer2(z)
        z = self.fclayer3(z)
        z = self.fclayer4(z)
        z = self.fclayer5(z)

        return [z, input]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the spatial coordinates loss function
        :param args: results data and input matrix
        :return:
        """
        cord_pred = args[0]
        input = args[1]

        loss = F.mse_loss(cord_pred, input[1])

        return {'loss': loss}



class DNNordinal(nn.Module):
    def __init__(self,
                 # in_channels: int,
                 in_channels: int,
                 num_classes: int,
                 hidden_dims: List = None,
                 importance_weights: List = None,
                 **kwargs) -> None:
        super(DNNordinal, self).__init__(in_channels, hidden_dims, **kwargs)

        if hidden_dims is None:
            hidden_dims = [200, 100, 50]

        self.fclayer1 = nn.Sequential(
            nn.Linear(in_channels, hidden_dims[0]),
            nn.Dropout(0.25),
            nn.ReLU())
        self.fclayer2 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU())
        self.fclayer3 = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU())
        self.fclayer4 = nn.Sequential(
            nn.Linear(hidden_dims[2], 1))

        self.coral_bias = torch.nn.Parameter(
            torch.arange(num_classes - 1, 0, -1).float() / (num_classes - 1))

        self.importance_weights = importance_weights






class KAN3_LIBD(DNN3):  # DNNordinal_v2
    """
     This model seperate the white matters from the grey matters (L1-L6)
    """

    def __init__(self,
                 # in_channels: int,
                 in_channels: int,
                 num_classes: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(KAN3_LIBD, self).__init__(in_channels, hidden_dims, **kwargs)

        if hidden_dims is None:
            hidden_dims = [500, 250, 100, 50, 15]

        self.fclayer1 = nn.Sequential(
            KANLinear(in_channels, hidden_dims[0]),
        )
        self.fclayer2 = nn.Sequential(
            KANLinear(hidden_dims[0], hidden_dims[1]),
        )
        self.fclayer3 = nn.Sequential(
            KANLinear(hidden_dims[1], hidden_dims[2]),
        )

        self.fclayer4 = nn.Sequential(
            KANLinear(hidden_dims[2], 1))

        self.coral_bias = torch.nn.Parameter(
            torch.arange(num_classes - 1, 0, -1).float() / (num_classes - 1))
        # To GPU
        self.to(device)

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        """
        Computes forward pass.
        Parameters
        -----------
        x : torch.tensor, shape=(num_examples, num_features)
            Input features.
        Returns
        -----------
        logits : torch.tensor, shape=(num_examples, num_classes-1)
        """
        #
        if isinstance(input, (tuple, list)) and len(input) > 0 and isinstance(input[0], torch.Tensor):
            # input = input.to(device)
            input = [x.to(device) for x in input]
        #
        z = self.fclayer1(input[0])
        z = self.fclayer2(z)
        z = self.fclayer3(z)
        z = self.fclayer4(z)

        logits = z[0, 0] + self.coral_bias
        logitWM = z[0, 0]
        return [logits, logitWM, input]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """Computes the CORAL loss described in
        Cao, Mirjalili, and Raschka (2020)
        *Rank Consistent Ordinal Regression for Neural Networks
           with Application to Age Estimation*
        Pattern Recognition Letters, https://doi.org/10.1016/j.patrec.2020.11.008
        Parameters
        ----------
        logits : torch.tensor, shape(num_examples, num_classes-1)
            Outputs of the CORAL layer.
        levels : torch.tensor, shape(num_examples, num_classes-1)
            True labels represented as extended binary vectors
            (via `coral_pytorch.dataset.levels_from_labelbatch`).
        importance_weights : torch.tensor, shape=(num_classes-1,) (default=None)
            Optional weights for the different labels in levels.
            A tensor of ones, i.e.,
            `torch.ones(num_classes-1, dtype=torch.float32)`
            will result in uniform weights that have the same effect as None.
        reduction : str or None (default='mean')
            If 'mean' or 'sum', returns the averaged or summed loss value across
            all data points (rows) in logits. If None, returns a vector of
            shape (num_examples,)

        """
        logits = args[0]
        logitWM = args[1]
        levelALL = args[2][1]

        levels = levelALL[0, :(levelALL.shape[1] - 1)]
        levelWM = levelALL[0, levelALL.shape[1] - 1]

        if not logits.shape == levels.shape:
            raise ValueError("Please ensure that logits (%s) has the same shape as levels (%s). "
                             % (logits.shape, levels.shape))
        term1 = (F.logsigmoid(logits) * levels + (F.logsigmoid(logits) - logits) * (1 - levels))
        term2 = F.logsigmoid(logitWM) * levelWM + (F.logsigmoid(logitWM) - logitWM + term1) * (1 - levelWM)

        val = (-torch.sum(term2, dim=0))

        # loss = torch.sum(val)
        return {'loss': val}


class WKAN3_LIBD(DNN3):  # DNNordinal_v2
    """
     This model seperate the white matters from the grey matters (L1-L6)
    """

    def __init__(self,
                 # in_channels: int,
                 in_channels: int,
                 num_classes: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(WKAN3_LIBD, self).__init__(in_channels, hidden_dims, **kwargs)

        if hidden_dims is None:
            hidden_dims = [500, 250, 100, 50, 15]

        self.fclayer1 = nn.Sequential(
            KANLinear(in_channels, hidden_dims[0]),
            nn.ReLU()
        )
        self.fclayer2 = nn.Sequential(
            KANLinear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )
        self.fclayer3 = nn.Sequential(
            KANLinear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU()
        )

        self.fclayer4 = nn.Sequential(
            KANLinear(hidden_dims[2], 1))

        self.coral_bias = torch.nn.Parameter(
            torch.arange(num_classes - 1, 0, -1).float() / (num_classes - 1))
        # To GPU
        self.to(device)

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        """
        Computes forward pass.
        Parameters
        -----------
        x : torch.tensor, shape=(num_examples, num_features)
            Input features.
        Returns
        -----------
        logits : torch.tensor, shape=(num_examples, num_classes-1)
        """
        #
        if isinstance(input, (tuple, list)) and len(input) > 0 and isinstance(input[0], torch.Tensor):
            # input = input.to(device)
            input = [x.to(device) for x in input]
        #
        z = self.fclayer1(input[0])
        z = self.fclayer2(z)
        z = self.fclayer3(z)
        z = self.fclayer4(z)

        logits = z[0, 0] + self.coral_bias
        logitWM = z[0, 0]
        return [logits, logitWM, input]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """Computes the CORAL loss described in
        Cao, Mirjalili, and Raschka (2020)
        *Rank Consistent Ordinal Regression for Neural Networks
           with Application to Age Estimation*
        Pattern Recognition Letters, https://doi.org/10.1016/j.patrec.2020.11.008
        Parameters
        ----------
        logits : torch.tensor, shape(num_examples, num_classes-1)
            Outputs of the CORAL layer.
        levels : torch.tensor, shape(num_examples, num_classes-1)
            True labels represented as extended binary vectors
            (via `coral_pytorch.dataset.levels_from_labelbatch`).
        importance_weights : torch.tensor, shape=(num_classes-1,) (default=None)
            Optional weights for the different labels in levels.
            A tensor of ones, i.e.,
            `torch.ones(num_classes-1, dtype=torch.float32)`
            will result in uniform weights that have the same effect as None.
        reduction : str or None (default='mean')
            If 'mean' or 'sum', returns the averaged or summed loss value across
            all data points (rows) in logits. If None, returns a vector of
            shape (num_examples,)

        """
        logits = args[0]
        logitWM = args[1]
        levelALL = args[2][1]

        levels = levelALL[0, :(levelALL.shape[1] - 1)]
        levelWM = levelALL[0, levelALL.shape[1] - 1]

        if not logits.shape == levels.shape:
            raise ValueError("Please ensure that logits (%s) has the same shape as levels (%s). "
                             % (logits.shape, levels.shape))
        term1 = (F.logsigmoid(logits) * levels + (F.logsigmoid(logits) - logits) * (1 - levels))
        term2 = F.logsigmoid(logitWM) * levelWM + (F.logsigmoid(logitWM) - logitWM + term1) * (1 - levelWM)

        val = (-torch.sum(term2, dim=0))

        # loss = torch.sum(val)
        return {'loss': val}




class KAN4_LIBD(DNN4):  # DNNordinal_v2
    """
     This model seperate the white matters from the grey matters (L1-L6)
    """

    def __init__(self,
                 # in_channels: int,
                 in_channels: int,
                 num_classes: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(KAN4_LIBD, self).__init__(in_channels, hidden_dims, **kwargs)

        if hidden_dims is None:
            hidden_dims = [200, 100, 50]

        self.fclayer1 = nn.Sequential(
            KANLinear(in_channels, hidden_dims[0]),
        )
        self.fclayer2 = nn.Sequential(
            KANLinear(hidden_dims[0], hidden_dims[1]),
        )
        self.fclayer3 = nn.Sequential(
            KANLinear(hidden_dims[1], hidden_dims[2]),
        )
        self.fclayer4 = nn.Sequential(
            KANLinear(hidden_dims[2], hidden_dims[3]),
        )
        self.fclayer5 = nn.Sequential(
            KANLinear(hidden_dims[3], 1))

        self.coral_bias = torch.nn.Parameter(
            torch.arange(num_classes - 1, 0, -1).float() / (num_classes - 1))
        # To GPU
        self.to(device)

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        """
        Computes forward pass.
        Parameters
        -----------
        x : torch.tensor, shape=(num_examples, num_features)
            Input features.
        Returns
        -----------
        logits : torch.tensor, shape=(num_examples, num_classes-1)
        """
        #
        if isinstance(input, (tuple, list)) and len(input) > 0 and isinstance(input[0], torch.Tensor):
            # input = input.to(device)
            input = [x.to(device) for x in input]
        #
        z = self.fclayer1(input[0])
        z = self.fclayer2(z)
        z = self.fclayer3(z)
        z = self.fclayer4(z)
        z = self.fclayer5(z)
        logits = z[0, 0] + self.coral_bias
        logitWM = z[0, 0]
        return [logits, logitWM, input]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """Computes the CORAL loss described in
        Cao, Mirjalili, and Raschka (2020)
        *Rank Consistent Ordinal Regression for Neural Networks
           with Application to Age Estimation*
        Pattern Recognition Letters, https://doi.org/10.1016/j.patrec.2020.11.008
        Parameters
        ----------
        logits : torch.tensor, shape(num_examples, num_classes-1)
            Outputs of the CORAL layer.
        levels : torch.tensor, shape(num_examples, num_classes-1)
            True labels represented as extended binary vectors
            (via `coral_pytorch.dataset.levels_from_labelbatch`).
        importance_weights : torch.tensor, shape=(num_classes-1,) (default=None)
            Optional weights for the different labels in levels.
            A tensor of ones, i.e.,
            `torch.ones(num_classes-1, dtype=torch.float32)`
            will result in uniform weights that have the same effect as None.
        reduction : str or None (default='mean')
            If 'mean' or 'sum', returns the averaged or summed loss value across
            all data points (rows) in logits. If None, returns a vector of
            shape (num_examples,)

        """
        logits = args[0]
        logitWM = args[1]
        levelALL = args[2][1]

        levels = levelALL[0, :(levelALL.shape[1] - 1)]
        levelWM = levelALL[0, levelALL.shape[1] - 1]

        if not logits.shape == levels.shape:
            raise ValueError("Please ensure that logits (%s) has the same shape as levels (%s). "
                             % (logits.shape, levels.shape))
        term1 = (F.logsigmoid(logits) * levels + (F.logsigmoid(logits) - logits) * (1 - levels))
        term2 = F.logsigmoid(logitWM) * levelWM + (F.logsigmoid(logitWM) - logitWM + term1) * (1 - levelWM)

        val = (-torch.sum(term2, dim=0))

        # loss = torch.sum(val)
        return {'loss': val}


class WKAN4_LIBD(DNN4):  # DNNordinal_v2
    """
     This model seperate the white matters from the grey matters (L1-L6)
    """

    def __init__(self,
                 # in_channels: int,
                 in_channels: int,
                 num_classes: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(WKAN4_LIBD, self).__init__(in_channels, hidden_dims, **kwargs)

        if hidden_dims is None:
            hidden_dims = [200, 100, 50]

        self.fclayer1 = nn.Sequential(
            KANLinear(in_channels, hidden_dims[0]),
            nn.ReLU()
        )
        self.fclayer2 = nn.Sequential(
            KANLinear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )
        self.fclayer3 = nn.Sequential(
            KANLinear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU()
        )
        self.fclayer4 = nn.Sequential(
            KANLinear(hidden_dims[2], hidden_dims[3]),
            nn.ReLU()
        )
        self.fclayer5 = nn.Sequential(
            KANLinear(hidden_dims[3], 1))

        self.coral_bias = torch.nn.Parameter(
            torch.arange(num_classes - 1, 0, -1).float() / (num_classes - 1))
        # To GPU
        self.to(device)

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        """
        Computes forward pass.
        Parameters
        -----------
        x : torch.tensor, shape=(num_examples, num_features)
            Input features.
        Returns
        -----------
        logits : torch.tensor, shape=(num_examples, num_classes-1)
        """
        #
        if isinstance(input, (tuple, list)) and len(input) > 0 and isinstance(input[0], torch.Tensor):
            # input = input.to(device)
            input = [x.to(device) for x in input]
        #
        z = self.fclayer1(input[0])
        z = self.fclayer2(z)
        z = self.fclayer3(z)
        z = self.fclayer4(z)
        z = self.fclayer5(z)
        logits = z[0, 0] + self.coral_bias
        logitWM = z[0, 0]
        return [logits, logitWM, input]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """Computes the CORAL loss described in
        Cao, Mirjalili, and Raschka (2020)
        *Rank Consistent Ordinal Regression for Neural Networks
           with Application to Age Estimation*
        Pattern Recognition Letters, https://doi.org/10.1016/j.patrec.2020.11.008
        Parameters
        ----------
        logits : torch.tensor, shape(num_examples, num_classes-1)
            Outputs of the CORAL layer.
        levels : torch.tensor, shape(num_examples, num_classes-1)
            True labels represented as extended binary vectors
            (via `coral_pytorch.dataset.levels_from_labelbatch`).
        importance_weights : torch.tensor, shape=(num_classes-1,) (default=None)
            Optional weights for the different labels in levels.
            A tensor of ones, i.e.,
            `torch.ones(num_classes-1, dtype=torch.float32)`
            will result in uniform weights that have the same effect as None.
        reduction : str or None (default='mean')
            If 'mean' or 'sum', returns the averaged or summed loss value across
            all data points (rows) in logits. If None, returns a vector of
            shape (num_examples,)

        """
        logits = args[0]
        logitWM = args[1]
        levelALL = args[2][1]

        levels = levelALL[0, :(levelALL.shape[1] - 1)]
        levelWM = levelALL[0, levelALL.shape[1] - 1]

        if not logits.shape == levels.shape:
            raise ValueError("Please ensure that logits (%s) has the same shape as levels (%s). "
                             % (logits.shape, levels.shape))
        term1 = (F.logsigmoid(logits) * levels + (F.logsigmoid(logits) - logits) * (1 - levels))
        term2 = F.logsigmoid(logitWM) * levelWM + (F.logsigmoid(logitWM) - logitWM + term1) * (1 - levelWM)

        val = (-torch.sum(term2, dim=0))

        # loss = torch.sum(val)
        return {'loss': val}






class KAN5_LIBD(DNN3):

    def __init__(self,
                 # in_channels: int,
                 in_channels: int,
                 num_classes: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(KAN5_LIBD, self).__init__(in_channels, hidden_dims, **kwargs)

        if hidden_dims is None:
            hidden_dims = [500, 250, 100, 50, 15]

        self.fclayer1 = nn.Sequential(
            KANLinear(in_channels, hidden_dims[0]),
        )
        self.fclayer2 = nn.Sequential(
            KANLinear(hidden_dims[0], hidden_dims[1]),
        )
        self.fclayer3 = nn.Sequential(
            KANLinear(hidden_dims[1], hidden_dims[2]),
        )
        self.fclayer4 = nn.Sequential(
            KANLinear(hidden_dims[2], hidden_dims[3]),
        )

        self.fclayer5 = nn.Sequential(
            KANLinear(hidden_dims[3], hidden_dims[4]),
        )
        self.fclayer6 = nn.Sequential(
            KANLinear(hidden_dims[4], 1))

        self.coral_bias = torch.nn.Parameter(
            torch.arange(num_classes - 1, 0, -1).float() / (num_classes - 1))
        # To GPU
        self.to(device)

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        """
        Computes forward pass.
        Parameters
        -----------
        x : torch.tensor, shape=(num_examples, num_features)
            Input features.
        Returns
        -----------
        logits : torch.tensor, shape=(num_examples, num_classes-1)
        """
        #
        if isinstance(input, (tuple, list)) and len(input) > 0 and isinstance(input[0], torch.Tensor):
            # input = input.to(device)
            input = [x.to(device) for x in input]
        #
        z = self.fclayer1(input[0])
        z = self.fclayer2(z)
        z = self.fclayer3(z)
        z = self.fclayer4(z)
        z = self.fclayer5(z)
        z = self.fclayer6(z)
        logits = z[0, 0] + self.coral_bias
        logitWM = z[0, 0]
        return [logits, logitWM, input]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """Computes the CORAL loss described in
        Cao, Mirjalili, and Raschka (2020)
        *Rank Consistent Ordinal Regression for Neural Networks
           with Application to Age Estimation*
        Pattern Recognition Letters, https://doi.org/10.1016/j.patrec.2020.11.008
        Parameters
        ----------
        logits : torch.tensor, shape(num_examples, num_classes-1)
            Outputs of the CORAL layer.
        levels : torch.tensor, shape(num_examples, num_classes-1)
            True labels represented as extended binary vectors
            (via `coral_pytorch.dataset.levels_from_labelbatch`).
        importance_weights : torch.tensor, shape=(num_classes-1,) (default=None)
            Optional weights for the different labels in levels.
            A tensor of ones, i.e.,
            `torch.ones(num_classes-1, dtype=torch.float32)`
            will result in uniform weights that have the same effect as None.
        reduction : str or None (default='mean')
            If 'mean' or 'sum', returns the averaged or summed loss value across
            all data points (rows) in logits. If None, returns a vector of
            shape (num_examples,)

        """
        logits = args[0]
        logitWM = args[1]
        levelALL = args[2][1]

        levels = levelALL[0, :(levelALL.shape[1] - 1)]
        levelWM = levelALL[0, levelALL.shape[1] - 1]

        if not logits.shape == levels.shape:
            raise ValueError("Please ensure that logits (%s) has the same shape as levels (%s). "
                             % (logits.shape, levels.shape))
        term1 = (F.logsigmoid(logits) * levels + (F.logsigmoid(logits) - logits) * (1 - levels))
        term2 = F.logsigmoid(logitWM) * levelWM + (F.logsigmoid(logitWM) - logitWM + term1) * (1 - levelWM)

        val = (-torch.sum(term2, dim=0))

        # loss = torch.sum(val)
        return {'loss': val}


class WKAN5_LIBD(DNN3):  # DNNordinal_v2
    """
     This model seperate the white matters from the grey matters (L1-L6)
    """

    def __init__(self,
                 # in_channels: int,
                 in_channels: int,
                 num_classes: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(WKAN5_LIBD, self).__init__(in_channels, hidden_dims, **kwargs)

        if hidden_dims is None:
            hidden_dims = [500, 250, 100, 50, 15]

        self.fclayer1 = nn.Sequential(
            KANLinear(in_channels, hidden_dims[0]),
            nn.ReLU()
        )
        self.fclayer2 = nn.Sequential(
            KANLinear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )
        self.fclayer3 = nn.Sequential(
            KANLinear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU()
        )
        self.fclayer4 = nn.Sequential(
            KANLinear(hidden_dims[2], hidden_dims[3]),
            nn.ReLU()
        )

        self.fclayer5 = nn.Sequential(
            KANLinear(hidden_dims[3], hidden_dims[4]),
            nn.ReLU()
        )
        self.fclayer6 = nn.Sequential(
            KANLinear(hidden_dims[4], 1))

        self.coral_bias = torch.nn.Parameter(
            torch.arange(num_classes - 1, 0, -1).float() / (num_classes - 1))
        # To GPU
        self.to(device)

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        """
        Computes forward pass.
        Parameters
        -----------
        x : torch.tensor, shape=(num_examples, num_features)
            Input features.
        Returns
        -----------
        logits : torch.tensor, shape=(num_examples, num_classes-1)
        """
        #
        if isinstance(input, (tuple, list)) and len(input) > 0 and isinstance(input[0], torch.Tensor):
            # input = input.to(device)
            input = [x.to(device) for x in input]
        #
        z = self.fclayer1(input[0])
        z = self.fclayer2(z)
        z = self.fclayer3(z)
        z = self.fclayer4(z)
        z = self.fclayer5(z)
        z = self.fclayer6(z)
        logits = z[0, 0] + self.coral_bias
        logitWM = z[0, 0]
        return [logits, logitWM, input]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """Computes the CORAL loss described in
        Cao, Mirjalili, and Raschka (2020)
        *Rank Consistent Ordinal Regression for Neural Networks
           with Application to Age Estimation*
        Pattern Recognition Letters, https://doi.org/10.1016/j.patrec.2020.11.008
        Parameters
        ----------
        logits : torch.tensor, shape(num_examples, num_classes-1)
            Outputs of the CORAL layer.
        levels : torch.tensor, shape(num_examples, num_classes-1)
            True labels represented as extended binary vectors
            (via `coral_pytorch.dataset.levels_from_labelbatch`).
        importance_weights : torch.tensor, shape=(num_classes-1,) (default=None)
            Optional weights for the different labels in levels.
            A tensor of ones, i.e.,
            `torch.ones(num_classes-1, dtype=torch.float32)`
            will result in uniform weights that have the same effect as None.
        reduction : str or None (default='mean')
            If 'mean' or 'sum', returns the averaged or summed loss value across
            all data points (rows) in logits. If None, returns a vector of
            shape (num_examples,)

        """
        logits = args[0]
        logitWM = args[1]
        levelALL = args[2][1]

        levels = levelALL[0, :(levelALL.shape[1] - 1)]
        levelWM = levelALL[0, levelALL.shape[1] - 1]

        if not logits.shape == levels.shape:
            raise ValueError("Please ensure that logits (%s) has the same shape as levels (%s). "
                             % (logits.shape, levels.shape))
        term1 = (F.logsigmoid(logits) * levels + (F.logsigmoid(logits) - logits) * (1 - levels))
        term2 = F.logsigmoid(logitWM) * levelWM + (F.logsigmoid(logitWM) - logitWM + term1) * (1 - levelWM)

        val = (-torch.sum(term2, dim=0))

        # loss = torch.sum(val)
        return {'loss': val}


class DNN(nn.Module):

    def __init__(self,
                 in_channels: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(DNN, self).__init__()

        if hidden_dims is None:
            hidden_dims = [200, 100, 50]

        self.fclayer1 = nn.Sequential(
            nn.Linear(in_channels, hidden_dims[0]),
            # nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU())
        self.fclayer2 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            # nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU())
        self.fclayer3 = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            # nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU())
        self.fclayer4 = nn.Sequential(
            nn.Linear(hidden_dims[2], 2),
            # nn.BatchNorm1d(2),
            nn.Sigmoid())

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        z = self.fclayer1(input[0])
        z = self.fclayer2(z)
        z = self.fclayer3(z)
        z = self.fclayer4(z)
        return [z, input]



class DNN5_LIBD(DNN):  # DNNordinal_v2
    """
     This model seperate the white matters from the grey matters (L1-L6)
    """

    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(DNN5_LIBD, self).__init__(in_channels, hidden_dims, **kwargs)

        if hidden_dims is None:
            hidden_dims = [200, 100, 50]

        self.fclayer1 = nn.Sequential(
            nn.Linear(in_channels, hidden_dims[0]),
            nn.ReLU()
        )
        self.fclayer2 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )
        self.fclayer3 = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU()
        )
        self.fclayer4 = nn.Sequential(
            nn.Linear(hidden_dims[2], hidden_dims[3]),
            nn.ReLU()
        )
        self.fclayer5 = nn.Sequential(
            nn.Linear(hidden_dims[3], hidden_dims[4]),
            nn.ReLU()
        )
        self.fclayer6 = nn.Sequential(
            nn.Linear(hidden_dims[4], 1))

        self.coral_bias = torch.nn.Parameter(
            torch.arange(num_classes - 1, 0, -1).float() / (num_classes - 1))
        # To GPU
        self.to(device)

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        """
        Computes forward pass.
        Parameters
        -----------
        x : torch.tensor, shape=(num_examples, num_features)
            Input features.
        Returns
        -----------
        logits : torch.tensor, shape=(num_examples, num_classes-1)
        """
        #
        if isinstance(input, (tuple, list)) and len(input) > 0 and isinstance(input[0], torch.Tensor):
            # input = input.to(device)
            input = [x.to(device) for x in input]
        #
        z = self.fclayer1(input[0])
        z = self.fclayer2(z)
        z = self.fclayer3(z)
        z = self.fclayer4(z)
        z = self.fclayer5(z)
        z = self.fclayer6(z)
        logits = z[0, 0] + self.coral_bias
        logitWM = z[0, 0]
        return [logits, logitWM, input]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """Computes the CORAL loss described in
        Cao, Mirjalili, and Raschka (2020)
        *Rank Consistent Ordinal Regression for Neural Networks
           with Application to Age Estimation*
        Pattern Recognition Letters, https://doi.org/10.1016/j.patrec.2020.11.008
        Parameters
        ----------
        logits : torch.tensor, shape(num_examples, num_classes-1)
            Outputs of the CORAL layer.
        levels : torch.tensor, shape(num_examples, num_classes-1)
            True labels represented as extended binary vectors
            (via `coral_pytorch.dataset.levels_from_labelbatch`).
        importance_weights : torch.tensor, shape=(num_classes-1,) (default=None)
            Optional weights for the different labels in levels.
            A tensor of ones, i.e.,
            `torch.ones(num_classes-1, dtype=torch.float32)`
            will result in uniform weights that have the same effect as None.
        reduction : str or None (default='mean')
            If 'mean' or 'sum', returns the averaged or summed loss value across
            all data points (rows) in logits. If None, returns a vector of
            shape (num_examples,)

        """
        logits = args[0]
        logitWM = args[1]
        levelALL = args[2][1]

        levels = levelALL[0, :(levelALL.shape[1] - 1)]
        levelWM = levelALL[0, levelALL.shape[1] - 1]

        if not logits.shape == levels.shape:
            raise ValueError("Please ensure that logits (%s) has the same shape as levels (%s). "
                             % (logits.shape, levels.shape))
        term1 = (F.logsigmoid(logits) * levels + (F.logsigmoid(logits) - logits) * (1 - levels))
        term2 = F.logsigmoid(logitWM) * levelWM + (F.logsigmoid(logitWM) - logitWM + term1) * (1 - levelWM)

        val = (-torch.sum(term2, dim=0))

        # loss = torch.sum(val)
        return {'loss': val}

class DNN4_LIBD(nn.Module):  # DNNordinal_v2
    """
     This model seperate the white matters from the grey matters (L1-L6)
    """

    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(DNN4_LIBD, self).__init__(in_channels, hidden_dims, **kwargs)

        if hidden_dims is None:
            hidden_dims = [200, 100, 50]

        self.fclayer1 = nn.Sequential(
            nn.Linear(in_channels, hidden_dims[0]),
            nn.ReLU()
        )
        self.fclayer2 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )
        self.fclayer3 = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU()
        )
        self.fclayer4 = nn.Sequential(
            nn.Linear(hidden_dims[2], hidden_dims[3]),
            nn.ReLU()
        )

        self.fclayer5 = nn.Sequential(
            nn.Linear(hidden_dims[3], 1))

        self.coral_bias = torch.nn.Parameter(
            torch.arange(num_classes - 1, 0, -1).float() / (num_classes - 1))
        # To GPU
        self.to(device)

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        """
        Computes forward pass.
        Parameters
        -----------
        x : torch.tensor, shape=(num_examples, num_features)
            Input features.
        Returns
        -----------
        logits : torch.tensor, shape=(num_examples, num_classes-1)
        """
        #
        if isinstance(input, (tuple, list)) and len(input) > 0 and isinstance(input[0], torch.Tensor):
            # input = input.to(device)
            input = [x.to(device) for x in input]
        #
        z = self.fclayer1(input[0])
        z = self.fclayer2(z)
        z = self.fclayer3(z)
        z = self.fclayer4(z)
        z = self.fclayer5(z)
        logits = z[0, 0] + self.coral_bias
        logitWM = z[0, 0]
        return [logits, logitWM, input]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """Computes the CORAL loss described in
        Cao, Mirjalili, and Raschka (2020)
        *Rank Consistent Ordinal Regression for Neural Networks
           with Application to Age Estimation*
        Pattern Recognition Letters, https://doi.org/10.1016/j.patrec.2020.11.008
        Parameters
        ----------
        logits : torch.tensor, shape(num_examples, num_classes-1)
            Outputs of the CORAL layer.
        levels : torch.tensor, shape(num_examples, num_classes-1)
            True labels represented as extended binary vectors
            (via `coral_pytorch.dataset.levels_from_labelbatch`).
        importance_weights : torch.tensor, shape=(num_classes-1,) (default=None)
            Optional weights for the different labels in levels.
            A tensor of ones, i.e.,
            `torch.ones(num_classes-1, dtype=torch.float32)`
            will result in uniform weights that have the same effect as None.
        reduction : str or None (default='mean')
            If 'mean' or 'sum', returns the averaged or summed loss value across
            all data points (rows) in logits. If None, returns a vector of
            shape (num_examples,)

        """
        logits = args[0]
        logitWM = args[1]
        levelALL = args[2][1]

        levels = levelALL[0, :(levelALL.shape[1] - 1)]
        levelWM = levelALL[0, levelALL.shape[1] - 1]

        if not logits.shape == levels.shape:
            raise ValueError("Please ensure that logits (%s) has the same shape as levels (%s). "
                             % (logits.shape, levels.shape))
        term1 = (F.logsigmoid(logits) * levels + (F.logsigmoid(logits) - logits) * (1 - levels))
        term2 = F.logsigmoid(logitWM) * levelWM + (F.logsigmoid(logitWM) - logitWM + term1) * (1 - levelWM)

        val = (-torch.sum(term2, dim=0))

        # loss = torch.sum(val)
        return {'loss': val}


class DNN3_LIBD(nn.Module):  # DNNordinal_v2
    """
     This model seperate the white matters from the grey matters (L1-L6)
    """

    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(DNN3_LIBD, self).__init__(in_channels, hidden_dims, **kwargs)

        if hidden_dims is None:
            hidden_dims = [200, 100, 50]

        self.fclayer1 = nn.Sequential(
            nn.Linear(in_channels, hidden_dims[0]),
            nn.ReLU()
        )
        self.fclayer2 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )
        self.fclayer3 = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU()
        )
        self.fclayer4 = nn.Sequential(
            nn.Linear(hidden_dims[2], 1))

        self.coral_bias = torch.nn.Parameter(
            torch.arange(num_classes - 1, 0, -1).float() / (num_classes - 1))
        # To GPU
        self.to(device)

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        """
        Computes forward pass.
        Parameters
        -----------
        x : torch.tensor, shape=(num_examples, num_features)
            Input features.
        Returns
        -----------
        logits : torch.tensor, shape=(num_examples, num_classes-1)
        """
        #
        if isinstance(input, (tuple, list)) and len(input) > 0 and isinstance(input[0], torch.Tensor):
            # input = input.to(device)
            input = [x.to(device) for x in input]
        #
        z = self.fclayer1(input[0])
        z = self.fclayer2(z)
        z = self.fclayer3(z)
        z = self.fclayer4(z)
        logits = z[0, 0] + self.coral_bias
        logitWM = z[0, 0]
        return [logits, logitWM, input]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """Computes the CORAL loss described in
        Cao, Mirjalili, and Raschka (2020)
        *Rank Consistent Ordinal Regression for Neural Networks
           with Application to Age Estimation*
        Pattern Recognition Letters, https://doi.org/10.1016/j.patrec.2020.11.008
        Parameters
        ----------
        logits : torch.tensor, shape(num_examples, num_classes-1)
            Outputs of the CORAL layer.
        levels : torch.tensor, shape(num_examples, num_classes-1)
            True labels represented as extended binary vectors
            (via `coral_pytorch.dataset.levels_from_labelbatch`).
        importance_weights : torch.tensor, shape=(num_classes-1,) (default=None)
            Optional weights for the different labels in levels.
            A tensor of ones, i.e.,
            `torch.ones(num_classes-1, dtype=torch.float32)`
            will result in uniform weights that have the same effect as None.
        reduction : str or None (default='mean')
            If 'mean' or 'sum', returns the averaged or summed loss value across
            all data points (rows) in logits. If None, returns a vector of
            shape (num_examples,)

        """
        logits = args[0]
        logitWM = args[1]
        levelALL = args[2][1]

        levels = levelALL[0, :(levelALL.shape[1] - 1)]
        levelWM = levelALL[0, levelALL.shape[1] - 1]

        if not logits.shape == levels.shape:
            raise ValueError("Please ensure that logits (%s) has the same shape as levels (%s). "
                             % (logits.shape, levels.shape))
        term1 = (F.logsigmoid(logits) * levels + (F.logsigmoid(logits) - logits) * (1 - levels))
        term2 = F.logsigmoid(logitWM) * levelWM + (F.logsigmoid(logitWM) - logitWM + term1) * (1 - levelWM)

        val = (-torch.sum(term2, dim=0))

        # loss = torch.sum(val)
        return {'loss': val}





class KAN_LIBD(nn.Module):
    """
     This model seperate the white matters from the grey matters (L1-L6)
    """

    def __init__(self,
                 # in_channels: int,
                 in_channels: int,
                 num_classes: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(KAN_LIBD, self).__init__(in_channels, hidden_dims, **kwargs)

        if hidden_dims is None:
            hidden_dims = [200, 100, 50]

        self.fclayer1 = nn.Sequential(
            KANLinear(in_channels, hidden_dims[0]),
        )
        self.fclayer2 = nn.Sequential(
            KANLinear(hidden_dims[0], hidden_dims[1]),
        )
        self.fclayer3 = nn.Sequential(
            KANLinear(hidden_dims[1], hidden_dims[2]),
        )
        self.fclayer4 = nn.Sequential(
            KANLinear(hidden_dims[2], hidden_dims[3]),
        )
        self.fclayer5 = nn.Sequential(
            KANLinear(hidden_dims[3], hidden_dims[4]),
        )
        self.fclayer6 = nn.Sequential(
            KANLinear(hidden_dims[4], 1))

        self.coral_bias = torch.nn.Parameter(
            torch.arange(num_classes - 1, 0, -1).float() / (num_classes - 1))
        # To GPU
        self.to(device)

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        """
        Computes forward pass.
        Parameters
        -----------
        x : torch.tensor, shape=(num_examples, num_features)
            Input features.
        Returns
        -----------
        logits : torch.tensor, shape=(num_examples, num_classes-1)
        """
        #
        if isinstance(input, (tuple, list)) and len(input) > 0 and isinstance(input[0], torch.Tensor):
            # input = input.to(device)
            input = [x.to(device) for x in input]
        #
        z = self.fclayer1(input[0])
        z = self.fclayer2(z)
        z = self.fclayer3(z)
        z = self.fclayer4(z)
        z = self.fclayer5(z)
        z = self.fclayer6(z)
        logits = z[0, 0] + self.coral_bias
        logitWM = z[0, 0]
        return [logits, logitWM, input]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """Computes the CORAL loss described in
        Cao, Mirjalili, and Raschka (2020)
        *Rank Consistent Ordinal Regression for Neural Networks
           with Application to Age Estimation*
        Pattern Recognition Letters, https://doi.org/10.1016/j.patrec.2020.11.008
        Parameters
        ----------
        logits : torch.tensor, shape(num_examples, num_classes-1)
            Outputs of the CORAL layer.
        levels : torch.tensor, shape(num_examples, num_classes-1)
            True labels represented as extended binary vectors
            (via `coral_pytorch.dataset.levels_from_labelbatch`).
        importance_weights : torch.tensor, shape=(num_classes-1,) (default=None)
            Optional weights for the different labels in levels.
            A tensor of ones, i.e.,
            `torch.ones(num_classes-1, dtype=torch.float32)`
            will result in uniform weights that have the same effect as None.
        reduction : str or None (default='mean')
            If 'mean' or 'sum', returns the averaged or summed loss value across
            all data points (rows) in logits. If None, returns a vector of
            shape (num_examples,)

        """
        logits = args[0]
        logitWM = args[1]
        levelALL = args[2][1]

        levels = levelALL[0, :(levelALL.shape[1] - 1)]
        levelWM = levelALL[0, levelALL.shape[1] - 1]

        if not logits.shape == levels.shape:
            raise ValueError("Please ensure that logits (%s) has the same shape as levels (%s). "
                             % (logits.shape, levels.shape))
        term1 = (F.logsigmoid(logits) * levels + (F.logsigmoid(logits) - logits) * (1 - levels))
        term2 = F.logsigmoid(logitWM) * levelWM + (F.logsigmoid(logitWM) - logitWM + term1) * (1 - levelWM)

        val = (-torch.sum(term2, dim=0))

        # loss = torch.sum(val)
        return {'loss': val}



