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





class FlexibleKANLIBD(nn.Module):
    def __init__(self, in_channels, hidden_dims, task_type='regression', num_classes=None, use_relu=False, layer_type='KAN'):
        super(FlexibleKANLIBD, self).__init__()
        self.task_type = task_type
        self.use_relu = use_relu
        self.layer_type = layer_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 检查分类任务是否提供了 num_classes
        if task_type == 'classification' and num_classes is None:
            raise ValueError("分类任务必须指定 num_classes！")

        # 动态构建网络层
        layers = []
        input_dim = in_channels
        for hidden_dim in hidden_dims:
            if layer_type == 'KAN':
                layer = KANLinear(input_dim, hidden_dim)  # 假设 KANLinear 已定义
            elif layer_type == 'MLP':
                layer = nn.Linear(input_dim, hidden_dim)
            else:
                raise ValueError("layer_type 只能是 'KAN' 或 'MLP'！")
            layers.append(layer)
            if use_relu:
                layers.append(nn.ReLU())
            input_dim = hidden_dim

        # 根据任务类型设置输出层
        if task_type == 'regression':
            if layer_type == 'KAN':
                layers.append(KANLinear(input_dim, 2))
            else:
                layers.append(nn.Linear(input_dim, 2))
            layers.append(nn.Sigmoid())
        elif task_type == 'classification':
            if layer_type == 'KAN':
                layers.append(KANLinear(input_dim, 1))
            else:
                layers.append(nn.Linear(input_dim, 1))

        self.network = nn.Sequential(*layers)

        # 分类任务加 CORAL 偏置
        if task_type == 'classification':
            self.coral_bias = nn.Parameter(
                torch.arange(num_classes - 1, 0, -1).float() / (num_classes - 1)
            )

        self.to(self.device)

    def forward(self, input):
        # 处理输入格式
        if isinstance(input, (tuple, list)) and len(input) > 0 and isinstance(input[0], torch.Tensor):
            input = [x.to(self.device) for x in input]
        z = self.network(input[0])

        if self.task_type == 'regression':
            return [z, input]
        elif self.task_type == 'classification':
            logits = z[0, 0] + self.coral_bias
            logitWM = z[0, 0]
            screwed = input
            return [logits, logitWM, screwed]

    def loss_function(self, *args, **kwargs):
        if self.task_type == 'regression':
            cord_pred = args[0]
            input = args[1]
            loss = F.mse_loss(cord_pred, input[1])
            return {'loss': loss}
        elif self.task_type == 'classification':
            logits = args[0]
            logitWM = args[1]
            levelALL = args[2][1]
            levels = levelALL[0, :(levelALL.shape[1] - 1)]
            levelWM = levelALL[0, levelALL.shape[1] - 1]
            if not logits.shape == levels.shape:
                raise ValueError(f"Logits 形状 {logits.shape} 和 levels 形状 {levels.shape} 不匹配！")
            term1 = (F.logsigmoid(logits) * levels + (F.logsigmoid(logits) - logits) * (1 - levels))
            term2 = F.logsigmoid(logitWM) * levelWM + (F.logsigmoid(logitWM) - logitWM + term1) * (1 - levelWM)
            val = -torch.sum(term2, dim=0)
            return {'loss': val}



