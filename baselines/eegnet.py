"""Compact EEGNet implementation for binary/multiclass raw EEG baselines.

The implementation follows the EEGNet-8,2 design: temporal convolution,
depthwise spatial filtering, separable temporal convolution, and a constrained
linear classifier.  Inputs have shape ``(batch, 1, channels, time)``.
"""
import torch
from torch import nn


class Conv2dMaxNorm(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        self.max_norm = kwargs.pop("max_norm", None)
        super().__init__(*args, **kwargs)

    def forward(self, inputs):
        if self.max_norm is not None:
            with torch.no_grad():
                self.weight.copy_(torch.renorm(self.weight, p=2, dim=0, maxnorm=self.max_norm))
        return super().forward(inputs)


class LinearMaxNorm(nn.Linear):
    def __init__(self, *args, **kwargs):
        self.max_norm = kwargs.pop("max_norm", None)
        super().__init__(*args, **kwargs)

    def forward(self, inputs):
        if self.max_norm is not None:
            with torch.no_grad():
                self.weight.copy_(torch.renorm(self.weight, p=2, dim=0, maxnorm=self.max_norm))
        return super().forward(inputs)


class EEGNet(nn.Module):
    """EEGNet-8,2 with sampling-rate-scaled temporal kernels."""

    def __init__(
        self,
        n_channels=3,
        n_time=1000,
        n_classes=2,
        sampling_rate=250,
        f1=8,
        depth_multiplier=2,
        f2=16,
        dropout=0.5,
    ):
        super().__init__()
        if n_channels < 1 or n_time < 32 or n_classes < 2:
            raise ValueError("Invalid EEGNet input dimensions")
        temporal_kernel = max(3, int(round(sampling_rate / 2)))
        separable_kernel = max(3, int(round(sampling_rate / 8)))
        temporal_kernel += temporal_kernel % 2 == 0
        separable_kernel += separable_kernel % 2 == 0
        depthwise_filters = f1 * depth_multiplier

        self.features = nn.Sequential(
            nn.Conv2d(
                1,
                f1,
                kernel_size=(1, temporal_kernel),
                padding=(0, temporal_kernel // 2),
                bias=False,
            ),
            nn.BatchNorm2d(f1),
            Conv2dMaxNorm(
                f1,
                depthwise_filters,
                kernel_size=(n_channels, 1),
                groups=f1,
                bias=False,
                max_norm=1.0,
            ),
            nn.BatchNorm2d(depthwise_filters),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(dropout),
            nn.Conv2d(
                depthwise_filters,
                depthwise_filters,
                kernel_size=(1, separable_kernel),
                padding=(0, separable_kernel // 2),
                groups=depthwise_filters,
                bias=False,
            ),
            nn.Conv2d(depthwise_filters, f2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            feature_count = self.features(torch.zeros(1, 1, n_channels, n_time)).numel()
        if feature_count < 1:
            raise ValueError("EEGNet input window is too short")
        self.classifier = LinearMaxNorm(feature_count, n_classes, max_norm=0.25)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs):
        features = self.features(inputs)
        logits = self.classifier(torch.flatten(features, start_dim=1))
        return self.log_softmax(logits)
