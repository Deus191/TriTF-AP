"""Deep ConvNet baseline following the four-block Schirrmeister design."""
import torch
from torch import nn

try:
    from .eegnet import Conv2dMaxNorm, LinearMaxNorm
except ImportError:  # Direct execution from the baselines directory.
    from eegnet import Conv2dMaxNorm, LinearMaxNorm


class ConvBlock(nn.Sequential):
    def __init__(self, in_filters, out_filters, dropout):
        super().__init__(
            Conv2dMaxNorm(
                in_filters, out_filters, kernel_size=(1, 10), bias=False, max_norm=2.0
            ),
            nn.BatchNorm2d(out_filters, momentum=0.1, affine=True),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            nn.Dropout(dropout),
        )


class DeepConvNet(nn.Module):
    """Four temporal/spatial convolution blocks for ``(N,1,C,T)`` EEG."""

    def __init__(self, n_channels=3, n_time=1000, n_classes=2, dropout=0.5):
        super().__init__()
        if n_channels < 1 or n_time < 128 or n_classes < 2:
            raise ValueError("Invalid DeepConvNet input dimensions")
        self.features = nn.Sequential(
            Conv2dMaxNorm(1, 25, kernel_size=(1, 10), bias=False, max_norm=2.0),
            Conv2dMaxNorm(
                25, 25, kernel_size=(n_channels, 1), bias=False, max_norm=2.0
            ),
            nn.BatchNorm2d(25, momentum=0.1, affine=True),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            nn.Dropout(dropout),
            ConvBlock(25, 50, dropout),
            ConvBlock(50, 100, dropout),
            ConvBlock(100, 200, dropout),
        )
        with torch.no_grad():
            feature_count = self.features(torch.zeros(1, 1, n_channels, n_time)).numel()
        if feature_count < 1:
            raise ValueError("DeepConvNet input window is too short")
        self.classifier = LinearMaxNorm(feature_count, n_classes, max_norm=0.5)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs):
        features = self.features(inputs)
        logits = self.classifier(torch.flatten(features, start_dim=1))
        return self.log_softmax(logits)
