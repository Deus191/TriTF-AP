"""Compact CTNet model extracted from the supplied comparison notebook.

Reference: Zhao et al., CTNet: a convolutional transformer network for
EEG-based motor imagery classification, Scientific Reports 14, 20237 (2024).
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class PatchEmbeddingCNN(nn.Module):
    def __init__(self, number_channel=3, f1=8, kernel_size=64, depth_multiplier=2,
                 pooling_size1=8, pooling_size2=8, dropout_rate=0.25):
        super().__init__()
        f2 = depth_multiplier * f1
        self.layers = nn.Sequential(
            nn.Conv2d(1, f1, (1, kernel_size), padding="same", bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f2, (number_channel, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, pooling_size1)),
            nn.Dropout(dropout_rate),
            nn.Conv2d(f2, f2, (1, 16), padding="same", bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, pooling_size2)),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        x = self.layers(x)
        return x.flatten(2).transpose(1, 2)


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding, heads, dropout):
        super().__init__()
        if embedding % heads:
            raise ValueError("embedding must be divisible by heads")
        self.embedding = embedding
        self.heads = heads
        self.keys = nn.Linear(embedding, embedding)
        self.queries = nn.Linear(embedding, embedding)
        self.values = nn.Linear(embedding, embedding)
        self.attention_dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(embedding, embedding)

    def forward(self, x):
        batch, tokens, _ = x.shape
        head_dim = self.embedding // self.heads

        def split_heads(tensor):
            return tensor.reshape(batch, tokens, self.heads, head_dim).permute(0, 2, 1, 3)

        queries = split_heads(self.queries(x))
        keys = split_heads(self.keys(x))
        values = split_heads(self.values(x))
        energy = torch.einsum("bhqd,bhkd->bhqk", queries, keys)
        attention = self.attention_dropout(F.softmax(energy / math.sqrt(head_dim), dim=-1))
        output = torch.einsum("bhqk,bhkd->bhqd", attention, values)
        output = output.permute(0, 2, 1, 3).reshape(batch, tokens, self.embedding)
        return self.projection(output)


class ResidualBlock(nn.Module):
    def __init__(self, module, embedding, dropout):
        super().__init__()
        self.module = module
        self.dropout = nn.Dropout(dropout)
        self.normalization = nn.LayerNorm(embedding)

    def forward(self, x):
        return self.normalization(x + self.dropout(self.module(x)))


class TransformerBlock(nn.Module):
    def __init__(self, embedding=16, heads=2, dropout=0.5, expansion=4):
        super().__init__()
        self.attention = ResidualBlock(MultiHeadAttention(embedding, heads, dropout), embedding, dropout)
        self.feed_forward = ResidualBlock(
            nn.Sequential(
                nn.Linear(embedding, expansion * embedding),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(expansion * embedding, embedding),
            ),
            embedding,
            dropout,
        )

    def forward(self, x):
        return self.feed_forward(self.attention(x))


class CTNet(nn.Module):
    """Three-channel, two-class CTNet configuration used by the public runner."""

    def __init__(self, n_channels=3, n_time=1000, n_classes=2, embedding=16,
                 heads=2, depth=6, dropout=0.25):
        super().__init__()
        self.embedding = embedding
        self.patch_embedding = PatchEmbeddingCNN(
            number_channel=n_channels,
            f1=8,
            depth_multiplier=2,
            dropout_rate=dropout,
        )
        with torch.no_grad():
            token_shape = self.patch_embedding(torch.zeros(1, 1, n_channels, n_time)).shape
        self.position = nn.Parameter(torch.randn(1, token_shape[1], embedding))
        self.position_dropout = nn.Dropout(0.1)
        self.transformer = nn.Sequential(
            *[TransformerBlock(embedding, heads) for _ in range(depth)]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(token_shape[1] * embedding, n_classes),
        )

    def forward(self, x):
        tokens = self.patch_embedding(x) * math.sqrt(self.embedding)
        tokens = self.position_dropout(tokens + self.position[:, :tokens.shape[1]])
        features = tokens + self.transformer(tokens)
        return F.log_softmax(self.classifier(features), dim=1)
