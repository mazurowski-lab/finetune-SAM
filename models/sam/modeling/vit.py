from einops.layers.torch import Rearrange, Reduce
import torch.nn.functional as F
from einops import repeat
from torchvision.transforms.functional import to_pil_image
import numpy as np
import torch.nn as nn
from einops import rearrange
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
import os
import torchvision
import torch


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size=256):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size,
                      kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e')
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.position = nn.Parameter(torch.randn(
            (img_size//patch_size)**2+1, emb_size))

    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.projection(x)
        # cls token added  x batch times and appended
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.position
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 64*64, num_heads: int = 16, dropout: float = 0.1):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix

        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x, mask=None):
        # split keys, queries and values in num_heads
        # 3 x batch x no_head x sequence_length x emb_size
        qkv = rearrange(
            self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        # batch, num_heads, query_len, key_len
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForward(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.1):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
            nn.Dropout(drop_p),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size=256, drop_p: float = 0.1, forward_expansion: int = 4, forward_drop_p: float = 0.1, ** kwargs):
        super().__init__(
            ResidualBlock(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualBlock(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForward(emb_size),
                nn.Dropout(drop_p)
            )),
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 6, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs)
                           for _ in range(depth)])


class ViT(nn.Sequential):
    def __init__(self,
                 in_channels: int = 3,
                 patch_size: int = 4,
                 emb_size: int = 256,
                 img_size: int = 256,
                 depth: int = 6,
                 **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs)
        )


# model = ViT()
# print(model(torch.randn([1, 3, 32, 32])).shape)