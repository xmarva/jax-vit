import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().init()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        
    def forward(self, x):
        # (B, C, H, W) -> (B, E, H/P, W/P) -> (B, E, N) -> (B, N, E)
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        
        return x