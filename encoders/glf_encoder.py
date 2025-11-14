"""
Global Low-Frequency (GLF) Encoder

This module implements a lightweight tri-plane based encoder that captures 
global low-frequency features to complement the local high-frequency details 
from hash grid encoding.

Key Features:
- Tri-plane factorization (XY, XZ, YZ planes) for memory efficiency
- Low-rank decomposition for each plane to reduce parameters
- Smooth bilinear interpolation for continuous feature queries
- Small initialization to serve as a gentle bias rather than dominant signal

Architecture:
    Input: 3D coordinates (x, y, z) ∈ [-1, 1]³
    ↓
    Sample from 3 orthogonal planes (XY, XZ, YZ)
    ↓
    Concatenate plane features
    ↓
    Linear projection to output dimension
    ↓
    Output: Low-frequency feature vector

Author: TinyGS Project
License: See LICENSE.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GLFEncoder(nn.Module):
    """
    Global Low-Frequency Encoder using tri-plane factorization.
    
    This encoder captures smooth, global features across the 3D space by
    representing the feature field as three orthogonal 2D planes. Each plane
    uses low-rank decomposition for parameter efficiency.
    
    Args:
        resolution (int): Spatial resolution of each plane (e.g., 64 for 64x64).
        rank (int): Rank for low-rank factorization of plane features.
        out_channels (int): Number of output feature channels.
        init_scale (float): Initialization scale for plane parameters.
                           Small values ensure GLF acts as gentle bias.
    
    Attributes:
        plane_xy (nn.Parameter): XY plane features [resolution, resolution, rank]
        plane_xz (nn.Parameter): XZ plane features [resolution, resolution, rank]
        plane_yz (nn.Parameter): YZ plane features [resolution, resolution, rank]
        projection (nn.Linear): Final linear layer to project concatenated 
                               plane features to output dimension.
    
    Example:
        >>> encoder = GLFEncoder(resolution=64, rank=8, out_channels=8)
        >>> coords = torch.randn(1000, 3).cuda()  # [N, 3] in range [-1, 1]
        >>> features = encoder(coords)  # [N, 8]
    """
    
    def __init__(
        self,
        resolution: int = 64,
        rank: int = 8,
        out_channels: int = 8,
        init_scale: float = 0.01
    ):
        super().__init__()
        
        self.resolution = resolution
        self.rank = rank
        self.out_channels = out_channels
        
        # Initialize tri-plane parameters with small random values
        # Shape: [resolution, resolution, rank] for each plane
        self.plane_xy = nn.Parameter(
            torch.randn(resolution, resolution, rank) * init_scale
        )
        self.plane_xz = nn.Parameter(
            torch.randn(resolution, resolution, rank) * init_scale
        )
        self.plane_yz = nn.Parameter(
            torch.randn(resolution, resolution, rank) * init_scale
        )
        
        # Project concatenated tri-plane features to output dimension
        # Input: 3 * rank (from 3 planes), Output: out_channels
        self.projection = nn.Linear(3 * rank, out_channels, bias=True)
        
        # Initialize projection weights small to maintain gentle contribution
        nn.init.normal_(self.projection.weight, std=init_scale)
        nn.init.zeros_(self.projection.bias)
    
    def _sample_plane(
        self,
        plane: torch.Tensor,
        coordinates: torch.Tensor
    ) -> torch.Tensor:
        """
        Sample features from a 2D plane using bilinear interpolation.
        
        Args:
            plane (torch.Tensor): Plane parameters [H, W, rank]
            coordinates (torch.Tensor): 2D coordinates [N, 2] in range [-1, 1]
        
        Returns:
            torch.Tensor: Sampled features [N, rank]
        """
        # Reshape plane: [H, W, rank] -> [1, rank, H, W] for grid_sample
        plane_features = plane.permute(2, 0, 1).unsqueeze(0)
        
        # Reshape coordinates: [N, 2] -> [1, 1, N, 2] for grid_sample
        grid = coordinates.unsqueeze(0).unsqueeze(0)
        
        # Bilinear interpolation
        # Output: [1, rank, 1, N] -> [N, rank]
        sampled = F.grid_sample(
            plane_features,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )
        
        return sampled.squeeze(0).squeeze(1).transpose(0, 1)
    
    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute global low-frequency features.
        
        Args:
            coordinates (torch.Tensor): 3D coordinates [N, 3] in range [-1, 1]
                                       (x, y, z) format
        
        Returns:
            torch.Tensor: GLF features [N, out_channels]
        """
        # Clamp coordinates to [-1, 1] to prevent illegal memory access
        # This is critical for numerical stability and CUDA safety
        coordinates = torch.clamp(coordinates, -1.0, 1.0)
        
        # Extract coordinate components
        x, y, z = coordinates[:, 0:1], coordinates[:, 1:2], coordinates[:, 2:3]
        
        # Sample from each orthogonal plane
        xy_coords = torch.cat([x, y], dim=1)  # [N, 2]
        xz_coords = torch.cat([x, z], dim=1)  # [N, 2]
        yz_coords = torch.cat([y, z], dim=1)  # [N, 2]
        
        xy_features = self._sample_plane(self.plane_xy, xy_coords)  # [N, rank]
        xz_features = self._sample_plane(self.plane_xz, xz_coords)  # [N, rank]
        yz_features = self._sample_plane(self.plane_yz, yz_coords)  # [N, rank]
        
        # Concatenate features from all three planes
        triplane_features = torch.cat(
            [xy_features, xz_features, yz_features], 
            dim=1
        )  # [N, 3*rank]
        
        # Project to output dimension
        output = self.projection(triplane_features)  # [N, out_channels]
        
        # Clamp output to prevent extreme values that could cause numerical issues
        # This is a safety measure to ensure stable training
        output = torch.clamp(output, -10.0, 10.0)
        
        return output
