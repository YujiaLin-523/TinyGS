"""
Hierarchical Dual-Branch Encoder

This module implements a hierarchical feature encoding strategy that combines:
1. Hash Grid Branch: Captures local high-frequency geometric details (main branch)
2. GLF Branch: Provides global low-frequency smooth priors (auxiliary branch)

The two branches are fused through simple concatenation and linear projection,
with a fixed weighting factor (beta) controlling the GLF contribution.

Key Design Principles:
- Hash branch remains dominant for local detail preservation
- GLF branch acts as a gentle global bias for smoothness
- No complex gating or scheduling mechanisms for training stability
- Fixed beta ensures predictable behavior across training

Architecture:
    Input: 3D coordinates
    ↓
    ┌─────────────────┬─────────────────┐
    │   Hash Branch   │   GLF Branch    │
    │  (High-freq)    │  (Low-freq)     │
    └────────┬────────┴────────┬────────┘
             │                 │ (*beta)
             └────────┬────────┘
                      ↓
                Concatenate
                      ↓
               Linear Fusion
                      ↓
                  Output

Author: TinyGS Project
License: See LICENSE.md
"""

import torch
import torch.nn as nn
from .glf_encoder import GLFEncoder


class HierarchicalDualBranchEncoder(nn.Module):
    """
    Hierarchical encoding with dual branches: Hash (local) + GLF (global).
    
    This encoder wraps a pre-existing hash grid encoder and augments it with
    a lightweight global low-frequency encoder. The outputs are fused through
    concatenation and linear projection.
    
    Args:
        hash_encoder: Pre-initialized hash grid encoder (e.g., tcnn.Encoding).
                     Must have forward() method and n_output_dims property.
        out_dim (int): Desired output feature dimension (typically same as 
                      hash encoder's output dimension).
        glf_channels (int): Number of channels from GLF encoder.
        glf_resolution (int): Spatial resolution for GLF tri-planes.
        glf_rank (int): Rank for low-rank factorization in GLF.
        beta (float): Fixed weighting factor for GLF contribution.
                     Range: [0, 1], typical value: 0.1
                     beta=0 → pure hash, beta=1 → equal weighting
    
    Attributes:
        hash_encoder: The main hash grid encoder (frozen or trainable)
        glf_encoder: The auxiliary GLF encoder
        fusion_layer: Linear layer to fuse hash+GLF features to output dimension
        beta: GLF weighting factor (non-trainable)
    
    Example:
        >>> import tinycudann as tcnn
        >>> hash_enc = tcnn.Encoding(n_input_dims=3, encoding_config={...})
        >>> encoder = HierarchicalDualBranchEncoder(
        ...     hash_encoder=hash_enc,
        ...     out_dim=32,
        ...     glf_channels=8,
        ...     glf_resolution=64,
        ...     glf_rank=8,
        ...     beta=0.1
        ... )
        >>> coords = torch.randn(1000, 3).cuda()
        >>> features = encoder(coords)  # [1000, 32]
    """
    
    def __init__(
        self,
        hash_encoder,
        out_dim: int,
        glf_channels: int = 8,
        glf_resolution: int = 64,
        glf_rank: int = 8,
        beta: float = 0.05
    ):
        super().__init__()
        
        # Main branch: Hash grid encoder (pre-initialized)
        self.hash_encoder = hash_encoder
        self.hash_dim = hash_encoder.n_output_dims
        
        # Auxiliary branch: Global low-frequency encoder
        self.glf_encoder = GLFEncoder(
            resolution=glf_resolution,
            rank=glf_rank,
            out_channels=glf_channels,
            init_scale=0.01  # Small init for gentle contribution
        )
        self.glf_dim = glf_channels
        
        # Fusion layer: Concatenate hash + beta*GLF, then project to out_dim
        self.fusion_layer = nn.Linear(
            self.hash_dim + self.glf_dim,
            out_dim,
            bias=True
        )
        
        # Initialize fusion layer with small weights
        nn.init.normal_(self.fusion_layer.weight, std=0.01)
        nn.init.zeros_(self.fusion_layer.bias)
        
        # Fixed beta (non-trainable)
        self.register_buffer('beta', torch.tensor(beta, dtype=torch.float32))
        
        self._out_dim = out_dim
    
    @property
    def n_output_dims(self) -> int:
        """
        Output dimension property for compatibility with tcnn interface.
        
        Returns:
            int: Output feature dimension
        """
        return self._out_dim
    
    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hierarchical dual-branch encoder.
        
        Args:
            coordinates (torch.Tensor): Input 3D coordinates [N, 3]
                                       Expected range: normalized to [-1, 1]
        
        Returns:
            torch.Tensor: Fused features [N, out_dim]
        
        Note:
            The fusion formula is:
                output = Linear(concat([hash_features, beta * glf_features]))
        """
        # Hash branch: Local high-frequency features
        hash_features = self.hash_encoder(coordinates)  # [N, hash_dim]
        
        # GLF branch: Global low-frequency features
        glf_features = self.glf_encoder(coordinates)  # [N, glf_dim]
        
        # Scale GLF features by beta (controls auxiliary contribution)
        weighted_glf = self.beta * glf_features
        
        # Concatenate both branches
        combined_features = torch.cat(
            [hash_features, weighted_glf], 
            dim=1
        )  # [N, hash_dim + glf_dim]
        
        # Fuse through linear layer
        output = self.fusion_layer(combined_features)  # [N, out_dim]
        
        # Numerical stability: clamp output to reasonable range
        # This prevents extreme values from propagating through the network 
        output = torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)
        output = torch.clamp(output, min=-1e6, max=1e6)

        return output
