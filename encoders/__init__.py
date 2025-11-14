"""
Feature Encoders for TinyGS

This package provides hierarchical feature encoding components for 3D Gaussian Splatting:
- GLFEncoder: Global Low-Frequency encoder using tri-plane factorization
- HierarchicalDualBranchEncoder: Dual-branch architecture combining Hash + GLF

Author: TinyGS Project
License: See LICENSE.md
"""

from .glf_encoder import GLFEncoder
from .hierarchical_dual_branch import HierarchicalDualBranchEncoder

__all__ = ['GLFEncoder', 'HierarchicalDualBranchEncoder']
