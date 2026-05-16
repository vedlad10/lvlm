"""Data utilities for preprocessing and transformation."""

import torch
import numpy as np
from typing import Tuple, List
import torch.nn.functional as F


def normalize_tensor(tensor: torch.Tensor,
                    mean: List[float] = None,
                    std: List[float] = None) -> torch.Tensor:
    """Normalize tensor with ImageNet statistics.
    
    Args:
        tensor: Input tensor [C, H, W] or [B, C, H, W]
        mean: Normalization mean (default: ImageNet)
        std: Normalization std (default: ImageNet)
        
    Returns:
        Normalized tensor
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device)
    
    # Reshape for broadcasting
    if tensor.dim() == 4:  # [B, C, H, W]
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)
    elif tensor.dim() == 3:  # [C, H, W]
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)
    
    return (tensor - mean) / std


def denormalize_tensor(tensor: torch.Tensor,
                       mean: List[float] = None,
                       std: List[float] = None) -> torch.Tensor:
    """Denormalize tensor.
    
    Args:
        tensor: Normalized tensor
        mean: Normalization mean (default: ImageNet)
        std: Normalization std (default: ImageNet)
        
    Returns:
        Denormalized tensor
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device)
    
    # Reshape for broadcasting
    if tensor.dim() == 4:  # [B, C, H, W]
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)
    elif tensor.dim() == 3:  # [C, H, W]
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)
    
    return tensor * std + mean


def create_temporal_mask(seq_len: int, valid_len: int,
                        device: torch.device = None) -> torch.Tensor:
    """Create temporal mask for variable-length sequences.
    
    Args:
        seq_len: Maximum sequence length
        valid_len: Actual valid length for each sample in batch
        device: Device for tensor
        
    Returns:
        Mask tensor [batch_size, seq_len] with 0 for padding, 1 for valid
    """
    if isinstance(valid_len, int):
        batch_size = 1
        valid_len = [valid_len]
    else:
        batch_size = len(valid_len)
    
    mask = torch.zeros(batch_size, seq_len, device=device, dtype=torch.float32)
    
    for i, vlen in enumerate(valid_len):
        mask[i, :vlen] = 1.0
    
    return mask


def pad_sequence_batch(sequences: List[torch.Tensor],
                       pad_value: float = 0.0) -> Tuple[torch.Tensor, List[int]]:
    """Pad batch of sequences to same length.
    
    Args:
        sequences: List of tensors with shape [T, D] where T varies
        pad_value: Value to use for padding
        
    Returns:
        Padded tensor [batch_size, max_seq_len, D] and lengths list
    """
    lengths = [seq.shape[0] for seq in sequences]
    max_len = max(lengths)
    
    batch_size = len(sequences)
    feat_dim = sequences[0].shape[-1]
    device = sequences[0].device
    dtype = sequences[0].dtype
    
    padded = torch.full((batch_size, max_len, feat_dim),
                       pad_value, device=device, dtype=dtype)
    
    for i, (seq, length) in enumerate(zip(sequences, lengths)):
        padded[i, :length] = seq
    
    return padded, lengths


def create_attention_mask(lengths: List[int],
                         device: torch.device = None) -> torch.Tensor:
    """Create attention mask for variable-length sequences.
    
    Args:
        lengths: List of sequence lengths
        device: Device for tensor
        
    Returns:
        Mask tensor [batch_size, max_seq_len] (0 for padding, 1 for valid)
    """
    batch_size = len(lengths)
    max_len = max(lengths)
    
    mask = torch.zeros(batch_size, max_len, device=device, dtype=torch.bool)
    
    for i, length in enumerate(lengths):
        mask[i, :length] = True
    
    return mask


def aggregate_spatial_features(features: torch.Tensor,
                              method: str = "mean") -> torch.Tensor:
    """Aggregate spatial dimensions of feature maps.
    
    Args:
        features: Tensor [B, T, C, H, W] or [B, C, H, W]
        method: Aggregation method ("mean", "max", "flatten")
        
    Returns:
        Aggregated features [B, T, C] or [B, C]
    """
    if method == "mean":
        return features.mean(dim=(-2, -1))
    elif method == "max":
        result = features.max(dim=-1)[0].max(dim=-1)[0]
        return result
    elif method == "flatten":
        return features.flatten(start_dim=-2)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def interpolate_temporal(features: torch.Tensor,
                        target_length: int) -> torch.Tensor:
    """Interpolate features to target temporal length.
    
    Args:
        features: Tensor [B, T, D]
        target_length: Target temporal length
        
    Returns:
        Interpolated features [B, target_length, D]
    """
    batch_size, current_length, feat_dim = features.shape
    
    if current_length == target_length:
        return features
    
    # Reshape for interpolation
    features = features.transpose(1, 2)  # [B, D, T]
    
    # Use F.interpolate
    interpolated = F.interpolate(
        features,
        size=target_length,
        mode="linear",
        align_corners=False
    )
    
    interpolated = interpolated.transpose(1, 2)  # [B, T', D]
    return interpolated
