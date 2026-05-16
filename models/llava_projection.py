"""
Visual Projection Layer for LLaVA-style LVLM
Converts CLIP embeddings (B, 512) → (B, num_tokens, llm_dim)
"""

import torch
import torch.nn as nn
from typing import Optional


class VisualProjection(nn.Module):
    """
    MLP-based projection for visual token generation.
    
    Maps a single CLIP embedding into multiple LLaMA-compatible tokens.
    Architecture: Linear → GELU → Linear → GELU → Linear
    
    Args:
        clip_dim (int): Input dimension from CLIP (e.g., 512 for ViT-B/32)
        llm_dim (int): LLM hidden dimension (e.g., 4096 for LLaMA-7B)
        num_tokens (int): Number of visual tokens to generate per image
        hidden_dim (Optional[int]): MLP hidden dimension (defaults to llm_dim)
        dropout (float): Dropout probability in MLP layers
    """

    def __init__(
        self,
        clip_dim: int = 512,
        llm_dim: int = 4096,
        num_tokens: int = 32,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or llm_dim
        out_dim = num_tokens * llm_dim
        
        self.num_tokens = num_tokens
        self.llm_dim = llm_dim
        
        # 3-layer MLP with GELU activation
        self.mlp = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, clip_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clip_embeds (torch.Tensor): Shape (B, clip_dim)
            
        Returns:
            torch.Tensor: Shape (B, num_tokens, llm_dim)
        """
        batch_size = clip_embeds.size(0)
        out = self.mlp(clip_embeds)  # (B, num_tokens * llm_dim)
        return out.view(batch_size, self.num_tokens, self.llm_dim)
