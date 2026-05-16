"""
Frozen CLIP Visual Encoder for LLaVA-style LVLM
Extracts image embeddings using OpenAI CLIP (all parameters frozen).
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class FrozenCLIPEncoder(nn.Module):
    """
    Wraps OpenAI CLIP visual encoder with frozen parameters.
    
    All CLIP weights remain frozen during training.
    Handles preprocessing and produces L2-normalized embeddings.
    
    Args:
        model_name (str): CLIP model identifier (e.g., "ViT-B/32", "ViT-B/16")
        device (str): Compute device ("cuda" or "cpu")
    """

    # CLIP embedding dimensions for common variants
    EMBED_DIM: Dict[str, int] = {
        "ViT-B/32": 512,
        "ViT-B/16": 512,
        "ViT-L/14": 768,
        "ViT-L/14@336px": 768,
    }

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str = "cuda",
    ):
        """
        Args:
            model_name: CLIP model to load
            device: Device for computation
            
        Raises:
            ImportError: If clip is not installed
            RuntimeError: If model cannot be loaded
        """
        super().__init__()
        
        try:
            import clip
        except ImportError:
            raise ImportError(
                "clip package required. Install with: "
                "pip install git+https://github.com/openai/CLIP.git"
            )
        
        # Load CLIP model and preprocess function
        model, self.preprocess = clip.load(model_name, device=device)
        self.visual = model.visual
        self.embed_dim = self.EMBED_DIM.get(model_name, 512)
        self.device = device
        
        logger.info(f"Loaded frozen CLIP {model_name} | embed_dim={self.embed_dim}")
        
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract CLIP embeddings from images.
        
        Args:
            pixel_values (torch.Tensor): Shape (B, 3, H, W), CLIP-preprocessed images
            
        Returns:
            torch.Tensor: Shape (B, embed_dim), L2-normalized embeddings
        """
        # Ensure correct device
        pixel_values = pixel_values.to(self.device)
        
        # Extract visual features (no gradients)
        feats = self.visual(pixel_values)  # (B, embed_dim)
        
        # L2-normalize embeddings
        return feats / feats.norm(dim=-1, keepdim=True)
