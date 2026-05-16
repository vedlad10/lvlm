"""Feature extraction module for video frames."""

import torch
import torch.nn as nn
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor(nn.Module):
    """Feature extractor for video frames using pre-trained ViT or CLIP.
    
    Extracts visual embeddings from video frames for input to LVLM.
    """
    
    def __init__(
        self,
        model_type: str = "vit_base",
        pretrained: bool = True,
        frozen: bool = True,
        output_dim: int = 768,
    ):
        """Initialize feature extractor.
        
        Args:
            model_type: Type of model ("vit_base", "vit_large", "clip_vit", etc.)
            pretrained: Whether to use pretrained weights
            frozen: Whether to freeze model parameters
            output_dim: Output embedding dimension
        """
        super().__init__()
        
        self.model_type = model_type
        self.frozen = frozen
        self.output_dim = output_dim
        
        # Load model
        self.backbone = self._load_backbone(model_type, pretrained)
        
        # Feature projection to output dimension
        current_dim = self._get_backbone_output_dim()
        if current_dim != output_dim:
            self.projection = nn.Linear(current_dim, output_dim)
        else:
            self.projection = nn.Identity()
        
        # Freeze if needed
        if frozen:
            self._freeze_parameters()
        
        logger.info(f"Initialized {model_type} feature extractor (frozen={frozen})")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from frames.
        
        Args:
            x: Input frames [B, 3, H, W]
            
        Returns:
            Visual embeddings [B, output_dim]
        """
        # Extract backbone features
        features = self.backbone(x)
        
        # Project to output dimension
        features = self.projection(features)
        
        return features
    
    def _load_backbone(self, model_type: str, pretrained: bool) -> nn.Module:
        """Load backbone model.
        
        Args:
            model_type: Model type identifier
            pretrained: Whether to use pretrained weights
            
        Returns:
            Backbone model
        """
        if "vit" in model_type.lower():
            try:
                import timm
                model = timm.create_model(model_type, pretrained=pretrained)
                # Remove classification head, keep feature extraction
                model.head = nn.Identity()
                return model
            except ImportError:
                logger.warning("timm not installed, using mock model")
                return self._get_mock_model()
        elif "clip" in model_type.lower():
            try:
                import clip
                model, _ = clip.load("ViT-B/32")
                return model.visual
            except ImportError:
                logger.warning("CLIP not installed, using mock model")
                return self._get_mock_model()
        else:
            logger.warning(f"Unknown model type {model_type}, using mock model")
            return self._get_mock_model()
    
    def _get_mock_model(self) -> nn.Module:
        """Get mock model for testing.
        
        Returns:
            Mock feature extraction model
        """
        class MockModel(nn.Module):
            def forward(self, x):
                B = x.shape[0]
                return torch.randn(B, 768)
        
        return MockModel()
    
    def _get_backbone_output_dim(self) -> int:
        """Get output dimension of backbone.
        
        Returns:
            Output dimension
        """
        # Test forward pass
        dummy_input = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            output = self.backbone(dummy_input)
        
        if isinstance(output, torch.Tensor):
            return output.shape[-1] if output.dim() == 2 else 768
        else:
            return 768
    
    def _freeze_parameters(self) -> None:
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        logger.info("Froze feature extractor parameters")
