# models/__init__.py
"""Neural network models for LVLM."""

from .feature_extractor import FeatureExtractor
from .temporal_binding import TemporalBindingLayer, TemporalBindingModule
from .chimrt import CHIMRT, ReasoningGraph
from .adaptive_depth import AdaptiveDepthController
from .multimodal_vdb import MultimodalVDB
from .lvlm import LVLM
from .qwen_adapter import QwenAdapter, QwenLVLMFusion

# Research Modules (optional enhancements for visual token processing)
from .research_modules import (
    AdaptiveVisualTokenRouter,
    InstructionGuidedVisualAggregator,
    VisualTextGroundingHead,
    UnsupportedClaimPenalty,
    MemoryAwareProjection,
    TemporalFrameFusion,
    DynamicVisualTokenGenerator,
    RoutingStats,
)

__all__ = [
    "FeatureExtractor",
    "TemporalBindingLayer",
    "TemporalBindingModule",
    "CHIMRT",
    "ReasoningGraph",
    "AdaptiveDepthController",
    "MultimodalVDB",
    "LVLM",
    "QwenAdapter",
    "QwenLVLMFusion",
    # Research Modules
    "AdaptiveVisualTokenRouter",
    "InstructionGuidedVisualAggregator",
    "VisualTextGroundingHead",
    "UnsupportedClaimPenalty",
    "MemoryAwareProjection",
    "TemporalFrameFusion",
    "DynamicVisualTokenGenerator",
    "RoutingStats",
]
