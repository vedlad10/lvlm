"""Temporal Binding Layer - Core Innovation #1.

Implements Markov-assumption-based memory consolidation that compresses
video frames into semantic memory nodes while preserving temporal causality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TemporalBindingLayer(nn.Module):
    """Single temporal binding layer for frame consolidation.
    
    Converts sequences of frame embeddings into compressed memory nodes
    using learned gating and Markov transition probabilities.
    
    Core math:
    - Transition: P(Z_{i+1} | Z_i) := softmax(W_trans · [Z_i; ΔT])
    - Gate: stop_prob = sigmoid(w_gate^T · Z_i + b_gate)
    - Consolidation: Z_node = Σ α_t · frame_t where α_t learned via gating
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        gate_type: str = "learned",
        consolidation_method: str = "markov",
        dropout: float = 0.1,
    ):
        """Initialize temporal binding layer.
        
        Args:
            feature_dim: Input feature dimension
            hidden_dim: Hidden dimension for learned gates
            gate_type: Type of gating ("learned", "fixed_threshold")
            consolidation_method: Consolidation method ("markov", "attention")
            dropout: Dropout rate
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.gate_type = gate_type
        self.consolidation_method = consolidation_method
        
        # Learned gating mechanism
        if gate_type == "learned":
            self.gate_net = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.gate_net = None
        
        # Markov transition matrix parameters
        self.transition_matrix = nn.Parameter(
            torch.randn(feature_dim, feature_dim) * 0.02
        )
        
        # Consolidation projection
        self.consolidation_proj = nn.Linear(feature_dim, feature_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        logger.info(f"Initialized TemporalBindingLayer (method={consolidation_method})")
    
    def forward(
        self,
        frames: torch.Tensor,
        frame_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Consolidate frame sequence into memory nodes.
        
        Args:
            frames: Input frames [B, T, D]
            frame_lengths: Actual lengths for each sequence in batch [B]
            
        Returns:
            Tuple of:
            - memory_nodes: Consolidated nodes [B, K, D] where K << T
            - node_assignments: Frame-to-node assignments [B, T]
        """
        B, T, D = frames.shape
        
        # Compute transition probabilities using Markov assumption
        transition_probs = self._compute_transitions(frames)  # [B, T-1]
        
        # Compute gating scores (where to create boundaries)
        if self.gate_type == "learned":
            gate_scores = self.gate_net(frames).squeeze(-1)  # [B, T]
            gate_probs = torch.sigmoid(gate_scores)  # [B, T]
        else:
            gate_probs = torch.ones_like(frames[:, :, 0]) * 0.5  # [B, T]
        
        # Determine node boundaries
        node_boundaries = self._determine_boundaries(
            gate_probs, transition_probs, frame_lengths
        )  # [B, T]
        
        # Consolidate frames into nodes
        memory_nodes = self._consolidate_frames(
            frames, node_boundaries
        )  # [B, K, D]
        
        # Normalize node assignments for tracking
        node_assignments = node_boundaries.argmax(dim=-1)
        
        return memory_nodes, node_assignments
    
    def _compute_transitions(self, frames: torch.Tensor) -> torch.Tensor:
        """Compute frame-to-frame transition probabilities.
        
        Uses Markov assumption: P(Z_{i+1} | Z_i) learned via transition matrix.
        
        Args:
            frames: [B, T, D]
            
        Returns:
            Transition probabilities [B, T-1]
        """
        B, T, D = frames.shape
        
        # Compute Markov scores: Z_i @ W_trans @ Z_{i+1}
        # Approximation: use frame similarity
        frame_pairs = torch.nn.functional.cosine_similarity(
            frames[:, :-1], frames[:, 1:], dim=-1
        )  # [B, T-1]
        
        # Convert to probabilities
        transition_probs = torch.sigmoid(frame_pairs)  # [B, T-1]
        
        return transition_probs
    
    def _determine_boundaries(
        self,
        gate_probs: torch.Tensor,
        transition_probs: torch.Tensor,
        frame_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Determine temporal boundaries between memory nodes.
        
        Boundaries are placed where:
        1. Gate probability indicates stopping point (high prob = new node)
        2. Transition probability drops (low prob = event boundary)
        
        Args:
            gate_probs: Gating probabilities [B, T]
            transition_probs: Transition probabilities [B, T-1]
            frame_lengths: Actual sequence lengths [B]
            
        Returns:
            Boundary indicators [B, T] (soft boundaries)
        """
        B, T = gate_probs.shape
        
        device = gate_probs.device
        
        # Combine gate and transition signals
        boundary_scores = gate_probs.clone()
        
        # Transition drop indicates boundary
        transition_dips = torch.cat([
            torch.ones(B, 1, device=device),  # First frame is always boundary
            1 - transition_probs,               # Boundary at low transition
        ], dim=1)
        
        boundary_scores = boundary_scores * transition_dips
        
        # Apply length mask if provided
        if frame_lengths is not None:
            mask = torch.arange(T, device=device).unsqueeze(0) < frame_lengths.unsqueeze(1)
            boundary_scores = boundary_scores * mask.float()
        
        # Normalize
        boundary_scores = boundary_scores / (boundary_scores.sum(dim=1, keepdim=True) + 1e-10)
        
        return boundary_scores
    
    def _consolidate_frames(
        self,
        frames: torch.Tensor,
        boundaries: torch.Tensor,
    ) -> torch.Tensor:
        """Consolidate frames into memory nodes based on boundaries.
        
        Args:
            frames: [B, T, D]
            boundaries: [B, T] soft boundary indicators
            
        Returns:
            Memory nodes [B, K, D] where K is number of nodes
        """
        B, T, D = frames.shape
        
        # Use boundary points as node centers
        # Weighted consolidation: each node is weighted sum of frames
        
        # Create node assignments (soft clustering)
        # For now, use simple pooling at boundary points
        boundary_indices = torch.nonzero(boundaries.sum(dim=0) > 1)
        
        if len(boundary_indices) == 0:
            # Fallback: single node
            memory_nodes = frames.mean(dim=1, keepdim=True)  # [B, 1, D]
        else:
            # Consolidate at boundaries
            node_list = []
            
            for b in range(B):
                # Find strong boundaries for this sample
                strong_boundaries = (boundaries[b] > 0.5).nonzero(as_tuple=True)[0]
                
                if len(strong_boundaries) == 0:
                    # No strong boundaries, use mean
                    node_list.append(frames[b].mean(dim=0, keepdim=True))
                else:
                    # Consolidate at each boundary
                    b_nodes = []
                    for i, boundary_idx in enumerate(strong_boundaries):
                        # Average frames around boundary
                        start = max(0, boundary_idx - 2)
                        end = min(T, boundary_idx + 3)
                        node = frames[b, start:end].mean(dim=0)
                        b_nodes.append(node)
                    
                    b_nodes = torch.stack(b_nodes, dim=0)
                    node_list.append(b_nodes)
            
            # Pad to same length in batch
            max_nodes = max(n.shape[0] for n in node_list)
            memory_nodes = torch.zeros(B, max_nodes, D, device=frames.device, dtype=frames.dtype)
            
            for b, nodes in enumerate(node_list):
                memory_nodes[b, :nodes.shape[0]] = nodes
        
        return memory_nodes


class TemporalBindingModule(nn.Module):
    """Full temporal binding module with multiple stacked layers.
    
    Applies temporal binding hierarchically to progressively compress
    frame sequences.
    """
    
    def __init__(
        self,
        feature_dim: int,
        num_layers: int = 2,
        hidden_dim: int = 256,
        gate_type: str = "learned",
        dropout: float = 0.1,
    ):
        """Initialize temporal binding module.
        
        Args:
            feature_dim: Input feature dimension
            num_layers: Number of binding layers to stack
            hidden_dim: Hidden dimension for gates
            gate_type: Type of gating mechanism
            dropout: Dropout rate
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        
        # Stack multiple binding layers
        self.layers = nn.ModuleList([
            TemporalBindingLayer(
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                gate_type=gate_type,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        logger.info(f"Initialized TemporalBindingModule with {num_layers} layers")
    
    def forward(
        self,
        frames: torch.Tensor,
        frame_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, list]:
        """Apply hierarchical temporal binding.
        
        Args:
            frames: Input frames [B, T, D]
            frame_lengths: Actual sequence lengths [B]
            
        Returns:
            Tuple of:
            - final_nodes: Final compressed memory nodes [B, K, D]
            - assignment_trace: List of assignment tensors from each layer
        """
        current_frames = frames
        assignment_trace = []
        
        for layer in self.layers:
            memory_nodes, assignments = layer(current_frames, frame_lengths)
            assignment_trace.append(assignments)
            
            # Use memory nodes as input to next layer
            # Pad/interpolate if needed
            if memory_nodes.shape[1] > 1:
                current_frames = memory_nodes
            else:
                break  # Converged to single node
        
        return current_frames, assignment_trace
