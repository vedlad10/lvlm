"""CHIMRT - Conditional Hierarchical Integrated Multi-Relational Transformer.

The core reasoning engine that performs multi-hop semantic reasoning
over memory nodes with learned temporal graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class ReasoningGraph:
    """Temporal reasoning graph for memory nodes."""
    
    def __init__(self, num_nodes: int, device: torch.device):
        """Initialize reasoning graph.
        
        Args:
            num_nodes: Number of memory nodes
            device: Device for tensors
        """
        self.num_nodes = num_nodes
        self.device = device
        
        # Adjacency matrices for different relation types
        self.temporal_adjacency = torch.eye(
            num_nodes, device=device, dtype=torch.float32
        )  # Temporal connections
        
        self.semantic_adjacency = torch.ones(
            num_nodes, num_nodes, device=device, dtype=torch.float32
        ) / num_nodes  # Semantic similarity
        
        self.question_relevance = torch.ones(
            num_nodes, device=device, dtype=torch.float32
        ) / num_nodes  # Question relevance
    
    def update_temporal_edges(self, adjacency: torch.Tensor) -> None:
        """Update temporal adjacency matrix.
        
        Args:
            adjacency: Temporal adjacency [num_nodes, num_nodes]
        """
        self.temporal_adjacency = adjacency
    
    def update_semantic_edges(self, similarities: torch.Tensor) -> None:
        """Update semantic similarity matrix.
        
        Args:
            similarities: Semantic similarities [num_nodes, num_nodes]
        """
        self.semantic_adjacency = similarities
    
    def update_question_relevance(self, relevance: torch.Tensor) -> None:
        """Update question relevance scores.
        
        Args:
            relevance: Relevance scores [num_nodes]
        """
        self.question_relevance = relevance


class MultiHeadReasoningAttention(nn.Module):
    """Multi-head attention for reasoning over memory nodes."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 12,
        dropout: float = 0.1,
    ):
        """Initialize multi-head attention.
        
        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Multi-head attention forward pass.
        
        Args:
            query: Query tensor [B, 1, D] or [B, D]
            keys: Key tensor [B, T, D]
            values: Value tensor [B, T, D]
            mask: Optional attention mask [B, T]
            
        Returns:
            Tuple of (output, attention_weights)
        """
        B = query.shape[0]
        
        # Project inputs
        Q = self.query_proj(query)  # [B, 1, D] or [B, D]
        K = self.key_proj(keys)      # [B, T, D]
        V = self.value_proj(values)  # [B, T, D]
        
        # Reshape for multi-head attention
        if Q.dim() == 2:
            Q = Q.unsqueeze(1)  # [B, 1, D]
        
        Q = Q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, 1, D/H]
        K = K.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D/H]
        V = V.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D/H]
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, 1, T]
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, T]
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)  # [B, H, 1, T]
        attn_weights = self.dropout(attn_weights)
        
        # Output
        output = torch.matmul(attn_weights, V)  # [B, H, 1, D/H]
        output = output.transpose(1, 2).contiguous()  # [B, 1, H, D/H]
        output = output.view(B, -1, self.hidden_dim)  # [B, 1, D]
        output = self.output_proj(output.squeeze(1))  # [B, D]
        
        # Return average attention weights
        avg_attn = attn_weights.mean(dim=1).squeeze(1)  # [B, T]
        
        return output, avg_attn


class CHIMRTLayer(nn.Module):
    """Single CHIMRT layer for hierarchical reasoning."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 12,
        ffn_dim: int = 3072,
        dropout: float = 0.1,
    ):
        """Initialize CHIMRT layer.
        
        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            ffn_dim: Feed-forward network dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        # Reasoning attention
        self.reasoning_attn = MultiHeadReasoningAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        reasoning_state: torch.Tensor,
        memory_nodes: torch.Tensor,
        graph: Optional[ReasoningGraph] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform single reasoning hop.
        
        Args:
            reasoning_state: Current reasoning state [B, D]
            memory_nodes: Memory nodes [B, K, D]
            graph: Temporal reasoning graph (optional)
            mask: Valid nodes mask [B, K]
            
        Returns:
            Tuple of (refined_reasoning_state, attention_weights)
        """
        # Reasoning attention with residual connection
        attn_output, attn_weights = self.reasoning_attn(
            reasoning_state, memory_nodes, memory_nodes, mask
        )
        reasoning_state = self.norm1(reasoning_state + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(reasoning_state)
        reasoning_state = self.norm2(reasoning_state + self.dropout(ffn_output))
        
        return reasoning_state, attn_weights


class CHIMRT(nn.Module):
    """Conditional Hierarchical Integrated Multi-Relational Transformer.
    
    Performs hierarchical multi-hop reasoning over memory nodes,
    with learned temporal graphs and adaptive depth control.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 3,
        num_heads: int = 12,
        ffn_dim: int = 3072,
        dropout: float = 0.1,
        max_hops: int = 5,
    ):
        """Initialize CHIMRT.
        
        Args:
            hidden_dim: Hidden dimension
            num_layers: Number of CHIMRT layers per hop
            num_heads: Number of attention heads
            ffn_dim: Feed-forward dimension
            dropout: Dropout rate
            max_hops: Maximum reasoning hops
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_hops = max_hops
        
        # Stack CHIMRT layers
        self.layers = nn.ModuleList([
            CHIMRTLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Temporal adjacency learner
        self.temporal_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Question-conditioned refinement
        self.question_fusion = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        
        logger.info(f"Initialized CHIMRT ({num_layers} layers, {max_hops} max hops)")
    
    def forward(
        self,
        memory_nodes: torch.Tensor,
        question_embedding: torch.Tensor,
        frame_lengths: Optional[torch.Tensor] = None,
        max_hops: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """Perform hierarchical reasoning.
        
        Args:
            memory_nodes: Memory nodes [B, K, D]
            question_embedding: Question embedding [B, D]
            frame_lengths: Valid frame lengths [B]
            max_hops: Override max hops for this forward pass
            
        Returns:
            Tuple of:
            - reasoning_state: Final reasoning state [B, D]
            - reasoning_traces: List of reasoning states per hop [max_hops, (B, D)]
            - attention_traces: List of attention weights per hop
        """
        B, K, D = memory_nodes.shape
        max_hops = max_hops or self.max_hops
        
        # Initialize reasoning state from question
        reasoning_state = question_embedding  # [B, D]
        
        reasoning_traces = []
        attention_traces = []
        
        # Multi-hop reasoning
        for hop in range(max_hops):
            # Apply CHIMRT layers for this hop
            current_state = reasoning_state
            
            for layer in self.layers:
                current_state, attn_weights = layer(
                    current_state, memory_nodes, mask=self._get_mask(memory_nodes, frame_lengths)
                )
            
            reasoning_state = current_state
            reasoning_traces.append(reasoning_state.detach())
            attention_traces.append(attn_weights.detach())
            
            # Memory refinement for next hop
            if hop < max_hops - 1:
                memory_nodes = self._refine_memory(memory_nodes, reasoning_state)
        
        return reasoning_state, reasoning_traces, attention_traces
    
    def _get_mask(
        self,
        memory_nodes: torch.Tensor,
        frame_lengths: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Get valid nodes mask.
        
        Args:
            memory_nodes: [B, K, D]
            frame_lengths: [B]
            
        Returns:
            Mask [B, K] or None
        """
        if frame_lengths is None:
            return None
        
        B, K = memory_nodes.shape[:2]
        device = memory_nodes.device
        
        # For now, assume all memory nodes are valid
        # In practice, could prune based on importance
        mask = torch.ones(B, K, device=device, dtype=torch.bool)
        
        return mask
    
    def _refine_memory(
        self,
        memory_nodes: torch.Tensor,
        reasoning_state: torch.Tensor,
    ) -> torch.Tensor:
        """Refine memory nodes for next hop.
        
        Args:
            memory_nodes: Current nodes [B, K, D]
            reasoning_state: Current reasoning state [B, D]
            
        Returns:
            Refined memory nodes [B, K', D]
        """
        # For now, return unchanged
        # Can implement progressive refinement or pruning here
        return memory_nodes
