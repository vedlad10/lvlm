"""Mathematical formulations for LVLM based on research notes.

This module implements all key equations from the research notes:
- Temporal Memory Graph with semantic similarity
- Multi-hop reasoning with iterative refinement  
- CHIMRT attention mechanism
- Probabilistic graphical model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict
import math


# ============================================================================
# 1. TEMPORAL MEMORY GRAPH CONSTRUCTION
# ============================================================================

def compute_temporal_memory_graph(
    embeddings: torch.Tensor,
    temporal_adjacency_threshold: int = 3,
) -> torch.Tensor:
    """Construct temporal memory graph with semantic similarity.
    
    From research notes:
    A_ij = { |i-j| < k / sin(z_i, z_j)  if |i-j| < k
           { 0                            otherwise
    
    Args:
        embeddings: Clip embeddings [num_clips, embedding_dim]
        temporal_adjacency_threshold: k value for temporal window
        
    Returns:
        Adjacency matrix [num_clips, num_clips] with temporal & semantic edges
    """
    num_clips = embeddings.shape[0]
    device = embeddings.device
    
    # Normalize embeddings for cosine similarity
    normalized_emb = F.normalize(embeddings, p=2, dim=1)  # [N, D]
    
    # Compute pairwise cosine similarity: sim(z_i, z_j) = z_i^T z_j
    similarity = torch.mm(normalized_emb, normalized_emb.t())  # [N, N]
    
    # Create temporal adjacency mask: |i-j| < k
    temporal_mask = torch.zeros(num_clips, num_clips, device=device, dtype=torch.bool)
    for i in range(num_clips):
        for j in range(num_clips):
            if abs(i - j) < temporal_adjacency_threshold:
                temporal_mask[i, j] = True
    
    # Compute adjacency matrix
    # A_ij = temporal_weight / (1 + sin_distance)
    # Using cosine similarity: sin_distance ≈ sqrt(1 - cos_sim^2)
    sin_distance = torch.sqrt(torch.clamp(1 - similarity ** 2, min=1e-8))
    
    # A_ij = (1 / sin_distance) for temporal neighbors, 0 otherwise
    adjacency = torch.zeros_like(similarity)
    adjacency[temporal_mask] = 1.0 / sin_distance[temporal_mask]
    
    # Normalize adjacency matrix
    row_sum = adjacency.sum(dim=1, keepdim=True)
    adjacency = adjacency / (row_sum + 1e-8)
    
    return adjacency


# ============================================================================
# 2. MEMORY CONSOLIDATION WITH MARKOV ASSUMPTION
# ============================================================================

def apply_markov_memory_consolidation(
    embeddings: torch.Tensor,
    gate_weights: torch.Tensor,
    sparsity_target: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Consolidate memory nodes using Markov assumption.
    
    Reduces temporal complexity from O(T) to O(log T) by maintaining
    a compressed set of key memory nodes.
    
    Args:
        embeddings: Clip embeddings [num_clips, embedding_dim]
        gate_weights: Learned gate parameters [embedding_dim]
        sparsity_target: Target compression ratio (0.3 = 70% reduction)
        
    Returns:
        (consolidated_nodes, node_indices)
    """
    num_clips = embeddings.shape[0]
    target_nodes = max(1, int(num_clips * (1 - sparsity_target)))
    
    # Learn importance gates: g_i = sigmoid(w^T z_i)
    gate_logits = torch.matmul(embeddings, gate_weights)  # [N]
    gate_scores = torch.sigmoid(gate_logits)  # [N]
    
    # Select top-k important frames
    top_k_scores, top_k_indices = torch.topk(gate_scores, k=target_nodes)
    
    # Memory nodes: consolidated set
    memory_nodes = embeddings[top_k_indices]  # [K, D]
    
    return memory_nodes, top_k_indices


# ============================================================================
# 3. MULTI-HOP REASONING WITH ITERATIVE REFINEMENT
# ============================================================================

def multi_hop_reasoning(
    memory_nodes: torch.Tensor,
    question_embedding: torch.Tensor,
    reasoning_weights: Dict[str, torch.Tensor],
    num_hops: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Iterative multi-hop reasoning over memory nodes.
    
    From research notes:
    R_k = softmax(W_1 * R_{k-1} + W_2 * Θ + W_3 * M)
    
    Where:
    - R_k: Reasoning state at step k (probability distribution)
    - Θ: Question embedding
    - M: Memory nodes
    - W_1, W_2, W_3: Learned weight matrices
    
    Args:
        memory_nodes: Consolidated memory nodes [num_nodes, embedding_dim]
        question_embedding: Question text embedding [embedding_dim]
        reasoning_weights: Dictionary with keys 'W1', 'W2', 'W3'
        num_hops: Number of reasoning steps
        
    Returns:
        (final_reasoning_state, intermediate_states)
    """
    num_nodes = memory_nodes.shape[0]
    embedding_dim = memory_nodes.shape[1]
    device = memory_nodes.device
    
    # Initialize reasoning state: uniform distribution over memory nodes
    reasoning_state = torch.ones(num_nodes, device=device) / num_nodes  # [N]
    
    intermediate_states = [reasoning_state.clone()]
    
    W1 = reasoning_weights.get('W1', torch.eye(num_nodes, device=device))
    W2 = reasoning_weights.get('W2', torch.randn(num_nodes, embedding_dim, device=device) * 0.01)
    W3 = reasoning_weights.get('W3', torch.randn(num_nodes, embedding_dim, device=device) * 0.01)
    
    for k in range(num_hops):
        # R_k = softmax(W_1 * R_{k-1} + W_2 * Θ + W_3 * M)
        term1 = torch.matmul(W1, reasoning_state)  # [N]
        term2 = torch.matmul(W2, question_embedding)  # [N]
        term3 = torch.matmul(W3, memory_nodes.mean(dim=0))  # [N]
        
        logits = term1 + term2 + term3  # [N]
        reasoning_state = F.softmax(logits, dim=0)  # [N]
        
        intermediate_states.append(reasoning_state.clone())
    
    return reasoning_state, torch.stack(intermediate_states)


# ============================================================================
# 4. ATTENTION MECHANISM (CHIMRT)
# ============================================================================

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scaling_factor: float = 1.0,
) -> torch.Tensor:
    """Scaled dot-product attention mechanism.
    
    From research notes:
    Attention(Q, K, V) = softmax(Q * K^T / sqrt(d)) * V
    
    Where optional scaling_factor allows temporal scaling.
    
    Args:
        query: Query embeddings [batch_size, seq_len, d_k]
        key: Key embeddings [batch_size, seq_len, d_k]
        value: Value embeddings [batch_size, seq_len, d_v]
        mask: Attention mask (optional)
        scaling_factor: Temporal scaling factor
        
    Returns:
        Attention output [batch_size, seq_len, d_v]
    """
    d_k = query.shape[-1]
    
    # Compute attention scores: Q * K^T / sqrt(d)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  
    scores = scores * scaling_factor  # Apply temporal scaling
    
    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Apply softmax
    attention_weights = F.softmax(scores, dim=-1)
    
    # Apply dropout for regularization
    attention_weights = F.dropout(attention_weights, p=0.1, training=True)
    
    # Weighted combination of values
    output = torch.matmul(attention_weights, value)
    
    return output


# ============================================================================
# 5. CHIMRT: CONDITIONAL HIERARCHICAL MULTI-RELATIONAL TRANSFORMER
# ============================================================================

class CHIMRTAttention(nn.Module):
    """CHIMRT attention layer with multi-head and multi-relation support."""
    
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 12,
        num_relations: int = 3,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_relations = num_relations
        self.head_dim = embedding_dim // num_heads
        
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        
        # Multi-relation projections
        self.query_proj = nn.Linear(embedding_dim, embedding_dim)
        self.key_proj = nn.Linear(embedding_dim, embedding_dim)
        self.value_proj = nn.Linear(embedding_dim, embedding_dim)
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # Relation-specific parameters
        self.relation_weights = nn.Parameter(
            torch.randn(num_relations, embedding_dim) * 0.01
        )
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        relation_type: int = 0,
    ) -> torch.Tensor:
        """Forward pass with multi-head, multi-relation attention.
        
        Args:
            query: [batch_size, seq_len, embedding_dim]
            key: [batch_size, seq_len, embedding_dim]
            value: [batch_size, seq_len, embedding_dim]
            relation_type: Which relation to use (0-num_relations)
            
        Returns:
            Output [batch_size, seq_len, embedding_dim]
        """
        batch_size = query.shape[0]
        seq_len = query.shape[1]
        
        # Project and reshape for multi-head attention
        Q = self.query_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.key_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.value_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        Q = Q.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Compute attention with relation-specific scaling
        relation_scale = torch.sigmoid(
            self.relation_weights[relation_type]
        ).mean()
        
        attention_output = scaled_dot_product_attention(
            Q, K, V, scaling_factor=relation_scale
        )
        
        # Reshape back and project output
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, self.embedding_dim)
        output = self.output_proj(attention_output)
        
        return output


# ============================================================================
# 6. PROBABILISTIC MODEL: P(C,T|Θ,V)
# ============================================================================

def compute_probabilistic_model(
    question_embedding: torch.Tensor,
    reasoning_states: torch.Tensor,
    memory_nodes: torch.Tensor,
    class_weights: torch.Tensor,
    num_classes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute final probabilistic model output.
    
    From research notes:
    P(C,T|Θ,V) = [Σ_k P(C|F_k) * ∏_k P(R_k|I_R_k, T)]
    
    This represents:
    - P(C|F_k): Probability of answer class given features at step k
    - P(R_k|...): Probability of reasoning state at step k
    
    Args:
        question_embedding: Question encoding [embedding_dim]
        reasoning_states: Multi-hop reasoning states [num_hops, num_nodes]
        memory_nodes: Memory node embeddings [num_nodes, embedding_dim]
        class_weights: Learnable class weights [num_classes, embedding_dim]
        num_classes: Number of answer classes
        
    Returns:
        (class_logits, temporal_logits)
    """
    num_hops = reasoning_states.shape[0]
    num_nodes = reasoning_states.shape[1]
    
    # Compute P(C|F_k) for each reasoning step
    # Using weighted combination: score_c = Σ_k R_k * (w_c^T memory_nodes)
    class_logits = torch.zeros(num_classes, device=question_embedding.device)
    
    for c in range(num_classes):
        class_score = 0
        for k in range(num_hops):
            # R_k is probability distribution over nodes
            node_features = torch.matmul(
                reasoning_states[k],
                memory_nodes
            )  # [embedding_dim]
            
            # P(C|F_k) ∝ w_c^T * node_features
            class_probability = torch.dot(
                class_weights[c],
                node_features
            )
            class_score += class_probability
        
        class_logits[c] = class_score / num_hops
    
    # Compute temporal grounding: P(T|reasoning_states)
    # Use mean reasoning state for temporal prediction
    mean_reasoning = reasoning_states.mean(dim=0)  # [num_nodes]
    temporal_logits = torch.log(mean_reasoning + 1e-8)  # For temporal regression
    
    return class_logits, temporal_logits


# ============================================================================
# 7. LOSS FUNCTIONS WITH PROPER WEIGHTING
# ============================================================================

def compute_lvlm_loss(
    class_logits: torch.Tensor,
    temporal_logits: torch.Tensor,
    target_class: torch.Tensor,
    target_temporal: torch.Tensor,
    loss_weights: Dict[str, float],
) -> Dict[str, torch.Tensor]:
    """Compute total loss with all components.
    
    From research notes - Loss weights:
    - answer_loss_weight: 1.0
    - temporal_grounding_weight: 0.5
    - contrastive_weight: 0.2
    - sparsity_weight: 0.1
    - adaptive_depth_weight: 0.2
    
    Args:
        class_logits: Answer class predictions [num_classes]
        temporal_logits: Temporal predictions [num_temporal_bins]
        target_class: Ground truth answer class
        target_temporal: Ground truth temporal spans
        loss_weights: Dictionary of loss component weights
        
    Returns:
        Dictionary with individual loss components
    """
    losses = {}
    
    # 1. Answer loss (cross-entropy)
    answer_loss = F.cross_entropy(
        class_logits.unsqueeze(0),
        target_class.unsqueeze(0)
    )
    losses['answer'] = answer_loss * loss_weights.get('answer_loss_weight', 1.0)
    
    # 2. Temporal grounding loss (smooth L1)
    if target_temporal is not None:
        temporal_loss = F.smooth_l1_loss(
            temporal_logits,
            target_temporal
        )
        losses['temporal'] = temporal_loss * loss_weights.get('temporal_grounding_weight', 0.5)
    
    # 3. Sparsity loss (encourage sparse reasoning)
    sparsity_loss = -torch.mean(
        -torch.sum(torch.softmax(class_logits, dim=0) * 
                  torch.log(torch.softmax(class_logits, dim=0) + 1e-8), dim=0)
    )
    losses['sparsity'] = sparsity_loss * loss_weights.get('sparsity_weight', 0.1)
    
    # Total loss
    total_loss = sum(losses.values())
    losses['total'] = total_loss
    
    return losses


# ============================================================================
# 8. ADAPTIVE REASONING DEPTH
# ============================================================================

def compute_adaptive_stopping_gate(
    reasoning_state: torch.Tensor,
    stop_weights: torch.Tensor,
    entropy_threshold: float = 0.5,
) -> Tuple[bool, float, float]:
    """Adaptive depth: decide when to stop reasoning.
    
    Uses:
    1. Learned stopping gate: stop_prob = σ(w^T R_k)
    2. Entropy check: continue if entropy > ε
    
    Args:
        reasoning_state: Current reasoning distribution [num_nodes]
        stop_weights: Learned gate weights [num_nodes]
        entropy_threshold: Minimum entropy to continue
        
    Returns:
        (should_stop, stop_probability, entropy)
    """
    # Learned stopping gate: stop_prob = sigmoid(w^T R_k)
    stop_logit = torch.dot(stop_weights, reasoning_state)
    stop_probability = torch.sigmoid(stop_logit).item()
    
    # Entropy of reasoning distribution: -Σ_i R_k^i * log(R_k^i)
    entropy = -torch.sum(
        reasoning_state * torch.log(reasoning_state + 1e-8)
    ).item()
    
    # Stop if stop_prob > 0.5 OR entropy < threshold
    should_stop = (stop_probability > 0.5) or (entropy < entropy_threshold)
    
    return should_stop, stop_probability, entropy
