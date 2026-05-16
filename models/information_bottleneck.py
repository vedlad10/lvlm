"""Information Bottleneck Theory Integration for LVLM.

Theoretical Framework:
The Information Bottleneck (IB) objective by Tishby aims to extract
relevant information from input X about target Y, while compressing X.

Objective:
L = I(X; Z) - β*I(Z; Y)

Where:
- I(X; Z): mutual information between input and representation (compression)
- I(Z; Y): mutual information between representation and target (usefulness)
- β: trade-off parameter (Lagrange multiplier)
- Z: latent representation (bottleneck)

Practical Implementation: Variational Information Bottleneck (VIB)
L = E_q(z|x)[-log p(y|z)] + β*KL(q(z|x)||p(z))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# MUTUAL INFORMATION ESTIMATION
# ============================================================================

def estimate_mutual_information_lower_bound(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    num_samples: int = 100,
) -> torch.Tensor:
    """Estimate lower bound on mutual information I(X; Y).
    
    Uses MINE (Mutual Information Neural Estimation) approximation.
    
    Args:
        embeddings: Representation Z [B, D]
        labels: Target Y [B, num_classes]
        num_samples: Number of negative samples for MINE
        
    Returns:
        Lower bound on mutual information
    """
    batch_size = embeddings.shape[0]
    device = embeddings.device
    
    # Joint samples: p(z, y)
    joint_indices = torch.arange(batch_size, device=device)
    
    # Marginal samples: p(z)p(y) - shuffle labels for negative samples
    margin_indices = torch.randperm(batch_size, device=device)
    
    # Concatenate joint and marginal samples
    z_joint = embeddings[joint_indices]  # [B, D]
    y_joint = labels[joint_indices]  # [B]
    
    z_margin = embeddings[margin_indices]  # [B, D]
    y_margin = labels[margin_indices]  # [B]
    
    # Compute scores (simplified MINE)
    joint_score = torch.sum(z_joint * F.embedding(y_joint, torch.eye(labels.max().item() + 1, device=device)), dim=1)
    margin_score = torch.sum(z_margin * F.embedding(y_margin, torch.eye(labels.max().item() + 1, device=device)), dim=1)
    
    # MINE lower bound: E[T(x,y)] - log(E[e^T(x,y')])
    mi_lower_bound = joint_score.mean() - torch.log(torch.exp(margin_score).mean() + 1e-8)
    
    return torch.clamp(mi_lower_bound, min=0)


# ============================================================================
# VARIATIONAL INFORMATION BOTTLENECK (VIB) COMPONENTS
# ============================================================================

class VariationalBottleneck(nn.Module):
    """Variational Information Bottleneck layer.
    
    Implements stochastic bottleneck with learnable prior and posterior.
    Forces compression of information while maintaining task performance.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        beta: float = 1e-3,
    ):
        """Initialize VIB layer.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension for encoder
            latent_dim: Latent space dimension (bottleneck size)
            beta: Trade-off parameter (higher = more compression)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Encoder: q(z|x) - learn mean and log-variance
        self.encoder_mean = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        
        self.encoder_logvar = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        
        # Prior: p(z) = N(0, I)
        self.register_buffer('prior_mean', torch.zeros(latent_dim))
        self.register_buffer('prior_logvar', torch.zeros(latent_dim))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Variational bottleneck forward pass.
        
        Implements stochastic compression of input through latent bottleneck.
        
        Args:
            x: Input tensor [B, input_dim]
            
        Returns:
            (sampled_z, kl_divergence)
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Encode: q(z|x)
        mean = self.encoder_mean(x)  # [B, latent_dim]
        logvar = self.encoder_logvar(x)  # [B, latent_dim]
        
        # Reparameterization trick: z = mean + std * epsilon
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mean + std * epsilon  # [B, latent_dim]
        
        # KL divergence: KL(q(z|x) || p(z))
        # = 0.5 * sum(mean^2 + var - 1 - log(var))
        kl_div = -0.5 * torch.sum(
            1 + logvar - mean.pow(2) - logvar.exp(),
            dim=1
        ).mean()
        
        return z, kl_div


# ============================================================================
# INFORMATION BOTTLENECK FOR MEMORY CONSOLIDATION
# ============================================================================

def compute_ib_memory_consolidation_loss(
    original_frames: torch.Tensor,
    consolidated_memory: torch.Tensor,
    answer_logits: torch.Tensor,
    target_answer: torch.Tensor,
    beta: float = 0.01,
) -> Dict[str, torch.Tensor]:
    """Compute Information Bottleneck loss for memory consolidation.
    
    Objective: Compress video frames into memory nodes while preserving
    answer prediction capability.
    
    Theory:
    - I(frames; memory): Compression (minimize)
    - I(memory; answer): Task performance (maximize)
    - IB Loss = I(frames; memory) - β*I(memory; answer)
    
    Args:
        original_frames: Original video frames [B, T, D]
        consolidated_memory: Consolidated memory nodes [B, K, D]
        answer_logits: Predicted answer logits [B, num_classes]
        target_answer: Ground truth answers [B]
        beta: Trade-off parameter
        
    Returns:
        Dictionary with loss components
    """
    losses = {}
    
    # Compression term: I(X; Z) - how much did we compress?
    # Measured as reconstruction error
    num_frames = original_frames.shape[1]
    num_memory = consolidated_memory.shape[1]
    compression_ratio = 1.0 - (num_memory / num_frames)
    
    # Reconstruction loss (proxy for I(X; Z))
    # How well can we reconstruct original frames from memory?
    # Use cosine distance as proxy
    original_mean = original_frames.mean(dim=1)  # [B, D]
    memory_mean = consolidated_memory.mean(dim=1)  # [B, D]
    
    reconstruction_loss = 1.0 - F.cosine_similarity(original_mean, memory_mean).mean()
    losses['reconstruction_compression'] = reconstruction_loss
    
    # Task performance term: I(Z; Y) - how useful is Z for answer?
    # Measured as negative log probability of correct answer
    answer_loss = F.cross_entropy(answer_logits, target_answer)
    losses['answer_task_performance'] = answer_loss
    
    # Information Bottleneck objective
    # IB_loss = I(X; Z) - β*I(Z; Y)
    # We minimize this, so lower answer_loss (higher I(Z;Y)) reduces total loss
    ib_loss = reconstruction_loss - beta * (-answer_loss)
    losses['ib_memory_consolidation'] = ib_loss
    
    losses['compression_ratio'] = torch.tensor(
        compression_ratio, dtype=torch.float32, device=original_frames.device
    )
    
    return losses


# ============================================================================
# INFORMATION BOTTLENECK FOR REASONING DEPTH
# ============================================================================

def compute_ib_reasoning_depth_loss(
    reasoning_states: torch.Tensor,
    answer_logits_per_hop: list,
    target_answer: torch.Tensor,
    beta: float = 0.01,
) -> Dict[str, torch.Tensor]:
    """Compute IB loss for adaptive reasoning depth.
    
    Objective: Use minimal reasoning hops while maintaining accuracy.
    
    Theory:
    - I(num_hops; answer): Correlation between hops used and performance
    - Minimize: num_hops while maximize answer accuracy
    
    Args:
        reasoning_states: Reasoning states per hop [num_hops, B, K]
        answer_logits_per_hop: Answer logits per hop [num_hops, (B, num_classes)]
        target_answer: Ground truth answers [B]
        beta: Trade-off parameter
        
    Returns:
        Dictionary with loss components
    """
    losses = {}
    
    num_hops = len(reasoning_states)
    batch_size = reasoning_states[0].shape[0]
    device = reasoning_states[0].device
    
    # Measure reasoning state diversity across hops
    # High diversity = more information per hop
    diversity_loss = 0
    for hop in range(num_hops - 1):
        state_similarity = F.cosine_similarity(
            reasoning_states[hop].mean(dim=1),  # [B, D] → [D]
            reasoning_states[hop + 1].mean(dim=1)  # [B, D] → [D]
        ).mean()
        diversity_loss += state_similarity
    
    diversity_loss = diversity_loss / max(1, num_hops - 1)
    losses['reasoning_diversity'] = diversity_loss
    
    # Performance across hops
    # Later hops should perform better (convergence)
    hop_performance_loss = 0
    for hop, logits in enumerate(answer_logits_per_hop):
        hop_loss = F.cross_entropy(logits, target_answer)
        # Weight earlier hops less (they haven't converged yet)
        weight = (hop + 1) / num_hops
        hop_performance_loss += weight * hop_loss
    
    losses['hop_convergence'] = hop_performance_loss
    
    # IB depth loss: penalize using many hops
    hops_penalty = torch.tensor(
        num_hops / 5.0, dtype=torch.float32, device=device
    )  # Normalized to max 5 hops
    losses['depth_penalty'] = hops_penalty
    
    # Total IB depth loss
    ib_depth = hops_penalty - beta * (-hop_performance_loss)
    losses['ib_reasoning_depth'] = ib_depth
    
    return losses


# ============================================================================
# INFORMATION BOTTLENECK FOR MULTIMODAL ALIGNMENT
# ============================================================================

class MultimodalInformationBottleneck(nn.Module):
    """Information Bottleneck for multimodal fusion (vision + language).
    
    Compresses multimodal input (image + text) into shared embedding
    while preserving task-relevant information.
    """
    
    def __init__(
        self,
        vision_dim: int,
        language_dim: int,
        shared_dim: int,
        beta: float = 0.01,
    ):
        """Initialize multimodal IB.
        
        Args:
            vision_dim: Vision encoder output dimension
            language_dim: Language encoder output dimension
            shared_dim: Shared bottleneck dimension
            beta: Trade-off parameter
        """
        super().__init__()
        
        self.vision_bottleneck = VariationalBottleneck(
            input_dim=vision_dim,
            hidden_dim=vision_dim // 2,
            latent_dim=shared_dim,
            beta=beta,
        )
        
        self.language_bottleneck = VariationalBottleneck(
            input_dim=language_dim,
            hidden_dim=language_dim // 2,
            latent_dim=shared_dim,
            beta=beta,
        )
    
    def forward(
        self,
        vision_features: torch.Tensor,
        language_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compress multimodal features through bottleneck.
        
        Args:
            vision_features: Vision embeddings [B, vision_dim]
            language_features: Language embeddings [B, language_dim]
            
        Returns:
            Dictionary with bottleneck embeddings and KL losses
        """
        # Pass through bottlenecks
        vision_z, vision_kl = self.vision_bottleneck(vision_features)
        language_z, language_kl = self.language_bottleneck(language_features)
        
        # Shared representation: average bottleneck outputs
        shared_z = (vision_z + language_z) / 2
        
        return {
            'vision_z': vision_z,
            'language_z': language_z,
            'shared_z': shared_z,
            'vision_kl': vision_kl,
            'language_kl': language_kl,
            'total_kl': vision_kl + language_kl,
        }


# ============================================================================
# COMPLETE IB LOSS FOR LVLM
# ============================================================================

def compute_complete_ib_loss(
    output: Dict,
    targets: Dict,
    original_frames: torch.Tensor,
    consolidated_memory: torch.Tensor,
    beta_memory: float = 0.01,
    beta_depth: float = 0.01,
    beta_multimodal: float = 0.01,
) -> Dict[str, torch.Tensor]:
    """Compute complete Information Bottleneck loss for LVLM.
    
    Integrates IB principles across:
    1. Memory consolidation (video compression)
    2. Reasoning depth (number of hops used)
    3. Multimodal alignment (vision + language)
    
    Args:
        output: Model output dictionary
        targets: Target dictionary
        original_frames: Original video frames [B, T, D]
        consolidated_memory: Consolidated memory [B, K, D]
        beta_memory: Trade-off for memory IB
        beta_depth: Trade-off for depth IB
        beta_multimodal: Trade-off for multimodal IB
        
    Returns:
        Dictionary with all IB loss components
    """
    losses = {}
    
    # 1. Memory Consolidation IB Loss
    if 'answer_logits' in output and 'answers' in targets:
        memory_ib = compute_ib_memory_consolidation_loss(
            original_frames=original_frames,
            consolidated_memory=consolidated_memory,
            answer_logits=output['answer_logits'],
            target_answer=targets['answers'],
            beta=beta_memory,
        )
        losses.update(memory_ib)
    
    # 2. Reasoning Depth IB Loss
    if 'all_answer_logits' in output and 'answers' in targets:
        depth_ib = compute_ib_reasoning_depth_loss(
            reasoning_states=output.get('reasoning_states', []),
            answer_logits_per_hop=output['all_answer_logits'],
            target_answer=targets['answers'],
            beta=beta_depth,
        )
        losses.update(depth_ib)
    
    # Total IB loss (weighted sum)
    total_ib = (
        losses.get('ib_memory_consolidation', torch.tensor(0.0)) +
        losses.get('ib_reasoning_depth', torch.tensor(0.0))
    )
    losses['total_ib'] = total_ib
    
    return losses
