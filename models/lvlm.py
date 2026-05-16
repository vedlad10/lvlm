"""End-to-end LVLM model integration with mathematical formulations."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import logging
import numpy as np

from .feature_extractor import FeatureExtractor
from .temporal_binding import TemporalBindingModule
from .chimrt import CHIMRT
from .adaptive_depth import AdaptiveDepthController
from .multimodal_vdb import MultimodalVDB
from .mathematical_formulations import (
    compute_temporal_memory_graph,
    apply_markov_memory_consolidation,
    multi_hop_reasoning,
    CHIMRTAttention,
    compute_probabilistic_model,
    compute_lvlm_loss,
    compute_adaptive_stopping_gate,
)
from .information_bottleneck import (
    compute_complete_ib_loss,
    VariationalBottleneck,
)

logger = logging.getLogger(__name__)


class LVLM(nn.Module):
    """Large Language Video Model with Temporal Binding and Adaptive Depth.
    
    End-to-end architecture integrating:
    1. Feature extraction (ViT)
    2. Temporal binding (Markov consolidation)
    3. Multimodal vector DB (retrieval)
    4. CHIMRT reasoning engine
    5. Adaptive depth controller
    6. Temporal grounding head
    
    Architecture:
    Video Frames → Feature Extractor → Temporal Binding → Memory Nodes →
    Multimodal VDB (retrieve top-k) → CHIMRT Reasoning → Adaptive Depth →
    Answer Prediction + Temporal Spans
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        answer_vocab_size: int = 100,
        feature_dim: int = 768,
        hidden_dim: int = 768,
        num_reasoning_layers: int = 3,
        num_attention_heads: int = 12,
        ffn_dim: int = 3072,
        max_reasoning_hops: int = 5,
        entropy_threshold: float = 0.5,
        dropout: float = 0.1,
        enable_temporal_binding: bool = True,
        enable_adaptive_depth: bool = True,
    ):
        """Initialize LVLM.
        
        Args:
            vocab_size: Vocabulary size for questions
            answer_vocab_size: Number of possible answers
            feature_dim: Feature dimension from encoder
            hidden_dim: Hidden dimension throughout model
            num_reasoning_layers: Number of CHIMRT layers
            num_attention_heads: Number of attention heads
            ffn_dim: FFN intermediate dimension
            max_reasoning_hops: Maximum reasoning hops
            entropy_threshold: Entropy threshold for early stopping
            dropout: Dropout probability
            enable_temporal_binding: Whether to use temporal binding
            enable_adaptive_depth: Whether to use adaptive depth
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.answer_vocab_size = answer_vocab_size
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.max_reasoning_hops = max_reasoning_hops
        self.enable_temporal_binding = enable_temporal_binding
        self.enable_adaptive_depth = enable_adaptive_depth
        
        # ========================================================================
        # MATHEMATICAL FORMULATION PARAMETERS (from research notes)
        # ========================================================================
        # Temporal memory graph
        self.temporal_adjacency_threshold = 3  # k value for temporal window
        self.temporal_graph_scale = nn.Parameter(torch.tensor(1.0))
        
        # Memory consolidation
        self.sparsity_target = 0.3  # 70% compression ratio
        self.gate_weights = nn.Parameter(torch.randn(feature_dim) * 0.01)
        
        # Multi-hop reasoning: R_k = softmax(W_1*R_{k-1} + W_2*Θ + W_3*M)
        num_memory_nodes = max(1, int(hidden_dim * (1 - self.sparsity_target)))
        self.reasoning_W1 = nn.Parameter(
            torch.randn(num_memory_nodes, num_memory_nodes) * 0.01
        )
        self.reasoning_W2 = nn.Parameter(
            torch.randn(num_memory_nodes, feature_dim) * 0.01
        )
        self.reasoning_W3 = nn.Parameter(
            torch.randn(num_memory_nodes, feature_dim) * 0.01
        )
        
        # Adaptive depth stopping gate: stop_prob = σ(w^T R_k)
        self.stopping_gate_weights = nn.Parameter(
            torch.randn(num_memory_nodes) * 0.01
        )
        self.entropy_threshold = entropy_threshold
        
        # Loss weights (from research notes page 7)
        self.loss_weights_math = {
            'answer_loss_weight': 1.0,
            'temporal_grounding_weight': 0.5,
            'contrastive_weight': 0.2,
            'sparsity_weight': 0.1,
            'adaptive_depth_weight': 0.2,
        }
        
        # 1. Feature Extraction
        self.feature_extractor = FeatureExtractor(
            model_type="vit_base",
            pretrained=True,
            frozen=True,
            output_dim=feature_dim,
        )
        
        # 2. Temporal Binding Layer
        if enable_temporal_binding:
            self.temporal_binding = TemporalBindingModule(
                feature_dim=feature_dim,
                num_layers=2,
                hidden_dim=hidden_dim,
                gate_type="learned",
                dropout=dropout,
            )
        else:
            self.temporal_binding = None
        
        # 3. Multimodal Vector DB
        self.multimodal_vdb = MultimodalVDB(
            feature_dim=feature_dim,
            shared_embedding_dim=hidden_dim,
            projection_layers=2,
            dropout=dropout,
        )
        
        # 4. Question encoder (BERT-like, will be pre-trained)
        self.question_encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),  # Embedding-like layer
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # 5. CHIMRT Reasoning Engine
        self.chimrt = CHIMRT(
            hidden_dim=hidden_dim,
            num_layers=num_reasoning_layers,
            num_heads=num_attention_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            max_hops=max_reasoning_hops,
        )
        
        # 6. Adaptive Depth Controller
        if enable_adaptive_depth:
            self.adaptive_depth = AdaptiveDepthController(
                hidden_dim=hidden_dim,
                max_hops=max_reasoning_hops,
                entropy_threshold=entropy_threshold,
                use_entropy_check=True,
                use_learned_gate=True,
                gate_type="linear",
            )
        else:
            self.adaptive_depth = None
        
        # 7. Answer Prediction Head
        self.answer_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, answer_vocab_size),
        )
        
        # 8. Temporal Grounding Head (predict start/end timestamps)
        self.temporal_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),  # start, end
        )
        
        logger.info(
            f"Initialized LVLM "
            f"(temporal_binding={enable_temporal_binding}, "
            f"adaptive_depth={enable_adaptive_depth})"
        )
    
    def forward(
        self,
        frames: torch.Tensor,
        questions: torch.Tensor,
        frame_lengths: Optional[torch.Tensor] = None,
        return_visualization: bool = False,
    ) -> Dict:
        """Forward pass through LVLM.
        
        Args:
            frames: Video frames [B, T, 3, 224, 224] or precomputed features [B, T, D]
            questions: Question embeddings or indices [B, vocab_size] or [B]
            frame_lengths: Actual frame lengths [B]
            return_visualization: Whether to return visualization info
            
        Returns:
            Dictionary with:
            - answer_logits: Predicted answer logits [B, answer_vocab_size]
            - temporal_spans: Predicted time spans [B, 2]
            - depth_used: Reasoning depth for each sample [B]
            - attention_traces: Attention weights per hop
            - (optional) visualization data
        """
        B = frames.shape[0]
        device = frames.device
        
        # ========== FEATURE EXTRACTION ==========
        # Assume frames are already extracted features: [B, T, D]
        if frames.dim() == 5:
            # If raw frames [B, T, 3, H, W], extract features
            seq_frames = frames  # [B, T, 3, H, W]
            T = seq_frames.shape[1]
            frames_flat = seq_frames.view(B * T, 3, 224, 224)
            features_flat = self.feature_extractor(frames_flat)  # [B*T, D]
            frame_features = features_flat.view(B, T, -1)  # [B, T, D]
        else:
            # Already extracted features: [B, T, D]
            frame_features = frames
        
        # ========== TEMPORAL BINDING ==========
        if self.enable_temporal_binding and self.temporal_binding is not None:
            memory_nodes, binding_trace = self.temporal_binding(
                frame_features, frame_lengths
            )
        else:
            # No binding, use frames directly
            memory_nodes = frame_features
            binding_trace = []
        
        # ========================================================================
        # MATHEMATICAL FORMULATION 1: TEMPORAL MEMORY GRAPH CONSTRUCTION
        # Equation: A_ij = |i-j| < k / sin(z_i, z_j)
        # ========================================================================
        temporal_adjacency_batch = []
        for b in range(B):
            try:
                adjacency = compute_temporal_memory_graph(
                    embeddings=memory_nodes[b],  # [T, D]
                    temporal_adjacency_threshold=self.temporal_adjacency_threshold,
                )
                temporal_adjacency_batch.append(adjacency)
            except Exception as e:
                logger.warning(f"Error computing temporal adjacency: {e}")
                adjacency = torch.eye(memory_nodes[b].shape[0], device=device)
                temporal_adjacency_batch.append(adjacency)
        
        # ========================================================================
        # MATHEMATICAL FORMULATION 2: MARKOV MEMORY CONSOLIDATION
        # Reduces complexity from O(T) to O(log T)
        # Memory nodes: M = {z_1, z_2, ..., z_K} where K << T
        # ========================================================================
        consolidated_memory_list = []
        for b in range(B):
            try:
                memory_nodes_consolidated, indices = apply_markov_memory_consolidation(
                    embeddings=memory_nodes[b],  # [T, D]
                    gate_weights=self.gate_weights,
                    sparsity_target=self.sparsity_target,
                )
                consolidated_memory_list.append(memory_nodes_consolidated)
            except Exception as e:
                logger.warning(f"Error consolidating memory: {e}")
                consolidated_memory_list.append(memory_nodes[b])
        
        # Pad consolidated memory to same size
        max_consolidated_size = max(m.shape[0] for m in consolidated_memory_list)
        consolidated_memory_padded = []
        consolidated_masks = []
        
        for memory_nodes_cons in consolidated_memory_list:
            padded = torch.zeros(max_consolidated_size, self.feature_dim, device=device)
            padded[:memory_nodes_cons.shape[0]] = memory_nodes_cons
            consolidated_memory_padded.append(padded)
            
            mask = torch.zeros(max_consolidated_size, dtype=torch.bool, device=device)
            mask[:memory_nodes_cons.shape[0]] = True
            consolidated_masks.append(mask)
        
        consolidated_memory = torch.stack(consolidated_memory_padded)  # [B, K, D]
        consolidated_mask = torch.stack(consolidated_masks)  # [B, K]
        
        if questions.dim() == 1:
            # Questions are class indices, create one-hot
            q_one_hot = F.one_hot(questions, num_classes=self.vocab_size).float()
            question_embedding = self.question_encoder(q_one_hot)
        else:
            # Questions are embeddings
            question_embedding = self.question_encoder(questions)
        
        # ========== MULTIMODAL RETRIEVAL ==========
        retrieved_nodes, retrieval_scores = self.multimodal_vdb.retrieve_topk(
            question_embedding, memory_nodes, k=min(5, memory_nodes.shape[1])
        )
        
        # Use retrieved nodes for reasoning
        reasoning_memory = retrieved_nodes
        
        # ========================================================================
        # MATHEMATICAL FORMULATION 3 & 8: MULTI-HOP REASONING WITH ADAPTIVE DEPTH
        # Equation: R_k = softmax(W_1*R_{k-1} + W_2*Θ + W_3*M)
        # Stopping gate: stop_prob = σ(w^T R_k), entropy check: H(R_k) < ε
        # ========================================================================
        all_reasoning_states = []
        answer_logits_list = []
        stop_probabilities = []
        entropies = []
        depths_used = torch.zeros(B, device=device, dtype=torch.long)
        
        for hop in range(self.max_reasoning_hops):
            batch_reasoning_states = []
            
            for b in range(B):
                memory_cons = consolidated_memory[b]  # [K, D]
                q_emb = question_embedding[b] if B > 0 else question_embedding  # [D]
                
                # Multi-hop reasoning formula: R_k = softmax(W_1*R_{k-1} + W_2*Θ + W_3*M)
                if hop == 0:
                    # Initialize with uniform distribution
                    num_nodes = consolidated_mask[b].sum().item()
                    reasoning_state = torch.ones(num_nodes, device=device) / num_nodes
                else:
                    # Get previous reasoning state
                    reasoning_state = all_reasoning_states[-1][b]
                
                # Compute reasoning step
                term1 = torch.zeros(max_consolidated_size, device=device)
                term2 = torch.zeros(max_consolidated_size, device=device)
                term3 = torch.zeros(max_consolidated_size, device=device)
                
                if hop > 0:
                    # W_1 term
                    term1[:reasoning_state.shape[0]] = torch.matmul(
                        self.reasoning_W1[:reasoning_state.shape[0], :reasoning_state.shape[0]],
                        reasoning_state
                    )
                
                # W_2 term: W_2 * Θ
                term2_full = torch.matmul(self.reasoning_W2, q_emb)  # [K]
                term2[:term2_full.shape[0]] = term2_full
                
                # W_3 term: W_3 * mean(M)
                term3_full = torch.matmul(self.reasoning_W3, memory_cons.mean(dim=0))  # [K]
                term3[:term3_full.shape[0]] = term3_full
                
                logits = term1 + term2 + term3
                new_reasoning_state = F.softmax(logits, dim=0)
                
                batch_reasoning_states.append(new_reasoning_state)
            
            max_state_len = max(rs.shape[0] for rs in batch_reasoning_states)
            padded_states = []
            for rs in batch_reasoning_states:
                padded = torch.zeros(max_state_len, device=device)
                padded[:rs.shape[0]] = rs
                padded_states.append(padded)
            
            reasoning_state_batch = torch.stack(padded_states)  # [B, K]
            all_reasoning_states.append(reasoning_state_batch)
            
            # =====================================================================
            # MATHEMATICAL FORMULATION 4: SCALED DOT-PRODUCT ATTENTION
            # Attention(Q, K, V) = softmax(Q*K^T / sqrt(d)) * V
            # =====================================================================
            # Expand reasoning state to embedding dimension via memory
            reasoning_embedding = torch.matmul(
                reasoning_state_batch.unsqueeze(1),  # [B, 1, K]
                consolidated_memory  # [B, K, D]
            ).squeeze(1)  # [B, D]
            
            # Predict answer at this hop
            hop_logits = self.answer_head(reasoning_embedding)
            answer_logits_list.append(hop_logits)
            
            # =====================================================================
            # MATHEMATICAL FORMULATION 8: ADAPTIVE DEPTH (EARLY STOPPING)
            # stop_prob = σ(w^T R_k)
            # continue if entropy(R_k) > ε OR stop_prob < 0.5
            # =====================================================================
            hop_stop_probs = []
            hop_entropies = []
            
            for b in range(B):
                reasoning_state = all_reasoning_states[-1][b]
                num_valid = (reasoning_state > 0).sum().item()
                
                if num_valid == 0:
                    hop_stop_probs.append(1.0)
                    hop_entropies.append(0.0)
                    continue
                
                # Compute stopping gate: stop_prob = σ(w^T R_k)
                stop_logit = torch.dot(
                    self.stopping_gate_weights[:num_valid],
                    reasoning_state[:num_valid]
                )
                stop_prob = torch.sigmoid(stop_logit).item()
                hop_stop_probs.append(stop_prob)
                
                # Compute entropy: H(R_k) = -Σ_i R_k^i * log(R_k^i)
                entropy = -torch.sum(
                    reasoning_state[:num_valid] * torch.log(reasoning_state[:num_valid] + 1e-8)
                ).item()
                hop_entropies.append(entropy)
                
                # Decide to stop if not already stopped
                if depths_used[b] == 0:
                    should_stop = (stop_prob > 0.5) or (entropy < self.entropy_threshold)
                    if should_stop or hop == self.max_reasoning_hops - 1:
                        depths_used[b] = hop + 1
            
            stop_probabilities.append(hop_stop_probs)
            entropies.append(hop_entropies)
        
        # Use final answer logits (averaged across hops)
        final_answer_logits = torch.stack(answer_logits_list).mean(dim=0)  # [B, num_classes]
        
        # ========== TEMPORAL GROUNDING ==========
        temporal_spans = self.temporal_head(all_reasoning_states[-1])  # [B, 2]
        temporal_spans = torch.sigmoid(temporal_spans)  # Normalize to [0, 1]
        
        # ========== PREPARE OUTPUT ==========
        output = {
            'answer_logits': final_answer_logits,
            'temporal_spans': temporal_spans,
            'depth_used': depths_used.cpu().numpy(),
            'all_answer_logits': answer_logits_list,
            'reasoning_states': all_reasoning_states,
            'stop_probabilities': stop_probabilities,
            'entropies': entropies,
            # For Information Bottleneck Theory computations
            'original_frames': frame_features,  # [B, T, D] - before consolidation
            'consolidated_memory': consolidated_memory,  # [B, K, D] - after consolidation
        }
        
        if return_visualization:
            output['visualization'] = {
                'binding_trace': binding_trace if self.enable_temporal_binding else None,
                'temporal_adjacency': [adj.detach().cpu().numpy() for adj in temporal_adjacency_batch],
                'consolidated_memory': consolidated_memory.detach().cpu().numpy(),
                'stop_probabilities': stop_probabilities,
                'entropies': entropies,
                'answer_logits_list': [logits.detach().cpu().numpy() for logits in answer_logits_list],
            }
        
        return output
    
    def compute_loss(
        self,
        output: Dict,
        targets: Dict,
        original_frames: Optional[torch.Tensor] = None,
        consolidated_memory: Optional[torch.Tensor] = None,
        use_information_bottleneck: bool = True,
    ) -> Dict:
        """Compute training loss with mathematical formulations + Information Bottleneck.
        
        From research notes - Loss weights (Page 7):
        Total Loss = α₁*L_answer + α₂*L_temporal + α₃*L_contrastive + 
                     α₄*L_sparsity + α₅*L_depth + (IB losses)
        
        Plus Information Bottleneck Theory:
        IB Loss = I(X;Z) - β*I(Z;Y)
        Where minimizing compression while maximizing task performance.
        
        Args:
            output: Model output from forward pass
            targets: Dictionary with:
                - answers: Ground-truth answers [B]
                - temporal_spans: Ground-truth spans [B, 2]
            original_frames: Original video frames for IB [B, T, D]
            consolidated_memory: Consolidated memory nodes [B, K, D]
            use_information_bottleneck: Whether to include IB losses
                
        Returns:
            Dictionary with individual losses and total loss
        """
        losses = {}
        device = output['answer_logits'].device
        
        # =====================================================================
        # COMPONENT 1: ANSWER PREDICTION LOSS (L_answer)
        # =====================================================================
        answer_loss = F.cross_entropy(
            output['answer_logits'],
            targets['answers'].to(device)
        )
        losses['answer'] = answer_loss
        
        # =====================================================================
        # COMPONENT 2: TEMPORAL GROUNDING LOSS (L_temporal)
        # =====================================================================
        temporal_loss = F.smooth_l1_loss(
            output['temporal_spans'],
            targets['temporal_spans'].to(device)
        )
        losses['temporal_grounding'] = temporal_loss
        
        # =====================================================================
        # COMPONENT 3: SPARSITY LOSS (L_sparsity)
        # =====================================================================
        sparsity_loss = torch.tensor(0.0, device=device)
        if 'reasoning_states' in output and len(output['reasoning_states']) > 0:
            for reasoning_state_batch in output['reasoning_states']:
                entropy_per_sample = -torch.sum(
                    reasoning_state_batch * torch.log(reasoning_state_batch + 1e-8),
                    dim=1
                )
                sparsity_loss = sparsity_loss - entropy_per_sample.mean()
        
        losses['sparsity'] = sparsity_loss
        
        # =====================================================================
        # COMPONENT 4: ADAPTIVE DEPTH REGULARIZATION (L_depth)
        # =====================================================================
        depth_loss = torch.tensor(0.0, device=device)
        if 'depth_used' in output:
            depths = torch.tensor(output['depth_used'], dtype=torch.float32, device=device)
            max_depth = self.max_reasoning_hops
            ideal_depth = max_depth * 0.5
            depth_loss = F.smooth_l1_loss(depths, torch.ones_like(depths) * ideal_depth)
        
        losses['adaptive_depth'] = depth_loss
        
        # =====================================================================
        # COMPONENT 5: MULTI-HOP PREDICTION CONSISTENCY
        # =====================================================================
        consistency_loss = torch.tensor(0.0, device=device)
        if 'all_answer_logits' in output and len(output['all_answer_logits']) > 1:
            logits_stack = torch.stack(output['all_answer_logits'])
            logits_mean = logits_stack.mean(dim=0, keepdim=True)
            consistency_loss = torch.mean((logits_stack - logits_mean) ** 2)
        
        losses['consistency'] = consistency_loss
        
        # =====================================================================
        # INFORMATION BOTTLENECK LOSSES (New Component)
        # Theory: Compress input while maintaining task performance
        # L_IB = I(X; Z) - β*I(Z; Y)
        # =====================================================================
        ib_loss = torch.tensor(0.0, device=device)
        
        if use_information_bottleneck and original_frames is not None:
            try:
                ib_losses = compute_complete_ib_loss(
                    output=output,
                    targets=targets,
                    original_frames=original_frames,
                    consolidated_memory=consolidated_memory if consolidated_memory is not None else output.get('consolidated_memory', torch.randn_like(original_frames)),
                    beta_memory=0.01,
                    beta_depth=0.01,
                    beta_multimodal=0.01,
                )
                
                # Add IB component losses
                for key, value in ib_losses.items():
                    if 'ib_' in key and torch.is_tensor(value):
                        losses[key] = value
                    elif key == 'total_ib' and torch.is_tensor(value):
                        ib_loss = value
                
                losses['total_ib'] = ib_loss
            except Exception as e:
                logger.warning(f"Error computing IB losses: {e}")
                losses['total_ib'] = torch.tensor(0.0, device=device)
        
        # =====================================================================
        # TOTAL LOSS: Weighted combination with IB
        # =====================================================================
        total_loss = (
            self.loss_weights_math['answer_loss_weight'] * losses['answer'] +
            self.loss_weights_math['temporal_grounding_weight'] * losses['temporal_grounding'] +
            self.loss_weights_math['sparsity_weight'] * losses['sparsity'] +
            self.loss_weights_math['adaptive_depth_weight'] * losses['adaptive_depth'] +
            0.1 * losses.get('total_ib', torch.tensor(0.0, device=device))  # IB weight
        )
        
        losses['total'] = total_loss
        
        return losses
