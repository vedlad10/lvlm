"""Advanced research modules for LVLM visual token processing.

These modules are optional enhancements that can be enabled/disabled via config:
- AdaptiveVisualTokenRouter: Learns to select variable number of important tokens
- TemporalFrameFusion: Fuses per-frame tokens with temporal positioning
- InstructionGuidedVisualAggregator: Query-conditioned token selection  
- MemoryAwareProjection: Carries context across image/video chunks
- DynamicVisualTokenGenerator: Alternative learnable sparsity mechanism
- VisualTextGroundingHead: Auxiliary loss for hallucination reduction
- UnsupportedClaimPenalty: Lightweight evidence scoring

Research use: Test whether redundant visual tokens can be pruned while maintaining
answer quality and grounding, and whether task-specific visual processing improves
performance across OCR, counting, scene understanding, and identity questions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RoutingStats:
    """Statistics from token routing operations."""
    kept_tokens: torch.Tensor  # [batch]: number of tokens kept per sample
    token_scores: torch.Tensor  # [batch, tokens]: importance score for each token


class AdaptiveVisualTokenRouter(nn.Module):
    """Selects a variable number of visual tokens based on learned importance.

    Input shape:
        visual_tokens: [batch, tokens, dim]
        query_embed: optional [batch, dim]

    Output shape:
        routed_tokens: [batch, max_kept, dim]
        mask: [batch, max_kept], True for real tokens, False for padding

    Research use:
        Tests whether redundant visual prefix tokens can be removed while preserving
        answer quality and grounding. This is the core module for efficient visual
        token handling.
    """

    def __init__(
        self,
        dim: int,
        min_tokens: int = 8,
        max_tokens: int = 32,
        temperature: float = 1.0,
        query_conditioned: bool = True,
    ):
        super().__init__()
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.query_conditioned = query_conditioned
        in_dim = dim * 2 if query_conditioned else dim
        self.scorer = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )
        self.count_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
        )

    def forward(
        self,
        visual_tokens: torch.Tensor,
        query_embed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, RoutingStats]:
        batch, token_count, dim = visual_tokens.shape
        pooled = visual_tokens.mean(dim=1)

        if self.query_conditioned and query_embed is not None:
            q = query_embed.unsqueeze(1).expand(-1, token_count, -1)
            score_input = torch.cat([visual_tokens, q], dim=-1)
            pooled_for_count = 0.5 * (pooled + query_embed)
        else:
            score_input = visual_tokens
            pooled_for_count = pooled

        scores = self.scorer(score_input).squeeze(-1) / self.temperature
        count_alpha = torch.sigmoid(self.count_head(pooled_for_count)).squeeze(-1)
        counts = self.min_tokens + torch.round(count_alpha * (self.max_tokens - self.min_tokens)).long()
        counts = counts.clamp(1, min(self.max_tokens, token_count))
        max_kept = int(counts.max().item())

        routed = visual_tokens.new_zeros(batch, max_kept, dim)
        mask = torch.zeros(batch, max_kept, dtype=torch.bool, device=visual_tokens.device)
        for row in range(batch):
            keep = int(counts[row].item())
            idx = torch.topk(scores[row], k=keep, dim=0).indices
            routed[row, :keep] = visual_tokens[row, idx]
            mask[row, :keep] = True

        return routed, mask, RoutingStats(kept_tokens=counts, token_scores=scores)


class TemporalFrameFusion(nn.Module):
    """Fuses per-frame visual tokens with temporal position encoding and attention.

    Expected input:
        frame_tokens: [batch, frames, tokens_per_frame, dim]

    Output:
        fused_tokens: [batch, output_tokens, dim]

    Research use:
        Adds video/event continuity to the existing image pipeline with a lightweight
        temporal module instead of full video-LLM retraining.
    """

    def __init__(self, dim: int, num_heads: int = 8, depth: int = 2, output_tokens: int = 32, max_frames: int = 32):
        super().__init__()
        self.output_tokens = output_tokens
        self.temporal_pos = nn.Parameter(torch.zeros(1, max_frames, 1, dim))
        self.query_tokens = nn.Parameter(torch.randn(1, output_tokens, dim) / math.sqrt(dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, frame_tokens: torch.Tensor) -> torch.Tensor:
        batch, frames, tokens_per_frame, dim = frame_tokens.shape
        if frames > self.temporal_pos.size(1):
            raise ValueError(f"frames={frames} exceeds max_frames={self.temporal_pos.size(1)}")

        x = frame_tokens + self.temporal_pos[:, :frames]
        x = x.reshape(batch, frames * tokens_per_frame, dim)
        x = self.temporal_encoder(x)
        queries = self.query_tokens.expand(batch, -1, -1)
        fused, _ = self.cross_attn(queries, x, x, need_weights=False)
        return self.norm(fused)


class InstructionGuidedVisualAggregator(nn.Module):
    """Lets the text instruction decide which visual tokens matter.

    Expected input:
        visual_tokens: [batch, tokens, dim]
        instruction_embeds: [batch, text_tokens, dim]

    Output:
        aggregated_tokens: [batch, output_tokens, dim]

    Research use:
        Tests query-conditioned vision processing: OCR, counting, scene captioning,
        and identity questions should attend to different evidence.
    """

    def __init__(self, dim: int, output_tokens: int = 32, num_heads: int = 8):
        super().__init__()
        self.query_pool = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.Tanh(),
        )
        self.learned_queries = nn.Parameter(torch.randn(1, output_tokens, dim) / math.sqrt(dim))
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.gate = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(dim)

    def forward(self, visual_tokens: torch.Tensor, instruction_embeds: torch.Tensor) -> torch.Tensor:
        batch = visual_tokens.size(0)
        instruction_summary = self.query_pool(instruction_embeds.mean(dim=1))
        queries = self.learned_queries.expand(batch, -1, -1) + instruction_summary.unsqueeze(1)
        attended, _ = self.cross_attn(queries, visual_tokens, visual_tokens, need_weights=False)
        gate = self.gate(torch.cat([queries, attended], dim=-1))
        return self.norm(gate * attended + (1.0 - gate) * queries)


class MemoryAwareProjection(nn.Module):
    """Adds a small recurrent memory over images or video segments.

    Research use:
        Useful for multi-image or long-video settings where the model must carry
        context across chunks without feeding every frame token to the LLM.
    """

    def __init__(self, dim: int, memory_slots: int = 8, num_heads: int = 8):
        super().__init__()
        self.memory_slots = memory_slots
        self.initial_memory = nn.Parameter(torch.randn(1, memory_slots, dim) / math.sqrt(dim))
        self.update_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.write_gate = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(dim)

    def init_memory(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.initial_memory.expand(batch_size, -1, -1).to(device)

    def forward(self, visual_tokens: torch.Tensor, memory: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, _ = visual_tokens.shape
        if memory is None:
            memory = self.init_memory(batch, visual_tokens.device)

        update, _ = self.update_attn(memory, visual_tokens, visual_tokens, need_weights=False)
        gate = self.write_gate(torch.cat([memory, update], dim=-1))
        new_memory = self.norm(gate * update + (1.0 - gate) * memory)
        augmented_tokens = torch.cat([new_memory, visual_tokens], dim=1)
        return augmented_tokens, new_memory


class DynamicVisualTokenGenerator(nn.Module):
    """Generates visual prefix tokens with a learned complexity-dependent gate.

    This is an alternative to always producing exactly 32 active tokens. It keeps a
    fixed tensor shape for batching, but returns a token mask and sparsity loss so
    experiments can evaluate effective token count.
    """

    def __init__(self, clip_dim: int, llm_dim: int, max_tokens: int = 32, min_tokens: int = 8):
        super().__init__()
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.token_mlp = nn.Sequential(
            nn.Linear(clip_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, max_tokens * llm_dim),
        )
        self.complexity = nn.Sequential(
            nn.LayerNorm(clip_dim),
            nn.Linear(clip_dim, llm_dim // 4),
            nn.GELU(),
            nn.Linear(llm_dim // 4, max_tokens),
        )
        self.llm_dim = llm_dim

    def forward(self, clip_embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = clip_embeds.size(0)
        tokens = self.token_mlp(clip_embeds).view(batch, self.max_tokens, self.llm_dim)
        gates = torch.sigmoid(self.complexity(clip_embeds))

        hard_budget = self.min_tokens + torch.round(
            gates.mean(dim=-1) * (self.max_tokens - self.min_tokens)
        ).long()
        positions = torch.arange(self.max_tokens, device=clip_embeds.device).unsqueeze(0)
        mask = positions < hard_budget.unsqueeze(1)
        tokens = tokens * mask.unsqueeze(-1).to(tokens.dtype)
        sparsity_loss = gates.mean()
        return tokens, mask, sparsity_loss


class VisualTextGroundingHead(nn.Module):
    """Contrastive visual-text consistency head for hallucination reduction.

    Use this as an auxiliary loss:
        loss = lm_loss + lambda_ground * grounding_loss

    It pulls answer embeddings closer to visual tokens from the same sample and
    pushes them away from other samples in the batch.
    """

    def __init__(self, dim: int, projection_dim: int = 512, temperature: float = 0.07):
        super().__init__()
        self.visual_proj = nn.Linear(dim, projection_dim)
        self.text_proj = nn.Linear(dim, projection_dim)
        self.temperature = temperature

    def forward(self, visual_tokens: torch.Tensor, answer_hidden: torch.Tensor) -> torch.Tensor:
        visual = F.normalize(self.visual_proj(visual_tokens.mean(dim=1)), dim=-1)
        text = F.normalize(self.text_proj(answer_hidden.mean(dim=1)), dim=-1)
        logits = visual @ text.t() / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


class UnsupportedClaimPenalty(nn.Module):
    """Lightweight hallucination penalty using answer-token evidence scores.

    This module does not know factual truth by itself. It supplies a trainable
    scalar evidence score from visual and generated-token states. During training,
    pair it with negative examples whose answers contain unsupported entities.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.evidence = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

    def forward(self, visual_tokens: torch.Tensor, answer_hidden: torch.Tensor, supported_labels: torch.Tensor) -> torch.Tensor:
        visual_summary = visual_tokens.mean(dim=1).unsqueeze(1).expand(-1, answer_hidden.size(1), -1)
        scores = self.evidence(torch.cat([visual_summary, answer_hidden], dim=-1)).squeeze(-1)
        return F.binary_cross_entropy_with_logits(scores, supported_labels.float())
