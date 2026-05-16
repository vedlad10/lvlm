"""
LLaVA-style Large Vision-Language Model (LVLM)

Integrates:
- Frozen CLIP ViT-B/32 visual encoder
- Trainable MLP projection layer
- LoRA-finetuned LLaMA-2-7B language model
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from typing import Optional, Tuple
import logging

from .llava_clip_encoder import FrozenCLIPEncoder
from .llava_projection import VisualProjection

logger = logging.getLogger(__name__)


class LVLM(nn.Module):
    """
    Large Vision-Language Model combining CLIP, projection, and LoRA-LLaMA.
    
    Forward pass expects:
        - pixel_values (B, 3, H, W): CLIP-preprocessed images
        - input_ids (B, L_text): Tokenized text prompts
        - labels (B, L_total): Target token IDs with -100 masking for non-loss tokens
    
    Layout: [visual_tokens] + [text_tokens] concatenated and passed to LLaMA.
    Loss computed only on answer tokens; visual + prompt tokens masked with -100.
    """

    def __init__(
        self,
        llama_model: nn.Module,
        tokenizer: AutoTokenizer,
        clip_model_name: str = "ViT-B/32",
        num_visual_tokens: int = 32,
        llm_dim: int = 4096,
        device: str = "cuda",
    ):
        """
        Args:
            llama_model: LoRA-wrapped LLaMA model
            tokenizer: LLaMA tokenizer
            clip_model_name: CLIP variant to use
            num_visual_tokens: Number of visual tokens per image
            llm_dim: LLaMA hidden dimension
            device: Compute device
        """
        super().__init__()
        
        self.device = device
        self.num_visual_tokens = num_visual_tokens
        self.llm_dim = llm_dim
        
        # Component 1: Frozen CLIP encoder
        self.clip = FrozenCLIPEncoder(model_name=clip_model_name, device=device)
        
        # Component 2: Trainable projection layer
        self.projection = VisualProjection(
            clip_dim=self.clip.embed_dim,
            llm_dim=llm_dim,
            num_tokens=num_visual_tokens,
        )
        
        # Component 3: LoRA-patched LLaMA
        self.llama = llama_model
        self.tokenizer = tokenizer
        
        logger.info(
            f"✓ LVLM initialized: "
            f"CLIP({clip_model_name}→{self.clip.embed_dim}) → "
            f"Projection({num_visual_tokens}×{llm_dim}) → "
            f"LoRA-LLaMA"
        )

    def _get_model_dtype(self) -> torch.dtype:
        """Infer model dtype from LLaMA parameters."""
        return next(self.llama.parameters()).dtype

    def _get_llama_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Extract token embeddings from LLaMA's embedding table."""
        return self.llama.get_input_embeddings()(input_ids)

    def _extract_visual_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract and project visual tokens from images.
        
        CLIP always receives float32 input; output cast to match LLM dtype.
        """
        with torch.no_grad():
            # CLIP expects float32, produces (B, clip_dim)
            clip_embeds = self.clip(pixel_values)
        
        # Cast to LLaMA dtype (typically float16) before projection
        model_dtype = self._get_model_dtype()
        clip_embeds = clip_embeds.to(dtype=model_dtype)
        
        # Project to (B, num_visual_tokens, llm_dim)
        return self.projection(clip_embeds)

    def _build_inputs(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Concatenate visual and text embeddings.
        
        Returns:
            inputs_embeds (B, num_visual_tokens + L_text, llm_dim)
            attention_mask (B, num_visual_tokens + L_text)
        """
        batch_size = pixel_values.size(0)
        
        # Extract visual tokens
        visual_tokens = self._extract_visual_tokens(pixel_values)  # (B, V, llm_dim)
        
        # Extract text embeddings
        text_embeds = self._get_llama_embeddings(input_ids)  # (B, L, llm_dim)
        
        # Concatenate: visual tokens first, then text
        inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)
        
        # Attention mask: 1 for all visual tokens, then text mask
        vis_mask = torch.ones(
            batch_size, self.num_visual_tokens,
            device=input_ids.device, dtype=torch.long
        )
        txt_mask = (input_ids != self.tokenizer.pad_token_id).long()
        attention_mask = torch.cat([vis_mask, txt_mask], dim=1)
        
        return inputs_embeds, attention_mask

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for training/inference.
        
        Args:
            pixel_values (B, 3, H, W): CLIP-preprocessed images
            input_ids (B, L_text): Tokenized prompts
            labels (B, V+L_text): Target token IDs
                - Set to -100 for visual tokens and prompt tokens
                - Use actual token ID for answer portion
        
        Returns:
            CausalLMOutputWithPast: Contains loss when labels provided
        """
        inputs_embeds, attention_mask = self._build_inputs(pixel_values, input_ids)
        
        return self.llama(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate text responses given images and prompts.
        
        Args:
            pixel_values (B, 3, H, W): Images
            input_ids (B, L): Prompts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional generation parameters
            
        Returns:
            Generated token IDs
        """
        inputs_embeds, attention_mask = self._build_inputs(pixel_values, input_ids)
        
        return self.llama.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )


def build_lora_llama(
    model_name_or_path: str = "meta-llama/Llama-2-7b-hf",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[list] = None,
    torch_dtype: torch.dtype = torch.float16,
) -> Tuple[nn.Module, AutoTokenizer]:
    """
    Load LLaMA-2 and wrap with LoRA adapters.
    
    Only LoRA matrices are trainable; base weights frozen via PEFT.
    
    Args:
        model_name_or_path: HuggingFace model path
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling
        lora_dropout: LoRA dropout (acts on input)
        target_modules: Projection matrices to adapt
        torch_dtype: Model dtype (float16 for mixed precision)
    
    Returns:
        Tuple of (model, tokenizer)
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model with specified dtype
    logger.info(f"Loading {model_name_or_path} in {torch_dtype}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    
    # Configure LoRA
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    
    # Apply LoRA
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    
    logger.info(f"✓ LoRA-LLaMA initialized with r={lora_r}, alpha={lora_alpha}")
    return model, tokenizer
