"""Qwen 2.5 Adapter for LVLM Integration.

Integrates Qwen 2.5 as the reasoning backbone with:
- 4-bit quantization for efficiency
- LoRA fine-tuning for parameter efficiency
- Temporal binding compatibility
- Multi-modal feature fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import logging
from peft import get_peft_model, LoraConfig, TaskType

logger = logging.getLogger(__name__)


class QwenAdapter(nn.Module):
    """Qwen 2.5 adapter for LVLM reasoning pipeline.
    
    Loads Qwen 2.5 with 4-bit quantization and LoRA fine-tuning,
    integrates temporal binding features, and handles multi-hop reasoning.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        use_4bit: bool = True,
        max_seq_length: int = 4096,
        device: str = "cuda",
    ):
        """Initialize Qwen adapter.
        
        Args:
            model_name: Qwen model identifier on HuggingFace
            lora_rank: LoRA rank for parameter-efficient fine-tuning
            lora_alpha: LoRA scaling factor
            lora_dropout: LoRA dropout
            use_4bit: Whether to use 4-bit quantization
            max_seq_length: Maximum sequence length
            device: Computation device
        """
        super().__init__()
        
        self.model_name = model_name
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.max_seq_length = max_seq_length
        self.device = device
        
        logger.info(f"Loading {model_name}...")
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 4-bit quantization config
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            logger.info("Using 4-bit quantization for efficiency")
        else:
            bnb_config = None
        
        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config if use_4bit else None,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Apply LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=["q_proj", "v_proj"],  # Qwen attention modules
        )
        
        self.model = get_peft_model(self.base_model, lora_config)
        self.model.print_trainable_parameters()
        
        # Feature projection layer for temporal binding compatibility
        hidden_size = self.base_model.config.hidden_size
        self.feature_projector = nn.Linear(768, hidden_size)  # Project from LVLM features
        self.temporal_fusion = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            batch_first=True,
            dropout=0.1,
        )
        
        # Classification head for QA
        self.qa_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 512),  # Answer vocab
        )
        
        # Temporal grounding head
        self.grounding_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 2),  # Start, end timestamps
        )
        
        logger.info(f"Qwen adapter initialized with LoRA rank={lora_rank}")
    
    def forward(
        self,
        question_ids: torch.Tensor,
        temporal_features: Optional[torch.Tensor] = None,
        memory_nodes: Optional[torch.Tensor] = None,
        return_grounding: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for Qwen reasoning.
        
        Args:
            question_ids: Tokenized question input [B, seq_len]
            temporal_features: Video features from temporal binding [B, num_nodes, 768]
            memory_nodes: Compressed memory nodes [B, num_nodes, 768]
            return_grounding: Whether to return temporal grounding predictions
            
        Returns:
            Dictionary with:
                - 'logits': Answer logits [B, vocab_size]
                - 'hidden_states': Last hidden state [B, seq_len, hidden_size]
                - 'grounding': Temporal predictions [B, 2] if return_grounding=True
        """
        batch_size = question_ids.shape[0]
        
        # Base Qwen forward pass
        outputs = self.base_model(
            input_ids=question_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        
        hidden_states = outputs.hidden_states[-1]  # [B, seq_len, hidden_size]
        
        # Fuse temporal features if provided
        if temporal_features is not None and memory_nodes is not None:
            # Project temporal features to Qwen hidden size
            projected_features = self.feature_projector(temporal_features)  # [B, nodes, hidden_size]
            
            # Cross-attention fusion
            question_repr = hidden_states[:, -1:, :]  # [B, 1, hidden_size] - use last token
            fused_features, _ = self.temporal_fusion(
                question_repr,
                projected_features,
                projected_features,
            )
            
            # Blend with original hidden states
            hidden_states = hidden_states + 0.3 * fused_features
        
        # Generate predictions
        last_token_hidden = hidden_states[:, -1, :]  # [B, hidden_size]
        
        # QA head
        logits = self.qa_head(last_token_hidden)  # [B, vocab_size]
        
        # Temporal grounding head
        grounding_output = None
        if return_grounding:
            grounding_output = torch.sigmoid(
                self.grounding_head(last_token_hidden)
            )  # [B, 2] normalized to [0, 1]
        
        return {
            "logits": logits,
            "hidden_states": hidden_states,
            "grounding": grounding_output,
            "attention_weights": outputs.attentions if hasattr(outputs, "attentions") else None,
        }
    
    def generate_answer(
        self,
        question_text: str,
        context: Optional[str] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
    ) -> str:
        """Generate natural language answer using Qwen.
        
        Args:
            question_text: Input question
            context: Optional context (video description, etc.)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated answer string
        """
        prompt = question_text
        if context:
            prompt = f"Context: {context}\nQuestion: {question_text}"
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.base_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
            )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
    
    def enable_training(self):
        """Enable training mode for LoRA parameters."""
        self.model.train()
        for param in self.model.parameters():
            if "lora" in param.__class__.__name__.lower():
                param.requires_grad = True
    
    def save_lora_weights(self, save_path: str):
        """Save only LoRA weights (parameter-efficient).
        
        Args:
            save_path: Path to save LoRA weights
        """
        self.model.save_pretrained(save_path)
        logger.info(f"LoRA weights saved to {save_path}")
    
    def load_lora_weights(self, load_path: str):
        """Load LoRA weights.
        
        Args:
            load_path: Path to load LoRA weights from
        """
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.model, load_path)
        logger.info(f"LoRA weights loaded from {load_path}")


class QwenLVLMFusion(nn.Module):
    """End-to-end fusion of Qwen with LVLM temporal binding.
    
    Combines:
    - LVLM temporal binding for video compression
    - Qwen 2.5 for semantic reasoning
    - Adaptive depth controller for efficiency
    """
    
    def __init__(
        self,
        lvlm_model: nn.Module,
        qwen_adapter: QwenAdapter,
        fusion_type: str = "cross_attention",
    ):
        """Initialize Qwen-LVLM fusion.
        
        Args:
            lvlm_model: LVLM instance
            qwen_adapter: Qwen adapter instance
            fusion_type: How to fuse features ("cross_attention", "concatenation", "gating")
        """
        super().__init__()
        
        self.lvlm = lvlm_model
        self.qwen = qwen_adapter
        self.fusion_type = fusion_type
        
        # Fusion layer
        hidden_size = qwen_adapter.base_model.config.hidden_size
        if fusion_type == "gating":
            self.fusion_gate = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Sigmoid(),
            )
        
        logger.info(f"Qwen-LVLM fusion initialized with {fusion_type}")
    
    def forward(
        self,
        video_frames: torch.Tensor,
        question_ids: torch.Tensor,
        adaptive_depth: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for complete pipeline.
        
        Args:
            video_frames: Video frame embeddings [B, T, 768]
            question_ids: Tokenized questions [B, seq_len]
            adaptive_depth: Optional fixed depth (else learned)
            
        Returns:
            Dictionary with predictions and intermediate outputs
        """
        # LVLM forward: temporal binding
        lvlm_output = self.lvlm(
            video_frames=video_frames,
            questions=question_ids,
        )
        
        temporal_features = lvlm_output.get("temporal_features")
        memory_nodes = lvlm_output.get("memory_nodes")
        
        # Qwen reasoning with temporal fusion
        qwen_output = self.qwen(
            question_ids=question_ids,
            temporal_features=temporal_features,
            memory_nodes=memory_nodes,
        )
        
        return {
            "lvlm_output": lvlm_output,
            "qwen_output": qwen_output,
            "answer_logits": qwen_output["logits"],
            "grounding": qwen_output["grounding"],
        }
