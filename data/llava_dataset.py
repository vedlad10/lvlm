"""
LLaVA-style Dataset for instruction-tuning image-to-text tasks.

Loads images + QA pairs and produces sequences:
    [visual_tokens] + [question] + [answer]
    
Loss computed only on answer tokens using -100 masking.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import List, Dict, Optional, Callable
import os
import logging

logger = logging.getLogger(__name__)


class LLaVADataset(Dataset):
    """
    Instruction-tuning dataset in LLaVA format.
    
    Expected JSON structure:
        [
            {
                "image": "image_filename.jpg",
                "conversations": [
                    {"from": "human", "value": "What is in the image?"},
                    {"from": "gpt", "value": "The image shows..."}
                ]
            },
            ...
        ]
    
    Produces sequences with only answer tokens contributing to loss.
    """

    QUESTION_TEMPLATE = "Question: {question}\nAnswer:"

    def __init__(
        self,
        samples: List[Dict],
        image_dir: str,
        tokenizer,
        clip_preprocess: Callable,
        num_visual_tokens: int = 32,
        max_length: int = 512,
    ):
        """
        Args:
            samples: List of sample dicts with 'image' and 'conversations'
            image_dir: Directory containing images
            tokenizer: HuggingFace tokenizer
            clip_preprocess: CLIP preprocessing transform
            num_visual_tokens: Number of visual tokens (for label masking)
            max_length: Max sequence length
        """
        self.samples = samples
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.clip_preprocess = clip_preprocess
        self.num_visual_tokens = num_visual_tokens
        self.max_length = max_length
        
        logger.info(
            f"Initialized LLaVADataset with {len(samples)} samples, "
            f"max_length={max_length}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dict with 'pixel_values', 'input_ids', 'labels'
        """
        item = self.samples[idx]
        
        # ── Extract Q&A from conversations ────────────────────────
        conversations = item.get("conversations", [])
        question = next(
            (c["value"] for c in conversations if c["from"] == "human"),
            ""
        ).strip()
        answer = next(
            (c["value"] for c in conversations if c["from"] == "gpt"),
            ""
        ).strip()
        
        if not question or not answer:
            raise ValueError(
                f"Sample {idx} missing question or answer: {item}"
            )
        
        # ── Load and preprocess image ────────────────────────────
        img_path = os.path.join(self.image_dir, item["image"])
        try:
            image = Image.open(img_path).convert("RGB")
            pixel_values = self.clip_preprocess(image)  # (3, H, W)
        except Exception as e:
            raise IOError(f"Failed to load image {img_path}: {e}")
        
        # ── Tokenize prompt (question only) ──────────────────────
        prompt_text = self.QUESTION_TEMPLATE.format(question=question)
        
        # ── Tokenize full sequence (prompt + answer) ─────────────
        eos_token = self.tokenizer.eos_token or "</s>"
        full_text = prompt_text + " " + answer + eos_token
        
        # Tokenize with padding
        encoded = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)  # (max_length,)
        
        # ── Create labels (mask prompt, keep answer) ─────────────
        # First, tokenize ONLY the prompt to find its length
        prompt_encoded = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_len = prompt_encoded["input_ids"].shape[-1]
        
        # Labels: -100 for prompt portion, actual IDs for answer
        labels = input_ids.clone()
        labels[:prompt_len] = -100  # Mask prompt tokens in labels
        
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "labels": labels,
        }


def build_llava_dataloader(
    samples: List[Dict],
    image_dir: str,
    tokenizer,
    clip_preprocess: Callable,
    batch_size: int = 2,
    num_workers: int = 0,
    shuffle: bool = True,
    num_visual_tokens: int = 32,
) -> DataLoader:
    """
    Create a DataLoader for LLaVA training.
    
    Args:
        samples: List of sample dicts
        image_dir: Directory with images
        tokenizer: HuggingFace tokenizer
        clip_preprocess: CLIP preprocessing
        batch_size: Batch size
        num_workers: Num workers (0 for Colab)
        shuffle: Whether to shuffle
        num_visual_tokens: For label masking
    
    Returns:
        DataLoader with custom collate_fn
    """
    dataset = LLaVADataset(
        samples=samples,
        image_dir=image_dir,
        tokenizer=tokenizer,
        clip_preprocess=clip_preprocess,
        num_visual_tokens=num_visual_tokens,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: llava_collate_fn(batch, num_visual_tokens),
    )


def llava_collate_fn(batch: List[Dict], num_visual_tokens: int) -> Dict:
    """
    Collate function that stacks batch and prepends visual token masks to labels.
    
    Ensures all labels have shape (B, num_visual_tokens + L_text) with first
    num_visual_tokens positions set to -100.
    """
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    input_ids = torch.stack([b["input_ids"] for b in batch])
    text_labels = torch.stack([b["labels"] for b in batch])
    
    # Prepend visual token mask (-100) to labels
    batch_size = pixel_values.size(0)
    visual_mask = torch.full(
        (batch_size, num_visual_tokens),
        -100,
        dtype=torch.long,
    )
    labels = torch.cat([visual_mask, text_labels], dim=1)
    
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "labels": labels,
    }
