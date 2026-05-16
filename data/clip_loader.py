"""CLIP (ViT-B-32) Dataset loader."""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)


class CLIPDataset(Dataset):
    """CLIP (ViT-B-32) Dataset for Vision-Language Learning.
    
    Uses pretrained CLIP ViT-B-32 vision encoder for feature extraction.
    
    Dataset characteristics:
    - Integrates image-text pairs from multiple sources
    - Pretrained CLIP ViT-B-32 (768-dimensional embeddings)
    - No video time dimension (single frame per example)
    - Optimized for image-text alignment tasks
    """
    
    def __init__(
        self,
        annotation_file: str,
        feature_cache_dir: Optional[str] = None,
        split: str = "train",
        embedding_dim: int = 768,
        preprocess_features: bool = True,
    ):
        """Initialize CLIP dataset.
        
        Args:
            annotation_file: Path to annotation JSON file
            feature_cache_dir: Directory containing cached CLIP features (optional)
            split: Dataset split ("train", "val", "test")
            embedding_dim: Dimension of CLIP embeddings (ViT-B-32 = 768)
            preprocess_features: Whether to preprocess features
        """
        self.annotation_file = Path(annotation_file)
        self.feature_cache_dir = Path(feature_cache_dir) if feature_cache_dir else None
        self.split = split
        self.embedding_dim = embedding_dim
        self.preprocess_features = preprocess_features
        
        # Load annotations
        self.data = []
        self._load_annotations()
        
        logger.info(
            f"Loaded CLIP dataset with {len(self.data)} samples "
            f"(split: {split})"
        )
    
    def _load_annotations(self):
        """Load annotations from JSON file."""
        if not self.annotation_file.exists():
            logger.warning(f"Annotation file not found: {self.annotation_file}")
            return
        
        try:
            with open(self.annotation_file, 'r') as f:
                raw_data = json.load(f)
            
            # Handle different annotation formats
            if isinstance(raw_data, dict):
                # If dict with 'train', 'val', 'test' splits
                if self.split in raw_data:
                    self.data = raw_data[self.split]
                else:
                    self.data = raw_data.get('data', raw_data.get('samples', []))
            else:
                # Assume it's a list of samples
                self.data = raw_data
            
            logger.info(f"Loaded {len(self.data)} annotations for {self.split} split")
        except Exception as e:
            logger.error(f"Error loading annotations: {e}")
            self.data = []
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.
        
        Returns:
            Dictionary containing:
                - image_features: (embedding_dim,) CLIP image embeddings
                - text_features: (embedding_dim,) CLIP text embeddings
                - image_id: Image identifier
                - text: Raw text description
                - labels: Task-specific labels (if available)
        """
        sample = self.data[idx]
        
        # Extract data
        image_id = sample.get('image_id', str(idx))
        text = sample.get('text', sample.get('caption', ''))
        
        # Get or generate CLIP features
        image_features = self._get_image_features(sample)
        text_features = self._get_text_features(sample, text)
        
        # Prepare output
        output = {
            'image_features': torch.tensor(image_features, dtype=torch.float32),
            'text_features': torch.tensor(text_features, dtype=torch.float32),
            'image_id': image_id,
            'text': text,
        }
        
        # Add labels if available
        if 'labels' in sample:
            output['labels'] = torch.tensor(sample['labels'], dtype=torch.long)
        if 'score' in sample:
            output['score'] = torch.tensor(sample['score'], dtype=torch.float32)
        
        return output
    
    def _get_image_features(self, sample: Dict) -> np.ndarray:
        """Get CLIP image features from sample or cache.
        
        Args:
            sample: Data sample dictionary
            
        Returns:
            Image features as numpy array of shape (embedding_dim,)
        """
        # If features are directly in sample
        if 'image_features' in sample:
            features = np.array(sample['image_features'], dtype=np.float32)
            if features.shape != (self.embedding_dim,):
                features = features.reshape(-1) if features.size == self.embedding_dim else \
                          np.zeros(self.embedding_dim, dtype=np.float32)
            return features
        
        # If features are in cache
        if self.feature_cache_dir and 'image_id' in sample:
            cache_path = self.feature_cache_dir / f"{sample['image_id']}_image.npy"
            if cache_path.exists():
                try:
                    return np.load(cache_path).astype(np.float32)
                except Exception as e:
                    logger.warning(f"Error loading cached features: {e}")
        
        # Return zeros if no features available
        return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def _get_text_features(self, sample: Dict, text: str) -> np.ndarray:
        """Get CLIP text features from sample or cache.
        
        Args:
            sample: Data sample dictionary
            text: Raw text string
            
        Returns:
            Text features as numpy array of shape (embedding_dim,)
        """
        # If features are directly in sample
        if 'text_features' in sample:
            features = np.array(sample['text_features'], dtype=np.float32)
            if features.shape != (self.embedding_dim,):
                features = features.reshape(-1) if features.size == self.embedding_dim else \
                          np.zeros(self.embedding_dim, dtype=np.float32)
            return features
        
        # If features are in cache
        if self.feature_cache_dir and 'image_id' in sample:
            cache_path = self.feature_cache_dir / f"{sample['image_id']}_text.npy"
            if cache_path.exists():
                try:
                    return np.load(cache_path).astype(np.float32)
                except Exception as e:
                    logger.warning(f"Error loading cached features: {e}")
        
        # Return zeros if no features available
        return np.zeros(self.embedding_dim, dtype=np.float32)


class CLIPDataLoader:
    """Wrapper for CLIP data loading with standard PyTorch DataLoader.
    
    Can be directly used as a PyTorch DataLoader.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False,
    ):
        """Initialize data loader.
        
        Args:
            dataset: CLIP dataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory for GPU transfer
            drop_last: Whether to drop last incomplete batch
        """
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
    
    def __iter__(self):
        """Iterate over batches."""
        return iter(self.dataloader)
    
    def __len__(self):
        """Return number of batches."""
        return len(self.dataloader)
