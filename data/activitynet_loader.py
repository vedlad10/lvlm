"""ActivityNet-QA Dataset loader."""

import json
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)


class ActivityNetQADataset(Dataset):
    """ActivityNet-QA Dataset for Video Question Answering.
    
    Dataset: https://github.com/lixiangpeng/AGQA
    
    A large-scale video question answering dataset.
    - 58K QA pairs
    - 5.8K videos
    - Average video length: ~2 minutes
    - Longer and more complex videos than TVQA
    """
    
    def __init__(
        self,
        annotation_file: str,
        feature_file: Optional[str] = None,
        split: str = "train",
        frame_rate: float = 2.0,
        max_frames: Optional[int] = None,
        vocab_file: Optional[str] = None,
        preprocess_features: bool = True,
    ):
        """Initialize ActivityNet-QA dataset.
        
        Args:
            annotation_file: Path to ActivityNet-QA JSON annotation file
            feature_file: Path to HDF5 cached features (optional)
            split: Dataset split ("train", "val", "test")
            frame_rate: Frame extraction rate (fps)
            max_frames: Maximum frames per video (None = unlimited)
            vocab_file: Path to vocabulary file (optional)
            preprocess_features: Whether to preprocess features
        """
        self.annotation_file = Path(annotation_file)
        self.feature_file = Path(feature_file) if feature_file else None
        self.split = split
        self.frame_rate = frame_rate
        self.max_frames = max_frames
        self.preprocess_features = preprocess_features
        
        # Load annotations
        if not self.annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
        
        logger.info(f"Loading {split} split from {annotation_file}")
        with open(self.annotation_file, 'r') as f:
            annotations = json.load(f)
        
        # Handle both list and dict formats
        if isinstance(annotations, dict):
            self.data = list(annotations.values())
        else:
            self.data = annotations
        
        logger.info(f"Loaded {len(self.data)} QA pairs for {split} split")
        
        # Load features if available
        self.features = None
        if self.feature_file and self.feature_file.exists():
            logger.info(f"Loading features from {self.feature_file}")
            self.features_file = h5py.File(self.feature_file, 'r')
        else:
            self.features_file = None
        
        # Build vocabulary
        self.vocab = self._build_vocab()
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with:
                - video_id: Video identifier
                - question: Question text
                - frames: Frame features [num_frames, feat_dim]
                - frame_timestamps: Timestamps for each frame
                - answer: Ground-truth answer (int or text)
                - temporal_spans: List of answer-relevant temporal spans
        """
        sample = self.data[idx]
        
        video_id = sample.get('video_id', sample.get('vid', ''))
        question_id = sample.get('qid', idx)
        question = sample.get('question', sample.get('q', ''))
        
        # Handle different answer formats
        if isinstance(sample.get('answer'), int):
            answer_idx = sample['answer']
            answer_text = sample.get('answer_text', '')
        else:
            answer_idx = 0  # Placeholder
            answer_text = sample.get('answer', '')
        
        # Get temporal information
        temporal_span = sample.get('timestamp', [0.0, 1.0])
        if isinstance(temporal_span, list) and len(temporal_span) == 2:
            temporal_span = temporal_span
        else:
            temporal_span = [0.0, 1.0]
        
        # Load frame features
        if self.features_file is not None:
            frames, frame_timestamps = self._load_features(video_id)
        else:
            # Placeholder: would load from video file
            frames = np.zeros((1, 768), dtype=np.float32)
            frame_timestamps = np.array([0.0])
        
        # Apply max frames constraint
        if self.max_frames is not None and len(frames) > self.max_frames:
            indices = np.linspace(0, len(frames)-1, self.max_frames, dtype=int)
            frames = frames[indices]
            frame_timestamps = frame_timestamps[indices]
        
        # Preprocess features
        if self.preprocess_features:
            frames = self._preprocess_features(frames)
        
        return {
            'video_id': video_id,
            'question_id': question_id,
            'question': question,
            'question_tokens': self._tokenize(question),
            'frames': torch.from_numpy(frames).float(),
            'frame_timestamps': torch.from_numpy(frame_timestamps).float(),
            'answer': answer_idx,
            'answer_text': answer_text,
            'temporal_span': torch.tensor(temporal_span, dtype=torch.float32),
        }
    
    def _load_features(self, video_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load cached features for video.
        
        Args:
            video_id: Video identifier
            
        Returns:
            Tuple of (frames, timestamps)
        """
        try:
            group = self.features_file[video_id]
            frames = group['features'][:]
            timestamps = group.get('timestamps', np.arange(len(frames)) / self.frame_rate)[:]
            return frames, timestamps
        except KeyError:
            logger.warning(f"Features not found for video {video_id}")
            # Return dummy features
            return np.zeros((1, 768), dtype=np.float32), np.array([0.0])
    
    def _preprocess_features(self, frames: np.ndarray) -> np.ndarray:
        """Preprocess frame features.
        
        Args:
            frames: Raw frame features [T, D]
            
        Returns:
            Preprocessed features
        """
        # L2 normalization
        frames = frames / (np.linalg.norm(frames, axis=1, keepdims=True) + 1e-10)
        return frames.astype(np.float32)
    
    def _tokenize(self, text: str) -> List[int]:
        """Simple tokenization (placeholder).
        
        Args:
            text: Text to tokenize
            
        Returns:
            Token indices
        """
        # TODO: Implement proper tokenization
        words = text.lower().split()
        return [self.vocab.get(w, 0) for w in words[:20]]  # Max 20 tokens
    
    def _build_vocab(self) -> Dict[str, int]:
        """Build vocabulary from questions.
        
        Returns:
            Vocabulary dictionary (word -> index)
        """
        # TODO: Build vocabulary
        return {
            '<unk>': 0,
            '<pad>': 1,
        }


class ActivityNetQADataLoader(DataLoader):
    """Custom DataLoader for ActivityNet-QA with collation."""
    
    def __init__(
        self,
        dataset: ActivityNetQADataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        """Initialize ActivityNet-QA DataLoader.
        
        Args:
            dataset: ActivityNet-QA dataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of workers
            pin_memory: Whether to pin memory
        """
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self._collate_fn,
        )
    
    @staticmethod
    def _collate_fn(batch: List[Dict]) -> Dict:
        """Custom collate function for ActivityNet-QA batch.
        
        Args:
            batch: List of samples from dataset
            
        Returns:
            Batched data dictionary
        """
        # Handle variable-length sequences
        max_frames = max(sample['frames'].shape[0] for sample in batch)
        batch_size = len(batch)
        feat_dim = batch[0]['frames'].shape[1]
        
        # Pad frames
        frames_padded = torch.zeros(batch_size, max_frames, feat_dim, dtype=torch.float32)
        frame_lengths = []
        
        for i, sample in enumerate(batch):
            num_frames = sample['frames'].shape[0]
            frames_padded[i, :num_frames] = sample['frames']
            frame_lengths.append(num_frames)
        
        # Batch answers and metadata
        answers = torch.tensor([sample['answer'] for sample in batch], dtype=torch.long)
        temporal_spans = torch.stack([sample['temporal_span'] for sample in batch])
        
        return {
            'video_ids': [sample['video_id'] for sample in batch],
            'question_ids': [sample['question_id'] for sample in batch],
            'questions': [sample['question'] for sample in batch],
            'frames': frames_padded,
            'frame_lengths': torch.tensor(frame_lengths, dtype=torch.long),
            'answers': answers,
            'answer_texts': [sample['answer_text'] for sample in batch],
            'temporal_spans': temporal_spans,
        }
