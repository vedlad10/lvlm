"""Metrics utilities for LVLM evaluation."""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class Metrics:
    """Container for storing and computing various metrics."""
    
    accuracies: List[float] = field(default_factory=list)
    temporal_ious: List[float] = field(default_factory=list)
    reasoning_depths: List[int] = field(default_factory=list)
    inference_times: List[float] = field(default_factory=list)
    losses: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    
    def add_accuracy(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Add accuracy metric.
        
        Args:
            predictions: Predicted answers (batch_size,)
            targets: Ground-truth answers (batch_size,)
            
        Returns:
            Batch accuracy
        """
        batch_acc = np.mean(predictions == targets)
        self.accuracies.append(batch_acc)
        return batch_acc
    
    def add_temporal_iou(self, pred_spans: List[Tuple[float, float]], 
                        gt_spans: List[Tuple[float, float]]) -> float:
        """Add temporal IoU metric.
        
        Args:
            pred_spans: Predicted [start, end] timestamps
            gt_spans: Ground-truth [start, end] timestamps
            
        Returns:
            Batch average IoU
        """
        ious = []
        for pred, gt in zip(pred_spans, gt_spans):
            iou = compute_temporal_iou(pred, gt)
            ious.append(iou)
        
        batch_iou = np.mean(ious) if ious else 0.0
        self.temporal_ious.append(batch_iou)
        return batch_iou
    
    def add_reasoning_depth(self, depths: List[int]) -> float:
        """Add reasoning depth metric (hops used).
        
        Args:
            depths: Number of hops used per sample
            
        Returns:
            Batch average depth
        """
        self.reasoning_depths.extend(depths)
        return np.mean(depths) if depths else 0.0
    
    def add_inference_time(self, times: List[float]) -> float:
        """Add inference time metric (ms per sample).
        
        Args:
            times: Inference times per sample in milliseconds
            
        Returns:
            Batch average time
        """
        self.inference_times.extend(times)
        return np.mean(times) if times else 0.0
    
    def add_loss(self, loss_dict: Dict[str, float]) -> None:
        """Add loss values.
        
        Args:
            loss_dict: Dictionary of loss names and values
        """
        for loss_name, loss_value in loss_dict.items():
            self.losses[loss_name].append(loss_value)
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics for all metrics.
        
        Returns:
            Dictionary of metric summaries
        """
        summary = {}
        
        if self.accuracies:
            summary["accuracy_mean"] = np.mean(self.accuracies)
            summary["accuracy_std"] = np.std(self.accuracies)
        
        if self.temporal_ious:
            summary["temporal_iou_mean"] = np.mean(self.temporal_ious)
            summary["temporal_iou_std"] = np.std(self.temporal_ious)
        
        if self.reasoning_depths:
            summary["reasoning_depth_mean"] = np.mean(self.reasoning_depths)
            summary["reasoning_depth_std"] = np.std(self.reasoning_depths)
            summary["reasoning_depth_max"] = np.max(self.reasoning_depths)
            summary["reasoning_depth_min"] = np.min(self.reasoning_depths)
        
        if self.inference_times:
            summary["inference_time_mean"] = np.mean(self.inference_times)
            summary["inference_time_std"] = np.std(self.inference_times)
        
        for loss_name, loss_values in self.losses.items():
            if loss_values:
                summary[f"loss_{loss_name}_mean"] = np.mean(loss_values)
                summary[f"loss_{loss_name}_std"] = np.std(loss_values)
        
        return summary
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.accuracies.clear()
        self.temporal_ious.clear()
        self.reasoning_depths.clear()
        self.inference_times.clear()
        self.losses.clear()
    
    def __repr__(self) -> str:
        summary = self.get_summary()
        lines = ["Metrics Summary:"]
        for key, value in summary.items():
            lines.append(f"  {key}: {value:.4f}")
        return "\n".join(lines)


def compute_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute accuracy.
    
    Args:
        predictions: Predicted labels
        targets: Ground-truth labels
        
    Returns:
        Accuracy score [0, 1]
    """
    return np.mean(predictions == targets)


def compute_temporal_iou(pred_span: Tuple[float, float], 
                        gt_span: Tuple[float, float]) -> float:
    """Compute Intersection over Union (IoU) for temporal spans.
    
    Args:
        pred_span: Predicted [start, end] timestamps
        gt_span: Ground-truth [start, end] timestamps
        
    Returns:
        IoU score [0, 1]
    """
    pred_start, pred_end = pred_span
    gt_start, gt_end = gt_span
    
    # Ensure valid spans
    pred_start, pred_end = min(pred_start, pred_end), max(pred_start, pred_end)
    gt_start, gt_end = min(gt_start, gt_end), max(gt_start, gt_end)
    
    # Compute intersection
    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)
    intersection = max(0, inter_end - inter_start)
    
    # Compute union
    union = (pred_end - pred_start) + (gt_end - gt_start) - intersection
    
    # Handle edge case
    if union == 0:
        return 0.0
    
    iou = intersection / union
    return float(iou)


def compute_f1(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Compute precision, recall, and F1 score.
    
    Args:
        predictions: Predicted binary labels
        targets: Ground-truth binary labels
        
    Returns:
        Dictionary with precision, recall, f1
    """
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    precision = precision_score(targets, predictions, zero_division=0)
    recall = recall_score(targets, predictions, zero_division=0)
    f1 = f1_score(targets, predictions, zero_division=0)
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }


def compute_entropy(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute entropy of probability distribution.
    
    Args:
        logits: Model logits
        dim: Dimension to compute entropy over
        
    Returns:
        Entropy values
    """
    probs = torch.softmax(logits, dim=dim)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=dim)
    return entropy


def compute_confidence(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute maximum probability (confidence).
    
    Args:
        logits: Model logits
        dim: Dimension to compute over
        
    Returns:
        Confidence values (max probability)
    """
    probs = torch.softmax(logits, dim=dim)
    confidence, _ = torch.max(probs, dim=dim)
    return confidence


def compute_speedup(time_adaptive: float, time_baseline: float) -> float:
    """Compute speedup factor.
    
    Args:
        time_adaptive: Time with adaptive depth
        time_baseline: Time with fixed depth (baseline)
        
    Returns:
        Speedup factor (baseline / adaptive)
    """
    if time_adaptive == 0:
        return 0.0
    return time_baseline / time_adaptive


def compute_accuracy_drop(acc_adaptive: float, acc_baseline: float) -> float:
    """Compute accuracy drop percentage.
    
    Args:
        acc_adaptive: Accuracy with adaptive depth
        acc_baseline: Accuracy with fixed depth (baseline)
        
    Returns:
        Accuracy drop as percentage points
    """
    return (acc_baseline - acc_adaptive) * 100


class MetricsTracker:
    """Context manager for tracking metrics during training/evaluation."""
    
    def __init__(self):
        self.metrics = Metrics()
        self.batch_count = 0
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def add_batch(self, predictions: np.ndarray, targets: np.ndarray,
                 pred_spans: Optional[List[Tuple[float, float]]] = None,
                 gt_spans: Optional[List[Tuple[float, float]]] = None,
                 depths: Optional[List[int]] = None,
                 times: Optional[List[float]] = None) -> None:
        """Add batch metrics.
        
        Args:
            predictions: Batch predictions
            targets: Batch targets
            pred_spans: Predicted temporal spans (optional)
            gt_spans: Ground-truth temporal spans (optional)
            depths: Reasoning depths (optional)
            times: Inference times in ms (optional)
        """
        self.metrics.add_accuracy(predictions, targets)
        
        if pred_spans is not None and gt_spans is not None:
            self.metrics.add_temporal_iou(pred_spans, gt_spans)
        
        if depths is not None:
            self.metrics.add_reasoning_depth(depths)
        
        if times is not None:
            self.metrics.add_inference_time(times)
        
        self.batch_count += 1
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics."""
        return self.metrics.get_summary()
