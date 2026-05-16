"""Visualization utilities for LVLM."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict, Tuple, Optional
import torch


def plot_attention_heatmap(attention_weights: np.ndarray,
                          timesteps: Optional[List[float]] = None,
                          question: Optional[str] = None,
                          title: Optional[str] = None,
                          save_path: Optional[str] = None) -> plt.Figure:
    """Plot attention heatmap over temporal axis.
    
    Args:
        attention_weights: Attention weights [T] or [num_heads, T]
        timesteps: Optional list of timestamps
        question: Optional question text for title
        title: Optional plot title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    
    if attention_weights.ndim > 1:
        # Average over multiple heads
        attention_weights = attention_weights.mean(axis=0)
    
    x_pos = np.arange(len(attention_weights))
    
    ax.bar(x_pos, attention_weights, color='steelblue', alpha=0.8)
    ax.set_xlabel('Temporal Position')
    ax.set_ylabel('Attention Weight')
    
    if timesteps is not None:
        ax.set_xticks(x_pos[::max(1, len(x_pos)//10)])
        ax.set_xticklabels([f'{t:.1f}s' for t in np.array(timesteps)[::max(1, len(x_pos)//10)]])
    
    if question:
        ax.set_title(f'Attention Heatmap - Q: {question}')
    elif title:
        ax.set_title(title)
    else:
        ax.set_title('Attention Heatmap')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_reasoning_trace(reasoning_states: List[np.ndarray],
                        memory_labels: Optional[List[str]] = None,
                        hops: Optional[int] = None,
                        title: Optional[str] = None,
                        save_path: Optional[str] = None) -> plt.Figure:
    """Plot reasoning trace across hops.
    
    Args:
        reasoning_states: List of reasoning state vectors per hop [num_hops, num_memory_nodes]
        memory_labels: Optional labels for memory nodes
        hops: Number of hops performed
        title: Optional plot title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    num_hops = len(reasoning_states)
    
    fig, axes = plt.subplots(1, num_hops, figsize=(4*num_hops, 4))
    if num_hops == 1:
        axes = [axes]
    
    for k, state in enumerate(reasoning_states):
        ax = axes[k]
        
        if state.ndim > 1:
            state = state.mean(axis=-1)  # Average over feature dimension
        
        x_pos = np.arange(len(state))
        ax.bar(x_pos, state, color=plt.cm.viridis(k / max(1, num_hops-1)), alpha=0.8)
        
        ax.set_xlabel('Memory Node')
        ax.set_ylabel('Activation')
        ax.set_title(f'Hop {k+1}/{num_hops}')
        
        if memory_labels:
            ax.set_xticks(x_pos)
            ax.set_xticklabels(memory_labels, rotation=45, ha='right')
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    elif hops:
        fig.suptitle(f'Reasoning Trace ({hops} hops)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_temporal_binding(binding_assignments: np.ndarray,
                         frame_count: Optional[int] = None,
                         timesteps: Optional[np.ndarray] = None,
                         title: Optional[str] = None,
                         save_path: Optional[str] = None) -> plt.Figure:
    """Plot temporal binding assignments (frames to memory nodes).
    
    Args:
        binding_assignments: Assignment indices [num_frames]
        frame_count: Total number of frames
        timesteps: Timestamps for each frame
        title: Optional plot title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    
    x_pos = np.arange(len(binding_assignments))
    colors = plt.cm.tab20(binding_assignments / max(binding_assignments.max(), 1))
    
    ax.scatter(x_pos, binding_assignments, c=binding_assignments, cmap='tab20',
               s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Memory Node')
    ax.set_title(title or 'Temporal Binding Assignments')
    
    if timesteps is not None:
        ax.set_xticks(x_pos[::max(1, len(x_pos)//10)])
        ax.set_xticklabels([f'{t:.1f}s' for t in timesteps[::max(1, len(x_pos)//10)]])
    
    plt.colorbar(ax.scatter(x_pos, binding_assignments, c=binding_assignments,
                            cmap='tab20'), ax=ax, label='Memory Node')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_accuracy_speedup_tradeoff(accuracies: Dict[str, float],
                                   speedups: Dict[str, float],
                                   labels: Optional[List[str]] = None,
                                   title: Optional[str] = None,
                                   save_path: Optional[str] = None) -> plt.Figure:
    """Plot accuracy vs. speedup trade-off (Pareto frontier).
    
    Args:
        accuracies: Dictionary mapping experiment names to accuracies
        speedups: Dictionary mapping experiment names to speedups
        labels: Optional custom labels
        title: Optional plot title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    names = list(accuracies.keys())
    accs = [accuracies[name] for name in names]
    spds = [speedups[name] for name in names]
    
    ax.scatter(spds, accs, s=100, alpha=0.7, edgecolors='black', linewidth=1.5)
    
    # Add labels
    for i, name in enumerate(names):
        ax.annotate(labels[i] if labels else name,
                   (spds[i], accs[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, alpha=0.8)
    
    ax.set_xlabel('Speedup Factor', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(title or 'Accuracy vs. Speedup Trade-off', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_depth_distribution(depths: List[int],
                           bins: int = 20,
                           title: Optional[str] = None,
                           save_path: Optional[str] = None) -> plt.Figure:
    """Plot distribution of reasoning depths used.
    
    Args:
        depths: List of depths used per sample
        bins: Number of histogram bins
        title: Optional plot title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.hist(depths, bins=bins, color='steelblue', alpha=0.7, edgecolor='black')
    
    ax.axvline(np.mean(depths), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(depths):.2f}')
    ax.axvline(np.median(depths), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(depths):.2f}')
    
    ax.set_xlabel('Reasoning Depth (hops)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title or 'Distribution of Reasoning Depths', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_loss_curves(train_losses: Dict[str, List[float]],
                    val_losses: Optional[Dict[str, List[float]]] = None,
                    title: Optional[str] = None,
                    save_path: Optional[str] = None) -> plt.Figure:
    """Plot training and validation loss curves.
    
    Args:
        train_losses: Dictionary of training losses per component
        val_losses: Optional dictionary of validation losses
        title: Optional plot title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for loss_name, loss_values in train_losses.items():
        ax.plot(loss_values, label=f'Train {loss_name}', linewidth=2)
    
    if val_losses:
        for loss_name, loss_values in val_losses.items():
            ax.plot(loss_values, label=f'Val {loss_name}', linestyle='--', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title or 'Training and Validation Loss Curves', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
