"""Multimodal Vector Database for efficient memory retrieval."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Dict
import logging

logger = logging.getLogger(__name__)


class MultimodalVDB(nn.Module):
    """Multimodal Vector Database with contrastive learning.
    
    Learns joint embedding space for questions and memory nodes,
    enabling efficient retrieval of relevant segments for reasoning.
    """
    
    def __init__(
        self,
        feature_dim: int,
        shared_embedding_dim: int = 768,
        projection_layers: int = 2,
        dropout: float = 0.1,
        temperature: float = 0.07,
    ):
        """Initialize multimodal VDB.
        
        Args:
            feature_dim: Input feature dimension
            shared_embedding_dim: Shared embedding space dimension
            projection_layers: Number of projection layers
            dropout: Dropout rate
            temperature: Temperature for contrastive loss (lower = sharper distribution)
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.shared_dim = shared_embedding_dim
        self.temperature = temperature
        
        # Question encoder (usually pre-trained BERT)
        self.question_encoder = nn.Sequential(
            nn.Linear(feature_dim, shared_embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(shared_embedding_dim, shared_embedding_dim),
        )
        
        # Memory node encoder
        self.memory_encoder = nn.Sequential(
            nn.Linear(feature_dim, shared_embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(shared_embedding_dim, shared_embedding_dim),
        )
        
        # Store memory nodes and their embeddings
        self.memory_bank = None
        self.memory_embeddings = None
        
        logger.info(
            f"Initialized MultimodalVDB "
            f"(embedding_dim={shared_embedding_dim}, temperature={temperature})"
        )
    
    def encode_question(self, question_embedding: torch.Tensor) -> torch.Tensor:
        """Encode question to shared embedding space.
        
        Args:
            question_embedding: Question embedding [B, D]
            
        Returns:
            Encoded question [B, shared_dim]
        """
        encoded = self.question_encoder(question_embedding)
        # L2 normalize
        encoded = F.normalize(encoded, p=2, dim=-1)
        return encoded
    
    def encode_memory(self, memory_nodes: torch.Tensor) -> torch.Tensor:
        """Encode memory nodes to shared embedding space.
        
        Args:
            memory_nodes: Memory nodes [B, K, D]
            
        Returns:
            Encoded memory [B, K, shared_dim]
        """
        B, K, D = memory_nodes.shape
        
        # Flatten batch and sequence dimensions
        memory_flat = memory_nodes.view(B * K, D)
        encoded = self.memory_encoder(memory_flat)
        # L2 normalize
        encoded = F.normalize(encoded, p=2, dim=-1)
        # Reshape back
        encoded = encoded.view(B, K, self.shared_dim)
        
        return encoded
    
    def update_memory_bank(self, memory_nodes: torch.Tensor) -> None:
        """Update memory bank with new nodes.
        
        Args:
            memory_nodes: Memory nodes [B, K, D]
        """
        with torch.no_grad():
            self.memory_bank = memory_nodes
            self.memory_embeddings = self.encode_memory(memory_nodes)
        
        logger.debug(f"Updated memory bank with {memory_nodes.shape[0]} samples")
    
    def retrieve_topk(
        self,
        question_embedding: torch.Tensor,
        memory_nodes: torch.Tensor,
        k: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve top-k relevant memory nodes for question.
        
        Args:
            question_embedding: Question embedding [B, D]
            memory_nodes: Memory nodes [B, K, D]
            k: Number of nodes to retrieve
            
        Returns:
            Tuple of:
            - retrieved_nodes: Top-k nodes [B, k, D]
            - similarity_scores: Similarity scores [B, k]
        """
        B, K, D = memory_nodes.shape
        
        # Encode question and memory
        q_encoded = self.encode_question(question_embedding)  # [B, shared_dim]
        mem_encoded = self.encode_memory(memory_nodes)  # [B, K, shared_dim]
        
        # Compute similarity scores (cosine similarity in normalized space)
        # q_encoded: [B, 1, shared_dim]
        # mem_encoded: [B, K, shared_dim]
        similarities = torch.bmm(
            q_encoded.unsqueeze(1),  # [B, 1, shared_dim]
            mem_encoded.transpose(1, 2)  # [B, shared_dim, K]
        ).squeeze(1)  # [B, K]
        
        # Get top-k
        k = min(k, K)
        top_k_scores, top_k_indices = torch.topk(similarities, k, dim=1)  # [B, k]
        
        # Gather top-k nodes
        # This requires advanced indexing
        retrieved_nodes = []
        for b in range(B):
            nodes_b = memory_nodes[b, top_k_indices[b]]  # [k, D]
            retrieved_nodes.append(nodes_b)
        
        retrieved_nodes = torch.stack(retrieved_nodes, dim=0)  # [B, k, D]
        
        return retrieved_nodes, top_k_scores
    
    def compute_contrastive_loss(
        self,
        question_embeddings: torch.Tensor,
        memory_nodes: torch.Tensor,
        positive_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute NT-Xent (contrastive) loss for alignment.
        
        Args:
            question_embeddings: Question embeddings [B, D]
            memory_nodes: Memory nodes [B, K, D]
            positive_indices: Indices of positive pairs [B] (which node is relevant)
            
        Returns:
            Contrastive loss (scalar)
        """
        B, K, D = memory_nodes.shape
        
        # Encode
        q_encoded = self.encode_question(question_embeddings)  # [B, shared_dim]
        mem_encoded = self.encode_memory(memory_nodes)  # [B, K, shared_dim]
        
        # Compute similarity matrix
        sim_matrix = torch.bmm(
            q_encoded.unsqueeze(1),
            mem_encoded.transpose(1, 2)
        ).squeeze(1)  # [B, K]
        
        # Scale by temperature
        sim_matrix = sim_matrix / self.temperature
        
        # Create labels (positive=1, negative=0)
        labels = torch.zeros(B, K, device=sim_matrix.device, dtype=torch.long)
        labels[torch.arange(B), positive_indices] = 1
        
        # Cross-entropy loss
        # Treat as classification: which node is most relevant?
        loss = F.cross_entropy(sim_matrix, positive_indices)
        
        return loss
    
    def get_similarity_heatmap(
        self,
        question_embedding: torch.Tensor,
        memory_nodes: torch.Tensor,
    ) -> torch.Tensor:
        """Get similarity heatmap for visualization.
        
        Args:
            question_embedding: Question embedding [B, D]
            memory_nodes: Memory nodes [B, K, D]
            
        Returns:
            Similarity heatmap [B, K]
        """
        q_encoded = self.encode_question(question_embedding)  # [B, shared_dim]
        mem_encoded = self.encode_memory(memory_nodes)  # [B, K, shared_dim]
        
        similarities = torch.bmm(
            q_encoded.unsqueeze(1),
            mem_encoded.transpose(1, 2)
        ).squeeze(1)  # [B, K]
        
        return similarities


class FAISSRetriever:
    """FAISS-based retriever for large-scale memory banks.
    
    More efficient for large memory banks (100k+ nodes).
    """
    
    def __init__(self, embedding_dim: int, use_gpu: bool = False):
        """Initialize FAISS retriever.
        
        Args:
            embedding_dim: Embedding dimension
            use_gpu: Whether to use GPU
        """
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            logger.warning(
                "FAISS not installed. Install with: pip install faiss-cpu"
            )
            self.faiss = None
        
        self.embedding_dim = embedding_dim
        self.use_gpu = use_gpu
        self.index = None
        
        # Create index
        if self.faiss is not None:
            self.index = faiss.IndexFlatL2(embedding_dim)
            if use_gpu:
                import faiss as faiss_module
                res = faiss_module.StandardGpuResources()
                self.index = faiss_module.index_cpu_to_gpu(res, 0, self.index)
    
    def add(self, embeddings: np.ndarray) -> None:
        """Add embeddings to index.
        
        Args:
            embeddings: Embeddings array [N, embedding_dim]
        """
        if self.index is None:
            logger.warning("FAISS not available, cannot add embeddings")
            return
        
        # Normalize embeddings for L2 search
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
        self.index.add(embeddings.astype(np.float32))
        logger.info(f"Added {embeddings.shape[0]} embeddings to FAISS index")
    
    def search(self, query_embeddings: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Search for top-k nearest neighbors.
        
        Args:
            query_embeddings: Query embeddings [N, embedding_dim]
            k: Number of neighbors to retrieve
            
        Returns:
            Tuple of (distances, indices)
        """
        if self.index is None:
            logger.warning("FAISS index not available")
            return np.zeros((query_embeddings.shape[0], k)), np.zeros((query_embeddings.shape[0], k), dtype=int)
        
        # Normalize query embeddings
        query_embeddings = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-10)
        distances, indices = self.index.search(query_embeddings.astype(np.float32), k)
        
        return distances, indices
