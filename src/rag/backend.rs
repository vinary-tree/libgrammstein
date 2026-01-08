//! Retrieval backend trait for RAG index.
//!
//! This module defines the trait that all retrieval backends must implement.

use std::path::Path;

use super::{DocumentId, RagError, Result};

/// Trait for retrieval backends.
///
/// Backends are responsible for storing document embeddings and performing
/// similarity search.
pub trait RetrievalBackend: Send + Sync {
    /// Add a document embedding to the index.
    ///
    /// The embedding is automatically normalized to unit length internally.
    fn add(&mut self, id: DocumentId, embedding: &[f32]) -> Result<()>;

    /// Query for top-k similar documents.
    ///
    /// Returns document IDs and similarity scores, sorted by score descending.
    fn query(&self, embedding: &[f32], top_k: usize) -> Vec<(DocumentId, f32)>;

    /// Number of documents in the index.
    fn len(&self) -> usize;

    /// Check if the index is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the embedding dimension.
    fn embedding_dim(&self) -> usize;

    /// Save the backend to disk.
    fn save(&self, path: &Path) -> Result<()>;

    /// Load the backend from disk.
    fn load(path: &Path, embedding_dim: usize) -> Result<Self>
    where
        Self: Sized;

    /// Clear all documents from the index.
    fn clear(&mut self);

    /// Check if a document exists in the index.
    fn contains(&self, id: DocumentId) -> bool;

    /// Remove a document from the index (if supported).
    ///
    /// Returns true if the document was removed, false if not found.
    /// Some backends may not support removal efficiently.
    fn remove(&mut self, id: DocumentId) -> Result<bool> {
        let _ = id;
        Err(RagError::IndexError(
            "Removal not supported by this backend".to_string(),
        ))
    }
}

/// Backend selection for index construction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BackendType {
    /// Exact cosine similarity using ndarray (default).
    #[default]
    ExactCosine,
    /// HNSW approximate nearest neighbor (for large indices).
    #[cfg(feature = "rag-hnsw")]
    Hnsw,
}

impl BackendType {
    /// Get recommended backend based on document count.
    pub fn recommended_for_size(num_documents: usize) -> Self {
        if num_documents > 1_000_000 {
            #[cfg(feature = "rag-hnsw")]
            return BackendType::Hnsw;
        }
        BackendType::ExactCosine
    }
}

/// Normalize embedding to unit length.
pub fn normalize_embedding(embedding: &[f32]) -> Vec<f32> {
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm == 0.0 {
        embedding.to_vec()
    } else {
        embedding.iter().map(|x| x / norm).collect()
    }
}

/// Compute dot product between two vectors.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Compute cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product(a, b);
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_embedding() {
        let embedding = vec![3.0, 4.0];
        let normalized = normalize_embedding(&embedding);

        let norm: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        assert!((dot_product(&a, &b) - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![0.0, 1.0];
        assert!((cosine_similarity(&a, &c) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_recommended_backend() {
        assert_eq!(
            BackendType::recommended_for_size(1000),
            BackendType::ExactCosine
        );
        assert_eq!(
            BackendType::recommended_for_size(100_000),
            BackendType::ExactCosine
        );
    }
}
