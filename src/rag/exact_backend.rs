//! Exact cosine similarity backend using ndarray.
//!
//! This backend stores pre-normalized embeddings and uses matrix-vector
//! multiplication for exact similarity computation. Efficient for indices
//! up to ~1M documents.

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

use super::backend::{normalize_embedding, RetrievalBackend};
use super::{DocumentId, RagError, Result};

/// Exact cosine similarity backend using ndarray.
///
/// Features:
/// - Pre-normalized embeddings for efficient dot product = cosine similarity
/// - BLAS-accelerated matrix-vector multiplication (via ndarray)
/// - Memory-efficient storage
pub struct ExactCosineBackend {
    /// Pre-normalized embeddings matrix (n_docs × embedding_dim).
    embeddings: Array2<f32>,
    /// Document ID for each row.
    doc_ids: Vec<DocumentId>,
    /// Embedding dimension.
    embedding_dim: usize,
    /// Capacity for pre-allocation.
    capacity: usize,
}

impl ExactCosineBackend {
    /// Create a new exact cosine backend.
    pub fn new(embedding_dim: usize) -> Self {
        Self::with_capacity(embedding_dim, 1000)
    }

    /// Create a new backend with pre-allocated capacity.
    pub fn with_capacity(embedding_dim: usize, capacity: usize) -> Self {
        Self {
            embeddings: Array2::zeros((0, embedding_dim)),
            doc_ids: Vec::with_capacity(capacity),
            embedding_dim,
            capacity,
        }
    }

    /// Get the embeddings matrix.
    pub fn embeddings(&self) -> &Array2<f32> {
        &self.embeddings
    }

    /// Get all embeddings as a vector of vectors (for topic extraction).
    ///
    /// Returns embeddings in the same order as `doc_ids()`.
    pub fn get_all_embeddings(&self) -> Vec<Vec<f32>> {
        self.embeddings
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect()
    }

    /// Get document IDs.
    pub fn doc_ids(&self) -> &[DocumentId] {
        &self.doc_ids
    }

    /// Get the configured pre-allocation capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get the index of a document by ID.
    fn index_of(&self, id: DocumentId) -> Option<usize> {
        self.doc_ids.iter().position(|&d| d == id)
    }
}

impl RetrievalBackend for ExactCosineBackend {
    fn add(&mut self, id: DocumentId, embedding: &[f32]) -> Result<()> {
        if embedding.len() != self.embedding_dim {
            return Err(RagError::IndexError(format!(
                "Embedding dimension mismatch: expected {}, got {}",
                self.embedding_dim,
                embedding.len()
            )));
        }

        // Normalize embedding
        let normalized = normalize_embedding(embedding);
        let row = Array1::from_vec(normalized);

        // Append row to matrix
        self.embeddings
            .push(Axis(0), row.view())
            .map_err(|e| RagError::IndexError(format!("Failed to add embedding: {}", e)))?;

        self.doc_ids.push(id);

        Ok(())
    }

    fn query(&self, embedding: &[f32], top_k: usize) -> Vec<(DocumentId, f32)> {
        if self.embeddings.is_empty() || embedding.len() != self.embedding_dim {
            return vec![];
        }

        // Normalize query embedding
        let normalized = normalize_embedding(embedding);
        let query = Array1::from_vec(normalized);

        // Compute scores: embeddings @ query (dot product = cosine similarity for unit vectors)
        let scores = self.embeddings.dot(&query);

        // Get top-k indices by score
        let mut scored: Vec<(usize, f32)> =
            scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();

        // Partial sort for efficiency
        let k = top_k.min(scored.len());
        if k > 0 {
            scored.select_nth_unstable_by(k - 1, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        scored.truncate(k);
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .map(|(idx, score)| (self.doc_ids[idx], score))
            .collect()
    }

    fn len(&self) -> usize {
        self.doc_ids.len()
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    fn save(&self, path: &Path) -> Result<()> {
        std::fs::create_dir_all(path)?;

        // Save embeddings as binary
        let embeddings_path = path.join("embeddings.bin");
        let file = File::create(&embeddings_path)?;
        let mut writer = BufWriter::new(file);

        // Write header: num_docs, embedding_dim
        let header = EmbeddingsHeader {
            num_docs: self.doc_ids.len(),
            embedding_dim: self.embedding_dim,
        };
        crate::bincode_compat::serialize_into(&mut writer, &header)
            .map_err(|e| RagError::Serialization(e.to_string()))?;

        // Write embeddings as flat array
        let flat: Vec<f32> = self.embeddings.iter().copied().collect();
        crate::bincode_compat::serialize_into(&mut writer, &flat)
            .map_err(|e| RagError::Serialization(e.to_string()))?;

        // Save document IDs
        let ids_path = path.join("doc_ids.bin");
        let ids_file = File::create(&ids_path)?;
        let ids_writer = BufWriter::new(ids_file);
        crate::bincode_compat::serialize_into(ids_writer, &self.doc_ids)
            .map_err(|e| RagError::Serialization(e.to_string()))?;

        Ok(())
    }

    fn load(path: &Path, embedding_dim: usize) -> Result<Self> {
        // Load embeddings
        let embeddings_path = path.join("embeddings.bin");
        let file = File::open(&embeddings_path)?;
        let mut reader = BufReader::new(file);

        let header: EmbeddingsHeader = crate::bincode_compat::deserialize_from(&mut reader)
            .map_err(|e| RagError::Serialization(e.to_string()))?;

        if header.embedding_dim != embedding_dim {
            return Err(RagError::IndexError(format!(
                "Embedding dimension mismatch: expected {}, got {}",
                embedding_dim, header.embedding_dim
            )));
        }

        let flat: Vec<f32> = crate::bincode_compat::deserialize_from(&mut reader)
            .map_err(|e| RagError::Serialization(e.to_string()))?;

        let embeddings = Array2::from_shape_vec((header.num_docs, header.embedding_dim), flat)
            .map_err(|e| RagError::IndexError(format!("Failed to reshape embeddings: {}", e)))?;

        // Load document IDs
        let ids_path = path.join("doc_ids.bin");
        let ids_file = File::open(&ids_path)?;
        let ids_reader = BufReader::new(ids_file);
        let doc_ids: Vec<DocumentId> = crate::bincode_compat::deserialize_from(ids_reader)
            .map_err(|e| RagError::Serialization(e.to_string()))?;

        Ok(Self {
            embeddings,
            doc_ids,
            embedding_dim,
            capacity: header.num_docs,
        })
    }

    fn clear(&mut self) {
        self.embeddings = Array2::zeros((0, self.embedding_dim));
        self.doc_ids.clear();
    }

    fn contains(&self, id: DocumentId) -> bool {
        self.doc_ids.contains(&id)
    }

    fn remove(&mut self, id: DocumentId) -> Result<bool> {
        if let Some(idx) = self.index_of(id) {
            // Remove from doc_ids
            self.doc_ids.remove(idx);

            // Rebuild embeddings matrix without the removed row
            // This is O(n) but removal should be rare
            let mut new_embeddings = Array2::zeros((self.doc_ids.len(), self.embedding_dim));
            let mut new_idx = 0;
            for (i, row) in self.embeddings.rows().into_iter().enumerate() {
                if i != idx {
                    new_embeddings.row_mut(new_idx).assign(&row);
                    new_idx += 1;
                }
            }
            self.embeddings = new_embeddings;

            Ok(true)
        } else {
            Ok(false)
        }
    }
}

/// Header for embeddings file.
#[derive(Serialize, Deserialize)]
struct EmbeddingsHeader {
    num_docs: usize,
    embedding_dim: usize,
}

impl std::fmt::Debug for ExactCosineBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExactCosineBackend")
            .field("num_documents", &self.len())
            .field("embedding_dim", &self.embedding_dim)
            .field("capacity", &self.capacity)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_query() {
        let mut backend = ExactCosineBackend::new(3);

        // Add some documents
        backend.add(DocumentId::new(0), &[1.0, 0.0, 0.0]).unwrap();
        backend.add(DocumentId::new(1), &[0.0, 1.0, 0.0]).unwrap();
        backend.add(DocumentId::new(2), &[0.0, 0.0, 1.0]).unwrap();

        assert_eq!(backend.len(), 3);

        // Query with first dimension
        let results = backend.query(&[1.0, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, DocumentId::new(0));
        assert!((results[0].1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_empty_query() {
        let backend = ExactCosineBackend::new(3);
        let results = backend.query(&[1.0, 0.0, 0.0], 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_contains() {
        let mut backend = ExactCosineBackend::new(3);
        backend.add(DocumentId::new(42), &[1.0, 0.0, 0.0]).unwrap();

        assert!(backend.contains(DocumentId::new(42)));
        assert!(!backend.contains(DocumentId::new(99)));
    }

    #[test]
    fn test_remove() {
        let mut backend = ExactCosineBackend::new(3);
        backend.add(DocumentId::new(0), &[1.0, 0.0, 0.0]).unwrap();
        backend.add(DocumentId::new(1), &[0.0, 1.0, 0.0]).unwrap();

        assert_eq!(backend.len(), 2);
        assert!(backend.remove(DocumentId::new(0)).unwrap());
        assert_eq!(backend.len(), 1);
        assert!(!backend.contains(DocumentId::new(0)));
        assert!(backend.contains(DocumentId::new(1)));
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut backend = ExactCosineBackend::new(3);
        let result = backend.add(DocumentId::new(0), &[1.0, 0.0]); // Wrong dimension
        assert!(result.is_err());
    }
}
