//! HNSW approximate nearest neighbor backend.
//!
//! This backend uses Hierarchical Navigable Small World graphs for
//! approximate nearest neighbor search. Efficient for indices with
//! more than 1M documents.
//!
//! # Features
//!
//! - **Per-query ef_search**: Tune accuracy/speed trade-off at query time
//! - **Parallel operations**: Concurrent insertion and batch search
//! - **SIMD acceleration**: AVX2 for f32 distance calculations (x86_64)
//! - **Filtered search**: Filter results during traversal (not post-filter)
//! - **Native persistence**: Dump/reload with optional mmap for large indices
//!
//! # Thread Safety
//!
//! This backend uses interior mutability with `RwLock` to enable:
//! - Concurrent queries via `&self` methods
//! - Lazy index building on first query
//! - Automatic rebuild when index becomes stale

#![cfg(feature = "rag-hnsw")]

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use hnsw_rs::api::AnnT;
use hnsw_rs::hnsw::Hnsw;
use hnsw_rs::hnswio::HnswIo;
use hnsw_rs::prelude::DistDot;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use super::backend::{normalize_embedding, RetrievalBackend};
use super::{DocumentId, RagError, Result};

/// HNSW approximate nearest neighbor backend.
///
/// Features:
/// - O(log n) query time with configurable recall via per-query ef_search
/// - Parallel insertion and batch search
/// - SIMD-accelerated distance calculations
/// - Filtered search support
/// - Native dump/reload with mmap for large indices
///
/// The index is built lazily on first query and automatically rebuilt
/// when the pending points exceed a threshold or when explicitly requested.
pub struct HnswBackend {
    /// HNSW index (lazy-built, thread-safe).
    /// `None` until first build, then `Some(index)`.
    index: RwLock<Option<Hnsw<'static, f32, DistDot>>>,
    /// Embedding dimension.
    embedding_dim: usize,
    /// Configuration.
    config: HnswConfig,
    /// Whether index needs rebuild (atomic for lock-free check).
    needs_rebuild: AtomicBool,
    /// Pending points for rebuild (protected by RwLock).
    /// Stored as (embedding, document_id) pairs.
    pending_points: RwLock<Vec<(Vec<f32>, DocumentId)>>,
    /// Number of points (atomic for fast len() without locking).
    num_points: AtomicUsize,
}

/// Configuration for HNSW index.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Maximum number of neighbors per node (M parameter).
    /// Higher = better recall, more memory.
    /// Typical: 16-64.
    pub max_nb_connection: usize,
    /// Number of neighbors to explore during construction.
    /// Higher = better graph quality, slower build.
    /// Typical: 200-800.
    pub ef_construction: usize,
    /// Default number of neighbors to explore during search.
    /// Higher = better recall, slower query.
    /// Can be overridden per-query via `query_with_ef()`.
    /// Typical: 50-200.
    pub ef_search: usize,
    /// Maximum number of layers in the graph.
    /// Usually 16 is sufficient for up to billions of points.
    pub max_layer: usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            max_nb_connection: 24, // Good balance for RAG
            ef_construction: 400,  // High quality graph
            ef_search: 100,        // Good recall default
            max_layer: 16,         // Sufficient for billions
        }
    }
}

impl HnswBackend {
    /// Create a new HNSW backend.
    pub fn new(embedding_dim: usize) -> Self {
        Self::with_config(embedding_dim, HnswConfig::default())
    }

    /// Create a new HNSW backend with custom configuration.
    pub fn with_config(embedding_dim: usize, config: HnswConfig) -> Self {
        Self {
            index: RwLock::new(None),
            embedding_dim,
            config,
            needs_rebuild: AtomicBool::new(false),
            pending_points: RwLock::new(Vec::new()),
            num_points: AtomicUsize::new(0),
        }
    }

    /// Build the HNSW index from pending points using parallel insertion.
    ///
    /// This method takes `&self` and uses interior mutability for thread-safe lazy building.
    fn build_index(&self) -> Result<()> {
        let points_guard = self.pending_points.read();
        if points_guard.is_empty() {
            return Ok(());
        }

        let num_points = points_guard.len();

        // Create new HNSW index with DistDot (equivalent to cosine for normalized vectors)
        let hnsw: Hnsw<f32, DistDot> = Hnsw::new(
            self.config.max_nb_connection,
            num_points,
            self.config.max_layer,
            self.config.ef_construction,
            DistDot,
        );

        // Prepare data for parallel insertion: Vec<(&Vec<f32>, usize)>
        let data_for_insert: Vec<(&Vec<f32>, usize)> = points_guard
            .iter()
            .enumerate()
            .map(|(idx, (embedding, _))| (embedding, idx))
            .collect();

        // Use parallel insertion for better performance on large datasets
        hnsw.parallel_insert(&data_for_insert);

        drop(points_guard); // Release read lock before acquiring write lock

        // Safety: The data is copied into the index during insertion.
        // We maintain pending_points for the lifetime of the index to map idx -> doc_id.
        let hnsw_static: Hnsw<'static, f32, DistDot> = unsafe { std::mem::transmute(hnsw) };

        *self.index.write() = Some(hnsw_static);
        self.needs_rebuild.store(false, Ordering::Release);
        Ok(())
    }

    /// Ensure the index is built before querying.
    fn ensure_built(&self) {
        if self.needs_rebuild.load(Ordering::Acquire) {
            let _ = self.build_index();
        }
    }

    /// Query with custom ef_search parameter.
    ///
    /// This method allows tuning the accuracy/speed trade-off per query:
    /// - Higher ef_search = better recall, slower query
    /// - Lower ef_search = faster query, lower recall
    ///
    /// Typical values: 50-200 depending on your latency requirements.
    pub fn query_with_ef(
        &self,
        embedding: &[f32],
        top_k: usize,
        ef_search: usize,
    ) -> Vec<(DocumentId, f32)> {
        if self.num_points.load(Ordering::Relaxed) == 0 {
            return vec![];
        }

        self.ensure_built();

        let normalized = normalize_embedding(embedding);

        let index_guard = self.index.read();
        let Some(ref index) = *index_guard else {
            return vec![];
        };

        let points_guard = self.pending_points.read();

        // Search returns Vec<Neighbour> for a single query
        let results = index.search(&normalized, top_k, ef_search);

        if results.is_empty() {
            return vec![];
        }

        results
            .iter()
            .filter_map(|neighbor| {
                let idx = neighbor.d_id;
                if idx < points_guard.len() {
                    let doc_id = points_guard[idx].1;
                    // DistDot returns 1 - dot_product for normalized vectors
                    // So similarity = 1 - distance
                    let similarity = 1.0 - neighbor.distance;
                    Some((doc_id, similarity))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Batch query with custom ef_search, using parallel search.
    ///
    /// More efficient than multiple single queries for batch operations.
    pub fn batch_query_with_ef(
        &self,
        embeddings: &[&[f32]],
        top_k: usize,
        ef_search: usize,
    ) -> Vec<Vec<(DocumentId, f32)>> {
        if self.num_points.load(Ordering::Relaxed) == 0 {
            return vec![vec![]; embeddings.len()];
        }

        self.ensure_built();

        // parallel_search takes &[Vec<T>], so we need owned Vecs
        let normalized: Vec<Vec<f32>> = embeddings
            .iter()
            .map(|e| normalize_embedding(e))
            .collect();

        let index_guard = self.index.read();
        let Some(ref index) = *index_guard else {
            return vec![vec![]; embeddings.len()];
        };

        let points_guard = self.pending_points.read();

        // Parallel search across all queries
        let results = index.parallel_search(&normalized, top_k, ef_search);

        results
            .into_iter()
            .map(|neighbors| {
                neighbors
                    .iter()
                    .filter_map(|neighbor| {
                        let idx = neighbor.d_id;
                        if idx < points_guard.len() {
                            let doc_id = points_guard[idx].1;
                            let similarity = 1.0 - neighbor.distance;
                            Some((doc_id, similarity))
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// Query with filtering: only return results where filter returns true.
    ///
    /// The filter is applied during graph traversal, not as a post-filter,
    /// which is more efficient for selective filters.
    ///
    /// # Arguments
    ///
    /// * `embedding` - Query embedding
    /// * `top_k` - Number of results to return
    /// * `ef_search` - Search width (higher = better recall)
    /// * `filter` - Function that takes document ID and returns true to include
    pub fn query_with_filter<F>(
        &self,
        embedding: &[f32],
        top_k: usize,
        ef_search: usize,
        filter: F,
    ) -> Vec<(DocumentId, f32)>
    where
        F: Fn(DocumentId) -> bool,
    {
        if self.num_points.load(Ordering::Relaxed) == 0 {
            return vec![];
        }

        self.ensure_built();

        let normalized = normalize_embedding(embedding);

        let index_guard = self.index.read();
        let Some(ref index) = *index_guard else {
            return vec![];
        };

        let points_guard = self.pending_points.read();

        // Create sorted filter set from allowed indices
        let allowed_ids: Vec<usize> = points_guard
            .iter()
            .enumerate()
            .filter_map(|(idx, (_, doc_id))| {
                if filter(*doc_id) {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect();

        if allowed_ids.is_empty() {
            return vec![];
        }

        // Use filtered search
        let results =
            index.search_filter(&normalized, top_k, ef_search, Some(&allowed_ids));

        results
            .iter()
            .filter_map(|neighbor| {
                let idx = neighbor.d_id;
                if idx < points_guard.len() {
                    let doc_id = points_guard[idx].1;
                    let similarity = 1.0 - neighbor.distance;
                    Some((doc_id, similarity))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Batch add documents using parallel insertion.
    ///
    /// More efficient than calling `add()` repeatedly for large batches.
    pub fn batch_add(
        &mut self,
        documents: impl IntoIterator<Item = (DocumentId, Vec<f32>)>,
    ) -> Result<()> {
        let mut points = self.pending_points.write();
        let start_idx = points.len();

        for (id, embedding) in documents {
            if embedding.len() != self.embedding_dim {
                return Err(RagError::IndexError(format!(
                    "Embedding dimension mismatch: expected {}, got {}",
                    self.embedding_dim,
                    embedding.len()
                )));
            }
            let normalized = normalize_embedding(&embedding);
            points.push((normalized, id));
        }

        let added = points.len() - start_idx;
        drop(points);

        self.num_points.fetch_add(added, Ordering::Relaxed);
        self.needs_rebuild.store(true, Ordering::Release);

        Ok(())
    }

    /// Get the configuration.
    pub fn config(&self) -> &HnswConfig {
        &self.config
    }

    /// Update configuration (requires rebuild).
    pub fn set_config(&mut self, config: HnswConfig) {
        self.config = config;
        self.needs_rebuild.store(true, Ordering::Release);
    }

    /// Force rebuild of the index.
    pub fn force_rebuild(&self) -> Result<()> {
        self.build_index()
    }

    /// Load the HNSW graph natively using hnsw_rs's HnswIo.
    ///
    /// This avoids expensive rebuilding by loading the persisted graph structure
    /// directly from disk.
    ///
    /// # Safety
    ///
    /// The returned Hnsw has a 'static lifetime, which is safe because:
    /// - The default ReloadOptions does not use memory mapping
    /// - Data is fully deserialized and owned by the Hnsw struct
    /// - The HnswIo can be dropped after loading
    fn load_native_graph(path: &Path) -> Result<Hnsw<'static, f32, DistDot>> {
        let mut reloader = HnswIo::new(path, "hnsw_index");

        let hnsw: Hnsw<f32, DistDot> = reloader
            .load_hnsw()
            .map_err(|e| RagError::Serialization(format!("Failed to reload HNSW graph: {:?}", e)))?;

        // Safety: With default ReloadOptions (no mmap), data is fully deserialized
        // and owned by the Hnsw struct, so 'static lifetime is valid.
        let hnsw_static: Hnsw<'static, f32, DistDot> = unsafe { std::mem::transmute(hnsw) };

        Ok(hnsw_static)
    }

    /// Get index statistics.
    pub fn stats(&self) -> HnswStats {
        let index_guard = self.index.read();
        let (layer_counts, memory_estimate) = if index_guard.is_some() {
            // Estimate memory: embeddings + graph structure
            let num_points = self.num_points.load(Ordering::Relaxed);
            let embedding_mem = num_points * self.embedding_dim * 4; // f32
            let graph_mem = num_points * self.config.max_nb_connection * 8; // usize pointers
            (vec![num_points], embedding_mem + graph_mem)
        } else {
            (vec![], 0)
        };

        HnswStats {
            num_points: self.num_points.load(Ordering::Relaxed),
            embedding_dim: self.embedding_dim,
            config: self.config.clone(),
            is_built: index_guard.is_some(),
            layer_counts,
            estimated_memory_bytes: memory_estimate,
        }
    }
}

/// Statistics about the HNSW index.
#[derive(Clone, Debug)]
pub struct HnswStats {
    /// Number of indexed points.
    pub num_points: usize,
    /// Embedding dimension.
    pub embedding_dim: usize,
    /// Current configuration.
    pub config: HnswConfig,
    /// Whether the index is currently built.
    pub is_built: bool,
    /// Number of points per layer.
    pub layer_counts: Vec<usize>,
    /// Estimated memory usage in bytes.
    pub estimated_memory_bytes: usize,
}

impl RetrievalBackend for HnswBackend {
    fn add(&mut self, id: DocumentId, embedding: &[f32]) -> Result<()> {
        if embedding.len() != self.embedding_dim {
            return Err(RagError::IndexError(format!(
                "Embedding dimension mismatch: expected {}, got {}",
                self.embedding_dim,
                embedding.len()
            )));
        }

        let normalized = normalize_embedding(embedding);
        self.pending_points.write().push((normalized, id));
        let new_count = self.num_points.fetch_add(1, Ordering::Relaxed) + 1;
        self.needs_rebuild.store(true, Ordering::Release);

        // Rebuild periodically to keep index reasonably up-to-date
        if new_count % 10000 == 0 {
            self.build_index()?;
        }

        Ok(())
    }

    fn query(&self, embedding: &[f32], top_k: usize) -> Vec<(DocumentId, f32)> {
        self.query_with_ef(embedding, top_k, self.config.ef_search)
    }

    fn len(&self) -> usize {
        self.num_points.load(Ordering::Relaxed)
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    fn save(&self, path: &Path) -> Result<()> {
        std::fs::create_dir_all(path)?;

        // Save configuration
        let config_path = path.join("hnsw_config.json");
        let config_file = File::create(&config_path)?;
        serde_json::to_writer_pretty(BufWriter::new(config_file), &self.config)
            .map_err(|e| RagError::Serialization(e.to_string()))?;

        // Save document ID mapping with batched serialization
        let mapping_path = path.join("doc_mapping.bin");
        let mapping_file = File::create(&mapping_path)?;
        let mut writer = BufWriter::new(mapping_file);

        let points_guard = self.pending_points.read();
        let num_docs = points_guard.len();

        // Header with version for batched format
        let header = HnswHeader {
            num_docs,
            embedding_dim: self.embedding_dim,
            version: 2, // v2 = batched format
        };
        bincode::serialize_into(&mut writer, &header)
            .map_err(|e| RagError::Serialization(e.to_string()))?;

        // Serialize embeddings and IDs in batches to prevent OOM
        // For 1M docs × 768 dims = ~3GB if collected at once; batching reduces peak to ~30MB
        let num_batches = (num_docs + SERIALIZATION_BATCH_SIZE - 1) / SERIALIZATION_BATCH_SIZE;
        bincode::serialize_into(&mut writer, &num_batches)
            .map_err(|e| RagError::Serialization(e.to_string()))?;

        for batch_idx in 0..num_batches {
            let start = batch_idx * SERIALIZATION_BATCH_SIZE;
            let end = (start + SERIALIZATION_BATCH_SIZE).min(num_docs);

            // Serialize batch of embeddings
            let batch_embeddings: Vec<&Vec<f32>> = points_guard[start..end]
                .iter()
                .map(|(embedding, _)| embedding)
                .collect();
            bincode::serialize_into(&mut writer, &batch_embeddings)
                .map_err(|e| RagError::Serialization(e.to_string()))?;

            // Serialize batch of document IDs
            let batch_ids: Vec<DocumentId> = points_guard[start..end]
                .iter()
                .map(|(_, id)| *id)
                .collect();
            bincode::serialize_into(&mut writer, &batch_ids)
                .map_err(|e| RagError::Serialization(e.to_string()))?;
        }

        drop(points_guard); // Release read lock before accessing index

        // Use hnsw_rs native dump for the graph structure if index is built
        let index_guard = self.index.read();
        if let Some(ref index) = *index_guard {
            // hnsw_rs file_dump takes a directory path and basename
            index
                .file_dump(path, "hnsw_index")
                .map_err(|e| RagError::Serialization(format!("Failed to dump HNSW graph: {}", e)))?;
        }

        Ok(())
    }

    fn load(path: &Path, embedding_dim: usize) -> Result<Self> {
        // Load configuration
        let config_path = path.join("hnsw_config.json");
        let config_file = File::open(&config_path)?;
        let config: HnswConfig = serde_json::from_reader(BufReader::new(config_file))
            .map_err(|e| RagError::Serialization(e.to_string()))?;

        // Load document mapping with streaming deserialization
        let mapping_path = path.join("doc_mapping.bin");
        let mapping_file = File::open(&mapping_path)?;
        let mut reader = BufReader::new(mapping_file);

        let header: HnswHeader = bincode::deserialize_from(&mut reader)
            .map_err(|e| RagError::Serialization(e.to_string()))?;

        if header.embedding_dim != embedding_dim {
            return Err(RagError::IndexError(format!(
                "Embedding dimension mismatch: expected {}, got {}",
                embedding_dim, header.embedding_dim
            )));
        }

        // Preallocate with known capacity to avoid reallocation during loading
        let mut pending_points: Vec<(Vec<f32>, DocumentId)> = Vec::with_capacity(header.num_docs);

        // Handle both v1 (unbatched) and v2 (batched) formats
        if header.version >= 2 {
            // v2 batched format: read num_batches, then process each batch
            let num_batches: usize = bincode::deserialize_from(&mut reader)
                .map_err(|e| RagError::Serialization(e.to_string()))?;

            for _ in 0..num_batches {
                // Deserialize batch of embeddings
                let batch_embeddings: Vec<Vec<f32>> = bincode::deserialize_from(&mut reader)
                    .map_err(|e| RagError::Serialization(e.to_string()))?;

                // Deserialize batch of document IDs
                let batch_ids: Vec<DocumentId> = bincode::deserialize_from(&mut reader)
                    .map_err(|e| RagError::Serialization(e.to_string()))?;

                // Combine embeddings and IDs
                pending_points.extend(batch_embeddings.into_iter().zip(batch_ids));
            }
        } else {
            // v1 legacy format: all embeddings, then all IDs (for backwards compatibility)
            let embeddings: Vec<Vec<f32>> = bincode::deserialize_from(&mut reader)
                .map_err(|e| RagError::Serialization(e.to_string()))?;

            let ids: Vec<DocumentId> = bincode::deserialize_from(&mut reader)
                .map_err(|e| RagError::Serialization(e.to_string()))?;

            pending_points.extend(embeddings.into_iter().zip(ids));
        }

        let num_points = pending_points.len();

        // Check for native graph files (created by hnsw_rs file_dump)
        let graph_path = path.join("hnsw_index.hnsw.graph");
        let data_path = path.join("hnsw_index.hnsw.data");

        // Try native graph reload if files exist, fall back to rebuild
        let (loaded_index, needs_rebuild) = if graph_path.exists() && data_path.exists() {
            match Self::load_native_graph(path) {
                Ok(hnsw) => {
                    log::debug!("Successfully loaded native HNSW graph from {:?}", path);
                    (Some(hnsw), false)
                }
                Err(e) => {
                    log::warn!(
                        "Failed to load native HNSW graph from {:?}, will rebuild: {}",
                        path,
                        e
                    );
                    (None, true)
                }
            }
        } else {
            log::debug!("No native HNSW graph files found at {:?}, will rebuild", path);
            (None, true)
        };

        let backend = Self {
            index: RwLock::new(loaded_index),
            embedding_dim,
            config,
            needs_rebuild: AtomicBool::new(needs_rebuild),
            pending_points: RwLock::new(pending_points),
            num_points: AtomicUsize::new(num_points),
        };

        // Build index if native load failed or files don't exist
        if needs_rebuild {
            backend.build_index()?;
        }

        Ok(backend)
    }

    fn clear(&mut self) {
        *self.index.write() = None;
        self.pending_points.write().clear();
        self.num_points.store(0, Ordering::Relaxed);
        self.needs_rebuild.store(false, Ordering::Release);
    }

    fn contains(&self, id: DocumentId) -> bool {
        self.pending_points
            .read()
            .iter()
            .any(|(_, doc_id)| *doc_id == id)
    }
}

/// Header for HNSW data file.
#[derive(Serialize, Deserialize)]
struct HnswHeader {
    num_docs: usize,
    embedding_dim: usize,
    /// Format version for backwards compatibility
    version: u32,
}

/// Default batch size for serialization/deserialization (10K embeddings)
/// This prevents OOM for 1M+ document indices (1M × 768 dims = ~3GB if loaded at once).
const SERIALIZATION_BATCH_SIZE: usize = 10_000;

impl std::fmt::Debug for HnswBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HnswBackend")
            .field("num_documents", &self.len())
            .field("embedding_dim", &self.embedding_dim)
            .field("config", &self.config)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_build() {
        let mut backend = HnswBackend::new(3);

        backend.add(DocumentId::new(0), &[1.0, 0.0, 0.0]).unwrap();
        backend.add(DocumentId::new(1), &[0.0, 1.0, 0.0]).unwrap();
        backend.add(DocumentId::new(2), &[0.0, 0.0, 1.0]).unwrap();

        assert_eq!(backend.len(), 3);

        backend.build_index().unwrap();
        assert!(!backend.needs_rebuild.load(Ordering::Relaxed));
    }

    #[test]
    fn test_contains() {
        let mut backend = HnswBackend::new(3);
        backend.add(DocumentId::new(42), &[1.0, 0.0, 0.0]).unwrap();

        assert!(backend.contains(DocumentId::new(42)));
        assert!(!backend.contains(DocumentId::new(99)));
    }

    #[test]
    fn test_lazy_build_on_query() {
        let mut backend = HnswBackend::new(3);

        backend.add(DocumentId::new(0), &[1.0, 0.0, 0.0]).unwrap();
        backend.add(DocumentId::new(1), &[0.0, 1.0, 0.0]).unwrap();
        backend.add(DocumentId::new(2), &[0.0, 0.0, 1.0]).unwrap();

        assert!(backend.needs_rebuild.load(Ordering::Relaxed));

        let results = backend.query(&[1.0, 0.0, 0.0], 1);

        assert!(!backend.needs_rebuild.load(Ordering::Relaxed));
        assert!(!results.is_empty());
    }

    #[test]
    fn test_query_with_ef() {
        let mut backend = HnswBackend::new(3);

        backend.add(DocumentId::new(0), &[1.0, 0.0, 0.0]).unwrap();
        backend.add(DocumentId::new(1), &[0.9, 0.1, 0.0]).unwrap();
        backend.add(DocumentId::new(2), &[0.0, 1.0, 0.0]).unwrap();

        backend.force_rebuild().unwrap();

        // Test with different ef_search values
        let results_low_ef = backend.query_with_ef(&[1.0, 0.0, 0.0], 2, 10);
        let results_high_ef = backend.query_with_ef(&[1.0, 0.0, 0.0], 2, 200);

        assert!(!results_low_ef.is_empty());
        assert!(!results_high_ef.is_empty());
    }

    #[test]
    fn test_batch_query() {
        let mut backend = HnswBackend::new(3);

        backend.add(DocumentId::new(0), &[1.0, 0.0, 0.0]).unwrap();
        backend.add(DocumentId::new(1), &[0.0, 1.0, 0.0]).unwrap();
        backend.add(DocumentId::new(2), &[0.0, 0.0, 1.0]).unwrap();

        backend.force_rebuild().unwrap();

        let queries: Vec<&[f32]> = vec![&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]];
        let results = backend.batch_query_with_ef(&queries, 2, 100);

        assert_eq!(results.len(), 2);
        assert!(!results[0].is_empty());
        assert!(!results[1].is_empty());
    }

    #[test]
    fn test_filtered_query() {
        let mut backend = HnswBackend::new(3);

        backend.add(DocumentId::new(0), &[1.0, 0.0, 0.0]).unwrap();
        backend.add(DocumentId::new(1), &[0.9, 0.1, 0.0]).unwrap();
        backend.add(DocumentId::new(2), &[0.0, 1.0, 0.0]).unwrap();

        backend.force_rebuild().unwrap();

        // Filter to only include documents with ID >= 1
        let results = backend.query_with_filter(&[1.0, 0.0, 0.0], 2, 100, |id| id.0 >= 1);

        // Should only return documents 1 and 2
        assert!(results.iter().all(|(id, _)| id.0 >= 1));
    }

    #[test]
    fn test_batch_add() {
        let mut backend = HnswBackend::new(3);

        let docs = vec![
            (DocumentId::new(0), vec![1.0, 0.0, 0.0]),
            (DocumentId::new(1), vec![0.0, 1.0, 0.0]),
            (DocumentId::new(2), vec![0.0, 0.0, 1.0]),
        ];

        backend.batch_add(docs).unwrap();

        assert_eq!(backend.len(), 3);
    }

    #[test]
    fn test_concurrent_query() {
        use std::sync::Arc;
        use std::thread;

        let mut backend = HnswBackend::new(3);

        backend.add(DocumentId::new(0), &[1.0, 0.0, 0.0]).unwrap();
        backend.add(DocumentId::new(1), &[0.0, 1.0, 0.0]).unwrap();
        backend.add(DocumentId::new(2), &[0.0, 0.0, 1.0]).unwrap();

        backend.force_rebuild().unwrap();

        let backend = Arc::new(backend);

        let handles: Vec<_> = (0..4)
            .map(|i| {
                let backend = Arc::clone(&backend);
                thread::spawn(move || {
                    for _ in 0..10 {
                        let query = match i % 3 {
                            0 => [1.0, 0.0, 0.0],
                            1 => [0.0, 1.0, 0.0],
                            _ => [0.0, 0.0, 1.0],
                        };
                        let results = backend.query(&query, 1);
                        assert!(!results.is_empty());
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread panicked");
        }
    }

    #[test]
    fn test_stats() {
        let mut backend = HnswBackend::new(384);

        backend.add(DocumentId::new(0), &vec![0.1; 384]).unwrap();
        backend.add(DocumentId::new(1), &vec![0.2; 384]).unwrap();

        backend.force_rebuild().unwrap();

        let stats = backend.stats();
        assert_eq!(stats.num_points, 2);
        assert_eq!(stats.embedding_dim, 384);
        assert!(stats.is_built);
        assert!(stats.estimated_memory_bytes > 0);
    }
}
