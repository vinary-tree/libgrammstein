//! Equation retrieval-augmented generation (RAG) for LaTeX correction.
//!
//! This module provides retrieval of similar equations from a reference corpus
//! to aid in correction decisions.

use std::collections::HashMap;

/// Configuration for equation retrieval.
#[derive(Debug, Clone)]
pub struct RetrievalConfig {
    /// Number of similar equations to retrieve.
    pub top_k: usize,
    /// Minimum similarity threshold.
    pub min_similarity: f32,
    /// Whether to use approximate nearest neighbor search.
    pub use_ann: bool,
    /// Maximum number of equations in the index.
    pub max_index_size: usize,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            top_k: 5,
            min_similarity: 0.5,
            use_ann: false,
            max_index_size: 1_000_000,
        }
    }
}

/// A document representing an equation in the corpus.
#[derive(Debug, Clone)]
pub struct EquationDocument {
    /// Unique identifier.
    pub id: String,
    /// The equation LaTeX source.
    pub latex: String,
    /// The embedding vector.
    pub embedding: Vec<f32>,
    /// Source document (e.g., arXiv paper ID).
    pub source: Option<String>,
    /// Equation label or reference.
    pub label: Option<String>,
    /// Mathematical domain (e.g., "calculus", "linear_algebra").
    pub domain: Option<String>,
    /// Additional metadata.
    pub metadata: HashMap<String, String>,
}

impl EquationDocument {
    /// Create a new equation document.
    pub fn new(id: String, latex: String, embedding: Vec<f32>) -> Self {
        Self {
            id,
            latex,
            embedding,
            source: None,
            label: None,
            domain: None,
            metadata: HashMap::new(),
        }
    }

    /// Set the source document.
    pub fn with_source(mut self, source: String) -> Self {
        self.source = Some(source);
        self
    }

    /// Set the equation label.
    pub fn with_label(mut self, label: String) -> Self {
        self.label = Some(label);
        self
    }

    /// Set the mathematical domain.
    pub fn with_domain(mut self, domain: String) -> Self {
        self.domain = Some(domain);
        self
    }

    /// Add metadata.
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Result of an equation retrieval query.
#[derive(Debug, Clone)]
pub struct RetrievalResult {
    /// The retrieved equation document.
    pub document: EquationDocument,
    /// Similarity score (0.0 to 1.0).
    pub similarity: f32,
    /// Rank in the result set.
    pub rank: usize,
}

impl RetrievalResult {
    /// Create a new retrieval result.
    pub fn new(document: EquationDocument, similarity: f32, rank: usize) -> Self {
        Self {
            document,
            similarity,
            rank,
        }
    }
}

/// Index for equation retrieval.
pub struct EquationRagIndex {
    /// All indexed equations.
    equations: Vec<EquationDocument>,
    /// Index by ID for fast lookup.
    id_index: HashMap<String, usize>,
    /// Index by domain for filtered retrieval.
    domain_index: HashMap<String, Vec<usize>>,
    /// Configuration.
    config: RetrievalConfig,
    /// Embedding dimension.
    dimension: usize,
}

impl EquationRagIndex {
    /// Create a new empty index.
    pub fn new(dimension: usize) -> Self {
        Self::with_config(dimension, RetrievalConfig::default())
    }

    /// Create an index with custom configuration.
    pub fn with_config(dimension: usize, config: RetrievalConfig) -> Self {
        Self {
            equations: Vec::new(),
            id_index: HashMap::new(),
            domain_index: HashMap::new(),
            config,
            dimension,
        }
    }

    /// Add an equation to the index.
    pub fn add(&mut self, doc: EquationDocument) -> Result<(), &'static str> {
        if doc.embedding.len() != self.dimension {
            return Err("Embedding dimension mismatch");
        }

        if self.equations.len() >= self.config.max_index_size {
            return Err("Index size limit reached");
        }

        let idx = self.equations.len();
        self.id_index.insert(doc.id.clone(), idx);

        if let Some(ref domain) = doc.domain {
            self.domain_index
                .entry(domain.clone())
                .or_default()
                .push(idx);
        }

        self.equations.push(doc);
        Ok(())
    }

    /// Add multiple equations to the index.
    pub fn add_batch(&mut self, docs: Vec<EquationDocument>) -> Result<usize, &'static str> {
        let mut added = 0;
        for doc in docs {
            if self.add(doc).is_ok() {
                added += 1;
            }
        }
        Ok(added)
    }

    /// Retrieve similar equations for a query embedding.
    pub fn retrieve(&self, query_embedding: &[f32]) -> Vec<RetrievalResult> {
        self.retrieve_with_filter(query_embedding, None)
    }

    /// Retrieve similar equations with domain filter.
    pub fn retrieve_in_domain(
        &self,
        query_embedding: &[f32],
        domain: &str,
    ) -> Vec<RetrievalResult> {
        self.retrieve_with_filter(query_embedding, Some(domain))
    }

    /// Retrieve similar equations with optional domain filter.
    fn retrieve_with_filter(
        &self,
        query_embedding: &[f32],
        domain: Option<&str>,
    ) -> Vec<RetrievalResult> {
        if query_embedding.len() != self.dimension {
            return Vec::new();
        }

        // Get candidate indices
        let candidates: Vec<usize> = if let Some(domain) = domain {
            self.domain_index.get(domain).cloned().unwrap_or_default()
        } else {
            (0..self.equations.len()).collect()
        };

        // Compute similarities
        let mut results: Vec<(usize, f32)> = candidates
            .iter()
            .map(|&idx| {
                let similarity = cosine_similarity(query_embedding, &self.equations[idx].embedding);
                (idx, similarity)
            })
            .filter(|(_, sim)| *sim >= self.config.min_similarity)
            .collect();

        // Sort by similarity (descending)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top-k
        results.truncate(self.config.top_k);

        // Convert to RetrievalResult
        results
            .into_iter()
            .enumerate()
            .map(|(rank, (idx, similarity))| {
                RetrievalResult::new(self.equations[idx].clone(), similarity, rank)
            })
            .collect()
    }

    /// Get an equation by ID.
    pub fn get(&self, id: &str) -> Option<&EquationDocument> {
        self.id_index.get(id).map(|&idx| &self.equations[idx])
    }

    /// Get the number of equations in the index.
    pub fn len(&self) -> usize {
        self.equations.len()
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.equations.is_empty()
    }

    /// Get all unique domains in the index.
    pub fn domains(&self) -> Vec<&str> {
        self.domain_index.keys().map(|s| s.as_str()).collect()
    }

    /// Get the count of equations in a domain.
    pub fn domain_count(&self, domain: &str) -> usize {
        self.domain_index.get(domain).map(|v| v.len()).unwrap_or(0)
    }

    /// Get the embedding dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the configuration.
    pub fn config(&self) -> &RetrievalConfig {
        &self.config
    }

    /// Clear the index.
    pub fn clear(&mut self) {
        self.equations.clear();
        self.id_index.clear();
        self.domain_index.clear();
    }
}

/// Retriever that wraps an index with additional functionality.
pub struct EquationRetriever {
    index: EquationRagIndex,
    /// Cache of recent queries.
    cache: HashMap<Vec<u8>, Vec<RetrievalResult>>,
    /// Maximum cache size.
    max_cache_size: usize,
}

impl EquationRetriever {
    /// Create a new retriever with the given index.
    pub fn new(index: EquationRagIndex) -> Self {
        Self {
            index,
            cache: HashMap::new(),
            max_cache_size: 1000,
        }
    }

    /// Retrieve similar equations.
    pub fn retrieve(&mut self, query_embedding: &[f32]) -> &[RetrievalResult] {
        let cache_key = embedding_to_cache_key(query_embedding);

        if !self.cache.contains_key(&cache_key) {
            let results = self.index.retrieve(query_embedding);

            // Evict old entries if cache is full
            if self.cache.len() >= self.max_cache_size {
                self.cache.clear();
            }

            self.cache.insert(cache_key.clone(), results);
        }

        self.cache.get(&cache_key).unwrap()
    }

    /// Retrieve with domain filter (not cached).
    pub fn retrieve_in_domain(
        &self,
        query_embedding: &[f32],
        domain: &str,
    ) -> Vec<RetrievalResult> {
        self.index.retrieve_in_domain(query_embedding, domain)
    }

    /// Get a reference to the underlying index.
    pub fn index(&self) -> &EquationRagIndex {
        &self.index
    }

    /// Get a mutable reference to the underlying index.
    pub fn index_mut(&mut self) -> &mut EquationRagIndex {
        self.cache.clear(); // Invalidate cache when index changes
        &mut self.index
    }

    /// Clear the cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

/// Compute cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom > 0.0 {
        dot / denom
    } else {
        0.0
    }
}

/// Convert an embedding to a cache key.
fn embedding_to_cache_key(embedding: &[f32]) -> Vec<u8> {
    // Quantize to reduce cache key size and improve hit rate
    embedding
        .iter()
        .map(|&x| (x * 127.0).clamp(-128.0, 127.0) as i8 as u8)
        .collect()
}

/// Builder for equation RAG index.
pub struct EquationRagIndexBuilder {
    dimension: usize,
    config: RetrievalConfig,
    equations: Vec<EquationDocument>,
}

impl EquationRagIndexBuilder {
    /// Create a new builder.
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            config: RetrievalConfig::default(),
            equations: Vec::new(),
        }
    }

    /// Set the configuration.
    pub fn with_config(mut self, config: RetrievalConfig) -> Self {
        self.config = config;
        self
    }

    /// Set top-k retrieval count.
    pub fn top_k(mut self, k: usize) -> Self {
        self.config.top_k = k;
        self
    }

    /// Set minimum similarity threshold.
    pub fn min_similarity(mut self, threshold: f32) -> Self {
        self.config.min_similarity = threshold;
        self
    }

    /// Add an equation.
    pub fn add_equation(mut self, doc: EquationDocument) -> Self {
        self.equations.push(doc);
        self
    }

    /// Add multiple equations.
    pub fn add_equations(mut self, docs: Vec<EquationDocument>) -> Self {
        self.equations.extend(docs);
        self
    }

    /// Build the index.
    pub fn build(self) -> Result<EquationRagIndex, &'static str> {
        let mut index = EquationRagIndex::with_config(self.dimension, self.config);
        index.add_batch(self.equations)?;
        Ok(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_embedding(seed: u32, dim: usize) -> Vec<f32> {
        let mut embedding = vec![0.0f32; dim];
        for (i, v) in embedding.iter_mut().enumerate() {
            *v = ((seed as f32 * 0.1 + i as f32 * 0.01) % 1.0) * 2.0 - 1.0;
        }
        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        for v in &mut embedding {
            *v /= norm;
        }
        embedding
    }

    #[test]
    fn test_add_and_retrieve() {
        let mut index = EquationRagIndex::new(8);

        let doc1 = EquationDocument::new(
            "eq1".to_string(),
            "x^2".to_string(),
            create_test_embedding(1, 8),
        );
        let doc2 = EquationDocument::new(
            "eq2".to_string(),
            "y^2".to_string(),
            create_test_embedding(2, 8),
        );

        index.add(doc1).unwrap();
        index.add(doc2).unwrap();

        assert_eq!(index.len(), 2);

        let query = create_test_embedding(1, 8);
        let results = index.retrieve(&query);

        assert!(!results.is_empty());
        assert_eq!(results[0].document.id, "eq1");
    }

    #[test]
    fn test_domain_filter() {
        let mut index = EquationRagIndex::new(8);

        let doc1 = EquationDocument::new(
            "eq1".to_string(),
            "x^2".to_string(),
            create_test_embedding(1, 8),
        )
        .with_domain("algebra".to_string());

        let doc2 = EquationDocument::new(
            "eq2".to_string(),
            "\\int x dx".to_string(),
            create_test_embedding(2, 8),
        )
        .with_domain("calculus".to_string());

        index.add(doc1).unwrap();
        index.add(doc2).unwrap();

        assert_eq!(index.domain_count("algebra"), 1);
        assert_eq!(index.domain_count("calculus"), 1);

        let query = create_test_embedding(1, 8);
        let results = index.retrieve_in_domain(&query, "algebra");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].document.domain, Some("algebra".to_string()));
    }

    #[test]
    fn test_similarity_threshold() {
        let config = RetrievalConfig {
            min_similarity: 0.99, // Very high threshold
            ..Default::default()
        };
        let mut index = EquationRagIndex::with_config(8, config);

        // Create orthogonal embeddings to ensure low similarity
        let mut doc_emb = vec![0.0f32; 8];
        doc_emb[0] = 1.0; // Unit vector along first axis

        let doc = EquationDocument::new("eq1".to_string(), "x^2".to_string(), doc_emb);
        index.add(doc).unwrap();

        // Query with an orthogonal embedding (unit vector along second axis)
        let mut query = vec![0.0f32; 8];
        query[1] = 1.0;
        let results = index.retrieve(&query);

        // Should be empty due to high threshold and orthogonal vectors
        assert!(results.is_empty());
    }

    #[test]
    fn test_builder() {
        let index = EquationRagIndexBuilder::new(8)
            .top_k(3)
            .min_similarity(0.5)
            .add_equation(EquationDocument::new(
                "eq1".to_string(),
                "x^2".to_string(),
                create_test_embedding(1, 8),
            ))
            .build()
            .unwrap();

        assert_eq!(index.len(), 1);
        assert_eq!(index.config().top_k, 3);
    }

    #[test]
    fn test_retriever_cache() {
        let mut index = EquationRagIndex::new(8);
        index
            .add(EquationDocument::new(
                "eq1".to_string(),
                "x^2".to_string(),
                create_test_embedding(1, 8),
            ))
            .unwrap();

        let mut retriever = EquationRetriever::new(index);

        let query = create_test_embedding(1, 8);

        // First call
        let len1 = retriever.retrieve(&query).len();
        assert!(len1 > 0);

        // Second call (should use cache)
        let len2 = retriever.retrieve(&query).len();
        assert_eq!(len1, len2);

        retriever.clear_cache();
    }
}
