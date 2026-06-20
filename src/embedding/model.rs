//! Subword embedding model (FastText-style).
//!
//! Implements word embeddings with subword (character n-gram) enrichment,
//! providing robust representations for OOV words by composing subword vectors.

use super::bpe::{extract_subwords, hash_subword, BpeTokenizer};
use dashmap::DashMap;
use ndarray::{Array1, Array2, ArrayView1, Axis};
use ordered_float::OrderedFloat;
use std::collections::HashMap;
use std::sync::Arc;

#[cfg(feature = "serde-extras")]
use std::path::Path;

/// Default embedding dimension.
pub const DEFAULT_EMBEDDING_DIM: usize = 100;

/// Default number of subword buckets.
pub const DEFAULT_BUCKET_COUNT: usize = 2_000_000;

/// Default minimum subword length (character n-gram).
pub const DEFAULT_MIN_SUBWORD_LEN: usize = 3;

/// Default maximum subword length (character n-gram).
pub const DEFAULT_MAX_SUBWORD_LEN: usize = 6;

/// FastText-style subword embedding model.
///
/// Combines word-level embeddings with subword (character n-gram) embeddings:
/// - Known words: Use learned word embedding + average of subword embeddings
/// - OOV words: Use only average of subword embeddings
///
/// # Example
///
/// ```ignore
/// use libgrammstein::embedding::SubwordEmbedding;
///
/// // Load pre-trained model
/// let model = SubwordEmbedding::load("model.bin")?;
///
/// // Get embedding for a word
/// let vec = model.word_vector("hello");
///
/// // Find similar words
/// let similar = model.most_similar("king", 10);
/// ```
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct SubwordEmbedding {
    /// Word embeddings [vocab_size, dim].
    word_embeddings: Array2<f32>,

    /// Subword (character n-gram) embeddings [bucket_count, dim].
    subword_embeddings: Array2<f32>,

    /// Word to index mapping.
    word_to_idx: HashMap<String, usize>,

    /// Index to word mapping.
    idx_to_word: Vec<String>,

    /// Embedding dimension.
    dim: usize,

    /// Number of subword hash buckets.
    bucket_count: usize,

    /// Minimum subword (character n-gram) length.
    min_subword_len: usize,

    /// Maximum subword (character n-gram) length.
    max_subword_len: usize,

    /// Optional BPE tokenizer for tokenization.
    tokenizer: Option<BpeTokenizer>,

    /// Cache for computed word vectors (not serialized - reconstructed on load).
    #[serde(skip)]
    cache: Arc<DashMap<String, Array1<f32>>>,

    /// Maximum cache size.
    max_cache_size: usize,
}

impl SubwordEmbedding {
    /// Create a new subword embedding model.
    ///
    /// # Arguments
    ///
    /// * `vocab` - Vocabulary (word to index mapping)
    /// * `dim` - Embedding dimension
    /// * `bucket_count` - Number of subword hash buckets
    pub fn new(vocab: Vec<String>, dim: usize, bucket_count: usize) -> Self {
        let vocab_size = vocab.len();

        // Build word to index mapping
        let word_to_idx: HashMap<String, usize> = vocab
            .iter()
            .enumerate()
            .map(|(i, w)| (w.clone(), i))
            .collect();

        Self {
            word_embeddings: Array2::zeros((vocab_size, dim)),
            subword_embeddings: Array2::zeros((bucket_count, dim)),
            word_to_idx,
            idx_to_word: vocab,
            dim,
            bucket_count,
            min_subword_len: DEFAULT_MIN_SUBWORD_LEN,
            max_subword_len: DEFAULT_MAX_SUBWORD_LEN,
            tokenizer: None,
            cache: Arc::new(DashMap::new()),
            max_cache_size: 100_000,
        }
    }

    /// Create from pre-initialized embeddings.
    pub fn from_embeddings(
        word_embeddings: Array2<f32>,
        subword_embeddings: Array2<f32>,
        vocab: Vec<String>,
    ) -> Self {
        let dim = word_embeddings.ncols();
        let bucket_count = subword_embeddings.nrows();

        let word_to_idx: HashMap<String, usize> = vocab
            .iter()
            .enumerate()
            .map(|(i, w)| (w.clone(), i))
            .collect();

        Self {
            word_embeddings,
            subword_embeddings,
            word_to_idx,
            idx_to_word: vocab,
            dim,
            bucket_count,
            min_subword_len: DEFAULT_MIN_SUBWORD_LEN,
            max_subword_len: DEFAULT_MAX_SUBWORD_LEN,
            tokenizer: None,
            cache: Arc::new(DashMap::new()),
            max_cache_size: 100_000,
        }
    }

    /// Set subword length range.
    pub fn with_subword_range(mut self, min_len: usize, max_len: usize) -> Self {
        self.min_subword_len = min_len;
        self.max_subword_len = max_len;
        self
    }

    /// Set BPE tokenizer.
    pub fn with_tokenizer(mut self, tokenizer: BpeTokenizer) -> Self {
        self.tokenizer = Some(tokenizer);
        self
    }

    /// Set maximum cache size.
    pub fn with_cache_size(mut self, size: usize) -> Self {
        self.max_cache_size = size;
        self
    }

    /// Get embedding dimension.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get vocabulary size.
    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.idx_to_word.len()
    }

    /// Get the vocabulary in index order (`idx_to_word`).
    ///
    /// The slice is indexed by the same word ids the model's embedding rows use —
    /// resumed training relies on this to keep indices aligned.
    #[inline]
    pub fn vocab(&self) -> &[String] {
        &self.idx_to_word
    }

    /// Get number of subword buckets.
    #[inline]
    pub fn bucket_count(&self) -> usize {
        self.bucket_count
    }

    /// Check if word is in vocabulary.
    #[inline]
    pub fn contains(&self, word: &str) -> bool {
        self.word_to_idx.contains_key(word)
    }

    /// Get word index.
    #[inline]
    pub fn word_index(&self, word: &str) -> Option<usize> {
        self.word_to_idx.get(word).copied()
    }

    /// Get word by index.
    #[inline]
    pub fn index_to_word(&self, idx: usize) -> Option<&str> {
        self.idx_to_word.get(idx).map(|s| s.as_str())
    }

    /// Get word embedding by index (without subword enrichment).
    #[inline]
    pub fn embedding_by_index(&self, idx: usize) -> Option<ArrayView1<'_, f32>> {
        if idx < self.word_embeddings.nrows() {
            Some(self.word_embeddings.row(idx))
        } else {
            None
        }
    }

    /// Compute subword embedding for a word.
    ///
    /// Averages the embeddings of all character n-grams.
    fn subword_vector(&self, word: &str) -> Array1<f32> {
        let subwords = extract_subwords(word, self.min_subword_len, self.max_subword_len);

        if subwords.is_empty() {
            return Array1::zeros(self.dim);
        }

        let mut sum = Array1::zeros(self.dim);
        for subword in &subwords {
            let bucket = hash_subword(subword, self.bucket_count);
            sum = sum + self.subword_embeddings.row(bucket);
        }

        sum / subwords.len() as f32
    }

    /// Get word vector (with subword enrichment).
    ///
    /// For known words: word embedding + average subword embedding
    /// For OOV words: average subword embedding only
    pub fn word_vector(&self, word: &str) -> Array1<f32> {
        // Check cache first
        if let Some(cached) = self.cache.get(word) {
            return cached.clone();
        }

        let vector = if let Some(&idx) = self.word_to_idx.get(word) {
            // Known word: combine word embedding with subword embedding
            let word_emb = self.word_embeddings.row(idx).to_owned();
            let subword_emb = self.subword_vector(word);

            // Simple average of word and subword vectors
            (word_emb + subword_emb) / 2.0
        } else {
            // OOV word: use only subword embedding
            self.subword_vector(word)
        };

        // Update cache
        if self.cache.len() < self.max_cache_size {
            self.cache.insert(word.to_string(), vector.clone());
        }

        vector
    }

    /// Get word vector without caching.
    pub fn word_vector_uncached(&self, word: &str) -> Array1<f32> {
        if let Some(&idx) = self.word_to_idx.get(word) {
            let word_emb = self.word_embeddings.row(idx).to_owned();
            let subword_emb = self.subword_vector(word);
            (word_emb + subword_emb) / 2.0
        } else {
            self.subword_vector(word)
        }
    }

    /// Compute cosine similarity between two vectors.
    #[inline]
    fn cosine_similarity(a: ArrayView1<f32>, b: ArrayView1<f32>) -> f32 {
        let dot = a.dot(&b);
        let norm_a = a.dot(&a).sqrt();
        let norm_b = b.dot(&b).sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    /// Compute similarity between two words.
    pub fn similarity(&self, word1: &str, word2: &str) -> f32 {
        let v1 = self.word_vector(word1);
        let v2 = self.word_vector(word2);
        Self::cosine_similarity(v1.view(), v2.view())
    }

    /// Find most similar words to a query word.
    ///
    /// # Arguments
    ///
    /// * `word` - Query word
    /// * `k` - Number of results to return
    ///
    /// # Returns
    ///
    /// Vector of (word, similarity) pairs, sorted by descending similarity.
    pub fn most_similar(&self, word: &str, k: usize) -> Vec<(String, f32)> {
        let query_vec = self.word_vector(word);
        self.most_similar_to_vector(query_vec.view(), k, Some(word))
    }

    /// Find most similar words to a vector.
    ///
    /// # Arguments
    ///
    /// * `vector` - Query vector
    /// * `k` - Number of results to return
    /// * `exclude` - Optional word to exclude from results (e.g., the query word)
    pub fn most_similar_to_vector(
        &self,
        vector: ArrayView1<f32>,
        k: usize,
        exclude: Option<&str>,
    ) -> Vec<(String, f32)> {
        // Compute similarities to all words
        let mut similarities: Vec<(usize, OrderedFloat<f32>)> = self
            .word_embeddings
            .axis_iter(Axis(0))
            .enumerate()
            .filter_map(|(idx, word_vec)| {
                // Skip excluded word
                if let Some(ex) = exclude {
                    if self.idx_to_word.get(idx).map(|s| s.as_str()) == Some(ex) {
                        return None;
                    }
                }
                let sim = Self::cosine_similarity(vector, word_vec);
                Some((idx, OrderedFloat(sim)))
            })
            .collect();

        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.1.cmp(&a.1));

        // Take top k
        similarities
            .into_iter()
            .take(k)
            .map(|(idx, sim)| {
                let word = self.idx_to_word[idx].clone();
                (word, sim.0)
            })
            .collect()
    }

    /// Perform analogy computation: a is to b as c is to ?
    ///
    /// Computes: b - a + c, then finds most similar words.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // "king" - "man" + "woman" ≈ "queen"
    /// let results = model.analogy("man", "king", "woman", 5);
    /// ```
    pub fn analogy(&self, a: &str, b: &str, c: &str, k: usize) -> Vec<(String, f32)> {
        let va = self.word_vector(a);
        let vb = self.word_vector(b);
        let vc = self.word_vector(c);

        // result = b - a + c
        let result = &vb - &va + &vc;

        // Exclude the input words from results
        let mut results = self.most_similar_to_vector(result.view(), k + 3, None);
        results.retain(|(w, _)| w != a && w != b && w != c);
        results.truncate(k);
        results
    }

    /// Get sentence vector by averaging word vectors.
    pub fn sentence_vector(&self, words: &[&str]) -> Array1<f32> {
        if words.is_empty() {
            return Array1::zeros(self.dim);
        }

        let mut sum = Array1::zeros(self.dim);
        for word in words {
            sum = sum + self.word_vector(word);
        }

        sum / words.len() as f32
    }

    /// Update word embedding (for training).
    pub(crate) fn update_word_embedding(&mut self, idx: usize, delta: &Array1<f32>, lr: f32) {
        let mut row = self.word_embeddings.row_mut(idx);
        for (i, d) in delta.iter().enumerate() {
            row[i] += lr * d;
        }
    }

    /// Update subword embedding (for training).
    pub(crate) fn update_subword_embedding(&mut self, bucket: usize, delta: &Array1<f32>, lr: f32) {
        let mut row = self.subword_embeddings.row_mut(bucket);
        for (i, d) in delta.iter().enumerate() {
            row[i] += lr * d;
        }
    }

    /// Get mutable reference to word embeddings (for training).
    pub(crate) fn word_embeddings_mut(&mut self) -> &mut Array2<f32> {
        &mut self.word_embeddings
    }

    /// Get mutable reference to subword embeddings (for training).
    pub(crate) fn subword_embeddings_mut(&mut self) -> &mut Array2<f32> {
        &mut self.subword_embeddings
    }

    /// Clear the embedding cache.
    pub fn clear_cache(&self) {
        self.cache.clear();
    }
}

impl Clone for SubwordEmbedding {
    fn clone(&self) -> Self {
        Self {
            word_embeddings: self.word_embeddings.clone(),
            subword_embeddings: self.subword_embeddings.clone(),
            word_to_idx: self.word_to_idx.clone(),
            idx_to_word: self.idx_to_word.clone(),
            dim: self.dim,
            bucket_count: self.bucket_count,
            min_subword_len: self.min_subword_len,
            max_subword_len: self.max_subword_len,
            tokenizer: self.tokenizer.clone(),
            cache: Arc::new(DashMap::new()), // Don't clone cache
            max_cache_size: self.max_cache_size,
        }
    }
}

// Serialization support (requires bincode via serde-extras feature)
#[cfg(feature = "serde-extras")]
impl SubwordEmbedding {
    /// Save the embedding model to a binary file.
    ///
    /// Uses bincode for efficient binary serialization.
    ///
    /// # Example
    ///
    /// ```ignore
    /// model.save("embeddings.bin")?;
    /// ```
    pub fn save<P: AsRef<Path>>(&self, path: P) -> crate::Result<()> {
        let file = std::fs::File::create(path)?;
        let writer = std::io::BufWriter::new(file);
        bincode::serialize_into(writer, self)?;
        Ok(())
    }

    /// Load an embedding model from a binary file.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let model = SubwordEmbedding::load("embeddings.bin")?;
    /// ```
    pub fn load<P: AsRef<Path>>(path: P) -> crate::Result<Self> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let model = bincode::deserialize_from(reader)?;
        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_model() -> SubwordEmbedding {
        let vocab = vec![
            "king".to_string(),
            "queen".to_string(),
            "man".to_string(),
            "woman".to_string(),
        ];
        let mut model = SubwordEmbedding::new(vocab, 4, 1000);

        // Set up some simple embeddings for testing
        // These are not realistic but allow testing the API
        model.word_embeddings[[0, 0]] = 1.0; // king
        model.word_embeddings[[0, 1]] = 1.0;
        model.word_embeddings[[1, 0]] = 1.0; // queen
        model.word_embeddings[[1, 2]] = 1.0;
        model.word_embeddings[[2, 1]] = 1.0; // man
        model.word_embeddings[[2, 3]] = 1.0;
        model.word_embeddings[[3, 2]] = 1.0; // woman
        model.word_embeddings[[3, 3]] = 1.0;

        model
    }

    #[test]
    fn test_basic_properties() {
        let model = create_test_model();
        assert_eq!(model.dim(), 4);
        assert_eq!(model.vocab_size(), 4);
        assert_eq!(model.bucket_count(), 1000);
    }

    #[test]
    fn test_contains() {
        let model = create_test_model();
        assert!(model.contains("king"));
        assert!(model.contains("queen"));
        assert!(!model.contains("prince"));
    }

    #[test]
    fn test_word_index() {
        let model = create_test_model();
        assert_eq!(model.word_index("king"), Some(0));
        assert_eq!(model.word_index("queen"), Some(1));
        assert_eq!(model.word_index("prince"), None);
    }

    #[test]
    fn test_index_to_word() {
        let model = create_test_model();
        assert_eq!(model.index_to_word(0), Some("king"));
        assert_eq!(model.index_to_word(1), Some("queen"));
        assert_eq!(model.index_to_word(100), None);
    }

    #[test]
    fn test_word_vector() {
        let model = create_test_model();
        let vec = model.word_vector("king");
        assert_eq!(vec.len(), 4);
    }

    #[test]
    fn test_oov_word_vector() {
        let model = create_test_model();
        // OOV word should still produce a vector from subwords
        let vec = model.word_vector("prince");
        assert_eq!(vec.len(), 4);
    }

    #[test]
    fn test_similarity() {
        let model = create_test_model();
        let sim = model.similarity("king", "king");
        // Self-similarity should be high (approximately 1.0)
        assert!(sim > 0.9);
    }

    #[test]
    fn test_most_similar() {
        let model = create_test_model();
        let similar = model.most_similar("king", 2);
        assert_eq!(similar.len(), 2);
        // Results should not include the query word
        assert!(!similar.iter().any(|(w, _)| w == "king"));
    }

    #[test]
    fn test_sentence_vector() {
        let model = create_test_model();
        let vec = model.sentence_vector(&["king", "queen"]);
        assert_eq!(vec.len(), 4);
    }

    #[test]
    fn test_empty_sentence_vector() {
        let model = create_test_model();
        let vec = model.sentence_vector(&[]);
        assert_eq!(vec.len(), 4);
        assert!(vec.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_cache() {
        let model = create_test_model();

        // First call populates cache
        let vec1 = model.word_vector("king");

        // Second call should use cache
        let vec2 = model.word_vector("king");

        assert_eq!(vec1, vec2);
        assert!(model.cache.len() > 0);

        // Clear cache
        model.clear_cache();
        assert_eq!(model.cache.len(), 0);
    }

    #[test]
    fn test_clone() {
        let model = create_test_model();
        let cloned = model.clone();

        assert_eq!(model.dim(), cloned.dim());
        assert_eq!(model.vocab_size(), cloned.vocab_size());
        assert_eq!(model.word_embeddings, cloned.word_embeddings);
    }

    #[cfg(feature = "serde-extras")]
    #[test]
    fn test_embedding_save_load_roundtrip() {
        let model = create_test_model();
        let temp_file = tempfile::NamedTempFile::new().expect("Failed to create temp file");

        // Save the model
        model.save(temp_file.path()).expect("Failed to save model");

        // Verify file was created with content
        let metadata = std::fs::metadata(temp_file.path()).expect("Failed to get file metadata");
        assert!(metadata.len() > 0, "Saved model file should not be empty");

        // Load the model
        let loaded = SubwordEmbedding::load(temp_file.path()).expect("Failed to load model");

        // Verify properties match
        assert_eq!(model.dim(), loaded.dim());
        assert_eq!(model.vocab_size(), loaded.vocab_size());
        assert_eq!(model.bucket_count(), loaded.bucket_count());

        // Verify embeddings match
        assert_eq!(model.word_embeddings, loaded.word_embeddings);
        assert_eq!(model.subword_embeddings, loaded.subword_embeddings);

        // Verify vocabulary matches
        for word in &model.idx_to_word {
            assert!(
                loaded.contains(word),
                "Word '{}' should be in loaded model",
                word
            );
            assert_eq!(
                model.word_index(word),
                loaded.word_index(word),
                "Word indices should match for '{}'",
                word
            );
        }

        // Verify similarity computations match
        let orig_sim = model.similarity("king", "queen");
        let loaded_sim = loaded.similarity("king", "queen");
        assert!(
            (orig_sim - loaded_sim).abs() < 1e-6,
            "Similarity should match after roundtrip: {} vs {}",
            orig_sim,
            loaded_sim
        );
    }
}
