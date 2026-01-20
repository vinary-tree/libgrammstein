//! Code-specific neural embeddings for code similarity and retrieval.
//!
//! This module provides neural code embeddings using pre-trained code models:
//! - **CodeT5+**: Salesforce's encoder-decoder model, optimized for code understanding
//! - **UniXcoder**: Microsoft's unified cross-modal encoder for code
//! - **GraphCodeBERT**: Microsoft's model leveraging code structure (data flow)
//!
//! # Features
//!
//! - **Code Embedding**: Generate semantic embeddings for code snippets
//! - **Batch Processing**: Efficient batched GPU inference
//! - **Multi-Language**: Support for 10+ programming languages
//! - **Ensemble**: Combine multiple models for improved accuracy
//!
//! # Example
//!
//! ```ignore
//! use libgrammstein::neural::code::{CodeEmbedder, CodeT5Embedder, CodeLanguage};
//!
//! let embedder = CodeT5Embedder::load("Salesforce/codet5p-110m-embedding", Device::Cuda(0))?;
//! let embedding = embedder.embed_code("fn main() { println!(\"hello\"); }", CodeLanguage::Rust)?;
//! ```

mod codet5;
mod ensemble;
mod graphcodebert;
mod unixcoder;

pub use codet5::{CodeT5Embedder, CodeT5Config};
pub use ensemble::{EnsembleCodeEmbedder, EnsembleStrategy};
pub use graphcodebert::{GraphCodeBertEmbedder, GraphCodeBertConfig};
pub use unixcoder::{UniXcoderEmbedder, UniXcoderConfig};

use std::sync::Arc;

/// Result type for code embedding operations.
pub type Result<T> = std::result::Result<T, CodeEmbeddingError>;

/// Error type for code embedding operations.
#[derive(Debug, thiserror::Error)]
pub enum CodeEmbeddingError {
    /// Model loading failed.
    #[error("Failed to load model: {0}")]
    ModelLoad(String),

    /// Tokenization failed.
    #[error("Tokenization error: {0}")]
    Tokenization(String),

    /// Inference failed.
    #[error("Inference error: {0}")]
    Inference(String),

    /// ONNX Runtime error.
    #[error("ONNX Runtime error: {0}")]
    Onnx(String),

    /// Unsupported language.
    #[error("Unsupported language: {0}")]
    UnsupportedLanguage(String),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

impl From<ort::Error> for CodeEmbeddingError {
    fn from(e: ort::Error) -> Self {
        CodeEmbeddingError::Onnx(e.to_string())
    }
}

impl From<tokenizers::Error> for CodeEmbeddingError {
    fn from(e: tokenizers::Error) -> Self {
        CodeEmbeddingError::Tokenization(e.to_string())
    }
}

/// Programming language for code embeddings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CodeLanguage {
    /// Python
    Python,
    /// Java
    Java,
    /// JavaScript
    JavaScript,
    /// TypeScript
    TypeScript,
    /// Go
    Go,
    /// Ruby
    Ruby,
    /// PHP
    Php,
    /// C
    C,
    /// C++
    Cpp,
    /// C#
    CSharp,
    /// Rust
    Rust,
    /// Kotlin
    Kotlin,
    /// Scala
    Scala,
    /// Swift
    Swift,
    /// Haskell
    Haskell,
    /// OCaml
    OCaml,
    /// Elixir
    Elixir,
    /// Bash / Shell
    Bash,
    /// Rholang (F1R3FLY.io process calculus)
    Rholang,
    /// MeTTa (F1R3FLY.io meta-language)
    MeTTa,
    /// Unknown or unspecified language
    Unknown,
}

impl CodeLanguage {
    /// Get the language prefix for models that support it.
    pub fn prefix(&self) -> &'static str {
        match self {
            Self::Python => "<python>",
            Self::Java => "<java>",
            Self::JavaScript => "<javascript>",
            Self::TypeScript => "<typescript>",
            Self::Go => "<go>",
            Self::Ruby => "<ruby>",
            Self::Php => "<php>",
            Self::C => "<c>",
            Self::Cpp => "<cpp>",
            Self::CSharp => "<c_sharp>",
            Self::Rust => "<rust>",
            Self::Kotlin => "<kotlin>",
            Self::Scala => "<scala>",
            Self::Swift => "<swift>",
            Self::Haskell => "<haskell>",
            Self::OCaml => "<ocaml>",
            Self::Elixir => "<elixir>",
            Self::Bash => "<bash>",
            Self::Rholang => "<rholang>",
            Self::MeTTa => "<metta>",
            Self::Unknown => "",
        }
    }

    /// Get the language name as a string.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Python => "python",
            Self::Java => "java",
            Self::JavaScript => "javascript",
            Self::TypeScript => "typescript",
            Self::Go => "go",
            Self::Ruby => "ruby",
            Self::Php => "php",
            Self::C => "c",
            Self::Cpp => "cpp",
            Self::CSharp => "csharp",
            Self::Rust => "rust",
            Self::Kotlin => "kotlin",
            Self::Scala => "scala",
            Self::Swift => "swift",
            Self::Haskell => "haskell",
            Self::OCaml => "ocaml",
            Self::Elixir => "elixir",
            Self::Bash => "bash",
            Self::Rholang => "rholang",
            Self::MeTTa => "metta",
            Self::Unknown => "unknown",
        }
    }

    /// Parse language from file extension.
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "py" => Self::Python,
            "java" => Self::Java,
            "js" | "mjs" | "cjs" => Self::JavaScript,
            "ts" | "tsx" => Self::TypeScript,
            "go" => Self::Go,
            "rb" => Self::Ruby,
            "php" => Self::Php,
            "c" | "h" => Self::C,
            "cpp" | "cc" | "cxx" | "hpp" | "hxx" => Self::Cpp,
            "cs" => Self::CSharp,
            "rs" => Self::Rust,
            "kt" | "kts" => Self::Kotlin,
            "scala" | "sc" => Self::Scala,
            "swift" => Self::Swift,
            "hs" | "lhs" => Self::Haskell,
            "ml" | "mli" => Self::OCaml,
            "ex" | "exs" => Self::Elixir,
            "sh" | "bash" => Self::Bash,
            "rho" => Self::Rholang,
            "metta" | "mtt" => Self::MeTTa,
            _ => Self::Unknown,
        }
    }
}

/// Trait for code embedding models.
///
/// Implementations generate dense vector representations of code snippets
/// that capture semantic meaning for similarity search and retrieval.
pub trait CodeEmbedder: Send + Sync {
    /// Embed a single code snippet.
    ///
    /// # Arguments
    /// * `code` - The source code to embed
    /// * `language` - The programming language (optional for some models)
    ///
    /// # Returns
    /// A dense vector embedding of the code
    fn embed_code(&self, code: &str, language: CodeLanguage) -> Result<Vec<f32>>;

    /// Embed multiple code snippets in a batch.
    ///
    /// More efficient than calling `embed_code` repeatedly for multiple snippets.
    fn embed_code_batch(
        &self,
        codes: &[&str],
        languages: &[CodeLanguage],
    ) -> Result<Vec<Vec<f32>>>;

    /// Get the embedding dimension.
    fn embedding_dim(&self) -> usize;

    /// Get the model name.
    fn model_name(&self) -> &str;

    /// Get the maximum sequence length supported.
    fn max_sequence_length(&self) -> usize;

    /// Get supported languages.
    fn supported_languages(&self) -> &[CodeLanguage];

    /// Check if a language is supported.
    fn supports_language(&self, language: CodeLanguage) -> bool {
        let supported = self.supported_languages();
        supported.is_empty() || supported.contains(&language)
    }
}

/// Configuration for code embedding caching.
#[derive(Clone, Debug)]
pub struct CodeEmbeddingCacheConfig {
    /// Maximum number of embeddings to cache.
    pub max_entries: usize,
    /// Whether to hash code for cache keys (saves memory for long code).
    pub hash_keys: bool,
}

impl Default for CodeEmbeddingCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 10000,
            hash_keys: true,
        }
    }
}

/// Thread-safe cache for code embeddings.
pub struct CodeEmbeddingCache {
    cache: dashmap::DashMap<u64, Arc<[f32]>>,
    config: CodeEmbeddingCacheConfig,
}

impl CodeEmbeddingCache {
    /// Create a new cache with the given configuration.
    pub fn new(config: CodeEmbeddingCacheConfig) -> Self {
        Self {
            cache: dashmap::DashMap::with_capacity(config.max_entries),
            config,
        }
    }

    /// Get an embedding from the cache.
    pub fn get(&self, code: &str, language: CodeLanguage) -> Option<Arc<[f32]>> {
        let key = self.compute_key(code, language);
        self.cache.get(&key).map(|v| Arc::clone(&v))
    }

    /// Insert an embedding into the cache.
    pub fn insert(&self, code: &str, language: CodeLanguage, embedding: Vec<f32>) {
        // Simple eviction: if over capacity, remove a random entry
        if self.cache.len() >= self.config.max_entries {
            if let Some(entry) = self.cache.iter().next() {
                let key = *entry.key();
                drop(entry);
                self.cache.remove(&key);
            }
        }

        let key = self.compute_key(code, language);
        self.cache.insert(key, Arc::from(embedding.into_boxed_slice()));
    }

    /// Clear the cache.
    pub fn clear(&self) {
        self.cache.clear();
    }

    /// Get the number of cached embeddings.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Compute a cache key for code and language.
    fn compute_key(&self, code: &str, language: CodeLanguage) -> u64 {
        use crate::util::hash::{fnv1a, GXHASH_MIN_SIZE};
        use std::hash::{Hash, Hasher};

        if code.len() >= GXHASH_MIN_SIZE {
            let mut hasher = gxhash::GxHasher::default();
            code.hash(&mut hasher);
            language.hash(&mut hasher);
            hasher.finish()
        } else {
            // FNV-1a for short code, mix in language discriminant
            let mut hash = fnv1a(code.as_bytes());
            hash ^= language as u64;
            hash.wrapping_mul(0x100000001b3) // FNV_PRIME
        }
    }
}

/// Compute cosine similarity between two embeddings.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Embedding dimensions must match");

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// Normalize an embedding to unit length.
pub fn normalize_embedding(embedding: &mut [f32]) {
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in embedding.iter_mut() {
            *x /= norm;
        }
    }
}

/// Normalize an embedding and return a new vector.
pub fn normalize_embedding_clone(embedding: &[f32]) -> Vec<f32> {
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        embedding.iter().map(|x| x / norm).collect()
    } else {
        embedding.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_from_extension() {
        assert_eq!(CodeLanguage::from_extension("py"), CodeLanguage::Python);
        assert_eq!(CodeLanguage::from_extension("rs"), CodeLanguage::Rust);
        assert_eq!(CodeLanguage::from_extension("rho"), CodeLanguage::Rholang);
        assert_eq!(CodeLanguage::from_extension("metta"), CodeLanguage::MeTTa);
        assert_eq!(CodeLanguage::from_extension("xyz"), CodeLanguage::Unknown);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c) - 0.0).abs() < 1e-6);

        let d = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &d) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_embedding() {
        let mut embedding = vec![3.0, 4.0];
        normalize_embedding(&mut embedding);

        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_embedding_cache() {
        let cache = CodeEmbeddingCache::new(CodeEmbeddingCacheConfig {
            max_entries: 10,
            hash_keys: true,
        });

        let embedding = vec![1.0, 2.0, 3.0];
        cache.insert("fn main() {}", CodeLanguage::Rust, embedding.clone());

        let retrieved = cache.get("fn main() {}", CodeLanguage::Rust);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().as_ref(), &embedding[..]);

        // Different language should miss
        let missed = cache.get("fn main() {}", CodeLanguage::Python);
        assert!(missed.is_none());
    }
}
