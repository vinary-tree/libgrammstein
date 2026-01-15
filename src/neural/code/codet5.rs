//! CodeT5+ embedder using ONNX Runtime.
//!
//! CodeT5+ is a family of open code large language models from Salesforce,
//! trained on 9 programming languages. The embedding model generates
//! dense vectors for code similarity and retrieval.
//!
//! Model variants:
//! - `Salesforce/codet5p-110m-embedding`: 110M parameters, 256-dim embeddings
//! - `Salesforce/codet5p-220m`: 220M parameters, encoder-decoder
//!
//! Reference: https://huggingface.co/Salesforce/codet5p-110m-embedding
//!
//! # Example
//!
//! ```ignore
//! use libgrammstein::neural::code::{CodeT5Embedder, CodeT5Config, CodeLanguage, CodeEmbedder};
//!
//! let config = CodeT5Config::codet5p_110m_embedding("/path/to/model");
//! let embedder = CodeT5Embedder::load(config)?;
//! let embedding = embedder.embed_code("fn main() {}", CodeLanguage::Rust)?;
//! ```

use std::path::Path;
use std::sync::Arc;

use ndarray::Array2;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use parking_lot::Mutex;
use tokenizers::Tokenizer;

use super::{CodeEmbedder, CodeEmbeddingCache, CodeEmbeddingCacheConfig, CodeLanguage, Result, CodeEmbeddingError};

/// Configuration for CodeT5+ embedder.
#[derive(Clone, Debug)]
pub struct CodeT5Config {
    /// Path to ONNX model file.
    pub model_path: String,
    /// Path to tokenizer.json file.
    pub tokenizer_path: String,
    /// Maximum sequence length (default: 512).
    pub max_length: usize,
    /// Whether to use language prefix tokens.
    pub use_language_prefix: bool,
    /// Number of threads for inference.
    pub num_threads: usize,
    /// Graph optimization level (0-3).
    pub optimization_level: u8,
    /// Cache configuration (None to disable caching).
    pub cache_config: Option<CodeEmbeddingCacheConfig>,
    /// Whether to normalize embeddings.
    pub normalize: bool,
    /// Embedding dimension (detected from model or set explicitly).
    pub embedding_dim: Option<usize>,
}

impl Default for CodeT5Config {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            tokenizer_path: String::new(),
            max_length: 512,
            use_language_prefix: false,
            num_threads: 4,
            optimization_level: 3, // All optimizations
            cache_config: Some(CodeEmbeddingCacheConfig::default()),
            normalize: true,
            embedding_dim: None, // Auto-detect from model
        }
    }
}

impl CodeT5Config {
    /// Create config for the 110M embedding model.
    pub fn codet5p_110m_embedding(model_dir: impl AsRef<Path>) -> Self {
        let model_dir = model_dir.as_ref();
        Self {
            model_path: model_dir.join("model.onnx").to_string_lossy().to_string(),
            tokenizer_path: model_dir.join("tokenizer.json").to_string_lossy().to_string(),
            max_length: 512,
            use_language_prefix: false,
            embedding_dim: Some(256), // CodeT5+ 110M uses 256-dim embeddings
            ..Default::default()
        }
    }

    /// Map optimization level config to ort GraphOptimizationLevel.
    fn graph_optimization_level(&self) -> GraphOptimizationLevel {
        match self.optimization_level {
            0 => GraphOptimizationLevel::Disable,
            1 => GraphOptimizationLevel::Level1,
            2 => GraphOptimizationLevel::Level2,
            _ => GraphOptimizationLevel::Level3,
        }
    }
}

/// CodeT5+ embedder using ONNX Runtime for inference.
///
/// This embedder generates semantic embeddings for code snippets
/// using the CodeT5+ model family. It uses ONNX Runtime for efficient
/// CPU inference with configurable threading and optimization levels.
///
/// # Thread Safety
///
/// The embedder is thread-safe. The ONNX session is protected by a mutex
/// to ensure safe concurrent access across threads.
pub struct CodeT5Embedder {
    /// ONNX Runtime session (mutex-protected for thread safety).
    session: Arc<Mutex<Session>>,
    /// Tokenizer for converting code to tokens.
    tokenizer: Tokenizer,
    /// Configuration.
    config: CodeT5Config,
    /// Embedding cache (thread-safe via DashMap).
    cache: Option<CodeEmbeddingCache>,
    /// Embedding dimension.
    embedding_dim: usize,
    /// Name of the input_ids input node.
    input_ids_name: String,
    /// Name of the attention_mask input node.
    attention_mask_name: String,
    /// Name of the output node.
    output_name: String,
}

impl CodeT5Embedder {
    /// Load a CodeT5+ model from files.
    ///
    /// # Arguments
    /// * `config` - Configuration specifying model and tokenizer paths
    ///
    /// # Returns
    /// A new CodeT5Embedder instance
    ///
    /// # Errors
    /// Returns an error if the model or tokenizer cannot be loaded.
    pub fn load(config: CodeT5Config) -> Result<Self> {
        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&config.tokenizer_path)
            .map_err(|e| CodeEmbeddingError::ModelLoad(format!(
                "Failed to load tokenizer from {}: {}", config.tokenizer_path, e
            )))?;

        // Create ONNX session
        let session = Session::builder()
            .map_err(|e| CodeEmbeddingError::Onnx(format!("Failed to create session builder: {}", e)))?
            .with_optimization_level(config.graph_optimization_level())
            .map_err(|e| CodeEmbeddingError::Onnx(format!("Failed to set optimization level: {}", e)))?
            .with_intra_threads(config.num_threads)
            .map_err(|e| CodeEmbeddingError::Onnx(format!("Failed to set thread count: {}", e)))?
            .commit_from_file(&config.model_path)
            .map_err(|e| CodeEmbeddingError::ModelLoad(format!(
                "Failed to load ONNX model from {}: {}", config.model_path, e
            )))?;

        // Get input/output names from the model (iterate without cloning)
        // Find input names (typically "input_ids" and "attention_mask")
        let input_ids_name = session.inputs.iter()
            .find(|i| i.name.contains("input_ids") || i.name == "input_ids")
            .map(|i| i.name.to_string())
            .unwrap_or_else(|| "input_ids".to_string());

        let attention_mask_name = session.inputs.iter()
            .find(|i| i.name.contains("attention_mask") || i.name == "attention_mask")
            .map(|i| i.name.to_string())
            .unwrap_or_else(|| "attention_mask".to_string());

        // Find output name
        let output_name = session.outputs.first()
            .map(|o| o.name.to_string())
            .unwrap_or_else(|| "last_hidden_state".to_string());

        // Determine embedding dimension from config (output shape inspection not available in ort 2.0)
        let embedding_dim = config.embedding_dim.unwrap_or(256);

        // Create cache if configured
        let cache = config.cache_config.as_ref().map(|c| CodeEmbeddingCache::new(c.clone()));

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            tokenizer,
            config,
            cache,
            embedding_dim,
            input_ids_name,
            attention_mask_name,
            output_name,
        })
    }

    /// Load from a HuggingFace model directory (must contain model.onnx and tokenizer.json).
    pub fn from_directory(dir: impl AsRef<Path>) -> Result<Self> {
        let config = CodeT5Config::codet5p_110m_embedding(dir);
        Self::load(config)
    }

    /// Tokenize code for the model.
    fn tokenize(&self, code: &str, language: CodeLanguage) -> Result<(Vec<i64>, Vec<i64>)> {
        // Optionally prepend language prefix
        let input = if self.config.use_language_prefix && language != CodeLanguage::Unknown {
            format!("{} {}", language.prefix(), code)
        } else {
            code.to_string()
        };

        // Tokenize with truncation
        let encoding = self.tokenizer
            .encode(input, true)
            .map_err(|e| CodeEmbeddingError::Tokenization(e.to_string()))?;

        // Truncate to max length
        let max_len = self.config.max_length;
        let ids = encoding.get_ids();
        let attention = encoding.get_attention_mask();

        let (ids, attention) = if ids.len() > max_len {
            (
                ids[..max_len].iter().map(|&x| x as i64).collect(),
                attention[..max_len].iter().map(|&x| x as i64).collect(),
            )
        } else {
            (
                ids.iter().map(|&x| x as i64).collect(),
                attention.iter().map(|&x| x as i64).collect(),
            )
        };

        Ok((ids, attention))
    }

    /// Run ONNX inference to generate embedding.
    fn run_inference(&self, input_ids: Vec<i64>, attention_mask: Vec<i64>) -> Result<Vec<f32>> {
        use ort::value::Tensor;
        use std::borrow::Cow;

        let seq_len = input_ids.len();

        // Create input tensors as 2D arrays [batch=1, seq_len]
        let input_ids_array = Array2::from_shape_vec((1, seq_len), input_ids)
            .map_err(|e| CodeEmbeddingError::Inference(format!("Failed to create input_ids array: {}", e)))?;
        let attention_mask_array = Array2::from_shape_vec((1, seq_len), attention_mask)
            .map_err(|e| CodeEmbeddingError::Inference(format!("Failed to create attention_mask array: {}", e)))?;

        // Convert ndarray to ort Tensor values
        let input_ids_tensor = Tensor::from_array(input_ids_array)
            .map_err(|e| CodeEmbeddingError::Onnx(format!("Failed to create input_ids tensor: {}", e)))?;
        let attention_mask_tensor = Tensor::from_array(attention_mask_array)
            .map_err(|e| CodeEmbeddingError::Onnx(format!("Failed to create attention_mask tensor: {}", e)))?;

        // Build inputs as Vec of named values (ort 2.0 API)
        let inputs: Vec<(Cow<'_, str>, ort::value::DynValue)> = vec![
            (Cow::Owned(self.input_ids_name.clone()), input_ids_tensor.into_dyn()),
            (Cow::Owned(self.attention_mask_name.clone()), attention_mask_tensor.into_dyn()),
        ];

        // Run inference with mutex lock
        let mut session = self.session.lock();
        let outputs = session.run(inputs)
            .map_err(|e| CodeEmbeddingError::Inference(format!("Inference failed: {}", e)))?;

        // Extract output tensor
        let output = outputs.get(&self.output_name)
            .ok_or_else(|| CodeEmbeddingError::Inference(format!(
                "Output '{}' not found in model outputs", self.output_name
            )))?;

        // Extract as f32 array - ort 2.0 try_extract_tensor returns (&Shape, &[T])
        let (shape, data) = output.try_extract_tensor::<f32>()
            .map_err(|e| CodeEmbeddingError::Inference(format!("Failed to extract output tensor: {}", e)))?;

        // Convert shape to Vec<usize> for easier handling
        let shape_dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();

        // Handle different output shapes:
        // - [batch, hidden_dim]: Direct embedding
        // - [batch, seq_len, hidden_dim]: Need to pool (mean or CLS token)
        let embedding: Vec<f32> = match shape_dims.len() {
            2 => {
                // [batch, hidden_dim] - direct embedding
                data.to_vec()
            }
            3 => {
                // [batch, seq_len, hidden_dim] - need to pool
                // Use mean pooling over sequence dimension
                let batch_size = shape_dims[0];
                let seq_length = shape_dims[1];
                let hidden_dim = shape_dims[2];

                if batch_size != 1 {
                    return Err(CodeEmbeddingError::Inference(
                        "Unexpected batch size > 1".to_string()
                    ));
                }

                // Mean pooling: average over sequence dimension
                let mut embedding = vec![0.0f32; hidden_dim];
                for seq_idx in 0..seq_length {
                    for dim_idx in 0..hidden_dim {
                        embedding[dim_idx] += data[seq_idx * hidden_dim + dim_idx];
                    }
                }
                for val in &mut embedding {
                    *val /= seq_length as f32;
                }
                embedding
            }
            _ => {
                return Err(CodeEmbeddingError::Inference(format!(
                    "Unexpected output shape: {:?}", shape_dims
                )));
            }
        };

        Ok(embedding)
    }

    /// Get the configuration.
    pub fn config(&self) -> &CodeT5Config {
        &self.config
    }

    /// Clear the embedding cache.
    pub fn clear_cache(&self) {
        if let Some(cache) = &self.cache {
            cache.clear();
        }
    }

    /// Get cache statistics.
    pub fn cache_stats(&self) -> Option<usize> {
        self.cache.as_ref().map(|c| c.len())
    }

    /// Get the input node names detected from the model.
    pub fn input_names(&self) -> (&str, &str) {
        (&self.input_ids_name, &self.attention_mask_name)
    }

    /// Get the output node name detected from the model.
    pub fn output_name(&self) -> &str {
        &self.output_name
    }
}

impl CodeEmbedder for CodeT5Embedder {
    fn embed_code(&self, code: &str, language: CodeLanguage) -> Result<Vec<f32>> {
        // Check cache first
        if let Some(cache) = &self.cache {
            if let Some(embedding) = cache.get(code, language) {
                return Ok(embedding.to_vec());
            }
        }

        // Tokenize
        let (input_ids, attention_mask) = self.tokenize(code, language)?;

        // Run ONNX inference
        let mut embedding = self.run_inference(input_ids, attention_mask)?;

        // Normalize if configured
        if self.config.normalize {
            super::normalize_embedding(&mut embedding);
        }

        // Cache the result
        if let Some(cache) = &self.cache {
            cache.insert(code, language, embedding.clone());
        }

        Ok(embedding)
    }

    fn embed_code_batch(
        &self,
        codes: &[&str],
        languages: &[CodeLanguage],
    ) -> Result<Vec<Vec<f32>>> {
        if codes.is_empty() {
            return Ok(vec![]);
        }

        // For now, process sequentially (batch inference optimization can be added later)
        // The mutex-based session access makes true batching complex
        codes
            .iter()
            .zip(languages.iter().chain(std::iter::repeat(&CodeLanguage::Unknown)))
            .map(|(code, lang)| self.embed_code(code, *lang))
            .collect()
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    fn model_name(&self) -> &str {
        "CodeT5+"
    }

    fn max_sequence_length(&self) -> usize {
        self.config.max_length
    }

    fn supported_languages(&self) -> &[CodeLanguage] {
        // CodeT5+ was trained on these languages
        &[
            CodeLanguage::Python,
            CodeLanguage::Java,
            CodeLanguage::JavaScript,
            CodeLanguage::Go,
            CodeLanguage::Ruby,
            CodeLanguage::Php,
            CodeLanguage::C,
            CodeLanguage::Cpp,
            CodeLanguage::CSharp,
        ]
    }
}

// Implement Send + Sync manually since we use Arc<Mutex<Session>>
unsafe impl Send for CodeT5Embedder {}
unsafe impl Sync for CodeT5Embedder {}

impl std::fmt::Debug for CodeT5Embedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CodeT5Embedder")
            .field("model_path", &self.config.model_path)
            .field("embedding_dim", &self.embedding_dim)
            .field("max_length", &self.config.max_length)
            .field("input_ids_name", &self.input_ids_name)
            .field("attention_mask_name", &self.attention_mask_name)
            .field("output_name", &self.output_name)
            .field("cache_size", &self.cache.as_ref().map(|c| c.len()))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = CodeT5Config::default();
        assert_eq!(config.max_length, 512);
        assert!(!config.use_language_prefix);
        assert!(config.normalize);
        assert_eq!(config.num_threads, 4);
        assert_eq!(config.optimization_level, 3);
    }

    #[test]
    fn test_config_from_directory() {
        let config = CodeT5Config::codet5p_110m_embedding("/tmp/codet5p");
        assert!(config.model_path.contains("model.onnx"));
        assert!(config.tokenizer_path.contains("tokenizer.json"));
        assert_eq!(config.embedding_dim, Some(256));
    }

    #[test]
    fn test_graph_optimization_levels() {
        let mut config = CodeT5Config::default();

        config.optimization_level = 0;
        assert!(matches!(config.graph_optimization_level(), GraphOptimizationLevel::Disable));

        config.optimization_level = 1;
        assert!(matches!(config.graph_optimization_level(), GraphOptimizationLevel::Level1));

        config.optimization_level = 2;
        assert!(matches!(config.graph_optimization_level(), GraphOptimizationLevel::Level2));

        config.optimization_level = 3;
        assert!(matches!(config.graph_optimization_level(), GraphOptimizationLevel::Level3));

        config.optimization_level = 99;
        assert!(matches!(config.graph_optimization_level(), GraphOptimizationLevel::Level3));
    }
}
