//! GraphCodeBERT embedder using ONNX Runtime.
//!
//! GraphCodeBERT is a pre-trained model from Microsoft that incorporates
//! code structure through data flow graphs. It achieves strong performance
//! on code search, clone detection, and code completion tasks.
//!
//! Model: microsoft/graphcodebert-base
//! - 125M parameters
//! - 768-dim embeddings
//! - 512 max sequence length
//! - Leverages data flow information
//!
//! Reference: https://huggingface.co/microsoft/graphcodebert-base

use std::path::Path;
use std::sync::Arc;

use ndarray::Array2;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use parking_lot::Mutex;
use tokenizers::Tokenizer;

use super::{
    CodeEmbedder, CodeEmbeddingCache, CodeEmbeddingCacheConfig, CodeEmbeddingError, CodeLanguage,
    Result,
};

/// Configuration for GraphCodeBERT embedder.
#[derive(Clone, Debug)]
pub struct GraphCodeBertConfig {
    /// Path to ONNX model file.
    pub model_path: String,
    /// Path to tokenizer.json file.
    pub tokenizer_path: String,
    /// Maximum sequence length (default: 512).
    pub max_length: usize,
    /// Number of threads for inference.
    pub num_threads: usize,
    /// Graph optimization level (0-3).
    pub optimization_level: u8,
    /// Cache configuration (None to disable caching).
    pub cache_config: Option<CodeEmbeddingCacheConfig>,
    /// Whether to normalize embeddings.
    pub normalize: bool,
    /// Embedding dimension (768 for graphcodebert-base).
    pub embedding_dim: usize,
    /// Whether to use data flow input (if model supports it).
    pub use_data_flow: bool,
}

impl Default for GraphCodeBertConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            tokenizer_path: String::new(),
            max_length: 512,
            num_threads: 4,
            optimization_level: 3,
            cache_config: Some(CodeEmbeddingCacheConfig::default()),
            normalize: true,
            embedding_dim: 768,
            use_data_flow: false, // Most ONNX exports don't include DFG inputs
        }
    }
}

impl GraphCodeBertConfig {
    /// Create config for graphcodebert-base model.
    pub fn graphcodebert_base(model_dir: impl AsRef<Path>) -> Self {
        let model_dir = model_dir.as_ref();
        Self {
            model_path: model_dir.join("model.onnx").to_string_lossy().to_string(),
            tokenizer_path: model_dir
                .join("tokenizer.json")
                .to_string_lossy()
                .to_string(),
            embedding_dim: 768,
            ..Default::default()
        }
    }

    fn graph_optimization_level(&self) -> GraphOptimizationLevel {
        match self.optimization_level {
            0 => GraphOptimizationLevel::Disable,
            1 => GraphOptimizationLevel::Level1,
            2 => GraphOptimizationLevel::Level2,
            _ => GraphOptimizationLevel::Level3,
        }
    }
}

/// GraphCodeBERT embedder using ONNX Runtime for inference.
///
/// GraphCodeBERT excels at tasks where code structure matters:
/// - Code search with structural understanding
/// - Clone detection (including semantic clones)
/// - Code completion
/// - Variable misuse detection
///
/// The model was pre-trained on data flow graphs from source code,
/// making it particularly effective at understanding variable relationships.
///
/// # Thread Safety
///
/// The embedder is thread-safe via mutex-protected ONNX session.
pub struct GraphCodeBertEmbedder {
    session: Arc<Mutex<Session>>,
    tokenizer: Tokenizer,
    config: GraphCodeBertConfig,
    cache: Option<CodeEmbeddingCache>,
    embedding_dim: usize,
    input_ids_name: String,
    attention_mask_name: String,
    output_name: String,
    /// Whether the model has position_ids input
    has_position_ids: bool,
}

impl GraphCodeBertEmbedder {
    /// Load a GraphCodeBERT model from files.
    pub fn load(config: GraphCodeBertConfig) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(&config.tokenizer_path).map_err(|e| {
            CodeEmbeddingError::ModelLoad(format!(
                "Failed to load tokenizer from {}: {}",
                config.tokenizer_path, e
            ))
        })?;

        let session = Session::builder()
            .map_err(|e| {
                CodeEmbeddingError::Onnx(format!("Failed to create session builder: {}", e))
            })?
            .with_optimization_level(config.graph_optimization_level())
            .map_err(|e| {
                CodeEmbeddingError::Onnx(format!("Failed to set optimization level: {}", e))
            })?
            .with_intra_threads(config.num_threads)
            .map_err(|e| CodeEmbeddingError::Onnx(format!("Failed to set thread count: {}", e)))?
            .commit_from_file(&config.model_path)
            .map_err(|e| {
                CodeEmbeddingError::ModelLoad(format!(
                    "Failed to load ONNX model from {}: {}",
                    config.model_path, e
                ))
            })?;

        // Detect input/output names
        let input_ids_name = session
            .inputs()
            .iter()
            .find(|i| i.name().contains("input_ids"))
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "input_ids".to_string());

        let attention_mask_name = session
            .inputs()
            .iter()
            .find(|i| i.name().contains("attention_mask"))
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "attention_mask".to_string());

        let has_position_ids = session
            .inputs()
            .iter()
            .any(|i| i.name().contains("position_ids"));

        let output_name = session
            .outputs()
            .first()
            .map(|o| o.name().to_string())
            .unwrap_or_else(|| "last_hidden_state".to_string());

        let embedding_dim = config.embedding_dim;
        let cache = config
            .cache_config
            .as_ref()
            .map(|c| CodeEmbeddingCache::new(c.clone()));

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            tokenizer,
            config,
            cache,
            embedding_dim,
            input_ids_name,
            attention_mask_name,
            output_name,
            has_position_ids,
        })
    }

    /// Load from a HuggingFace model directory.
    pub fn from_directory(dir: impl AsRef<Path>) -> Result<Self> {
        let config = GraphCodeBertConfig::graphcodebert_base(dir);
        Self::load(config)
    }

    fn tokenize(&self, code: &str) -> Result<(Vec<i64>, Vec<i64>)> {
        let encoding = self
            .tokenizer
            .encode(code, true)
            .map_err(|e| CodeEmbeddingError::Tokenization(e.to_string()))?;

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

    fn run_inference(&self, input_ids: Vec<i64>, attention_mask: Vec<i64>) -> Result<Vec<f32>> {
        use std::borrow::Cow;

        let seq_len = input_ids.len();

        let input_ids_array =
            Array2::from_shape_vec((1, seq_len), input_ids.clone()).map_err(|e| {
                CodeEmbeddingError::Inference(format!("Failed to create input_ids array: {}", e))
            })?;
        let attention_mask_array =
            Array2::from_shape_vec((1, seq_len), attention_mask).map_err(|e| {
                CodeEmbeddingError::Inference(format!(
                    "Failed to create attention_mask array: {}",
                    e
                ))
            })?;

        let input_ids_tensor = Tensor::from_array(input_ids_array).map_err(|e| {
            CodeEmbeddingError::Onnx(format!("Failed to create input_ids tensor: {}", e))
        })?;
        let attention_mask_tensor = Tensor::from_array(attention_mask_array).map_err(|e| {
            CodeEmbeddingError::Onnx(format!("Failed to create attention_mask tensor: {}", e))
        })?;

        let mut inputs: Vec<(Cow<'_, str>, ort::value::DynValue)> = vec![
            (
                Cow::Owned(self.input_ids_name.clone()),
                input_ids_tensor.into_dyn(),
            ),
            (
                Cow::Owned(self.attention_mask_name.clone()),
                attention_mask_tensor.into_dyn(),
            ),
        ];

        // Add position_ids if model requires it
        if self.has_position_ids {
            let position_ids: Vec<i64> = (0..seq_len as i64).collect();
            let position_ids_array =
                Array2::from_shape_vec((1, seq_len), position_ids).map_err(|e| {
                    CodeEmbeddingError::Inference(format!(
                        "Failed to create position_ids array: {}",
                        e
                    ))
                })?;
            let position_ids_tensor = Tensor::from_array(position_ids_array).map_err(|e| {
                CodeEmbeddingError::Onnx(format!("Failed to create position_ids tensor: {}", e))
            })?;
            inputs.push((
                Cow::Borrowed("position_ids"),
                position_ids_tensor.into_dyn(),
            ));
        }

        let mut session = self.session.lock();
        let outputs = session
            .run(inputs)
            .map_err(|e| CodeEmbeddingError::Inference(format!("Inference failed: {}", e)))?;

        let output = outputs.get(&self.output_name).ok_or_else(|| {
            CodeEmbeddingError::Inference(format!("Output '{}' not found", self.output_name))
        })?;

        let (shape, data) = output.try_extract_tensor::<f32>().map_err(|e| {
            CodeEmbeddingError::Inference(format!("Failed to extract tensor: {}", e))
        })?;

        let shape_dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();

        // GraphCodeBERT outputs [batch, seq_len, hidden_dim]
        // Use CLS token embedding (first token)
        let embedding: Vec<f32> = match shape_dims.len() {
            2 => data.to_vec(),
            3 => {
                let hidden_dim = shape_dims[2];
                data[..hidden_dim].to_vec()
            }
            _ => {
                return Err(CodeEmbeddingError::Inference(format!(
                    "Unexpected output shape: {:?}",
                    shape_dims
                )));
            }
        };

        Ok(embedding)
    }

    /// Get the configuration.
    pub fn config(&self) -> &GraphCodeBertConfig {
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

    /// Check if model has position_ids input.
    pub fn has_position_ids(&self) -> bool {
        self.has_position_ids
    }
}

impl CodeEmbedder for GraphCodeBertEmbedder {
    fn embed_code(&self, code: &str, language: CodeLanguage) -> Result<Vec<f32>> {
        if let Some(cache) = &self.cache {
            if let Some(embedding) = cache.get(code, language) {
                return Ok(embedding.to_vec());
            }
        }

        let (input_ids, attention_mask) = self.tokenize(code)?;
        let mut embedding = self.run_inference(input_ids, attention_mask)?;

        if self.config.normalize {
            super::normalize_embedding(&mut embedding);
        }

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

        codes
            .iter()
            .zip(
                languages
                    .iter()
                    .chain(std::iter::repeat(&CodeLanguage::Unknown)),
            )
            .map(|(code, lang)| self.embed_code(code, *lang))
            .collect()
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    fn model_name(&self) -> &str {
        "GraphCodeBERT"
    }

    fn max_sequence_length(&self) -> usize {
        self.config.max_length
    }

    fn supported_languages(&self) -> &[CodeLanguage] {
        // GraphCodeBERT was trained on CodeSearchNet (6 languages)
        &[
            CodeLanguage::Python,
            CodeLanguage::Java,
            CodeLanguage::JavaScript,
            CodeLanguage::Go,
            CodeLanguage::Ruby,
            CodeLanguage::Php,
        ]
    }
}

unsafe impl Send for GraphCodeBertEmbedder {}
unsafe impl Sync for GraphCodeBertEmbedder {}

impl std::fmt::Debug for GraphCodeBertEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphCodeBertEmbedder")
            .field("model_path", &self.config.model_path)
            .field("embedding_dim", &self.embedding_dim)
            .field("max_length", &self.config.max_length)
            .field("has_position_ids", &self.has_position_ids)
            .field("cache_size", &self.cache.as_ref().map(|c| c.len()))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = GraphCodeBertConfig::default();
        assert_eq!(config.max_length, 512);
        assert_eq!(config.embedding_dim, 768);
        assert!(config.normalize);
        assert!(!config.use_data_flow);
    }

    #[test]
    fn test_config_from_directory() {
        let config = GraphCodeBertConfig::graphcodebert_base("/tmp/graphcodebert");
        assert!(config.model_path.contains("model.onnx"));
        assert!(config.tokenizer_path.contains("tokenizer.json"));
        assert_eq!(config.embedding_dim, 768);
    }
}
