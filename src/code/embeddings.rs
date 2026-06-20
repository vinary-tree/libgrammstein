//! Neural code embeddings using UniXcoder/GraphCodeBERT.
//!
//! This module provides integration with code-specific transformer models
//! for semantic code understanding. Requires the `code-neural` feature.
//!
//! Supported models:
//! - UniXcoder: Best overall performance (45% MRR on code search)
//! - GraphCodeBERT: Data flow-aware embeddings
//! - CodeBERT: Original code BERT
//! - ModernBERT: General-purpose encoder with 8K context
//!
//! All inference is performed locally using Candle (SafeTensors) or ONNX Runtime.
//!
//! ## Loading Models from Local Paths
//!
//! The [`CodeEmbedder::from_path`] method supports loading models from local directories.
//! It auto-detects the model format and architecture:
//!
//! - **ONNX format**: `model.onnx` + `tokenizer.json`
//! - **SafeTensors format**: `model.safetensors` + `config.json` + `tokenizer.json`
//!
//! For SafeTensors models, the architecture is auto-detected from `config.json`:
//! - `model_type: "modernbert"` → ModernBERT
//! - `model_type: "roberta"` → RoBERTa (UniXcoder, GraphCodeBERT, CodeBERT)
//! - `model_type: "bert"` → BERT

use candle_core::{DType, Device as CandleDevice, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use dashmap::DashMap;
use serde::Deserialize;
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;

// Re-use the existing neural infrastructure from libgrammstein
use crate::neural::{
    Device, EmbeddingConfig, ModernBertConfig, ModernBertEmbedder, ModernBertModel,
};

/// Detected model format in a directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    /// ONNX model (UniXcoder, GraphCodeBERT style): model.onnx + tokenizer.json
    Onnx,
    /// SafeTensors model (ModernBERT/Candle style): model.safetensors + config.json + tokenizer.json
    SafeTensors,
}

impl ModelFormat {
    /// Detect model format from directory contents.
    ///
    /// Returns `Some(format)` if a supported format is detected, `None` otherwise.
    pub fn detect(path: &Path) -> Option<Self> {
        let has_onnx = path.join("model.onnx").exists();
        let has_safetensors = path.join("model.safetensors").exists();
        let has_tokenizer = path.join("tokenizer.json").exists();
        let has_config = path.join("config.json").exists();

        if has_onnx && has_tokenizer {
            Some(ModelFormat::Onnx)
        } else if has_safetensors && has_config && has_tokenizer {
            Some(ModelFormat::SafeTensors)
        } else {
            None
        }
    }

    /// Returns the expected files for this format.
    pub fn expected_files(&self) -> &'static [&'static str] {
        match self {
            ModelFormat::Onnx => &["model.onnx", "tokenizer.json"],
            ModelFormat::SafeTensors => &["model.safetensors", "config.json", "tokenizer.json"],
        }
    }
}

/// Detected model architecture from config.json.
///
/// This enum represents the underlying transformer architecture, which determines
/// how the model weights are loaded and how inference is performed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArchitecture {
    /// ModernBERT architecture (8K context, efficient attention)
    ModernBert,
    /// RoBERTa architecture (used by UniXcoder, GraphCodeBERT, CodeBERT)
    Roberta,
    /// Standard BERT architecture
    Bert,
    /// Unknown or unsupported architecture
    Unknown,
}

impl ModelArchitecture {
    /// Detect architecture from config.json contents.
    ///
    /// Parses the `model_type` field from the JSON configuration.
    pub fn from_config(config_json: &str) -> Self {
        #[derive(Deserialize)]
        struct MinimalConfig {
            model_type: Option<String>,
        }

        let config: MinimalConfig =
            serde_json::from_str(config_json).unwrap_or(MinimalConfig { model_type: None });

        match config.model_type.as_deref() {
            Some("modernbert") => ModelArchitecture::ModernBert,
            Some("roberta") => ModelArchitecture::Roberta,
            Some("bert") => ModelArchitecture::Bert,
            _ => ModelArchitecture::Unknown,
        }
    }

    /// Returns a human-readable name for the architecture.
    pub fn name(&self) -> &'static str {
        match self {
            ModelArchitecture::ModernBert => "ModernBERT",
            ModelArchitecture::Roberta => "RoBERTa",
            ModelArchitecture::Bert => "BERT",
            ModelArchitecture::Unknown => "Unknown",
        }
    }
}

/// Available code embedding model types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EmbeddingModel {
    /// UniXcoder - unified cross-modal model
    UniXcoder,
    /// GraphCodeBERT - data flow-aware embeddings
    GraphCodeBERT,
    /// CodeBERT - original code BERT
    CodeBERT,
    /// Custom/other model
    Custom,
}

impl EmbeddingModel {
    /// Returns the HuggingFace model ID.
    pub fn hf_model_id(&self) -> &str {
        match self {
            EmbeddingModel::UniXcoder => "microsoft/unixcoder-base",
            EmbeddingModel::GraphCodeBERT => "microsoft/graphcodebert-base",
            EmbeddingModel::CodeBERT => "microsoft/codebert-base",
            EmbeddingModel::Custom => "",
        }
    }

    /// Returns the embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        match self {
            EmbeddingModel::UniXcoder => 768,
            EmbeddingModel::GraphCodeBERT => 768,
            EmbeddingModel::CodeBERT => 768,
            EmbeddingModel::Custom => 768, // Default
        }
    }

    /// Returns the maximum sequence length.
    pub fn max_length(&self) -> usize {
        match self {
            EmbeddingModel::UniXcoder => 512,
            EmbeddingModel::GraphCodeBERT => 512,
            EmbeddingModel::CodeBERT => 512,
            EmbeddingModel::Custom => 512,
        }
    }
}

/// Configuration for the code embedder.
#[derive(Debug, Clone)]
pub struct CodeEmbedderConfig {
    /// Which model to use
    pub model: EmbeddingModel,
    /// Device for inference (CPU, CUDA, etc.)
    pub device: Device,
    /// Whether to use caching
    pub use_cache: bool,
    /// Maximum cache size (number of embeddings)
    pub cache_size: usize,
    /// Whether to normalize embeddings
    pub normalize: bool,
    /// Batch size for bulk embedding
    pub batch_size: usize,
}

impl Default for CodeEmbedderConfig {
    fn default() -> Self {
        Self {
            model: EmbeddingModel::UniXcoder,
            device: Device::Cpu,
            use_cache: true,
            cache_size: 10000,
            normalize: true,
            batch_size: 32,
        }
    }
}

/// Backend for code embeddings - supports multiple Candle architectures, plus an
/// ONNX backend (via `ort`) when the `code-neural` feature is enabled.
enum EmbedderBackend {
    /// ModernBERT via ModernBertEmbedder
    ModernBert(ModernBertEmbedder),
    /// BERT/RoBERTa via Candle (used for UniXcoder, GraphCodeBERT, CodeBERT)
    Bert {
        model: BertModel,
        tokenizer: Tokenizer,
        device: CandleDevice,
        hidden_size: usize,
    },
    /// ONNX-backed code embedder (UniXcoder / GraphCodeBERT) via `ort`.
    Onnx(Box<dyn crate::neural::code::CodeEmbedder>),
}

impl EmbedderBackend {
    /// Embed a single text and return the embedding vector.
    fn embed(&self, text: &str) -> Result<Vec<f32>, CodeEmbedderError> {
        match self {
            EmbedderBackend::ModernBert(embedder) => embedder
                .embed(text)
                .map_err(|e| CodeEmbedderError::Embedding(e.to_string())),
            EmbedderBackend::Bert {
                model,
                tokenizer,
                device,
                hidden_size,
            } => {
                // Tokenize
                let encoding = tokenizer.encode(text, true).map_err(|e| {
                    CodeEmbedderError::Embedding(format!("Tokenization error: {}", e))
                })?;

                let ids = encoding.get_ids();
                let token_type_ids: Vec<u32> = vec![0; ids.len()];

                // Create tensors
                let input_ids = Tensor::new(ids, device)
                    .map_err(|e| CodeEmbedderError::Embedding(e.to_string()))?
                    .unsqueeze(0)
                    .map_err(|e| CodeEmbedderError::Embedding(e.to_string()))?;

                let token_type_tensor = Tensor::new(&token_type_ids[..], device)
                    .map_err(|e| CodeEmbedderError::Embedding(e.to_string()))?
                    .unsqueeze(0)
                    .map_err(|e| CodeEmbedderError::Embedding(e.to_string()))?;

                // Forward pass
                let hidden_states = model
                    .forward(&input_ids, &token_type_tensor, None)
                    .map_err(|e| CodeEmbedderError::Embedding(e.to_string()))?;

                // Extract CLS token (first token)
                let cls_embedding = hidden_states
                    .i((0, 0))
                    .map_err(|e| CodeEmbedderError::Embedding(e.to_string()))?;

                let embedding = cls_embedding
                    .to_vec1()
                    .map_err(|e| CodeEmbedderError::Embedding(e.to_string()))?;

                if embedding.len() != *hidden_size {
                    return Err(CodeEmbedderError::Embedding(format!(
                        "BERT embedding dimension mismatch: expected {}, got {}",
                        hidden_size,
                        embedding.len()
                    )));
                }

                Ok(embedding)
            }
            EmbedderBackend::Onnx(embedder) => embedder
                .embed_code(text, crate::neural::code::CodeLanguage::Unknown)
                .map_err(|e| CodeEmbedderError::Embedding(e.to_string())),
        }
    }

    /// Embed a batch of texts.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, CodeEmbedderError> {
        match self {
            EmbedderBackend::ModernBert(embedder) => embedder
                .embed_batch(texts)
                .map_err(|e| CodeEmbedderError::Embedding(e.to_string())),
            EmbedderBackend::Bert {
                model,
                tokenizer,
                device,
                hidden_size,
            } => {
                if texts.is_empty() {
                    return Ok(vec![]);
                }

                // Tokenize all texts
                let encodings = tokenizer.encode_batch(texts.to_vec(), true).map_err(|e| {
                    CodeEmbedderError::Embedding(format!("Tokenization error: {}", e))
                })?;

                // Find max length and pad
                let max_len = encodings.iter().map(|e| e.len()).max().unwrap_or(0);
                let batch_size = encodings.len();

                let mut padded_ids: Vec<u32> = Vec::with_capacity(batch_size * max_len);
                let mut padded_types: Vec<u32> = Vec::with_capacity(batch_size * max_len);

                for enc in &encodings {
                    let ids = enc.get_ids();
                    let len = ids.len();
                    padded_ids.extend(ids.iter().copied());
                    padded_ids.extend(std::iter::repeat(0).take(max_len - len));
                    padded_types.extend(std::iter::repeat(0u32).take(max_len));
                }

                // Create tensors
                let input_tensor = Tensor::from_vec(padded_ids, (batch_size, max_len), device)
                    .map_err(|e| CodeEmbedderError::Embedding(e.to_string()))?;
                let type_tensor = Tensor::from_vec(padded_types, (batch_size, max_len), device)
                    .map_err(|e| CodeEmbedderError::Embedding(e.to_string()))?;

                // Forward pass
                let hidden_states = model
                    .forward(&input_tensor, &type_tensor, None)
                    .map_err(|e| CodeEmbedderError::Embedding(e.to_string()))?;

                // Extract CLS embeddings
                let mut embeddings = Vec::with_capacity(batch_size);
                for i in 0..batch_size {
                    let cls = hidden_states
                        .i((i, 0))
                        .map_err(|e| CodeEmbedderError::Embedding(e.to_string()))?;
                    let vec: Vec<f32> = cls
                        .to_vec1()
                        .map_err(|e| CodeEmbedderError::Embedding(e.to_string()))?;
                    if vec.len() != *hidden_size {
                        return Err(CodeEmbedderError::Embedding(format!(
                            "BERT embedding dimension mismatch: expected {}, got {}",
                            hidden_size,
                            vec.len()
                        )));
                    }
                    embeddings.push(vec);
                }

                Ok(embeddings)
            }
            EmbedderBackend::Onnx(embedder) => {
                let languages = vec![crate::neural::code::CodeLanguage::Unknown; texts.len()];
                embedder
                    .embed_code_batch(texts, &languages)
                    .map_err(|e| CodeEmbedderError::Embedding(e.to_string()))
            }
        }
    }
}

/// Code embedder using transformer models.
///
/// Supports multiple architectures via auto-detection:
/// - ModernBERT (general-purpose, 8K context)
/// - BERT/RoBERTa (UniXcoder, GraphCodeBERT, CodeBERT)
///
/// Features:
/// - Embedding caching
/// - Batch processing
/// - Similarity scoring for code
pub struct CodeEmbedder {
    config: CodeEmbedderConfig,
    backend: EmbedderBackend,
    cache: Option<DashMap<String, Vec<f32>>>,
    /// The detected architecture (for diagnostics)
    architecture: ModelArchitecture,
}

impl CodeEmbedder {
    /// Creates a new code embedder with default configuration.
    ///
    /// Uses ModernBERT as the default backend.
    pub fn new() -> Result<Self, CodeEmbedderError> {
        Self::with_config(CodeEmbedderConfig::default())
    }

    /// Creates a new code embedder with the given configuration.
    ///
    /// Uses ModernBERT as the default backend. For loading specific models
    /// from a local path, use [`CodeEmbedder::from_path`] instead.
    pub fn with_config(config: CodeEmbedderConfig) -> Result<Self, CodeEmbedderError> {
        // Create embedding config for ModernBertEmbedder
        let embed_config = EmbeddingConfig {
            normalize: config.normalize,
            batch_size: config.batch_size,
            ..Default::default()
        };

        // Create the underlying embedder using ModernBERT as default
        let embedder = ModernBertEmbedder::new(embed_config)
            .map_err(|e| CodeEmbedderError::ModelLoad(e.to_string()))?;

        let cache = if config.use_cache {
            Some(DashMap::with_capacity(config.cache_size))
        } else {
            None
        };

        Ok(Self {
            config,
            backend: EmbedderBackend::ModernBert(embedder),
            cache,
            architecture: ModelArchitecture::ModernBert,
        })
    }

    /// Loads embedder from a local model path.
    ///
    /// Automatically detects the model format based on files present in the directory:
    /// - **ONNX format**: `model.onnx` + `tokenizer.json`
    /// - **SafeTensors format**: `model.safetensors` + `config.json` + `tokenizer.json`
    ///
    /// # Arguments
    /// * `path` - Path to the model directory
    /// * `config` - Embedder configuration (cache settings, normalization, etc.)
    ///
    /// # Errors
    /// Returns an error if:
    /// - The path doesn't exist
    /// - No supported model format is detected
    /// - Model loading fails
    ///
    /// # Example
    /// ```ignore
    /// use libgrammstein::code::{CodeEmbedder, CodeEmbedderConfig};
    ///
    /// // Load a SafeTensors model
    /// let embedder = CodeEmbedder::from_path(
    ///     "/path/to/model",
    ///     CodeEmbedderConfig::default()
    /// )?;
    /// ```
    pub fn from_path(
        path: impl AsRef<Path>,
        config: CodeEmbedderConfig,
    ) -> Result<Self, CodeEmbedderError> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(CodeEmbedderError::ModelLoad(format!(
                "Model path does not exist: {}",
                path.display()
            )));
        }

        // Detect model format
        let format = ModelFormat::detect(path).ok_or_else(|| {
            CodeEmbedderError::ModelLoad(format!(
                "Could not detect model format in {}. Expected either:\n\
                 - ONNX: model.onnx + tokenizer.json\n\
                 - SafeTensors: model.safetensors + config.json + tokenizer.json",
                path.display()
            ))
        })?;

        match format {
            ModelFormat::Onnx => Self::load_onnx_model(path, config),
            ModelFormat::SafeTensors => Self::load_safetensors_model(path, config),
        }
    }

    /// Load an ONNX model from a directory.
    ///
    /// Delegates to the `ort`-backed embedders in [`crate::neural::code`] (selected by
    /// `config.model`) and wraps the result in [`EmbedderBackend::Onnx`], so an ONNX
    /// directory loads transparently through [`CodeEmbedder::from_path`]. (This module is
    /// only compiled under the `code-neural` feature, which provides the `ort` runtime.)
    fn load_onnx_model(path: &Path, config: CodeEmbedderConfig) -> Result<Self, CodeEmbedderError> {
        use crate::neural::code::{GraphCodeBertEmbedder, UniXcoderEmbedder};

        // Select the ONNX embedder by configured model. UniXcoder is the general default
        // for UniXcoder / CodeBERT / Custom; GraphCodeBERT has its own loader.
        let backend: Box<dyn crate::neural::code::CodeEmbedder> = match config.model {
            EmbeddingModel::GraphCodeBERT => Box::new(
                GraphCodeBertEmbedder::from_directory(path)
                    .map_err(|e| CodeEmbedderError::ModelLoad(e.to_string()))?,
            ),
            _ => Box::new(
                UniXcoderEmbedder::from_directory(path)
                    .map_err(|e| CodeEmbedderError::ModelLoad(e.to_string()))?,
            ),
        };

        let cache = if config.use_cache {
            Some(DashMap::with_capacity(config.cache_size))
        } else {
            None
        };

        Ok(Self {
            config,
            backend: EmbedderBackend::Onnx(backend),
            // UniXcoder / GraphCodeBERT / CodeBERT are RoBERTa-family architectures.
            architecture: ModelArchitecture::Roberta,
            cache,
        })
    }

    /// Load a SafeTensors model from a directory.
    ///
    /// Auto-detects the model architecture from `config.json` and uses
    /// the appropriate Candle model loader.
    fn load_safetensors_model(
        path: &Path,
        config: CodeEmbedderConfig,
    ) -> Result<Self, CodeEmbedderError> {
        let config_path = path.join("config.json");

        // Read config.json to detect architecture
        let config_json = std::fs::read_to_string(&config_path).map_err(|e| {
            CodeEmbedderError::ModelLoad(format!(
                "Failed to read config.json from {}: {}",
                path.display(),
                e
            ))
        })?;

        let architecture = ModelArchitecture::from_config(&config_json);

        match architecture {
            ModelArchitecture::ModernBert => Self::load_modernbert(path, config),
            ModelArchitecture::Roberta | ModelArchitecture::Bert => {
                Self::load_bert_family(path, config, architecture)
            }
            ModelArchitecture::Unknown => Err(CodeEmbedderError::ModelLoad(format!(
                "Unknown model architecture in {}. Supported architectures:\n\
                 - modernbert (ModernBERT)\n\
                 - roberta (RoBERTa, UniXcoder, GraphCodeBERT, CodeBERT)\n\
                 - bert (BERT)",
                path.display()
            ))),
        }
    }

    /// Load a ModernBERT model from a directory.
    fn load_modernbert(path: &Path, config: CodeEmbedderConfig) -> Result<Self, CodeEmbedderError> {
        let model_path = path.join("model.safetensors");
        let config_path = path.join("config.json");
        let tokenizer_path = path.join("tokenizer.json");

        // Create ModernBert config for the local model
        let bert_config = ModernBertConfig {
            model_id: path.display().to_string(),
            device: config.device,
            ..Default::default()
        };

        // Get the Candle device
        let device = config
            .device
            .to_candle()
            .map_err(|e| CodeEmbedderError::ModelLoad(format!("Device error: {}", e)))?;

        // Load the model from local files
        let model = ModernBertModel::load_from_files(
            &model_path,
            &config_path,
            &tokenizer_path,
            bert_config,
            device,
        )
        .map_err(|e| {
            CodeEmbedderError::ModelLoad(format!(
                "Failed to load ModernBERT model from {}: {}",
                path.display(),
                e
            ))
        })?;

        // Create embedding config
        let embed_config = EmbeddingConfig {
            normalize: config.normalize,
            batch_size: config.batch_size,
            ..Default::default()
        };

        // Wrap the model in a ModernBertEmbedder
        let embedder = ModernBertEmbedder::from_model(Arc::new(model), embed_config);

        let cache = if config.use_cache {
            Some(DashMap::with_capacity(config.cache_size))
        } else {
            None
        };

        Ok(Self {
            config,
            backend: EmbedderBackend::ModernBert(embedder),
            cache,
            architecture: ModelArchitecture::ModernBert,
        })
    }

    /// Load a BERT-family model (BERT, RoBERTa) from a directory.
    ///
    /// This handles UniXcoder, GraphCodeBERT, CodeBERT, and other BERT/RoBERTa models.
    fn load_bert_family(
        path: &Path,
        config: CodeEmbedderConfig,
        architecture: ModelArchitecture,
    ) -> Result<Self, CodeEmbedderError> {
        let model_path = path.join("model.safetensors");
        let config_path = path.join("config.json");
        let tokenizer_path = path.join("tokenizer.json");

        // Get the Candle device
        let device = config
            .device
            .to_candle()
            .map_err(|e| CodeEmbedderError::ModelLoad(format!("Device error: {}", e)))?;

        // Load BERT config
        let config_json = std::fs::read_to_string(&config_path)
            .map_err(|e| CodeEmbedderError::ModelLoad(format!("Failed to read config: {}", e)))?;

        let bert_config: BertConfig = serde_json::from_str(&config_json).map_err(|e| {
            CodeEmbedderError::ModelLoad(format!(
                "Failed to parse {} config from {}: {}",
                architecture.name(),
                path.display(),
                e
            ))
        })?;

        let hidden_size = bert_config.hidden_size;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            CodeEmbedderError::ModelLoad(format!(
                "Failed to load tokenizer from {}: {}",
                tokenizer_path.display(),
                e
            ))
        })?;

        // Load model weights
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&model_path], DType::F32, &device).map_err(
                |e| {
                    CodeEmbedderError::ModelLoad(format!(
                        "Failed to load model weights from {}: {}",
                        model_path.display(),
                        e
                    ))
                },
            )?
        };

        let model = BertModel::load(vb, &bert_config).map_err(|e| {
            CodeEmbedderError::ModelLoad(format!(
                "Failed to initialize {} model: {}",
                architecture.name(),
                e
            ))
        })?;

        let cache = if config.use_cache {
            Some(DashMap::with_capacity(config.cache_size))
        } else {
            None
        };

        Ok(Self {
            config,
            backend: EmbedderBackend::Bert {
                model,
                tokenizer,
                device,
                hidden_size,
            },
            cache,
            architecture,
        })
    }

    /// Returns the embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        self.config.model.embedding_dim()
    }

    /// Embeds a code snippet.
    pub fn embed(&self, code: &str) -> Result<Vec<f32>, CodeEmbedderError> {
        // Check cache first
        if let Some(ref cache) = self.cache {
            if let Some(embedding) = cache.get(code) {
                return Ok(embedding.clone());
            }
        }

        // Compute embedding
        let embedding = self.backend.embed(code)?;

        // Cache the result
        if let Some(ref cache) = self.cache {
            // Evict old entries if needed
            if cache.len() >= self.config.cache_size {
                // Simple eviction: remove ~10% of entries
                let to_remove: Vec<String> = cache
                    .iter()
                    .take(self.config.cache_size / 10)
                    .map(|e| e.key().clone())
                    .collect();
                for key in to_remove {
                    cache.remove(&key);
                }
            }
            cache.insert(code.to_string(), embedding.clone());
        }

        Ok(embedding)
    }

    /// Embeds multiple code snippets in batch.
    pub fn embed_batch(&self, codes: &[&str]) -> Result<Vec<Vec<f32>>, CodeEmbedderError> {
        // For uncached items, compute in batch
        let mut results = Vec::with_capacity(codes.len());
        let mut uncached_indices = Vec::new();
        let mut uncached_codes = Vec::new();

        for (i, code) in codes.iter().enumerate() {
            if let Some(ref cache) = self.cache {
                if let Some(embedding) = cache.get(*code) {
                    results.push(Some(embedding.clone()));
                    continue;
                }
            }
            results.push(None);
            uncached_indices.push(i);
            uncached_codes.push(*code);
        }

        // Batch embed uncached items
        if !uncached_codes.is_empty() {
            let batch_embeddings = self.backend.embed_batch(&uncached_codes)?;

            for (idx, embedding) in uncached_indices.into_iter().zip(batch_embeddings) {
                // Cache the result
                if let Some(ref cache) = self.cache {
                    cache.insert(codes[idx].to_string(), embedding.clone());
                }
                results[idx] = Some(embedding);
            }
        }

        Ok(results.into_iter().map(|e| e.unwrap()).collect())
    }

    /// Computes cosine similarity between two embeddings.
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    /// Scores the similarity between two code snippets.
    pub fn score_similarity(&self, code_a: &str, code_b: &str) -> Result<f32, CodeEmbedderError> {
        let embed_a = self.embed(code_a)?;
        let embed_b = self.embed(code_b)?;
        Ok(Self::cosine_similarity(&embed_a, &embed_b))
    }

    /// Scores how well a candidate completion fits a context.
    pub fn score_completion(
        &self,
        context: &str,
        candidate: &str,
    ) -> Result<f64, CodeEmbedderError> {
        let combined = format!("{}{}", context, candidate);
        let context_embed = self.embed(context)?;
        let combined_embed = self.embed(&combined)?;

        // Score based on how well the combined embedding aligns with the context
        // A good completion should produce a coherent embedding
        let similarity = Self::cosine_similarity(&context_embed, &combined_embed);

        // Convert to probability-like score
        Ok((similarity as f64 + 1.0) / 2.0)
    }

    /// Clears the embedding cache.
    pub fn clear_cache(&self) {
        if let Some(ref cache) = self.cache {
            cache.clear();
        }
    }

    /// Returns the current cache size.
    pub fn cache_size(&self) -> usize {
        self.cache.as_ref().map(|c| c.len()).unwrap_or(0)
    }

    /// Returns the model being used.
    pub fn model(&self) -> EmbeddingModel {
        self.config.model
    }

    /// Returns the detected model architecture.
    pub fn architecture(&self) -> ModelArchitecture {
        self.architecture
    }
}

impl Default for CodeEmbedder {
    fn default() -> Self {
        Self::new().expect("Failed to create default CodeEmbedder")
    }
}

/// Error types for code embedding operations.
#[derive(Debug, Clone)]
pub enum CodeEmbedderError {
    /// Model loading failed
    ModelLoad(String),
    /// Embedding computation failed
    Embedding(String),
    /// Invalid input
    InvalidInput(String),
    /// Cache error
    Cache(String),
}

impl std::fmt::Display for CodeEmbedderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CodeEmbedderError::ModelLoad(msg) => write!(f, "Model load error: {}", msg),
            CodeEmbedderError::Embedding(msg) => write!(f, "Embedding error: {}", msg),
            CodeEmbedderError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            CodeEmbedderError::Cache(msg) => write!(f, "Cache error: {}", msg),
        }
    }
}

impl std::error::Error for CodeEmbedderError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_model_config() {
        assert_eq!(EmbeddingModel::UniXcoder.embedding_dim(), 768);
        assert_eq!(EmbeddingModel::UniXcoder.max_length(), 512);
        assert_eq!(
            EmbeddingModel::UniXcoder.hf_model_id(),
            "microsoft/unixcoder-base"
        );
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((CodeEmbedder::cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0];
        assert!((CodeEmbedder::cosine_similarity(&a, &c) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_code_embedder_config_default() {
        let config = CodeEmbedderConfig::default();
        assert_eq!(config.model, EmbeddingModel::UniXcoder);
        assert!(config.use_cache);
        assert!(config.normalize);
    }

    #[test]
    fn test_architecture_detection() {
        // ModernBERT detection
        let modernbert = r#"{"model_type": "modernbert"}"#;
        assert_eq!(
            ModelArchitecture::from_config(modernbert),
            ModelArchitecture::ModernBert
        );

        // RoBERTa detection (UniXcoder, GraphCodeBERT, CodeBERT use this)
        let roberta = r#"{"model_type": "roberta"}"#;
        assert_eq!(
            ModelArchitecture::from_config(roberta),
            ModelArchitecture::Roberta
        );

        // BERT detection
        let bert = r#"{"model_type": "bert"}"#;
        assert_eq!(
            ModelArchitecture::from_config(bert),
            ModelArchitecture::Bert
        );

        // Unknown/missing model_type
        let unknown = r#"{"model_type": "gpt2"}"#;
        assert_eq!(
            ModelArchitecture::from_config(unknown),
            ModelArchitecture::Unknown
        );

        let missing = r#"{"hidden_size": 768}"#;
        assert_eq!(
            ModelArchitecture::from_config(missing),
            ModelArchitecture::Unknown
        );

        let invalid_json = "not valid json";
        assert_eq!(
            ModelArchitecture::from_config(invalid_json),
            ModelArchitecture::Unknown
        );
    }

    #[test]
    fn test_model_format_detection() {
        // This test verifies the ModelFormat struct logic
        // The actual detection requires filesystem access, so we test the expected_files method
        assert_eq!(
            ModelFormat::Onnx.expected_files(),
            &["model.onnx", "tokenizer.json"]
        );
        assert_eq!(
            ModelFormat::SafeTensors.expected_files(),
            &["model.safetensors", "config.json", "tokenizer.json"]
        );
    }

    /// Mock `ort`-backed embedder so the ONNX dispatch can be tested without a real
    /// ONNX runtime or model file.
    struct MockOnnxEmbedder;

    impl crate::neural::code::CodeEmbedder for MockOnnxEmbedder {
        fn embed_code(
            &self,
            _code: &str,
            _language: crate::neural::code::CodeLanguage,
        ) -> crate::neural::code::Result<Vec<f32>> {
            Ok(vec![0.5, 0.25, 0.75])
        }

        fn embed_code_batch(
            &self,
            codes: &[&str],
            _languages: &[crate::neural::code::CodeLanguage],
        ) -> crate::neural::code::Result<Vec<Vec<f32>>> {
            Ok(codes.iter().map(|_| vec![0.5, 0.25, 0.75]).collect())
        }

        fn embedding_dim(&self) -> usize {
            3
        }

        fn model_name(&self) -> &str {
            "mock-onnx"
        }

        fn max_sequence_length(&self) -> usize {
            512
        }

        fn supported_languages(&self) -> &[crate::neural::code::CodeLanguage] {
            &[]
        }
    }

    #[test]
    fn test_onnx_backend_dispatch() {
        let backend = EmbedderBackend::Onnx(Box::new(MockOnnxEmbedder));

        // Single embed routes through to the ONNX embedder.
        let v = backend.embed("fn main() {}").expect("embed");
        assert_eq!(v, vec![0.5, 0.25, 0.75]);

        // Batch embed routes through and preserves length per input.
        let batch = backend.embed_batch(&["a", "b"]).expect("embed_batch");
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0], vec![0.5, 0.25, 0.75]);
        assert_eq!(batch[1], vec![0.5, 0.25, 0.75]);
    }
}
