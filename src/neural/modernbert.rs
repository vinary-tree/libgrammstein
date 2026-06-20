//! ModernBERT model wrapper using Candle.
//!
//! This module provides a Rust-native wrapper for ModernBERT, a 149M parameter
//! encoder-only transformer model with 8,192 token context length.

use candle_core::{DType, Device as CandleDevice, IndexOp, Tensor};
use candle_nn::{layer_norm_no_bias, linear_no_bias, LayerNorm, Linear, VarBuilder};
use candle_transformers::models::modernbert::{Config as ModernBertConfigInner, ModernBert};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::path::Path;
use tokenizers::Tokenizer;

use super::{NeuralError, Result};

/// Device selection for model inference.
#[derive(Clone, Copy, Debug, Default)]
pub enum Device {
    /// CPU inference (default).
    #[default]
    Cpu,
    /// CUDA GPU inference with device index.
    Cuda(usize),
    /// Metal GPU inference (Apple Silicon).
    Metal,
}

impl Device {
    /// Convert to Candle device.
    pub fn to_candle(&self) -> std::result::Result<CandleDevice, candle_core::Error> {
        match self {
            Device::Cpu => Ok(CandleDevice::Cpu),
            Device::Cuda(idx) => CandleDevice::new_cuda(*idx),
            Device::Metal => CandleDevice::new_metal(0),
        }
    }
}

/// Configuration for ModernBERT model.
#[derive(Clone, Debug)]
pub struct ModernBertConfig {
    /// Model identifier on HuggingFace Hub (e.g., "answerdotai/ModernBERT-base").
    pub model_id: String,
    /// Device for inference.
    pub device: Device,
    /// Data type for model weights.
    pub dtype: DType,
    /// Maximum sequence length (default: 8192).
    pub max_seq_len: usize,
}

impl Default for ModernBertConfig {
    fn default() -> Self {
        Self {
            model_id: "answerdotai/ModernBERT-base".to_string(),
            device: Device::default(),
            dtype: DType::F32,
            max_seq_len: 8192,
        }
    }
}

/// ModernBERT model for encoding text.
pub struct ModernBertModel {
    model: ModernBert,
    /// Masked-LM head + tied decoder, replicated from candle's (private)
    /// `ModernBertHead` / `ModernBertDecoder`. This lets us project encoder hidden
    /// states to vocabulary logits without holding a second copy of the
    /// 149M-parameter encoder — candle only exposes the MLM head via
    /// `ModernBertForMaskedLM`, whose inner encoder is private.
    mlm_head_dense: Linear,
    mlm_head_norm: LayerNorm,
    mlm_decoder: Linear,
    tokenizer: Tokenizer,
    device: CandleDevice,
    config: ModernBertConfig,
    hidden_size: usize,
}

impl ModernBertModel {
    /// Load a ModernBERT model from HuggingFace Hub.
    pub fn load(config: ModernBertConfig) -> Result<Self> {
        let device = config
            .device
            .to_candle()
            .map_err(|e| NeuralError::DeviceNotAvailable(format!("{:?}: {}", config.device, e)))?;

        // Download model files from HuggingFace Hub
        let api = Api::new().map_err(|e| NeuralError::ModelLoad(e.to_string()))?;
        let repo = api.repo(Repo::new(config.model_id.clone(), RepoType::Model));

        let model_path = repo
            .get("model.safetensors")
            .map_err(|e| NeuralError::ModelLoad(format!("Failed to download model: {}", e)))?;

        let config_path = repo
            .get("config.json")
            .map_err(|e| NeuralError::ModelLoad(format!("Failed to download config: {}", e)))?;

        let tokenizer_path = repo
            .get("tokenizer.json")
            .map_err(|e| NeuralError::ModelLoad(format!("Failed to download tokenizer: {}", e)))?;

        Self::load_from_files(&model_path, &config_path, &tokenizer_path, config, device)
    }

    /// Load a ModernBERT model from local files.
    pub fn load_from_files(
        model_path: &Path,
        config_path: &Path,
        tokenizer_path: &Path,
        config: ModernBertConfig,
        device: CandleDevice,
    ) -> Result<Self> {
        // Load model configuration
        let config_json = std::fs::read_to_string(config_path)?;
        let model_config: ModernBertConfigInner = serde_json::from_str(&config_json)
            .map_err(|e| NeuralError::ModelLoad(format!("Invalid config: {}", e)))?;

        let hidden_size = model_config.hidden_size;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| NeuralError::Tokenization(format!("Failed to load tokenizer: {}", e)))?;

        // Load model weights
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], config.dtype, &device)? };

        let model = ModernBert::load(vb.clone(), &model_config)?;

        // Replicate candle's MLM head (`ModernBertHead`: dense → GELU → LayerNorm) and
        // decoder (`ModernBertDecoder`: a Linear whose weight is tied to the input token
        // embeddings). Both have private constructors in candle, so we build them from the
        // same `VarBuilder` using the identical weight paths.
        let mlm_head_dense = linear_no_bias(hidden_size, hidden_size, vb.pp("head").pp("dense"))?;
        let mlm_head_norm = layer_norm_no_bias(
            hidden_size,
            model_config.layer_norm_eps,
            vb.pp("head").pp("norm"),
        )?;
        let decoder_weights = vb.get(
            (model_config.vocab_size, hidden_size),
            "model.embeddings.tok_embeddings.weight",
        )?;
        let decoder_bias = vb.get(model_config.vocab_size, "decoder.bias")?;
        let mlm_decoder = Linear::new(decoder_weights, Some(decoder_bias));

        Ok(Self {
            model,
            mlm_head_dense,
            mlm_head_norm,
            mlm_decoder,
            tokenizer,
            device,
            config,
            hidden_size,
        })
    }

    /// Get the hidden size (embedding dimension) of the model.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Get the device the model is running on.
    pub fn device(&self) -> &CandleDevice {
        &self.device
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| NeuralError::Tokenization(e.to_string()))?;

        Ok(encoding.get_ids().to_vec())
    }

    /// Encode multiple texts to token IDs with padding.
    pub fn encode_batch(&self, texts: &[&str]) -> Result<(Vec<Vec<u32>>, Vec<usize>)> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| NeuralError::Tokenization(e.to_string()))?;

        let lengths: Vec<usize> = encodings.iter().map(|e| e.len()).collect();
        let ids: Vec<Vec<u32>> = encodings.iter().map(|e| e.get_ids().to_vec()).collect();

        Ok((ids, lengths))
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| NeuralError::Tokenization(e.to_string()))
    }

    /// Forward pass through the model to get hidden states.
    ///
    /// Returns the last hidden states tensor of shape (batch, seq_len, hidden_size).
    pub fn forward(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let mask = match attention_mask {
            Some(m) => m.clone(),
            None => {
                // Create a default attention mask of all 1s
                let shape = input_ids.dims();
                Tensor::ones(shape, DType::F32, &self.device)?
            }
        };
        let output = self.model.forward(input_ids, &mask)?;
        Ok(output)
    }

    /// Get the [CLS] token embedding for a text.
    ///
    /// Returns a vector of shape (hidden_size,).
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let ids = self.encode(text)?;
        let input_ids = Tensor::new(&ids[..], &self.device)?.unsqueeze(0)?;

        let hidden_states = self.forward(&input_ids, None)?;

        // Extract [CLS] token embedding (first token)
        let cls_embedding = hidden_states.i((0, 0))?;
        let embedding_vec: Vec<f32> = cls_embedding.to_vec1()?;

        Ok(embedding_vec)
    }

    /// Get the mean-pooled embedding for a text.
    ///
    /// Returns a vector of shape (hidden_size,).
    pub fn embed_mean_pooled(&self, text: &str) -> Result<Vec<f32>> {
        let ids = self.encode(text)?;
        let seq_len = ids.len();
        let input_ids = Tensor::new(&ids[..], &self.device)?.unsqueeze(0)?;

        let hidden_states = self.forward(&input_ids, None)?;

        // Mean pooling across sequence dimension
        let sum = hidden_states.sum(1)?;
        let mean = (sum / (seq_len as f64))?;
        let embedding_vec: Vec<f32> = mean.squeeze(0)?.to_vec1()?;

        Ok(embedding_vec)
    }

    /// Batch embed multiple texts using [CLS] token.
    ///
    /// Returns embeddings of shape (batch, hidden_size).
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let (ids_batch, lengths) = self.encode_batch(texts)?;

        // Pad sequences to max length
        let max_len = lengths.iter().copied().max().unwrap_or(0);
        let batch_size = ids_batch.len();

        let mut padded_ids: Vec<u32> = Vec::with_capacity(batch_size * max_len);
        let mut attention_mask: Vec<f32> = Vec::with_capacity(batch_size * max_len);

        for (ids, &len) in ids_batch.iter().zip(&lengths) {
            padded_ids.extend(ids.iter().copied());
            padded_ids.extend(std::iter::repeat(0).take(max_len - len));

            attention_mask.extend(std::iter::repeat(1.0).take(len));
            attention_mask.extend(std::iter::repeat(0.0).take(max_len - len));
        }

        let input_tensor = Tensor::from_vec(padded_ids, (batch_size, max_len), &self.device)?;
        let mask_tensor = Tensor::from_vec(attention_mask, (batch_size, max_len), &self.device)?;

        let hidden_states = self.forward(&input_tensor, Some(&mask_tensor))?;

        // Extract [CLS] embeddings for each sequence
        let mut embeddings = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let cls_embedding = hidden_states.i((i, 0))?;
            let embedding_vec: Vec<f32> = cls_embedding.to_vec1()?;
            embeddings.push(embedding_vec);
        }

        Ok(embeddings)
    }

    /// Get MLM logits for masked positions.
    ///
    /// Input should contain [MASK] tokens at positions to predict.
    /// Returns logits of shape (batch, seq_len, vocab_size).
    ///
    /// Runs the encoder, then the masked-LM head (dense → GELU(erf) → LayerNorm) and the
    /// tied decoder projection — equivalent to candle's `ModernBertForMaskedLM::forward`,
    /// but reusing our single encoder instance.
    pub fn get_mlm_logits(&self, input_ids: &Tensor) -> Result<Tensor> {
        let hidden_states = self.forward(input_ids, None)?;
        let logits = hidden_states
            .apply(&self.mlm_head_dense)?
            .gelu_erf()?
            .apply(&self.mlm_head_norm)?
            .apply(&self.mlm_decoder)?;
        Ok(logits)
    }

    /// Get the mask token ID.
    pub fn mask_token_id(&self) -> Option<u32> {
        self.tokenizer.token_to_id("[MASK]")
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(false)
    }

    /// Get the tokenizer reference.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Get the model configuration.
    pub fn config(&self) -> &ModernBertConfig {
        &self.config
    }
}

impl std::fmt::Debug for ModernBertModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModernBertModel")
            .field("model_id", &self.config.model_id)
            .field("device", &self.config.device)
            .field("hidden_size", &self.hidden_size)
            .field("vocab_size", &self.vocab_size())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = ModernBertConfig::default();

        assert_eq!(config.model_id, "answerdotai/ModernBERT-base");
        assert!(matches!(config.device, Device::Cpu));
        assert_eq!(config.dtype, DType::F32);
        assert_eq!(config.max_seq_len, 8192);
    }

    #[test]
    fn test_cpu_device_conversion() {
        let device = Device::Cpu.to_candle().expect("CPU device should exist");

        assert!(matches!(device, CandleDevice::Cpu));
    }

    #[test]
    fn test_load_from_files_rejects_invalid_config_json() {
        let dir = tempfile::tempdir().expect("tempdir");
        let model_path = dir.path().join("model.safetensors");
        let config_path = dir.path().join("config.json");
        let tokenizer_path = dir.path().join("tokenizer.json");

        std::fs::write(&model_path, b"").expect("empty model fixture");
        std::fs::write(&config_path, b"not json").expect("invalid config");
        std::fs::write(&tokenizer_path, b"{}").expect("minimal tokenizer fixture");

        let err = ModernBertModel::load_from_files(
            &model_path,
            &config_path,
            &tokenizer_path,
            ModernBertConfig::default(),
            CandleDevice::Cpu,
        )
        .expect_err("invalid config should be rejected before loading weights");

        match err {
            NeuralError::ModelLoad(message) => {
                assert!(message.contains("Invalid config"));
            }
            other => panic!("expected ModelLoad error, got {other:?}"),
        }
    }
}
