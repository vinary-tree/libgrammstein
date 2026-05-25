//! Document and query embedding generation using ModernBERT.
//!
//! This module provides embedding generation for RAG retrieval,
//! using ModernBERT's [CLS] token or mean pooling.

use std::sync::Arc;

use super::cache::EmbeddingCache;
use super::modernbert::{ModernBertConfig, ModernBertModel};
use super::Result;

/// Configuration for embedder.
#[derive(Clone, Debug)]
pub struct EmbeddingConfig {
    /// ModernBERT model configuration.
    pub model_config: ModernBertConfig,
    /// Pooling strategy for embeddings.
    pub pooling: PoolingStrategy,
    /// Whether to normalize embeddings to unit length.
    pub normalize: bool,
    /// Cache size for embedding lookups (0 to disable).
    pub cache_size: usize,
    /// Batch size for parallel embedding.
    pub batch_size: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model_config: ModernBertConfig::default(),
            pooling: PoolingStrategy::Cls,
            normalize: true,
            cache_size: 10000,
            batch_size: 32,
        }
    }
}

/// Pooling strategy for creating embeddings from token representations.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PoolingStrategy {
    /// Use [CLS] token embedding (default).
    #[default]
    Cls,
    /// Mean pooling across all tokens.
    MeanPooling,
    /// Max pooling across all tokens.
    MaxPooling,
}

/// Document/query embedder using ModernBERT.
pub struct ModernBertEmbedder {
    model: Arc<ModernBertModel>,
    config: EmbeddingConfig,
    cache: Option<EmbeddingCache>,
}

impl ModernBertEmbedder {
    /// Create a new embedder by loading a model.
    pub fn new(config: EmbeddingConfig) -> Result<Self> {
        let model = ModernBertModel::load(config.model_config.clone())?;
        let cache = if config.cache_size > 0 {
            Some(EmbeddingCache::new(config.cache_size))
        } else {
            None
        };

        Ok(Self {
            model: Arc::new(model),
            config,
            cache,
        })
    }

    /// Create an embedder from an existing model.
    pub fn from_model(model: Arc<ModernBertModel>, config: EmbeddingConfig) -> Self {
        let cache = if config.cache_size > 0 {
            Some(EmbeddingCache::new(config.cache_size))
        } else {
            None
        };

        Self {
            model,
            config,
            cache,
        }
    }

    /// Get the embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        self.model.hidden_size()
    }

    /// Embed a single text.
    ///
    /// This method takes `&self` instead of `&mut self`, enabling concurrent embedding
    /// by multiple threads without external synchronization.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        // Check cache first
        if let Some(cache) = &self.cache {
            if let Some(embedding) = cache.get(text) {
                return Ok(embedding.to_vec());
            }
        }

        // Generate embedding
        let embedding = match self.config.pooling {
            PoolingStrategy::Cls => self.model.embed(text)?,
            PoolingStrategy::MeanPooling => self.model.embed_mean_pooled(text)?,
            PoolingStrategy::MaxPooling => {
                // For max pooling, we'd need to implement it in ModernBertModel
                // For now, fall back to mean pooling
                self.model.embed_mean_pooled(text)?
            }
        };

        // Normalize if requested
        let embedding = if self.config.normalize {
            Self::normalize(&embedding)
        } else {
            embedding
        };

        // Cache the result
        if let Some(cache) = &self.cache {
            cache.insert(text, embedding.clone());
        }

        Ok(embedding)
    }

    /// Embed multiple texts in a batch.
    ///
    /// This method takes `&self` instead of `&mut self`, enabling concurrent embedding
    /// by multiple threads without external synchronization.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // Check cache for each text
        let mut results: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
        let mut uncached_indices: Vec<usize> = Vec::new();
        let mut uncached_texts: Vec<&str> = Vec::new();

        if let Some(cache) = &self.cache {
            for (i, text) in texts.iter().enumerate() {
                if let Some(embedding) = cache.get(text) {
                    results[i] = Some(embedding.to_vec());
                } else {
                    uncached_indices.push(i);
                    uncached_texts.push(text);
                }
            }
        } else {
            uncached_indices = (0..texts.len()).collect();
            uncached_texts = texts.to_vec();
        }

        // Embed uncached texts in batches
        if !uncached_texts.is_empty() {
            for chunk_start in (0..uncached_texts.len()).step_by(self.config.batch_size) {
                let chunk_end = (chunk_start + self.config.batch_size).min(uncached_texts.len());
                let chunk = &uncached_texts[chunk_start..chunk_end];

                let embeddings = match self.config.pooling {
                    PoolingStrategy::Cls => self.model.embed_batch(chunk)?,
                    PoolingStrategy::MeanPooling | PoolingStrategy::MaxPooling => {
                        // Batch mean pooling would need special handling
                        // For now, embed one by one
                        chunk
                            .iter()
                            .map(|t| self.model.embed_mean_pooled(t))
                            .collect::<Result<Vec<_>>>()?
                    }
                };

                // Normalize and store results
                for (j, embedding) in embeddings.into_iter().enumerate() {
                    let idx = uncached_indices[chunk_start + j];
                    let embedding = if self.config.normalize {
                        Self::normalize(&embedding)
                    } else {
                        embedding
                    };

                    // Cache the result
                    if let Some(cache) = &self.cache {
                        cache.insert(texts[idx], embedding.clone());
                    }

                    results[idx] = Some(embedding);
                }
            }
        }

        // Unwrap results (all should be Some now)
        Ok(results.into_iter().map(|r| r.unwrap()).collect())
    }

    /// Embed a document, optionally combining title and content.
    pub fn embed_document(&self, title: Option<&str>, content: &str) -> Result<Vec<f32>> {
        let text = match title {
            Some(t) => format!("{} {}", t, content),
            None => content.to_string(),
        };

        // Truncate to max sequence length if needed
        let truncated = self.truncate_text(&text);
        self.embed(&truncated)
    }

    /// Embed a query, optionally with query expansion prefix.
    pub fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        // Some models benefit from a query prefix like "query: "
        // ModernBERT doesn't require this, so we embed directly
        self.embed(query)
    }

    /// Compute cosine similarity between two embeddings.
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    /// Normalize embedding to unit length.
    pub fn normalize(embedding: &[f32]) -> Vec<f32> {
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm == 0.0 {
            embedding.to_vec()
        } else {
            embedding.iter().map(|x| x / norm).collect()
        }
    }

    /// Truncate text to fit within model's max sequence length.
    fn truncate_text(&self, text: &str) -> String {
        // Rough approximation: ~4 chars per token on average
        let max_chars = self.config.model_config.max_seq_len * 4;

        if text.len() <= max_chars {
            text.to_string()
        } else {
            // Truncate at word boundary
            let truncated = &text[..max_chars];
            match truncated.rfind(char::is_whitespace) {
                Some(pos) => truncated[..pos].to_string(),
                None => truncated.to_string(),
            }
        }
    }

    /// Get the underlying model.
    pub fn model(&self) -> &ModernBertModel {
        &self.model
    }

    /// Get a clone of the Arc-wrapped model.
    pub fn model_arc(&self) -> Arc<ModernBertModel> {
        Arc::clone(&self.model)
    }

    /// Get the configuration.
    pub fn config(&self) -> &EmbeddingConfig {
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
}

impl std::fmt::Debug for ModernBertEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModernBertEmbedder")
            .field("embedding_dim", &self.embedding_dim())
            .field("pooling", &self.config.pooling)
            .field("normalize", &self.config.normalize)
            .field("cache_size", &self.cache.as_ref().map(|c| c.len()))
            .finish()
    }
}

/// Wrapper for document embedding with metadata.
#[derive(Clone, Debug)]
pub struct DocumentEmbedding {
    /// The embedding vector.
    pub embedding: Vec<f32>,
    /// Document ID or URI.
    pub document_id: String,
    /// Optional title.
    pub title: Option<String>,
}

/// Batch document embedder for efficient corpus processing.
pub struct BatchDocumentEmbedder {
    embedder: ModernBertEmbedder,
}

impl BatchDocumentEmbedder {
    /// Create a new batch embedder.
    pub fn new(embedder: ModernBertEmbedder) -> Self {
        Self { embedder }
    }

    /// Embed a batch of documents.
    ///
    /// This method takes `&self` instead of `&mut self`, enabling concurrent embedding.
    pub fn embed_documents(
        &self,
        documents: &[(String, Option<String>, String)], // (id, title, content)
    ) -> Result<Vec<DocumentEmbedding>> {
        let texts: Vec<String> = documents
            .iter()
            .map(|(_, title, content)| match title {
                Some(t) => format!("{} {}", t, content),
                None => content.clone(),
            })
            .collect();

        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let embeddings = self.embedder.embed_batch(&text_refs)?;

        Ok(documents
            .iter()
            .zip(embeddings)
            .map(|((id, title, _), embedding)| DocumentEmbedding {
                embedding,
                document_id: id.clone(),
                title: title.clone(),
            })
            .collect())
    }

    /// Get the embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        self.embedder.embedding_dim()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize() {
        let embedding = vec![3.0, 4.0];
        let normalized = ModernBertEmbedder::normalize(&embedding);

        let norm: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);

        assert!((normalized[0] - 0.6).abs() < 1e-6);
        assert!((normalized[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((ModernBertEmbedder::cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0];
        assert!((ModernBertEmbedder::cosine_similarity(&a, &c) - 0.0).abs() < 1e-6);

        let d = vec![-1.0, 0.0, 0.0];
        assert!((ModernBertEmbedder::cosine_similarity(&a, &d) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_normalized() {
        // For normalized vectors, dot product = cosine similarity
        let a = ModernBertEmbedder::normalize(&vec![3.0, 4.0]);
        let b = ModernBertEmbedder::normalize(&vec![4.0, 3.0]);

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let cosine = ModernBertEmbedder::cosine_similarity(&a, &b);

        assert!((dot - cosine).abs() < 1e-6);
    }
}
