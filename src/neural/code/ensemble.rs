//! Ensemble code embedder combining multiple models.
//!
//! Ensemble strategies improve embedding quality by combining
//! multiple complementary models (e.g., CodeT5+, UniXcoder, GraphCodeBERT).

use std::sync::Arc;

use super::{CodeEmbedder, CodeEmbeddingError, CodeLanguage, Result};

/// Strategy for combining embeddings from multiple models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EnsembleStrategy {
    /// Concatenate all embeddings: [emb1 | emb2 | ... | embN]
    /// Results in embedding_dim = sum(model_dims)
    #[default]
    Concatenate,

    /// Weighted average: w1*emb1 + w2*emb2 + ... + wN*embN
    /// Results in embedding_dim = max(model_dims) (requires same dimensions)
    WeightedAverage,

    /// Element-wise maximum: max(emb1, emb2, ..., embN)
    /// Results in embedding_dim = model_dim (requires same dimensions)
    MaxPooling,

    /// Average pooling: (emb1 + emb2 + ... + embN) / N
    /// Results in embedding_dim = model_dim (requires same dimensions)
    MeanPooling,
}

/// Ensemble of multiple code embedders.
///
/// Combines embeddings from multiple models to improve
/// representation quality for code similarity tasks.
pub struct EnsembleCodeEmbedder {
    embedders: Vec<Arc<dyn CodeEmbedder>>,
    weights: Vec<f64>,
    strategy: EnsembleStrategy,
    embedding_dim: usize,
    normalize_final: bool,
}

impl EnsembleCodeEmbedder {
    /// Create a new ensemble with concatenation strategy.
    pub fn new(embedders: Vec<Arc<dyn CodeEmbedder>>) -> Self {
        let embedding_dim: usize = embedders.iter().map(|e| e.embedding_dim()).sum();
        let weights = vec![1.0; embedders.len()];

        Self {
            embedders,
            weights,
            strategy: EnsembleStrategy::Concatenate,
            embedding_dim,
            normalize_final: true,
        }
    }

    /// Create an ensemble with specified strategy and weights.
    pub fn with_strategy(
        embedders: Vec<Arc<dyn CodeEmbedder>>,
        strategy: EnsembleStrategy,
        weights: Option<Vec<f64>>,
    ) -> Result<Self> {
        if embedders.is_empty() {
            return Err(CodeEmbeddingError::Inference(
                "Ensemble requires at least one embedder".to_string(),
            ));
        }

        let weights =
            weights.unwrap_or_else(|| vec![1.0 / embedders.len() as f64; embedders.len()]);

        if weights.len() != embedders.len() {
            return Err(CodeEmbeddingError::Inference(format!(
                "Weight count ({}) must match embedder count ({})",
                weights.len(),
                embedders.len()
            )));
        }

        // Validate dimensions for non-concatenate strategies
        let first_dim = embedders[0].embedding_dim();
        if strategy != EnsembleStrategy::Concatenate {
            for (i, embedder) in embedders.iter().enumerate().skip(1) {
                if embedder.embedding_dim() != first_dim {
                    return Err(CodeEmbeddingError::Inference(format!(
                        "{:?} strategy requires equal embedding dimensions. \
                         Embedder {} has dim {} but expected {}",
                        strategy,
                        i,
                        embedder.embedding_dim(),
                        first_dim
                    )));
                }
            }
        }

        let embedding_dim = match strategy {
            EnsembleStrategy::Concatenate => embedders.iter().map(|e| e.embedding_dim()).sum(),
            _ => first_dim,
        };

        Ok(Self {
            embedders,
            weights,
            strategy,
            embedding_dim,
            normalize_final: true,
        })
    }

    /// Set whether to normalize the final embedding.
    pub fn set_normalize_final(&mut self, normalize: bool) {
        self.normalize_final = normalize;
    }

    /// Get the ensemble strategy.
    pub fn strategy(&self) -> EnsembleStrategy {
        self.strategy
    }

    /// Get the weights.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Get the number of embedders.
    pub fn num_embedders(&self) -> usize {
        self.embedders.len()
    }

    /// Combine embeddings according to the strategy.
    fn combine_embeddings(&self, embeddings: Vec<Vec<f32>>) -> Vec<f32> {
        if embeddings.is_empty() {
            return vec![];
        }

        let mut result = match self.strategy {
            EnsembleStrategy::Concatenate => {
                // Concatenate all embeddings
                embeddings.into_iter().flatten().collect()
            }

            EnsembleStrategy::WeightedAverage => {
                // Weighted sum
                let dim = embeddings[0].len();
                let mut combined = vec![0.0f32; dim];

                for (embedding, &weight) in embeddings.iter().zip(self.weights.iter()) {
                    let weight = weight as f32;
                    for (i, &val) in embedding.iter().enumerate() {
                        combined[i] += val * weight;
                    }
                }

                combined
            }

            EnsembleStrategy::MaxPooling => {
                // Element-wise maximum
                let dim = embeddings[0].len();
                let mut combined = vec![f32::NEG_INFINITY; dim];

                for embedding in &embeddings {
                    for (i, &val) in embedding.iter().enumerate() {
                        if val > combined[i] {
                            combined[i] = val;
                        }
                    }
                }

                combined
            }

            EnsembleStrategy::MeanPooling => {
                // Simple average
                let dim = embeddings[0].len();
                let n = embeddings.len() as f32;
                let mut combined = vec![0.0f32; dim];

                for embedding in &embeddings {
                    for (i, &val) in embedding.iter().enumerate() {
                        combined[i] += val / n;
                    }
                }

                combined
            }
        };

        // Normalize if configured
        if self.normalize_final {
            super::normalize_embedding(&mut result);
        }

        result
    }
}

impl CodeEmbedder for EnsembleCodeEmbedder {
    fn embed_code(&self, code: &str, language: CodeLanguage) -> Result<Vec<f32>> {
        // Collect embeddings from all models
        let embeddings: Vec<Vec<f32>> = self
            .embedders
            .iter()
            .map(|e| e.embed_code(code, language))
            .collect::<Result<Vec<_>>>()?;

        Ok(self.combine_embeddings(embeddings))
    }

    fn embed_code_batch(
        &self,
        codes: &[&str],
        languages: &[CodeLanguage],
    ) -> Result<Vec<Vec<f32>>> {
        if codes.is_empty() {
            return Ok(vec![]);
        }

        // Get batch embeddings from each model
        let all_model_embeddings: Vec<Vec<Vec<f32>>> = self
            .embedders
            .iter()
            .map(|e| e.embed_code_batch(codes, languages))
            .collect::<Result<Vec<_>>>()?;

        // Combine embeddings for each code snippet
        let num_codes = codes.len();
        let mut results = Vec::with_capacity(num_codes);

        for i in 0..num_codes {
            let embeddings: Vec<Vec<f32>> = all_model_embeddings
                .iter()
                .map(|model_embeddings| model_embeddings[i].clone())
                .collect();

            results.push(self.combine_embeddings(embeddings));
        }

        Ok(results)
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    fn model_name(&self) -> &str {
        "Ensemble"
    }

    fn max_sequence_length(&self) -> usize {
        // Use minimum across all models
        self.embedders
            .iter()
            .map(|e| e.max_sequence_length())
            .min()
            .unwrap_or(512)
    }

    fn supported_languages(&self) -> &[CodeLanguage] {
        // Return empty to indicate all languages are supported
        // (intersection of all models would be complex to track)
        &[]
    }
}

impl std::fmt::Debug for EnsembleCodeEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EnsembleCodeEmbedder")
            .field("num_embedders", &self.embedders.len())
            .field("strategy", &self.strategy)
            .field("embedding_dim", &self.embedding_dim)
            .field("weights", &self.weights)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock embedder for testing.
    struct MockEmbedder {
        dim: usize,
        value: f32,
    }

    impl MockEmbedder {
        fn new(dim: usize, value: f32) -> Self {
            Self { dim, value }
        }
    }

    impl CodeEmbedder for MockEmbedder {
        fn embed_code(&self, _code: &str, _language: CodeLanguage) -> Result<Vec<f32>> {
            Ok(vec![self.value; self.dim])
        }

        fn embed_code_batch(
            &self,
            codes: &[&str],
            languages: &[CodeLanguage],
        ) -> Result<Vec<Vec<f32>>> {
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
            self.dim
        }

        fn model_name(&self) -> &str {
            "Mock"
        }

        fn max_sequence_length(&self) -> usize {
            512
        }

        fn supported_languages(&self) -> &[CodeLanguage] {
            &[]
        }
    }

    #[test]
    fn test_concatenate_strategy() {
        let embedders: Vec<Arc<dyn CodeEmbedder>> = vec![
            Arc::new(MockEmbedder::new(3, 1.0)),
            Arc::new(MockEmbedder::new(2, 2.0)),
        ];

        let mut ensemble = EnsembleCodeEmbedder::new(embedders);
        ensemble.set_normalize_final(false); // Disable normalization for testing

        assert_eq!(ensemble.embedding_dim(), 5);

        let embedding = ensemble.embed_code("test", CodeLanguage::Rust).unwrap();
        assert_eq!(embedding.len(), 5);
        assert_eq!(&embedding[..3], &[1.0, 1.0, 1.0]);
        assert_eq!(&embedding[3..], &[2.0, 2.0]);
    }

    #[test]
    fn test_weighted_average_strategy() {
        let embedders: Vec<Arc<dyn CodeEmbedder>> = vec![
            Arc::new(MockEmbedder::new(3, 1.0)),
            Arc::new(MockEmbedder::new(3, 2.0)),
        ];

        let mut ensemble = EnsembleCodeEmbedder::with_strategy(
            embedders,
            EnsembleStrategy::WeightedAverage,
            Some(vec![0.5, 0.5]),
        )
        .unwrap();
        ensemble.set_normalize_final(false);

        assert_eq!(ensemble.embedding_dim(), 3);

        let embedding = ensemble.embed_code("test", CodeLanguage::Rust).unwrap();
        assert_eq!(embedding.len(), 3);
        // 0.5 * 1.0 + 0.5 * 2.0 = 1.5
        assert!((embedding[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_max_pooling_strategy() {
        let embedders: Vec<Arc<dyn CodeEmbedder>> = vec![
            Arc::new(MockEmbedder::new(3, 1.0)),
            Arc::new(MockEmbedder::new(3, 2.0)),
        ];

        let mut ensemble =
            EnsembleCodeEmbedder::with_strategy(embedders, EnsembleStrategy::MaxPooling, None)
                .unwrap();
        ensemble.set_normalize_final(false);

        let embedding = ensemble.embed_code("test", CodeLanguage::Rust).unwrap();
        assert_eq!(embedding.len(), 3);
        assert!((embedding[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_mean_pooling_strategy() {
        let embedders: Vec<Arc<dyn CodeEmbedder>> = vec![
            Arc::new(MockEmbedder::new(3, 1.0)),
            Arc::new(MockEmbedder::new(3, 3.0)),
        ];

        let mut ensemble =
            EnsembleCodeEmbedder::with_strategy(embedders, EnsembleStrategy::MeanPooling, None)
                .unwrap();
        ensemble.set_normalize_final(false);

        let embedding = ensemble.embed_code("test", CodeLanguage::Rust).unwrap();
        assert_eq!(embedding.len(), 3);
        // (1.0 + 3.0) / 2 = 2.0
        assert!((embedding[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let embedders: Vec<Arc<dyn CodeEmbedder>> = vec![
            Arc::new(MockEmbedder::new(3, 1.0)),
            Arc::new(MockEmbedder::new(4, 2.0)), // Different dimension
        ];

        let result =
            EnsembleCodeEmbedder::with_strategy(embedders, EnsembleStrategy::WeightedAverage, None);

        assert!(result.is_err());
    }
}
