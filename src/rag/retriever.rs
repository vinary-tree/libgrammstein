//! Query retrieval for RAG index.

use std::sync::Arc;

use super::backend::RetrievalBackend;
use super::document::DocumentMeta;
use super::exact_backend::ExactCosineBackend;
use super::index::RagIndex;
use super::Result;
use crate::neural::ModernBertEmbedder;

/// Configuration for retrieval.
#[derive(Clone, Debug)]
pub struct RetrievalConfig {
    /// Number of results to return.
    pub top_k: usize,
    /// Minimum similarity threshold (0.0 to 1.0).
    pub min_similarity: f32,
    /// Whether to include explicit synopsis in results.
    pub include_explicit_synopsis: bool,
    /// Whether to include generated synopsis in results.
    pub include_generated_synopsis: bool,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            top_k: 10,
            min_similarity: 0.0,
            include_explicit_synopsis: true,
            include_generated_synopsis: true,
        }
    }
}

/// A single retrieval result.
#[derive(Clone, Debug)]
pub struct RetrievalResult {
    /// Document URI.
    pub uri: String,
    /// Document title.
    pub title: Option<String>,
    /// Document synopsis.
    pub synopsis: String,
    /// Whether synopsis is explicit.
    pub synopsis_is_explicit: bool,
    /// Similarity score (0.0 to 1.0).
    pub score: f32,
    /// Rank in results (1 = best).
    pub rank: usize,
}

impl RetrievalResult {
    /// Create from document metadata and score.
    pub fn from_meta(meta: &DocumentMeta, score: f32, rank: usize) -> Self {
        Self {
            uri: meta.uri.clone(),
            title: meta.title.clone(),
            synopsis: meta.synopsis.clone(),
            synopsis_is_explicit: matches!(
                meta.synopsis_source,
                crate::neural::SynopsisSource::Explicit
            ),
            score,
            rank,
        }
    }

    /// Get display title (title or URI).
    pub fn display_title(&self) -> &str {
        self.title.as_deref().unwrap_or(&self.uri)
    }
}

/// Retriever for querying RAG index.
pub struct Retriever<B: RetrievalBackend = ExactCosineBackend> {
    /// The RAG index.
    index: Arc<RagIndex<B>>,
    /// Embedder for query encoding.
    embedder: ModernBertEmbedder,
    /// Configuration.
    config: RetrievalConfig,
}

impl<B: RetrievalBackend> Retriever<B> {
    /// Create a new retriever.
    pub fn new(
        index: Arc<RagIndex<B>>,
        embedder: ModernBertEmbedder,
        config: RetrievalConfig,
    ) -> Self {
        Self {
            index,
            embedder,
            config,
        }
    }

    /// Query the index with a text query.
    pub fn query(&mut self, query: &str) -> Result<Vec<RetrievalResult>> {
        // Embed query
        let embedding = self.embedder.embed_query(query)?;

        // Query index
        self.query_with_embedding(&embedding)
    }

    /// Query the index with a pre-computed embedding.
    pub fn query_with_embedding(&self, embedding: &[f32]) -> Result<Vec<RetrievalResult>> {
        let raw_results = self.index.query(embedding, self.config.top_k);

        let results: Vec<RetrievalResult> = raw_results
            .into_iter()
            .enumerate()
            .filter(|(_, (meta, score))| {
                // Filter by minimum similarity
                if *score < self.config.min_similarity {
                    return false;
                }

                // Filter by synopsis type
                let is_explicit = matches!(
                    meta.synopsis_source,
                    crate::neural::SynopsisSource::Explicit
                );

                if is_explicit && !self.config.include_explicit_synopsis {
                    return false;
                }
                if !is_explicit && !self.config.include_generated_synopsis {
                    return false;
                }

                true
            })
            .map(|(i, (meta, score))| RetrievalResult::from_meta(&meta, score, i + 1))
            .collect();

        Ok(results)
    }

    /// Get the configuration.
    pub fn config(&self) -> &RetrievalConfig {
        &self.config
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: RetrievalConfig) {
        self.config = config;
    }

    /// Get the index.
    pub fn index(&self) -> &RagIndex<B> {
        &self.index
    }

    /// Get the embedder.
    pub fn embedder(&self) -> &ModernBertEmbedder {
        &self.embedder
    }

    /// Get mutable embedder.
    pub fn embedder_mut(&mut self) -> &mut ModernBertEmbedder {
        &mut self.embedder
    }
}

impl<B: RetrievalBackend> std::fmt::Debug for Retriever<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Retriever")
            .field("index_size", &self.index.len())
            .field("config", &self.config)
            .finish()
    }
}

/// Batch retrieval for multiple queries.
pub struct BatchRetriever<B: RetrievalBackend = ExactCosineBackend> {
    retriever: Retriever<B>,
}

impl<B: RetrievalBackend> BatchRetriever<B> {
    /// Create a new batch retriever.
    pub fn new(retriever: Retriever<B>) -> Self {
        Self { retriever }
    }

    /// Query with multiple queries.
    pub fn query_batch(&mut self, queries: &[&str]) -> Result<Vec<Vec<RetrievalResult>>> {
        queries.iter().map(|q| self.retriever.query(q)).collect()
    }

    /// Get the inner retriever.
    pub fn retriever(&self) -> &Retriever<B> {
        &self.retriever
    }

    /// Get mutable inner retriever.
    pub fn retriever_mut(&mut self) -> &mut Retriever<B> {
        &mut self.retriever
    }
}

/// Format retrieval results for display.
pub fn format_results(results: &[RetrievalResult]) -> String {
    let mut output = String::new();

    for result in results {
        output.push_str(&format!(
            "{}. [{:.2}] {}\n",
            result.rank,
            result.score,
            result.display_title()
        ));

        output.push_str(&format!("   URI: {}\n", result.uri));

        let synopsis_type = if result.synopsis_is_explicit {
            "explicit"
        } else {
            "generated"
        };
        output.push_str(&format!(
            "   Synopsis ({}): {}\n",
            synopsis_type, result.synopsis
        ));

        output.push('\n');
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retrieval_result() {
        use super::super::document::{DocumentMeta, DocumentMetadata, LanguageTag};
        use crate::neural::SynopsisSource;

        let meta = DocumentMeta {
            uri: "test://doc".to_string(),
            title: Some("Test Document".to_string()),
            synopsis: "This is a test.".to_string(),
            synopsis_source: SynopsisSource::Explicit,
            language: LanguageTag::english_us(),
            metadata: DocumentMetadata::default(),
            topic_ids: Vec::new(),
        };

        let result = RetrievalResult::from_meta(&meta, 0.95, 1);

        assert_eq!(result.uri, "test://doc");
        assert_eq!(result.display_title(), "Test Document");
        assert!(result.synopsis_is_explicit);
        assert!((result.score - 0.95).abs() < 1e-6);
    }

    #[test]
    fn test_format_results() {
        let results = vec![
            RetrievalResult {
                uri: "test://1".to_string(),
                title: Some("First".to_string()),
                synopsis: "First doc.".to_string(),
                synopsis_is_explicit: true,
                score: 0.95,
                rank: 1,
            },
            RetrievalResult {
                uri: "test://2".to_string(),
                title: None,
                synopsis: "Second doc.".to_string(),
                synopsis_is_explicit: false,
                score: 0.80,
                rank: 2,
            },
        ];

        let output = format_results(&results);
        assert!(output.contains("First"));
        assert!(output.contains("0.95"));
        assert!(output.contains("explicit"));
        assert!(output.contains("generated"));
    }
}
