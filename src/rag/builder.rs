//! Index builder for constructing RAG indices from document collections.

use std::path::Path;

use rayon::prelude::*;

use super::backend::RetrievalBackend;
use super::document::{Document, DocumentBuilder, DocumentId};
use super::exact_backend::ExactCosineBackend;
use super::index::{RagIndex, RagIndexConfig};
use super::{RagError, Result, Synopsis};
use crate::neural::{EmbeddingConfig, ModernBertEmbedder, Summarizer, SummarizerConfig};

/// Configuration for index builder.
#[derive(Clone, Debug)]
pub struct IndexBuilderConfig {
    /// RAG index configuration.
    pub index_config: RagIndexConfig,
    /// Embedding configuration.
    pub embedding_config: EmbeddingConfig,
    /// Summarizer configuration.
    pub summarizer_config: SummarizerConfig,
    /// Batch size for parallel processing.
    pub batch_size: usize,
    /// Whether to generate summaries for documents without explicit synopsis.
    pub generate_summaries: bool,
    /// Progress callback interval (documents).
    pub progress_interval: usize,
}

impl Default for IndexBuilderConfig {
    fn default() -> Self {
        Self {
            index_config: RagIndexConfig::default(),
            embedding_config: EmbeddingConfig::default(),
            summarizer_config: SummarizerConfig::default(),
            batch_size: 32,
            generate_summaries: true,
            progress_interval: 100,
        }
    }
}

/// Builder for constructing RAG indices.
pub struct IndexBuilder {
    embedder: ModernBertEmbedder,
    summarizer: Summarizer,
    config: IndexBuilderConfig,
}

impl IndexBuilder {
    /// Create a new index builder.
    pub fn new(config: IndexBuilderConfig) -> Result<Self> {
        let embedder = ModernBertEmbedder::new(config.embedding_config.clone())?;
        let summarizer = Summarizer::from_model(
            embedder.model_arc(),
            config.summarizer_config.clone(),
        )?;

        Ok(Self {
            embedder,
            summarizer,
            config,
        })
    }

    /// Build index from document builders.
    ///
    /// This method takes `&self` instead of `&mut self`, enabling concurrent use.
    pub fn build_from_builders(
        &self,
        builders: Vec<DocumentBuilder>,
        progress: Option<&dyn Fn(usize, usize)>,
    ) -> Result<RagIndex<ExactCosineBackend>> {
        let mut index = RagIndex::with_exact_backend(self.config.index_config.clone());
        let total = builders.len();

        for (i, builder) in builders.into_iter().enumerate() {
            let doc = self.process_builder(builder, index.allocate_id())?;
            index.add_document(doc)?;

            if let Some(cb) = progress {
                if (i + 1) % self.config.progress_interval == 0 || i + 1 == total {
                    cb(i + 1, total);
                }
            }
        }

        Ok(index)
    }

    /// Process a document builder into a full document.
    fn process_builder(&self, builder: DocumentBuilder, id: DocumentId) -> Result<Document> {
        let content = builder.get_content().ok_or_else(|| {
            RagError::IndexError("Document builder missing content".to_string())
        })?;

        // Generate embedding
        let embedding = self.embedder.embed_document(builder.get_title(), content)?;

        // Generate or use explicit synopsis
        let synopsis = if self.config.generate_summaries {
            self.summarizer.create_synopsis(builder.get_explicit_synopsis(), content)?
        } else {
            match builder.get_explicit_synopsis() {
                Some(text) => Synopsis::explicit(text),
                None => Synopsis::generated(String::new()),
            }
        };

        Ok(builder.build(id, synopsis, embedding))
    }

    /// Build index from a directory of text files.
    ///
    /// This method takes `&self` instead of `&mut self`, enabling concurrent use.
    pub fn build_from_directory(
        &self,
        path: &Path,
        progress: Option<&dyn Fn(usize, usize)>,
    ) -> Result<RagIndex<ExactCosineBackend>> {
        let builders = self.scan_directory(path)?;
        self.build_from_builders(builders, progress)
    }

    /// Scan a directory for documents.
    fn scan_directory(&self, path: &Path) -> Result<Vec<DocumentBuilder>> {
        let mut builders = Vec::new();

        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let file_path = entry.path();

            if file_path.is_file() {
                if let Some(ext) = file_path.extension() {
                    if ext == "txt" || ext == "md" || ext == "html" {
                        let content = std::fs::read_to_string(&file_path)?;
                        let uri = file_path.to_string_lossy().to_string();
                        let title = file_path
                            .file_stem()
                            .map(|s| s.to_string_lossy().to_string());

                        let mut builder = DocumentBuilder::new(uri);
                        builder = builder.content(content);
                        if let Some(t) = title {
                            builder = builder.title(t);
                        }
                        builders.push(builder);
                    }
                }
            }
        }

        Ok(builders)
    }

    /// Extend an existing index with new documents.
    ///
    /// This method takes `&self` instead of `&mut self`, enabling concurrent use.
    pub fn extend_index<B: RetrievalBackend>(
        &self,
        index: &mut RagIndex<B>,
        builders: Vec<DocumentBuilder>,
        progress: Option<&dyn Fn(usize, usize)>,
    ) -> Result<usize> {
        let total = builders.len();
        let mut added = 0;

        for (i, builder) in builders.into_iter().enumerate() {
            let id = DocumentId::new(index.len() as u32 + added as u32);
            let doc = self.process_builder(builder, id)?;
            index.add_document(doc)?;
            added += 1;

            if let Some(cb) = progress {
                if (i + 1) % self.config.progress_interval == 0 || i + 1 == total {
                    cb(i + 1, total);
                }
            }
        }

        Ok(added)
    }

    /// Get the embedder.
    pub fn embedder(&self) -> &ModernBertEmbedder {
        &self.embedder
    }

    /// Get mutable embedder.
    pub fn embedder_mut(&mut self) -> &mut ModernBertEmbedder {
        &mut self.embedder
    }

    /// Get the summarizer.
    pub fn summarizer(&self) -> &Summarizer {
        &self.summarizer
    }

    /// Get mutable summarizer.
    pub fn summarizer_mut(&mut self) -> &mut Summarizer {
        &mut self.summarizer
    }

    /// Get the configuration.
    pub fn config(&self) -> &IndexBuilderConfig {
        &self.config
    }
}

impl std::fmt::Debug for IndexBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IndexBuilder")
            .field("batch_size", &self.config.batch_size)
            .field("generate_summaries", &self.config.generate_summaries)
            .finish()
    }
}

/// Parallel document processor for large corpora.
///
/// This builder shares a single embedder and summarizer across all threads,
/// leveraging their thread-safe `&self` methods for efficient concurrent processing.
pub struct ParallelIndexBuilder {
    embedder: ModernBertEmbedder,
    summarizer: Summarizer,
    config: IndexBuilderConfig,
}

impl ParallelIndexBuilder {
    /// Create a new parallel index builder.
    pub fn new(config: IndexBuilderConfig) -> Result<Self> {
        let embedder = ModernBertEmbedder::new(config.embedding_config.clone())?;
        let summarizer = Summarizer::from_model(
            embedder.model_arc(),
            config.summarizer_config.clone(),
        )?;

        Ok(Self {
            embedder,
            summarizer,
            config,
        })
    }

    /// Build index using parallel processing.
    ///
    /// Shares a single embedder across all threads using their thread-safe `&self` API.
    /// This is more efficient than creating separate embedders per thread.
    pub fn build_parallel(
        &self,
        builders: Vec<DocumentBuilder>,
    ) -> Result<RagIndex<ExactCosineBackend>> {
        use std::sync::atomic::{AtomicU32, Ordering};

        let next_id = AtomicU32::new(0);

        // Process all documents in parallel
        let results: Vec<Result<Document>> = builders
            .into_par_iter()
            .map(|builder| {
                let id = DocumentId::new(next_id.fetch_add(1, Ordering::Relaxed));
                let content = builder.get_content().ok_or_else(|| {
                    RagError::IndexError("Document builder missing content".to_string())
                })?;

                let embedding = self.embedder.embed_document(builder.get_title(), content)?;
                let synopsis = self.summarizer.create_synopsis(
                    builder.get_explicit_synopsis(),
                    content,
                )?;

                Ok(builder.build(id, synopsis, embedding))
            })
            .collect();

        // Build index from results
        let mut index = RagIndex::with_exact_backend(self.config.index_config.clone());

        for result in results {
            let doc = result?;
            index.add_document(doc)?;
        }

        Ok(index)
    }

    /// Get the embedder.
    pub fn embedder(&self) -> &ModernBertEmbedder {
        &self.embedder
    }

    /// Get the summarizer.
    pub fn summarizer(&self) -> &Summarizer {
        &self.summarizer
    }

    /// Get the configuration.
    pub fn config(&self) -> &IndexBuilderConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = IndexBuilderConfig::default();
        assert_eq!(config.batch_size, 32);
        assert!(config.generate_summaries);
    }

    #[test]
    fn test_document_builder_integration() {
        let builder = DocumentBuilder::new("test://doc")
            .title("Test")
            .content("This is test content.")
            .explicit_synopsis("Test summary.");

        assert_eq!(builder.get_uri(), "test://doc");
        assert_eq!(builder.get_title(), Some("Test"));
        assert_eq!(builder.get_content(), Some("This is test content."));
        assert_eq!(builder.get_explicit_synopsis(), Some("Test summary."));
    }
}
