//! RAG (Retrieval-Augmented Generation) module for document retrieval.
//!
//! This module provides dense retrieval capabilities for document search,
//! using ModernBERT embeddings with configurable backends.
//!
//! # Features
//!
//! - **Document Indexing**: Index documents with embeddings and metadata
//! - **Dense Retrieval**: Query documents using embedding similarity
//! - **Exact Backend**: Pre-normalized embeddings with BLAS-accelerated dot product
//! - **HNSW Backend**: Approximate nearest neighbor for large indices (optional)
//! - **Synopsis Generation**: Automatic document summarization
//!
//! # Example
//!
//! ```ignore
//! use libgrammstein::rag::{RagIndex, ExactCosineBackend, Document};
//!
//! let mut index = RagIndex::new(ExactCosineBackend::new(768));
//! index.add_document(document)?;
//!
//! let results = index.query("search query", 10)?;
//! ```

mod backend;
mod builder;
mod document;
mod exact_backend;
#[cfg(feature = "rag-hnsw")]
mod hnsw_backend;
mod index;
mod retriever;

pub use backend::RetrievalBackend;
pub use builder::{IndexBuilder, IndexBuilderConfig};
pub use document::{
    Document, DocumentBuilder, DocumentId, DocumentMeta, DocumentMetadata, LanguageTag,
};
pub use exact_backend::ExactCosineBackend;
#[cfg(feature = "rag-hnsw")]
pub use hnsw_backend::HnswBackend;
pub use index::{RagIndex, RagIndexConfig};
pub use retriever::{RetrievalConfig, RetrievalResult, Retriever};

// Re-export Synopsis from neural module for convenience
pub use crate::neural::{Synopsis, SynopsisSource};

/// Result type for RAG module operations.
pub type Result<T> = std::result::Result<T, RagError>;

/// Error type for RAG module operations.
#[derive(Debug, thiserror::Error)]
pub enum RagError {
    /// Document not found.
    #[error("Document not found: {0}")]
    DocumentNotFound(DocumentId),

    /// Index error.
    #[error("Index error: {0}")]
    IndexError(String),

    /// Embedding error.
    #[error("Embedding error: {0}")]
    EmbeddingError(String),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error.
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Neural module error.
    #[error("Neural error: {0}")]
    Neural(#[from] crate::neural::NeuralError),
}
