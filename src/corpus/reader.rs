//! Corpus reader trait and document types.

use std::path::PathBuf;

/// A document from the corpus (article, chapter, etc.).
#[derive(Clone, Debug)]
pub struct Document {
    /// Optional document identifier.
    pub id: Option<String>,

    /// Optional document title.
    pub title: Option<String>,

    /// Document text content.
    pub content: String,

    /// Source file path (if applicable).
    pub source: Option<PathBuf>,
}

impl Document {
    /// Create a new document with just content.
    pub fn new(content: String) -> Self {
        Self {
            id: None,
            title: None,
            content,
            source: None,
        }
    }

    /// Create a document with title and content.
    pub fn with_title(title: String, content: String) -> Self {
        Self {
            id: None,
            title: Some(title),
            content,
            source: None,
        }
    }
}

/// Trait for streaming corpus access without loading entire corpus into memory.
///
/// Implementors should provide efficient streaming iteration over documents
/// and sentences, minimizing memory usage.
///
/// # Example
///
/// ```ignore
/// use libgrammstein::corpus::{CorpusReader, PlaintextReader};
///
/// let reader = PlaintextReader::from_file("corpus.txt")?;
///
/// // Stream sentences
/// for sentence in reader.sentences() {
///     process_sentence(&sentence);
/// }
/// ```
pub trait CorpusReader: Send + Sync {
    /// Iterate over documents (articles, chapters, etc.).
    ///
    /// Each document is yielded once, in order. Documents are not cached.
    fn documents(&self) -> Box<dyn Iterator<Item = Document> + Send + '_>;

    /// Iterate over sentences across all documents.
    ///
    /// Sentences are extracted from documents using the default tokenizer.
    fn sentences(&self) -> Box<dyn Iterator<Item = String> + Send + '_>;

    /// Estimate total tokens (for progress tracking).
    ///
    /// Returns `None` if the estimate cannot be computed efficiently.
    fn estimated_tokens(&self) -> Option<usize> {
        None
    }

    /// Get the number of documents (if known).
    fn document_count(&self) -> Option<usize> {
        None
    }
}

/// Implement CorpusReader for boxed trait objects.
///
/// This enables dynamic dispatch with the ownership-based trainer APIs.
impl CorpusReader for Box<dyn CorpusReader> {
    fn documents(&self) -> Box<dyn Iterator<Item = Document> + Send + '_> {
        (**self).documents()
    }

    fn sentences(&self) -> Box<dyn Iterator<Item = String> + Send + '_> {
        (**self).sentences()
    }

    fn estimated_tokens(&self) -> Option<usize> {
        (**self).estimated_tokens()
    }

    fn document_count(&self) -> Option<usize> {
        (**self).document_count()
    }
}
