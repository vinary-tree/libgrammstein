//! Document representation for RAG index.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::Synopsis;
use crate::topic::TopicId;

/// Unique identifier for a document.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DocumentId(pub u32);

impl DocumentId {
    /// Create a new document ID.
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the raw ID value.
    pub fn as_u32(&self) -> u32 {
        self.0
    }
}

impl std::fmt::Display for DocumentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<u32> for DocumentId {
    fn from(id: u32) -> Self {
        Self(id)
    }
}

impl From<usize> for DocumentId {
    fn from(id: usize) -> Self {
        Self(id as u32)
    }
}

/// Language tag with optional dialect.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LanguageTag {
    /// ISO 639-1 language code (e.g., "en", "de", "es").
    pub language: String,
    /// Optional dialect/region (e.g., "US", "GB", "AU").
    pub dialect: Option<String>,
}

impl LanguageTag {
    /// Create a new language tag.
    pub fn new(language: impl Into<String>) -> Self {
        Self {
            language: language.into(),
            dialect: None,
        }
    }

    /// Create a new language tag with dialect.
    pub fn with_dialect(language: impl Into<String>, dialect: impl Into<String>) -> Self {
        Self {
            language: language.into(),
            dialect: Some(dialect.into()),
        }
    }

    /// English (US).
    pub fn english_us() -> Self {
        Self::with_dialect("en", "US")
    }

    /// English (UK).
    pub fn english_uk() -> Self {
        Self::with_dialect("en", "GB")
    }

    /// German.
    pub fn german() -> Self {
        Self::new("de")
    }

    /// Spanish.
    pub fn spanish() -> Self {
        Self::new("es")
    }

    /// French.
    pub fn french() -> Self {
        Self::new("fr")
    }

    /// Get the full tag string (e.g., "en-US").
    pub fn to_tag_string(&self) -> String {
        match &self.dialect {
            Some(d) => format!("{}-{}", self.language, d),
            None => self.language.clone(),
        }
    }

    /// Parse from tag string (e.g., "en-US").
    pub fn from_tag_string(tag: &str) -> Self {
        let parts: Vec<&str> = tag.split('-').collect();
        match parts.as_slice() {
            [lang] => Self::new(*lang),
            [lang, dialect, ..] => Self::with_dialect(*lang, *dialect),
            _ => Self::default(),
        }
    }
}

impl std::fmt::Display for LanguageTag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_tag_string())
    }
}

/// Document metadata.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DocumentMetadata {
    /// Content type (e.g., "text/plain", "text/html").
    pub content_type: Option<String>,
    /// Source corpus (e.g., "wikipedia", "gutenberg").
    pub source: Option<String>,
    /// Publication date.
    pub date: Option<String>,
    /// Authors.
    pub authors: Vec<String>,
    /// Additional key-value metadata.
    pub extra: HashMap<String, String>,
}

impl DocumentMetadata {
    /// Create new empty metadata.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set content type.
    pub fn with_content_type(mut self, content_type: impl Into<String>) -> Self {
        self.content_type = Some(content_type.into());
        self
    }

    /// Set source.
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Set date.
    pub fn with_date(mut self, date: impl Into<String>) -> Self {
        self.date = Some(date.into());
        self
    }

    /// Add author.
    pub fn with_author(mut self, author: impl Into<String>) -> Self {
        self.authors.push(author.into());
        self
    }

    /// Add extra metadata.
    pub fn with_extra(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra.insert(key.into(), value.into());
        self
    }
}

/// A document in the RAG index.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Document {
    /// Unique document identifier.
    pub id: DocumentId,
    /// Document URI (file path, URL, etc.).
    pub uri: String,
    /// Optional title.
    pub title: Option<String>,
    /// Document synopsis (summary).
    pub synopsis: Synopsis,
    /// Document language.
    pub language: LanguageTag,
    /// Document embedding (768-dim for ModernBERT).
    #[serde(skip)]
    pub embedding: Vec<f32>,
    /// Additional metadata.
    pub metadata: DocumentMetadata,
    /// Topic IDs assigned to this document.
    #[serde(default)]
    pub topic_ids: Vec<TopicId>,
}

impl Document {
    /// Create a new document.
    pub fn new(id: DocumentId, uri: impl Into<String>) -> Self {
        Self {
            id,
            uri: uri.into(),
            title: None,
            synopsis: Synopsis::generated(String::new()),
            language: LanguageTag::default(),
            embedding: Vec::new(),
            metadata: DocumentMetadata::default(),
            topic_ids: Vec::new(),
        }
    }

    /// Set title.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set synopsis.
    pub fn with_synopsis(mut self, synopsis: Synopsis) -> Self {
        self.synopsis = synopsis;
        self
    }

    /// Set language.
    pub fn with_language(mut self, language: LanguageTag) -> Self {
        self.language = language;
        self
    }

    /// Set embedding.
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = embedding;
        self
    }

    /// Set metadata.
    pub fn with_metadata(mut self, metadata: DocumentMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Set topic IDs.
    pub fn with_topic_ids(mut self, topic_ids: Vec<TopicId>) -> Self {
        self.topic_ids = topic_ids;
        self
    }

    /// Get display title (title or URI).
    pub fn display_title(&self) -> &str {
        self.title.as_deref().unwrap_or(&self.uri)
    }

    /// Check if synopsis is explicit.
    pub fn has_explicit_synopsis(&self) -> bool {
        self.synopsis.is_explicit()
    }
}

/// Document metadata stored in the index (without embedding).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DocumentMeta {
    /// Document URI.
    pub uri: String,
    /// Optional title.
    pub title: Option<String>,
    /// Synopsis text.
    pub synopsis: String,
    /// Synopsis source.
    pub synopsis_source: super::SynopsisSource,
    /// Language tag.
    pub language: LanguageTag,
    /// Additional metadata.
    pub metadata: DocumentMetadata,
    /// Topic IDs assigned to this document.
    #[serde(default)]
    pub topic_ids: Vec<TopicId>,
}

impl DocumentMeta {
    /// Create from a document.
    pub fn from_document(doc: &Document) -> Self {
        Self {
            uri: doc.uri.clone(),
            title: doc.title.clone(),
            synopsis: doc.synopsis.text.clone(),
            synopsis_source: doc.synopsis.source,
            language: doc.language.clone(),
            metadata: doc.metadata.clone(),
            topic_ids: doc.topic_ids.clone(),
        }
    }

    /// Get display title.
    pub fn display_title(&self) -> &str {
        self.title.as_deref().unwrap_or(&self.uri)
    }
}

/// Builder for creating documents.
#[derive(Clone)]
pub struct DocumentBuilder {
    uri: String,
    title: Option<String>,
    content: Option<String>,
    explicit_synopsis: Option<String>,
    language: LanguageTag,
    metadata: DocumentMetadata,
}

impl DocumentBuilder {
    /// Create a new document builder.
    pub fn new(uri: impl Into<String>) -> Self {
        Self {
            uri: uri.into(),
            title: None,
            content: None,
            explicit_synopsis: None,
            language: LanguageTag::default(),
            metadata: DocumentMetadata::default(),
        }
    }

    /// Set title.
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set content (for embedding and synopsis generation).
    pub fn content(mut self, content: impl Into<String>) -> Self {
        self.content = Some(content.into());
        self
    }

    /// Set explicit synopsis.
    pub fn explicit_synopsis(mut self, synopsis: impl Into<String>) -> Self {
        self.explicit_synopsis = Some(synopsis.into());
        self
    }

    /// Set language.
    pub fn language(mut self, language: LanguageTag) -> Self {
        self.language = language;
        self
    }

    /// Set metadata.
    pub fn metadata(mut self, metadata: DocumentMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Get the content.
    pub fn get_content(&self) -> Option<&str> {
        self.content.as_deref()
    }

    /// Get the explicit synopsis.
    pub fn get_explicit_synopsis(&self) -> Option<&str> {
        self.explicit_synopsis.as_deref()
    }

    /// Get the title.
    pub fn get_title(&self) -> Option<&str> {
        self.title.as_deref()
    }

    /// Get the URI.
    pub fn get_uri(&self) -> &str {
        &self.uri
    }

    /// Get the language.
    pub fn get_language(&self) -> &LanguageTag {
        &self.language
    }

    /// Get the metadata.
    pub fn get_metadata(&self) -> &DocumentMetadata {
        &self.metadata
    }

    /// Build the document with the given ID and synopsis.
    pub fn build(self, id: DocumentId, synopsis: Synopsis, embedding: Vec<f32>) -> Document {
        Document {
            id,
            uri: self.uri,
            title: self.title,
            synopsis,
            language: self.language,
            embedding,
            metadata: self.metadata,
            topic_ids: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_id() {
        let id = DocumentId::new(42);
        assert_eq!(id.as_u32(), 42);
        assert_eq!(format!("{}", id), "42");
    }

    #[test]
    fn test_language_tag() {
        let tag = LanguageTag::english_us();
        assert_eq!(tag.language, "en");
        assert_eq!(tag.dialect, Some("US".to_string()));
        assert_eq!(tag.to_tag_string(), "en-US");

        let parsed = LanguageTag::from_tag_string("de-AT");
        assert_eq!(parsed.language, "de");
        assert_eq!(parsed.dialect, Some("AT".to_string()));
    }

    #[test]
    fn test_document_builder() {
        let builder = DocumentBuilder::new("file:///test.txt")
            .title("Test Document")
            .content("This is the content.")
            .language(LanguageTag::english_us());

        assert_eq!(builder.get_uri(), "file:///test.txt");
        assert_eq!(builder.get_title(), Some("Test Document"));
        assert_eq!(builder.get_content(), Some("This is the content."));
    }

    #[test]
    fn test_document_meta() {
        let doc = Document::new(DocumentId::new(1), "test.txt")
            .with_title("Test")
            .with_synopsis(Synopsis::explicit("A test document."));

        let meta = DocumentMeta::from_document(&doc);
        assert_eq!(meta.uri, "test.txt");
        assert_eq!(meta.title, Some("Test".to_string()));
        assert_eq!(meta.synopsis, "A test document.");
    }
}
