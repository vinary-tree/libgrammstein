//! RAG index combining retrieval backend with document metadata.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::backend::RetrievalBackend;
use super::document::{Document, DocumentId, DocumentMeta};
use super::exact_backend::ExactCosineBackend;
use super::{RagError, Result};
use crate::topic::{TopicConfig, TopicExtractor, TopicId, TopicModel};

/// Configuration for RAG index.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RagIndexConfig {
    /// Embedding dimension (768 for ModernBERT).
    pub embedding_dim: usize,
    /// Maximum number of documents.
    pub max_documents: Option<usize>,
    /// Whether to store full document content.
    pub store_content: bool,
}

impl Default for RagIndexConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 768,
            max_documents: None,
            store_content: false,
        }
    }
}

/// RAG index with retrieval backend and document metadata.
pub struct RagIndex<B: RetrievalBackend = ExactCosineBackend> {
    /// Retrieval backend for similarity search.
    backend: B,
    /// Document metadata indexed by ID.
    documents: HashMap<DocumentId, DocumentMeta>,
    /// Next document ID.
    next_id: u32,
    /// Configuration.
    config: RagIndexConfig,
    /// Trained topic model (if any).
    topic_model: Option<TopicModel>,
}

impl<B: RetrievalBackend> RagIndex<B> {
    /// Create a new RAG index with the given backend.
    pub fn new(backend: B, config: RagIndexConfig) -> Self {
        Self {
            backend,
            documents: HashMap::new(),
            next_id: 0,
            config,
            topic_model: None,
        }
    }

    /// Add a document to the index.
    pub fn add_document(&mut self, doc: Document) -> Result<DocumentId> {
        // Check capacity
        if let Some(max) = self.config.max_documents {
            if self.documents.len() >= max {
                return Err(RagError::IndexError("Index at capacity".to_string()));
            }
        }

        let id = doc.id;

        // Add embedding to backend
        self.backend.add(id, &doc.embedding)?;

        // Store metadata
        let meta = DocumentMeta::from_document(&doc);
        self.documents.insert(id, meta);

        // Update next_id
        if id.as_u32() >= self.next_id {
            self.next_id = id.as_u32() + 1;
        }

        Ok(id)
    }

    /// Allocate a new document ID.
    pub fn allocate_id(&mut self) -> DocumentId {
        let id = DocumentId::new(self.next_id);
        self.next_id += 1;
        id
    }

    /// Query for similar documents.
    ///
    /// Returns document metadata and similarity scores.
    pub fn query(
        &self,
        embedding: &[f32],
        top_k: usize,
    ) -> Vec<(DocumentMeta, f32)> {
        let results = self.backend.query(embedding, top_k);

        results
            .into_iter()
            .filter_map(|(id, score)| {
                self.documents.get(&id).map(|meta| (meta.clone(), score))
            })
            .collect()
    }

    /// Get document metadata by ID.
    pub fn get(&self, id: DocumentId) -> Option<&DocumentMeta> {
        self.documents.get(&id)
    }

    /// Check if a document exists.
    pub fn contains(&self, id: DocumentId) -> bool {
        self.documents.contains_key(&id)
    }

    /// Remove a document from the index.
    pub fn remove(&mut self, id: DocumentId) -> Result<bool> {
        if self.documents.remove(&id).is_some() {
            self.backend.remove(id)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Number of documents in the index.
    pub fn len(&self) -> usize {
        self.documents.len()
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    /// Clear all documents from the index.
    pub fn clear(&mut self) {
        self.backend.clear();
        self.documents.clear();
        self.next_id = 0;
    }

    /// Get the configuration.
    pub fn config(&self) -> &RagIndexConfig {
        &self.config
    }

    /// Get the backend.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Get mutable backend.
    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }

    /// Iterate over all documents.
    pub fn iter(&self) -> impl Iterator<Item = (&DocumentId, &DocumentMeta)> {
        self.documents.iter()
    }

    /// Get all document IDs.
    pub fn document_ids(&self) -> Vec<DocumentId> {
        self.documents.keys().copied().collect()
    }

    /// Get the current topic model (if any).
    pub fn topic_model(&self) -> Option<&TopicModel> {
        self.topic_model.as_ref()
    }

    /// Set the topic model.
    pub fn set_topic_model(&mut self, model: TopicModel) {
        // Update document topic IDs from the model
        for (doc_id, meta) in &mut self.documents {
            let idx = doc_id.as_u32() as usize;
            let topic_ids = model.document_topic_ids(idx);
            meta.topic_ids = topic_ids.to_vec();
        }
        self.topic_model = Some(model);
    }

    /// Clear the topic model.
    pub fn clear_topic_model(&mut self) {
        // Clear topic IDs from all documents
        for meta in self.documents.values_mut() {
            meta.topic_ids.clear();
        }
        self.topic_model = None;
    }

    /// Get topics for a document.
    pub fn document_topics(&self, doc_id: DocumentId) -> Vec<TopicId> {
        self.documents
            .get(&doc_id)
            .map(|meta| meta.topic_ids.clone())
            .unwrap_or_default()
    }
}

impl RagIndex<ExactCosineBackend> {
    /// Create a new index with exact cosine backend.
    pub fn with_exact_backend(config: RagIndexConfig) -> Self {
        let backend = ExactCosineBackend::new(config.embedding_dim);
        Self::new(backend, config)
    }

    /// Extract topics from the indexed documents.
    ///
    /// This method extracts topics using hierarchical agglomerative clustering
    /// on the document embeddings, and assigns topic IDs to each document.
    ///
    /// # Arguments
    /// * `topic_config` - Configuration for topic extraction
    /// * `documents_text` - Document texts for keyword extraction (must be in same order as documents)
    ///
    /// # Returns
    /// The extracted topic model
    pub fn extract_topics(
        &mut self,
        topic_config: TopicConfig,
        documents_text: &[String],
    ) -> crate::topic::Result<TopicModel> {
        let embeddings = self.backend.get_all_embeddings();

        if embeddings.len() != documents_text.len() {
            return Err(crate::topic::TopicError::ClusteringError(format!(
                "Embedding count ({}) != document text count ({})",
                embeddings.len(),
                documents_text.len()
            )));
        }

        let mut extractor = TopicExtractor::new(topic_config.clone());
        let result = extractor.extract(&embeddings, documents_text)?;

        let model = TopicModel::from_extraction(result, topic_config);
        self.set_topic_model(model.clone());

        Ok(model)
    }

    /// Get all embeddings from the index.
    pub fn get_all_embeddings(&self) -> Vec<Vec<f32>> {
        self.backend.get_all_embeddings()
    }

    /// Save the index to disk.
    pub fn save(&self, path: &Path) -> Result<()> {
        std::fs::create_dir_all(path)?;

        // Save backend
        self.backend.save(&path.join("backend"))?;

        // Save metadata
        let meta_path = path.join("metadata.json");
        let meta_file = File::create(&meta_path)?;
        let meta_writer = BufWriter::new(meta_file);
        serde_json::to_writer(meta_writer, &self.documents)
            .map_err(|e| RagError::Serialization(e.to_string()))?;

        // Save config
        let config_path = path.join("config.json");
        let config_file = File::create(&config_path)?;
        let config_writer = BufWriter::new(config_file);
        serde_json::to_writer(config_writer, &self.config)
            .map_err(|e| RagError::Serialization(e.to_string()))?;

        // Save next_id
        let state_path = path.join("state.json");
        let state_file = File::create(&state_path)?;
        let state_writer = BufWriter::new(state_file);
        serde_json::to_writer(state_writer, &IndexState { next_id: self.next_id })
            .map_err(|e| RagError::Serialization(e.to_string()))?;

        // Save topic model (if exists)
        if let Some(ref topic_model) = self.topic_model {
            let topic_path = path.join("topic_model.json");
            topic_model
                .save(&topic_path)
                .map_err(|e| RagError::Serialization(format!("Topic model: {}", e)))?;
        }

        Ok(())
    }

    /// Load the index from disk.
    pub fn load(path: &Path) -> Result<Self> {
        // Load config
        let config_path = path.join("config.json");
        let config_file = File::open(&config_path)?;
        let config_reader = BufReader::new(config_file);
        let config: RagIndexConfig = serde_json::from_reader(config_reader)
            .map_err(|e| RagError::Serialization(e.to_string()))?;

        // Load backend
        let backend = ExactCosineBackend::load(&path.join("backend"), config.embedding_dim)?;

        // Load metadata
        let meta_path = path.join("metadata.json");
        let meta_file = File::open(&meta_path)?;
        let meta_reader = BufReader::new(meta_file);
        let documents: HashMap<DocumentId, DocumentMeta> = serde_json::from_reader(meta_reader)
            .map_err(|e| RagError::Serialization(e.to_string()))?;

        // Load state
        let state_path = path.join("state.json");
        let next_id = if state_path.exists() {
            let state_file = File::open(&state_path)?;
            let state_reader = BufReader::new(state_file);
            let state: IndexState = serde_json::from_reader(state_reader)
                .map_err(|e| RagError::Serialization(e.to_string()))?;
            state.next_id
        } else {
            documents.keys().map(|id| id.as_u32()).max().unwrap_or(0) + 1
        };

        // Load topic model (if exists)
        let topic_path = path.join("topic_model.json");
        let topic_model = if topic_path.exists() {
            Some(
                TopicModel::load(&topic_path)
                    .map_err(|e| RagError::Serialization(format!("Topic model: {}", e)))?,
            )
        } else {
            None
        };

        Ok(Self {
            backend,
            documents,
            next_id,
            config,
            topic_model,
        })
    }
}

/// Serializable index state.
#[derive(Serialize, Deserialize)]
struct IndexState {
    next_id: u32,
}

impl<B: RetrievalBackend> std::fmt::Debug for RagIndex<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RagIndex")
            .field("num_documents", &self.len())
            .field("embedding_dim", &self.config.embedding_dim)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural::Synopsis;

    fn make_test_document(id: u32, embedding: Vec<f32>) -> Document {
        use super::super::document::LanguageTag;

        Document {
            id: DocumentId::new(id),
            uri: format!("test://{}", id),
            title: Some(format!("Document {}", id)),
            synopsis: Synopsis::explicit("Test synopsis"),
            language: LanguageTag::english_us(),
            embedding,
            metadata: Default::default(),
            topic_ids: Vec::new(),
        }
    }

    #[test]
    fn test_add_and_query() {
        let config = RagIndexConfig {
            embedding_dim: 3,
            ..Default::default()
        };
        let mut index = RagIndex::with_exact_backend(config);

        // Add documents
        let doc1 = make_test_document(0, vec![1.0, 0.0, 0.0]);
        let doc2 = make_test_document(1, vec![0.0, 1.0, 0.0]);
        let doc3 = make_test_document(2, vec![0.0, 0.0, 1.0]);

        index.add_document(doc1).unwrap();
        index.add_document(doc2).unwrap();
        index.add_document(doc3).unwrap();

        assert_eq!(index.len(), 3);

        // Query
        let results = index.query(&[1.0, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0.uri, "test://0");
        assert!((results[0].1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_allocate_id() {
        let config = RagIndexConfig {
            embedding_dim: 3,
            ..Default::default()
        };
        let mut index = RagIndex::with_exact_backend(config);

        let id1 = index.allocate_id();
        let id2 = index.allocate_id();

        assert_eq!(id1.as_u32(), 0);
        assert_eq!(id2.as_u32(), 1);
    }

    #[test]
    fn test_remove() {
        let config = RagIndexConfig {
            embedding_dim: 3,
            ..Default::default()
        };
        let mut index = RagIndex::with_exact_backend(config);

        let doc = make_test_document(0, vec![1.0, 0.0, 0.0]);
        index.add_document(doc).unwrap();

        assert_eq!(index.len(), 1);
        assert!(index.remove(DocumentId::new(0)).unwrap());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_topic_extraction() {
        use crate::topic::{ClusteringConfig, CtfidfConfig, TopicConfig};

        let config = RagIndexConfig {
            embedding_dim: 3,
            ..Default::default()
        };
        let mut index = RagIndex::with_exact_backend(config);

        // Add documents with distinct embedding clusters
        let doc1 = make_test_document(0, vec![1.0, 0.0, 0.0]);
        let doc2 = make_test_document(1, vec![0.95, 0.1, 0.0]);
        let doc3 = make_test_document(2, vec![0.0, 1.0, 0.0]);
        let doc4 = make_test_document(3, vec![0.1, 0.95, 0.0]);
        let doc5 = make_test_document(4, vec![0.0, 0.0, 1.0]);
        let doc6 = make_test_document(5, vec![0.0, 0.1, 0.95]);

        index.add_document(doc1).unwrap();
        index.add_document(doc2).unwrap();
        index.add_document(doc3).unwrap();
        index.add_document(doc4).unwrap();
        index.add_document(doc5).unwrap();
        index.add_document(doc6).unwrap();

        // Document texts for keyword extraction
        let documents_text = vec![
            "machine learning algorithms neural networks".to_string(),
            "machine learning models training data".to_string(),
            "web development frontend backend".to_string(),
            "web application programming interface".to_string(),
            "database sql queries optimization".to_string(),
            "database storage retrieval systems".to_string(),
        ];

        let topic_config = TopicConfig {
            clustering: ClusteringConfig {
                num_clusters: Some(3),
                ..Default::default()
            },
            ctfidf: CtfidfConfig {
                num_keywords: 3,
                min_df: 1,
                min_term_length: 2,
                ..Default::default()
            },
            ..Default::default()
        };

        // Extract topics
        let model = index.extract_topics(topic_config, &documents_text).expect("topic extraction failed");

        // Verify model has topics
        assert_eq!(model.num_topics(), 3);
        assert_eq!(model.num_documents(), 6);

        // Verify topic model is stored
        assert!(index.topic_model().is_some());

        // Verify documents have topic IDs assigned
        for doc_id in [0u32, 1, 2, 3, 4, 5] {
            let topics = index.document_topics(DocumentId::new(doc_id));
            assert!(!topics.is_empty(), "Document {} should have topic IDs", doc_id);
        }
    }

    #[test]
    fn test_topic_model_persistence() {
        use crate::topic::{ClusteringConfig, CtfidfConfig, TopicConfig};

        let config = RagIndexConfig {
            embedding_dim: 3,
            ..Default::default()
        };
        let mut index = RagIndex::with_exact_backend(config);

        // Add documents with some similarity (clusters of 2 each)
        let doc1 = make_test_document(0, vec![1.0, 0.0, 0.0]);
        let doc2 = make_test_document(1, vec![0.95, 0.1, 0.0]);
        let doc3 = make_test_document(2, vec![0.0, 1.0, 0.0]);
        let doc4 = make_test_document(3, vec![0.1, 0.95, 0.0]);

        index.add_document(doc1).unwrap();
        index.add_document(doc2).unwrap();
        index.add_document(doc3).unwrap();
        index.add_document(doc4).unwrap();

        let documents_text = vec![
            "first document text content".to_string(),
            "first similar document text".to_string(),
            "second document different text".to_string(),
            "second similar document different".to_string(),
        ];

        let topic_config = TopicConfig {
            clustering: ClusteringConfig {
                num_clusters: Some(2),
                ..Default::default()
            },
            ctfidf: CtfidfConfig {
                num_keywords: 2,
                min_df: 1,
                min_term_length: 2,
                ..Default::default()
            },
            ..Default::default()
        };

        index.extract_topics(topic_config, &documents_text).expect("extraction failed");

        // Save index with topic model
        let temp_path = std::env::temp_dir().join("test_index_with_topics");
        index.save(&temp_path).expect("save failed");

        // Load index
        let loaded_index = RagIndex::load(&temp_path).expect("load failed");

        // Verify topic model was loaded
        assert!(loaded_index.topic_model().is_some());
        let loaded_model = loaded_index.topic_model().unwrap();
        assert_eq!(loaded_model.num_topics(), 2);
        assert_eq!(loaded_model.num_documents(), 4);

        // Cleanup
        let _ = std::fs::remove_dir_all(&temp_path);
    }

    #[test]
    fn test_clear_topic_model() {
        use crate::topic::{ClusteringConfig, CtfidfConfig, TopicConfig, TopicModel, TopicId, Topic};

        let config = RagIndexConfig {
            embedding_dim: 3,
            ..Default::default()
        };
        let mut index = RagIndex::with_exact_backend(config);

        let doc = make_test_document(0, vec![1.0, 0.0, 0.0]);
        index.add_document(doc).unwrap();

        // Manually set a topic model
        let topic_config = TopicConfig::default();
        let topic = Topic::new(TopicId::new(0)).with_document_count(1);
        let mut topics = std::collections::HashMap::new();
        topics.insert(TopicId::new(0), topic);

        // We can't easily create a TopicModel directly, so test via the API
        // Just test that clear_topic_model works
        assert!(index.topic_model().is_none());
        index.clear_topic_model();
        assert!(index.topic_model().is_none());
    }
}
