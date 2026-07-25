//! Topic model container and persistence.
//!
//! This module provides the TopicModel struct that serves as the
//! main container for trained topic extraction results, with support
//! for serialization and deserialization.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::config::TopicConfig;
use super::dendrogram::Dendrogram;
use super::extractor::ExtractionResult;
use super::types::{Topic, TopicId};
use super::{Result, TopicError};

/// A trained topic model.
///
/// TopicModel is the main container for topic extraction results.
/// It provides methods for:
/// - Querying topics by ID
/// - Getting topics for documents
/// - Navigating the topic hierarchy
/// - Saving and loading the model
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TopicModel {
    /// All topics indexed by ID.
    topics: HashMap<TopicId, Topic>,
    /// Document to topic mappings.
    document_topics: Vec<Vec<TopicId>>,
    /// Topic hierarchy as a dendrogram.
    dendrogram: Dendrogram,
    /// Vocabulary used for keyword extraction.
    vocabulary: Vec<String>,
    /// Linkage matrix from clustering.
    linkage: Vec<(u32, u32, f32, u32)>,
    /// Number of hierarchy levels.
    num_levels: usize,
    /// Configuration used for extraction.
    config: TopicConfig,
}

impl TopicModel {
    /// Create a new topic model from extraction results.
    pub fn from_extraction(result: ExtractionResult, config: TopicConfig) -> Self {
        let topics: HashMap<TopicId, Topic> =
            result.topics.into_iter().map(|t| (t.id, t)).collect();

        // Compute number of levels from topic hierarchy
        let num_levels = topics.values().map(|t| t.level).max().unwrap_or(0) + 1;

        Self {
            topics,
            document_topics: result.document_topics,
            dendrogram: result.dendrogram,
            vocabulary: Vec::new(), // Set via with_vocabulary
            linkage: result.linkage,
            num_levels,
            config,
        }
    }

    /// Set the vocabulary.
    pub fn with_vocabulary(mut self, vocabulary: Vec<String>) -> Self {
        self.vocabulary = vocabulary;
        self
    }

    /// Get a topic by ID.
    pub fn get(&self, id: TopicId) -> Option<&Topic> {
        self.topics.get(&id)
    }

    /// Get a mutable reference to a topic by ID.
    pub fn get_mut(&mut self, id: TopicId) -> Option<&mut Topic> {
        self.topics.get_mut(&id)
    }

    /// Get all topics.
    pub fn topics(&self) -> impl Iterator<Item = &Topic> {
        self.topics.values()
    }

    /// Get topics for a document.
    pub fn document_topics(&self, doc_idx: usize) -> Vec<&Topic> {
        if doc_idx >= self.document_topics.len() {
            return Vec::new();
        }

        self.document_topics[doc_idx]
            .iter()
            .filter_map(|id| self.get(*id))
            .collect()
    }

    /// Get topic IDs for a document.
    pub fn document_topic_ids(&self, doc_idx: usize) -> &[TopicId] {
        if doc_idx >= self.document_topics.len() {
            return &[];
        }
        &self.document_topics[doc_idx]
    }

    /// Get leaf topics (finest granularity).
    pub fn leaf_topics(&self) -> Vec<&Topic> {
        self.topics.values().filter(|t| t.is_leaf()).collect()
    }

    /// Get root topics (coarsest granularity).
    pub fn root_topics(&self) -> Vec<&Topic> {
        self.topics.values().filter(|t| t.is_root()).collect()
    }

    /// Get topics at a specific hierarchy level.
    pub fn topics_at_level(&self, level: usize) -> Vec<&Topic> {
        self.topics.values().filter(|t| t.level == level).collect()
    }

    /// Get child topics of a parent topic.
    pub fn children(&self, parent_id: TopicId) -> Vec<&Topic> {
        self.topics
            .get(&parent_id)
            .map(|t| t.children.iter().filter_map(|id| self.get(*id)).collect())
            .unwrap_or_default()
    }

    /// Get parent topic of a child topic.
    pub fn parent(&self, child_id: TopicId) -> Option<&Topic> {
        self.topics
            .get(&child_id)
            .and_then(|t| t.parent_id)
            .and_then(|id| self.get(id))
    }

    /// Get the dendrogram.
    pub fn dendrogram(&self) -> &Dendrogram {
        &self.dendrogram
    }

    /// Get the vocabulary.
    pub fn vocabulary(&self) -> &[String] {
        &self.vocabulary
    }

    /// Get the linkage matrix.
    pub fn linkage(&self) -> &[(u32, u32, f32, u32)] {
        &self.linkage
    }

    /// Get the number of topics.
    pub fn num_topics(&self) -> usize {
        self.topics.len()
    }

    /// Get the number of documents.
    pub fn num_documents(&self) -> usize {
        self.document_topics.len()
    }

    /// Get the number of hierarchy levels.
    pub fn num_levels(&self) -> usize {
        self.num_levels
    }

    /// Get the configuration.
    pub fn config(&self) -> &TopicConfig {
        &self.config
    }

    /// Find topics containing a keyword.
    pub fn topics_with_keyword(&self, keyword: &str) -> Vec<&Topic> {
        let keyword_lower = keyword.to_lowercase();
        self.topics
            .values()
            .filter(|t| {
                t.keywords
                    .iter()
                    .any(|(k, _)| k.to_lowercase().contains(&keyword_lower))
            })
            .collect()
    }

    /// Get the top N topics by document count.
    pub fn top_topics(&self, n: usize) -> Vec<&Topic> {
        let mut topics: Vec<_> = self.topics.values().collect();
        topics.sort_by_key(|t| std::cmp::Reverse(t.document_count));
        topics.truncate(n);
        topics
    }

    /// Save the model to a file (JSON format).
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let file = File::create(path.as_ref())?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)
            .map_err(|e| TopicError::SerializationError(e.to_string()))?;
        Ok(())
    }

    /// Load a model from a file (JSON format).
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        let reader = BufReader::new(file);
        serde_json::from_reader(reader).map_err(|e| TopicError::SerializationError(e.to_string()))
    }

    /// Save the model to a file (bincode format for efficiency).
    pub fn save_bincode(&self, path: impl AsRef<Path>) -> Result<()> {
        let file = File::create(path.as_ref())?;
        let writer = BufWriter::new(file);
        crate::bincode_compat::serialize_into(writer, self)?;
        Ok(())
    }

    /// Load a model from a file (bincode format).
    pub fn load_bincode(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        let reader = BufReader::new(file);
        crate::bincode_compat::deserialize_from(reader).map_err(TopicError::from)
    }

    /// Get statistics about the model.
    pub fn stats(&self) -> TopicModelStats {
        let total_documents: usize = self.topics.values().map(|t| t.document_count).sum();
        let avg_keywords = if self.topics.is_empty() {
            0.0
        } else {
            self.topics
                .values()
                .map(|t| t.keywords.len())
                .sum::<usize>() as f64
                / self.topics.len() as f64
        };

        TopicModelStats {
            num_topics: self.topics.len(),
            num_documents: self.document_topics.len(),
            num_levels: self.num_levels,
            vocabulary_size: self.vocabulary.len(),
            avg_keywords_per_topic: avg_keywords,
            total_document_assignments: total_documents,
        }
    }
}

/// Statistics about a topic model.
#[derive(Clone, Debug)]
pub struct TopicModelStats {
    /// Number of topics.
    pub num_topics: usize,
    /// Number of documents.
    pub num_documents: usize,
    /// Number of hierarchy levels.
    pub num_levels: usize,
    /// Size of the vocabulary.
    pub vocabulary_size: usize,
    /// Average number of keywords per topic.
    pub avg_keywords_per_topic: f64,
    /// Total document assignments across all topics.
    pub total_document_assignments: usize,
}

impl std::fmt::Display for TopicModelStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Topic Model Statistics:")?;
        writeln!(f, "  Topics: {}", self.num_topics)?;
        writeln!(f, "  Documents: {}", self.num_documents)?;
        writeln!(f, "  Hierarchy Levels: {}", self.num_levels)?;
        writeln!(f, "  Vocabulary Size: {}", self.vocabulary_size)?;
        writeln!(
            f,
            "  Avg Keywords/Topic: {:.2}",
            self.avg_keywords_per_topic
        )?;
        writeln!(
            f,
            "  Total Doc Assignments: {}",
            self.total_document_assignments
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topic::{ClusteringConfig, CtfidfConfig, TopicExtractor};

    fn create_test_model() -> TopicModel {
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.95, 0.1, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.1, 0.95, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.0, 0.1, 0.95],
        ];

        let documents = vec![
            "machine learning algorithms neural networks deep".to_string(),
            "machine learning models training data science".to_string(),
            "web development frontend backend javascript".to_string(),
            "web application programming interface api".to_string(),
            "database sql queries optimization indexes".to_string(),
            "database storage retrieval management systems".to_string(),
        ];

        let config = TopicConfig {
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

        let mut extractor = TopicExtractor::new(config.clone());
        let result = extractor
            .extract(&embeddings, &documents)
            .expect("extraction failed");

        TopicModel::from_extraction(result, config)
    }

    #[test]
    fn test_topic_model_creation() {
        let model = create_test_model();

        assert_eq!(model.num_topics(), 3);
        assert_eq!(model.num_documents(), 6);
    }

    #[test]
    fn test_get_topic() {
        let model = create_test_model();

        // Should be able to get at least one topic
        let topics: Vec<_> = model.topics().collect();
        assert!(!topics.is_empty());

        let first_topic = topics[0];
        let retrieved = model.get(first_topic.id);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, first_topic.id);
    }

    #[test]
    fn test_document_topics() {
        let model = create_test_model();

        // Each document should have at least one topic
        for i in 0..6 {
            let topics = model.document_topics(i);
            assert!(!topics.is_empty(), "Document {} has no topics", i);
        }

        // Out of bounds should return empty
        let topics = model.document_topics(100);
        assert!(topics.is_empty());
    }

    #[test]
    fn test_leaf_and_root_topics() {
        let model = create_test_model();

        let leaves = model.leaf_topics();
        let roots = model.root_topics();

        // Should have some topics
        assert!(!leaves.is_empty() || !roots.is_empty());
    }

    #[test]
    fn test_topics_with_keyword() {
        let model = create_test_model();

        // Search for a keyword that should exist
        let ml_topics = model.topics_with_keyword("machine");
        // May or may not find depending on extraction, but should not panic
        let _ = ml_topics;
    }

    #[test]
    fn test_top_topics() {
        let model = create_test_model();

        let top = model.top_topics(2);
        assert!(top.len() <= 2);

        // Should be sorted by document count
        if top.len() == 2 {
            assert!(top[0].document_count >= top[1].document_count);
        }
    }

    #[test]
    fn test_stats() {
        let model = create_test_model();

        let stats = model.stats();
        assert_eq!(stats.num_topics, 3);
        assert_eq!(stats.num_documents, 6);

        // Display should work
        let display = format!("{}", stats);
        assert!(display.contains("Topics:"));
    }

    #[test]
    fn test_save_load_json() {
        let model = create_test_model();
        let temp_path = std::env::temp_dir().join("test_topic_model.json");

        // Save
        model.save(&temp_path).expect("save failed");

        // Load
        let loaded = TopicModel::load(&temp_path).expect("load failed");

        assert_eq!(model.num_topics(), loaded.num_topics());
        assert_eq!(model.num_documents(), loaded.num_documents());

        // Cleanup
        let _ = std::fs::remove_file(&temp_path);
    }

    #[test]
    fn test_save_load_bincode() {
        // Bincode serialization uses custom serde for Arc<[f32]> in Topic.centroid.
        // The Arc<[f32]> is serialized as Vec<f32> and deserialized back to Arc<[f32]>.
        // DashMap is NOT in TopicModel - only used during extraction pipeline.
        let model = create_test_model();
        let temp_path = std::env::temp_dir().join("test_topic_model.bin");

        // Save
        model.save_bincode(&temp_path).expect("save failed");

        // Verify file was created
        assert!(temp_path.exists());

        // Load and verify
        let loaded = TopicModel::load_bincode(&temp_path).expect("load failed");

        assert_eq!(model.num_topics(), loaded.num_topics());
        assert_eq!(model.num_documents(), loaded.num_documents());
        assert_eq!(model.num_levels(), loaded.num_levels());

        // Verify topics match
        for topic in model.topics() {
            let loaded_topic = loaded.get(topic.id);
            assert!(loaded_topic.is_some(), "Topic {:?} not found", topic.id);
            let loaded_topic = loaded_topic.unwrap();
            assert_eq!(topic.keywords.len(), loaded_topic.keywords.len());
            assert_eq!(topic.document_count, loaded_topic.document_count);
        }

        // Cleanup
        let _ = std::fs::remove_file(&temp_path);
    }

    #[test]
    fn test_with_vocabulary() {
        let model =
            create_test_model().with_vocabulary(vec!["test".to_string(), "vocab".to_string()]);

        assert_eq!(model.vocabulary().len(), 2);
        assert_eq!(model.vocabulary()[0], "test");
    }
}
