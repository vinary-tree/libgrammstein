//! Topic data structures for BERTopic-like topic modeling.
//!
//! This module defines the core topic types used throughout the topic extraction system.

use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Unique identifier for a topic.
///
/// Topics are identified by a 32-bit unsigned integer. Topic IDs are assigned
/// during the hierarchical agglomerative clustering process, with lower IDs
/// typically representing leaf topics and higher IDs representing merged topics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TopicId(pub u32);

impl TopicId {
    /// Create a new topic ID.
    #[inline]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the raw ID value.
    #[inline]
    pub const fn as_u32(&self) -> u32 {
        self.0
    }

    /// Check if this is a leaf topic (ID < num_documents).
    #[inline]
    pub const fn is_leaf(&self, num_documents: u32) -> bool {
        self.0 < num_documents
    }
}

impl From<u32> for TopicId {
    fn from(id: u32) -> Self {
        Self(id)
    }
}

impl From<TopicId> for u32 {
    fn from(id: TopicId) -> Self {
        id.0
    }
}

impl std::fmt::Display for TopicId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Topic({})", self.0)
    }
}

/// A topic extracted from a document collection.
///
/// Topics contain:
/// - Keywords extracted via c-TF-IDF
/// - A natural language description
/// - Hierarchical structure (parent/children relationships)
/// - Cluster centroid embedding
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Topic {
    /// Unique topic identifier.
    pub id: TopicId,
    /// Parent topic ID (for hierarchy). None for root topics.
    pub parent_id: Option<TopicId>,
    /// Child topic IDs (for hierarchy).
    pub children: Vec<TopicId>,
    /// Hierarchy level (0 = root, increasing = deeper in tree).
    pub level: usize,
    /// Top keywords with their c-TF-IDF scores.
    pub keywords: Vec<(String, f32)>,
    /// Natural language description of the topic.
    pub description: String,
    /// Cluster centroid embedding (normalized).
    /// Note: Uses custom serde for Arc<[f32]> <-> Vec<f32> conversion.
    /// Always serializes (no skip_serializing_if) for bincode compatibility.
    #[serde(
        serialize_with = "serialize_arc_slice",
        deserialize_with = "deserialize_arc_slice",
        default
    )]
    pub centroid: Option<Arc<[f32]>>,
    /// Number of documents in this topic.
    pub document_count: usize,
    /// Topic coherence score (optional).
    /// Always serializes (no skip_serializing_if) for bincode compatibility.
    pub coherence: Option<f32>,
}

/// Serialize Option<Arc<[f32]>> as Option<Vec<f32>>.
fn serialize_arc_slice<S>(value: &Option<Arc<[f32]>>, serializer: S) -> std::result::Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    match value {
        Some(arc) => {
            let vec: Vec<f32> = arc.iter().copied().collect();
            serializer.serialize_some(&vec)
        }
        None => serializer.serialize_none(),
    }
}

/// Deserialize Option<Arc<[f32]>> from Option<Vec<f32>>.
fn deserialize_arc_slice<'de, D>(deserializer: D) -> std::result::Result<Option<Arc<[f32]>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let opt: Option<Vec<f32>> = Option::deserialize(deserializer)?;
    Ok(opt.map(|v| Arc::from(v.into_boxed_slice())))
}

impl Topic {
    /// Create a new topic with just an ID (builder pattern).
    pub fn new(id: TopicId) -> Self {
        Self {
            id,
            parent_id: None,
            children: Vec::new(),
            level: 0,
            keywords: Vec::new(),
            description: String::new(),
            centroid: None,
            document_count: 0,
            coherence: None,
        }
    }

    /// Create a new leaf topic.
    pub fn new_leaf(id: TopicId, keywords: Vec<(String, f32)>, centroid: Vec<f32>) -> Self {
        Self {
            id,
            parent_id: None,
            children: Vec::new(),
            level: 0,
            keywords,
            description: String::new(),
            centroid: Some(centroid.into()),
            document_count: 1,
            coherence: None,
        }
    }

    /// Create a new merged topic from two child topics.
    pub fn new_merged(
        id: TopicId,
        left: TopicId,
        right: TopicId,
        level: usize,
        keywords: Vec<(String, f32)>,
        centroid: Vec<f32>,
        document_count: usize,
    ) -> Self {
        Self {
            id,
            parent_id: None,
            children: vec![left, right],
            level,
            keywords,
            description: String::new(),
            centroid: Some(centroid.into()),
            document_count,
            coherence: None,
        }
    }

    /// Check if this is a leaf topic.
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    /// Check if this is a root topic.
    #[inline]
    pub fn is_root(&self) -> bool {
        self.parent_id.is_none()
    }

    /// Get a summary of the top N keywords.
    pub fn keyword_summary(&self, n: usize) -> String {
        self.keywords
            .iter()
            .take(n)
            .map(|(word, _)| word.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Set the topic description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Set the keywords.
    pub fn with_keywords(mut self, keywords: Vec<(String, f32)>) -> Self {
        self.keywords = keywords;
        self
    }

    /// Set the centroid embedding.
    pub fn with_centroid(mut self, centroid: Arc<[f32]>) -> Self {
        self.centroid = Some(centroid);
        self
    }

    /// Set the document count.
    pub fn with_document_count(mut self, count: usize) -> Self {
        self.document_count = count;
        self
    }

    /// Set the parent topic ID.
    pub fn with_parent(mut self, parent_id: TopicId) -> Self {
        self.parent_id = Some(parent_id);
        self
    }

    /// Set the coherence score.
    pub fn with_coherence(mut self, coherence: f32) -> Self {
        self.coherence = Some(coherence);
        self
    }

    /// Get the centroid embedding if available.
    pub fn centroid(&self) -> Option<&[f32]> {
        self.centroid.as_ref().map(|c| c.as_ref())
    }
}

impl std::fmt::Display for Topic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.description.is_empty() {
            write!(f, "{}: {}", self.id, self.keyword_summary(5))
        } else {
            write!(f, "{}: {}", self.id, self.description)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topic_id_creation() {
        let id = TopicId::new(42);
        assert_eq!(id.as_u32(), 42);
        assert_eq!(format!("{}", id), "Topic(42)");
    }

    #[test]
    fn test_topic_id_leaf_check() {
        let id = TopicId::new(5);
        assert!(id.is_leaf(10));
        assert!(!id.is_leaf(5));
        assert!(!id.is_leaf(3));
    }

    #[test]
    fn test_topic_leaf() {
        let keywords = vec![
            ("machine".to_string(), 0.9),
            ("learning".to_string(), 0.8),
            ("algorithm".to_string(), 0.7),
        ];
        let centroid = vec![0.1, 0.2, 0.3];
        let topic = Topic::new_leaf(TopicId::new(0), keywords, centroid);

        assert!(topic.is_leaf());
        assert!(topic.is_root());
        assert_eq!(topic.document_count, 1);
        assert_eq!(topic.keyword_summary(2), "machine, learning");
    }

    #[test]
    fn test_topic_merged() {
        let keywords = vec![
            ("data".to_string(), 0.85),
            ("science".to_string(), 0.75),
        ];
        let centroid = vec![0.15, 0.25, 0.35];
        let topic = Topic::new_merged(
            TopicId::new(10),
            TopicId::new(1),
            TopicId::new(2),
            1,
            keywords,
            centroid,
            5,
        );

        assert!(!topic.is_leaf());
        assert!(topic.is_root());
        assert_eq!(topic.children.len(), 2);
        assert_eq!(topic.document_count, 5);
    }

    #[test]
    fn test_topic_with_description() {
        let topic = Topic::new_leaf(TopicId::new(0), vec![], vec![])
            .with_description("Machine Learning and AI");

        assert_eq!(topic.description, "Machine Learning and AI");
        assert_eq!(format!("{}", topic), "Topic(0): Machine Learning and AI");
    }

    #[test]
    fn test_topic_serialization() {
        let keywords = vec![("test".to_string(), 0.5)];
        let topic = Topic::new_leaf(TopicId::new(1), keywords, vec![0.1, 0.2]);

        let json = serde_json::to_string(&topic).expect("serialization failed");
        let deserialized: Topic = serde_json::from_str(&json).expect("deserialization failed");

        assert_eq!(topic.id, deserialized.id);
        assert_eq!(topic.keywords.len(), deserialized.keywords.len());
    }
}
