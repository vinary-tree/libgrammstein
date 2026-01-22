//! Topic extraction orchestration.
//!
//! This module coordinates the full topic extraction pipeline:
//! 1. Clustering documents by embedding similarity
//! 2. Extracting keywords with c-TF-IDF
//! 3. Generating topic descriptions

use std::path::Path;
use std::sync::Arc;

use super::checkpoint::{ExtractionPhase, TopicExtractionCheckpoint};
use super::clustering::{compute_distance_matrix_parallel, ClusteringResult, HierarchicalClustering};
use super::config::TopicConfig;
use super::ctfidf::CtfIdf;
use super::dendrogram::Dendrogram;
use super::summarizer::TopicSummarizer;
use super::topic::{Topic, TopicId};
use super::{Result, TopicError};

/// Topic extraction result.
#[derive(Clone, Debug)]
pub struct ExtractionResult {
    /// Extracted topics.
    pub topics: Vec<Topic>,
    /// Document to topic assignments.
    pub document_topics: Vec<Vec<TopicId>>,
    /// Dendrogram for hierarchy.
    pub dendrogram: Dendrogram,
    /// Linkage matrix.
    pub linkage: Vec<(u32, u32, f32, u32)>,
}

impl ExtractionResult {
    /// Get topics at a specific hierarchy level.
    pub fn topics_at_level(&self, level: usize) -> Vec<&Topic> {
        self.topics.iter().filter(|t| t.level == level).collect()
    }

    /// Get leaf topics (finest granularity).
    pub fn leaf_topics(&self) -> Vec<&Topic> {
        self.topics.iter().filter(|t| t.children.is_empty()).collect()
    }

    /// Get root topics (coarsest granularity).
    pub fn root_topics(&self) -> Vec<&Topic> {
        self.topics.iter().filter(|t| t.parent_id.is_none()).collect()
    }

    /// Get topic by ID.
    pub fn get_topic(&self, id: TopicId) -> Option<&Topic> {
        self.topics.iter().find(|t| t.id == id)
    }

    /// Get topics for a document.
    pub fn topics_for_document(&self, doc_idx: usize) -> Vec<&Topic> {
        if doc_idx >= self.document_topics.len() {
            return Vec::new();
        }

        self.document_topics[doc_idx]
            .iter()
            .filter_map(|&id| self.get_topic(id))
            .collect()
    }
}

/// Topic extractor orchestrating the full pipeline.
pub struct TopicExtractor {
    /// Configuration.
    config: TopicConfig,
    /// Current checkpoint (if any).
    checkpoint: Option<TopicExtractionCheckpoint>,
    /// Checkpoint path.
    checkpoint_path: Option<std::path::PathBuf>,
}

impl TopicExtractor {
    /// Create a new topic extractor.
    pub fn new(config: TopicConfig) -> Self {
        Self {
            config,
            checkpoint: None,
            checkpoint_path: None,
        }
    }

    /// Set checkpoint path for resumable extraction.
    pub fn with_checkpoint_path(mut self, path: impl AsRef<Path>) -> Self {
        self.checkpoint_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Extract topics from embeddings and document texts.
    ///
    /// # Arguments
    /// * `embeddings` - Document embeddings (one per document)
    /// * `documents` - Document texts for keyword extraction
    ///
    /// # Returns
    /// Topic extraction result with topics and assignments
    pub fn extract(
        &mut self,
        embeddings: &[Vec<f32>],
        documents: &[String],
    ) -> Result<ExtractionResult> {
        let n = embeddings.len();

        if n != documents.len() {
            return Err(TopicError::ClusteringError(format!(
                "Embedding count ({}) != document count ({})",
                n,
                documents.len()
            )));
        }

        if n < 2 {
            return Err(TopicError::InsufficientDocuments {
                minimum: 2,
                actual: n,
            });
        }

        // Check embedding dimension consistency
        let embedding_dim = embeddings[0].len();
        for (_idx, emb) in embeddings.iter().enumerate() {
            if emb.len() != embedding_dim {
                return Err(TopicError::DimensionMismatch {
                    expected: embedding_dim,
                    actual: emb.len(),
                });
            }
        }

        // Initialize checkpoint
        let mut checkpoint = TopicExtractionCheckpoint::new(n, embedding_dim);

        // Phase 1: Clustering
        let clustering_result = self.run_clustering(embeddings, &mut checkpoint)?;

        // Phase 2: Build vocabulary and extract keywords
        let (keywords_per_topic, vocabulary) =
            self.run_keyword_extraction(documents, &clustering_result.assignments, &mut checkpoint)?;

        // Phase 3: Generate topic descriptions
        let topics = self.build_topics(
            &clustering_result,
            &keywords_per_topic,
            embeddings,
        )?;

        // Build document-topic assignments
        let document_topics = self.build_document_topics(&clustering_result.assignments);

        // Mark complete
        checkpoint.mark_complete();
        checkpoint.compute_checksum();

        if let Some(ref path) = self.checkpoint_path {
            let _ = checkpoint.save(path); // Ignore errors on final save
        }

        Ok(ExtractionResult {
            topics,
            document_topics,
            dendrogram: clustering_result.dendrogram,
            linkage: clustering_result.linkage,
        })
    }

    /// Run clustering phase.
    fn run_clustering(
        &self,
        embeddings: &[Vec<f32>],
        checkpoint: &mut TopicExtractionCheckpoint,
    ) -> Result<ClusteringResult> {
        checkpoint.phase = ExtractionPhase::DistanceMatrix;

        // Compute distance matrix
        let dist_matrix = compute_distance_matrix_parallel(embeddings, None);

        // Store distance matrix in checkpoint
        checkpoint.distance_matrix = Some(dist_matrix.to_vec());
        checkpoint.phase = ExtractionPhase::Clustering;

        // Save checkpoint if path configured
        if let Some(ref path) = self.checkpoint_path {
            checkpoint.compute_checksum();
            let _ = checkpoint.save(path);
        }

        // Run clustering
        let clustering = HierarchicalClustering::new(self.config.clustering.clone());
        let result = clustering.cluster_from_distances(&dist_matrix)?;

        // Update checkpoint with linkage
        checkpoint.linkage_matrix = result.linkage.clone();
        checkpoint.cluster_assignments = result.assignments.clone();

        if let Some(ref path) = self.checkpoint_path {
            checkpoint.compute_checksum();
            let _ = checkpoint.save(path);
        }

        Ok(result)
    }

    /// Run keyword extraction phase.
    fn run_keyword_extraction(
        &self,
        documents: &[String],
        assignments: &[u32],
        checkpoint: &mut TopicExtractionCheckpoint,
    ) -> Result<(Vec<Vec<(String, f32)>>, Vec<String>)> {
        checkpoint.phase = ExtractionPhase::VocabularyBuild;

        // Build c-TF-IDF
        let mut ctfidf = CtfIdf::new(self.config.ctfidf.clone());
        ctfidf.build_vocabulary(documents, assignments)?;

        // Store vocabulary in checkpoint
        checkpoint.vocabulary = ctfidf.export_vocabulary();

        if let Some(ref path) = self.checkpoint_path {
            checkpoint.compute_checksum();
            let _ = checkpoint.save(path);
        }

        checkpoint.phase = ExtractionPhase::KeywordExtraction;

        // Extract keywords for all topics
        let keywords = ctfidf.extract_all_keywords();

        // Store term frequencies in checkpoint
        if let Some(tf) = ctfidf.export_term_frequencies() {
            checkpoint.term_frequencies = tf;
        }

        if let Some(ref path) = self.checkpoint_path {
            checkpoint.compute_checksum();
            let _ = checkpoint.save(path);
        }

        Ok((keywords, checkpoint.vocabulary.clone()))
    }

    /// Build Topic structs from clustering and keywords.
    fn build_topics(
        &self,
        clustering_result: &ClusteringResult,
        keywords_per_topic: &[Vec<(String, f32)>],
        embeddings: &[Vec<f32>],
    ) -> Result<Vec<Topic>> {
        let summarizer = TopicSummarizer::new(self.config.summarization.clone());
        let mut topics = Vec::new();

        // Get unique clusters from assignments
        let unique_clusters = Dendrogram::unique_clusters(&clustering_result.assignments);

        for &cluster_id in &unique_clusters {
            let topic_id = TopicId::new(cluster_id);

            // Get keywords for this cluster
            let keywords = if (cluster_id as usize) < keywords_per_topic.len() {
                keywords_per_topic[cluster_id as usize].clone()
            } else {
                Vec::new()
            };

            // Generate description
            let description = summarizer.describe_from_keywords(&keywords);

            // Compute centroid
            let centroid = self.compute_centroid(cluster_id, &clustering_result.assignments, embeddings);

            // Count documents
            let document_count = clustering_result
                .assignments
                .iter()
                .filter(|&&a| a == cluster_id)
                .count();

            let topic = Topic::new(topic_id)
                .with_keywords(keywords)
                .with_description(description)
                .with_document_count(document_count);

            let topic = if let Some(centroid) = centroid {
                topic.with_centroid(Arc::from(centroid))
            } else {
                topic
            };

            topics.push(topic);
        }

        Ok(topics)
    }

    /// Compute centroid for a cluster.
    fn compute_centroid(
        &self,
        cluster_id: u32,
        assignments: &[u32],
        embeddings: &[Vec<f32>],
    ) -> Option<Vec<f32>> {
        let indices: Vec<usize> = assignments
            .iter()
            .enumerate()
            .filter(|(_, &a)| a == cluster_id)
            .map(|(i, _)| i)
            .collect();

        if indices.is_empty() {
            return None;
        }

        let dim = embeddings[0].len();
        let mut centroid = vec![0.0f32; dim];
        let count = indices.len() as f32;

        for idx in indices {
            for (i, &val) in embeddings[idx].iter().enumerate() {
                centroid[i] += val;
            }
        }

        for val in &mut centroid {
            *val /= count;
        }

        Some(centroid)
    }

    /// Build document-topic assignments.
    fn build_document_topics(&self, assignments: &[u32]) -> Vec<Vec<TopicId>> {
        assignments
            .iter()
            .map(|&cluster| vec![TopicId::new(cluster)])
            .collect()
    }

    /// Resume extraction from a checkpoint.
    pub fn resume(checkpoint_path: &Path) -> Result<(Self, TopicExtractionCheckpoint)> {
        let checkpoint = TopicExtractionCheckpoint::load(checkpoint_path)?;

        let config = TopicConfig::default(); // Would need to store config in checkpoint

        let extractor = Self {
            config,
            checkpoint: Some(checkpoint.clone()),
            checkpoint_path: Some(checkpoint_path.to_path_buf()),
        };

        Ok((extractor, checkpoint))
    }

    /// Get configuration.
    pub fn config(&self) -> &TopicConfig {
        &self.config
    }
}

impl Default for TopicExtractor {
    fn default() -> Self {
        Self::new(TopicConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_embeddings() -> Vec<Vec<f32>> {
        vec![
            vec![1.0, 0.0, 0.0],
            vec![0.95, 0.1, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.1, 0.95, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.0, 0.1, 0.95],
        ]
    }

    fn make_test_documents() -> Vec<String> {
        vec![
            "machine learning algorithms neural networks deep".to_string(),
            "machine learning models training data science".to_string(),
            "web development frontend backend javascript".to_string(),
            "web application programming interface api".to_string(),
            "database sql queries optimization indexes".to_string(),
            "database storage retrieval management systems".to_string(),
        ]
    }

    #[test]
    fn test_topic_extractor_basic() {
        let config = TopicConfig {
            clustering: super::super::config::ClusteringConfig {
                num_clusters: Some(3),
                ..Default::default()
            },
            ctfidf: super::super::config::CtfidfConfig {
                num_keywords: 3,
                min_df: 1,
                min_term_length: 2,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut extractor = TopicExtractor::new(config);
        let embeddings = make_test_embeddings();
        let documents = make_test_documents();

        let result = extractor.extract(&embeddings, &documents).expect("extraction failed");

        // Should have 3 topics
        assert_eq!(result.topics.len(), 3);

        // Each document should have an assignment
        assert_eq!(result.document_topics.len(), 6);

        // All documents should have at least one topic
        for topics in &result.document_topics {
            assert!(!topics.is_empty());
        }
    }

    #[test]
    fn test_extraction_result_methods() {
        let config = TopicConfig {
            clustering: super::super::config::ClusteringConfig {
                num_clusters: Some(2),
                ..Default::default()
            },
            ctfidf: super::super::config::CtfidfConfig {
                min_df: 1,
                min_term_length: 2,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut extractor = TopicExtractor::new(config);
        let embeddings = make_test_embeddings();
        let documents = make_test_documents();

        let result = extractor.extract(&embeddings, &documents).expect("extraction failed");

        // Test leaf_topics
        let leaves = result.leaf_topics();
        assert!(!leaves.is_empty());

        // Test topics_for_document
        let doc_topics = result.topics_for_document(0);
        assert!(!doc_topics.is_empty());

        // Test get_topic
        let topic_id = result.document_topics[0][0];
        let topic = result.get_topic(topic_id);
        assert!(topic.is_some());
    }

    #[test]
    fn test_extraction_error_mismatch() {
        let mut extractor = TopicExtractor::default();

        let embeddings = vec![vec![1.0, 0.0]];
        let documents = vec!["doc1".to_string(), "doc2".to_string()];

        let result = extractor.extract(&embeddings, &documents);
        assert!(result.is_err());
    }

    #[test]
    fn test_extraction_insufficient_documents() {
        let mut extractor = TopicExtractor::default();

        let embeddings = vec![vec![1.0, 0.0]];
        let documents = vec!["single doc".to_string()];

        let result = extractor.extract(&embeddings, &documents);
        assert!(matches!(result, Err(TopicError::InsufficientDocuments { .. })));
    }

    #[test]
    fn test_compute_centroid() {
        let extractor = TopicExtractor::default();

        let embeddings = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];
        let assignments = vec![0, 0, 1];

        // Cluster 0 has embeddings [1,0] and [0,1], centroid = [0.5, 0.5]
        let centroid = extractor.compute_centroid(0, &assignments, &embeddings);
        assert!(centroid.is_some());
        let c = centroid.unwrap();
        assert!((c[0] - 0.5).abs() < 1e-6);
        assert!((c[1] - 0.5).abs() < 1e-6);

        // Cluster 1 has embedding [1,1]
        let centroid = extractor.compute_centroid(1, &assignments, &embeddings);
        assert!(centroid.is_some());
        let c = centroid.unwrap();
        assert!((c[0] - 1.0).abs() < 1e-6);
        assert!((c[1] - 1.0).abs() < 1e-6);
    }
}
