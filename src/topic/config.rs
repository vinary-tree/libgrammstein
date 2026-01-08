//! Configuration for topic extraction and modeling.
//!
//! This module defines all configuration structures used by the topic
//! extraction system, including clustering, c-TF-IDF, and summarization settings.

use serde::{Deserialize, Serialize};

/// Linkage method for hierarchical agglomerative clustering.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum LinkageMethod {
    /// Ward's minimum variance method (default).
    /// Minimizes the total within-cluster variance.
    /// Best for producing compact, spherical clusters.
    #[default]
    Ward,
    /// Complete (maximum) linkage.
    /// Distance between clusters is the maximum distance between points.
    /// Produces compact, spherical clusters.
    Complete,
    /// Average linkage (UPGMA).
    /// Distance is the average of all pairwise distances.
    /// Good balance between single and complete linkage.
    Average,
    /// Single (minimum) linkage.
    /// Distance is the minimum distance between points.
    /// Can produce elongated, chain-like clusters.
    Single,
}

/// Configuration for hierarchical agglomerative clustering.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClusteringConfig {
    /// Target number of clusters. If None, determined automatically.
    pub num_clusters: Option<usize>,
    /// Distance threshold for cutting dendrogram. If None, uses num_clusters.
    pub distance_threshold: Option<f32>,
    /// Linkage method for computing cluster distances.
    pub linkage: LinkageMethod,
    /// Minimum number of documents in a cluster.
    pub min_cluster_size: usize,
    /// Enable parallel distance matrix computation.
    pub parallel: bool,
    /// Checkpoint interval (merge steps between saves). 0 = disabled.
    pub checkpoint_interval: usize,
    /// Enable verbose progress output.
    pub verbose: bool,
}

impl Default for ClusteringConfig {
    fn default() -> Self {
        Self {
            num_clusters: Some(10),
            distance_threshold: None,
            linkage: LinkageMethod::Ward,
            min_cluster_size: 5,
            parallel: true,
            checkpoint_interval: 100,
            verbose: false,
        }
    }
}

impl ClusteringConfig {
    /// Create config with a specific number of clusters.
    pub fn with_num_clusters(num_clusters: usize) -> Self {
        Self {
            num_clusters: Some(num_clusters),
            ..Default::default()
        }
    }

    /// Create config with a distance threshold.
    pub fn with_distance_threshold(threshold: f32) -> Self {
        Self {
            num_clusters: None,
            distance_threshold: Some(threshold),
            ..Default::default()
        }
    }
}

/// Configuration for c-TF-IDF keyword extraction.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CtfidfConfig {
    /// Number of keywords to extract per topic.
    pub num_keywords: usize,
    /// Minimum document frequency for a term to be included.
    pub min_df: usize,
    /// Maximum document frequency ratio (0.0 to 1.0) for filtering.
    pub max_df_ratio: f32,
    /// N-gram range (min, max).
    pub ngram_range: (usize, usize),
    /// Use sublinear TF scaling (1 + log(tf)).
    pub sublinear_tf: bool,
    /// Minimum term length.
    pub min_term_length: usize,
    /// Maximum term length.
    pub max_term_length: usize,
}

impl Default for CtfidfConfig {
    fn default() -> Self {
        Self {
            num_keywords: 10,
            min_df: 2,
            max_df_ratio: 0.95,
            ngram_range: (1, 1),
            sublinear_tf: true,
            min_term_length: 2,
            max_term_length: 50,
        }
    }
}

impl CtfidfConfig {
    /// Create config with custom number of keywords.
    pub fn with_num_keywords(num_keywords: usize) -> Self {
        Self {
            num_keywords,
            ..Default::default()
        }
    }

    /// Enable bigrams and trigrams.
    pub fn with_ngrams(max_n: usize) -> Self {
        Self {
            ngram_range: (1, max_n),
            ..Default::default()
        }
    }
}

/// Configuration for topic description generation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SummarizationConfig {
    /// Template type for generating descriptions from keywords.
    pub template: DescriptionTemplateType,
    /// Custom template string when template is DescriptionTemplateType::Custom.
    /// Use {keywords} placeholder for keyword insertion.
    #[serde(default)]
    pub custom_template: Option<String>,
    /// Number of representative documents to use for extractive summary.
    pub num_representative_docs: usize,
    /// Maximum description length in characters.
    pub max_description_length: usize,
    /// Include coherence scores in descriptions.
    pub include_coherence: bool,
}

impl Default for SummarizationConfig {
    fn default() -> Self {
        Self {
            template: DescriptionTemplateType::Keywords,
            custom_template: None,
            num_representative_docs: 3,
            max_description_length: 200,
            include_coherence: false,
        }
    }
}

/// Legacy type alias for backwards compatibility.
pub type DescriptionTemplate = DescriptionTemplateType;

/// Template type for generating topic descriptions.
///
/// For custom templates, set the type to `Custom` and provide the template
/// string in `SummarizationConfig::custom_template`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum DescriptionTemplateType {
    /// Simple keyword list: "keyword1, keyword2, keyword3"
    #[default]
    Keywords = 0,
    /// Topic label: "Topic covering: keyword1, keyword2, keyword3"
    Label = 1,
    /// Extractive summary from representative documents.
    Extractive = 2,
    /// Custom template string (use custom_template field for the pattern).
    Custom = 3,
}

/// Main configuration for topic extraction.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TopicConfig {
    /// Clustering configuration.
    pub clustering: ClusteringConfig,
    /// c-TF-IDF configuration.
    pub ctfidf: CtfidfConfig,
    /// Summarization configuration.
    pub summarization: SummarizationConfig,
    /// Number of hierarchy levels to retain.
    pub hierarchy_levels: usize,
    /// Minimum topic size (documents).
    pub min_topic_size: usize,
    /// Compute topic coherence scores.
    pub compute_coherence: bool,
    /// Enable verbose progress output.
    pub verbose: bool,
}

impl Default for TopicConfig {
    fn default() -> Self {
        Self {
            clustering: ClusteringConfig::default(),
            ctfidf: CtfidfConfig::default(),
            summarization: SummarizationConfig::default(),
            hierarchy_levels: 3,
            min_topic_size: 5,
            compute_coherence: true,
            verbose: false,
        }
    }
}

impl TopicConfig {
    /// Create configuration with specific cluster count.
    pub fn with_num_clusters(num_clusters: usize) -> Self {
        Self {
            clustering: ClusteringConfig::with_num_clusters(num_clusters),
            ..Default::default()
        }
    }

    /// Create configuration optimized for small corpora (< 1000 docs).
    pub fn for_small_corpus() -> Self {
        Self {
            clustering: ClusteringConfig {
                num_clusters: Some(5),
                min_cluster_size: 2,
                ..Default::default()
            },
            ctfidf: CtfidfConfig {
                min_df: 1,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Create configuration optimized for large corpora (> 10000 docs).
    pub fn for_large_corpus() -> Self {
        Self {
            clustering: ClusteringConfig {
                num_clusters: Some(50),
                min_cluster_size: 10,
                checkpoint_interval: 500,
                ..Default::default()
            },
            ctfidf: CtfidfConfig {
                min_df: 5,
                max_df_ratio: 0.8,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Enable checkpointing with specified interval.
    pub fn with_checkpointing(mut self, interval: usize) -> Self {
        self.clustering.checkpoint_interval = interval;
        self
    }

    /// Set verbose mode.
    pub fn verbose(mut self) -> Self {
        self.verbose = true;
        self
    }
}

/// Configuration for the complete topic model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TopicModelConfig {
    /// Topic extraction configuration.
    pub topic_config: TopicConfig,
    /// Embedding dimension (must match index embeddings).
    pub embedding_dim: usize,
}

impl Default for TopicModelConfig {
    fn default() -> Self {
        Self {
            topic_config: TopicConfig::default(),
            embedding_dim: 768, // Default ModernBERT dimension
        }
    }
}

impl TopicModelConfig {
    /// Create with specific embedding dimension.
    pub fn with_embedding_dim(embedding_dim: usize) -> Self {
        Self {
            embedding_dim,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clustering_config_default() {
        let config = ClusteringConfig::default();
        assert_eq!(config.num_clusters, Some(10));
        assert_eq!(config.linkage, LinkageMethod::Ward);
        assert!(config.parallel);
    }

    #[test]
    fn test_ctfidf_config_default() {
        let config = CtfidfConfig::default();
        assert_eq!(config.num_keywords, 10);
        assert_eq!(config.ngram_range, (1, 1));
        assert!(config.sublinear_tf);
    }

    #[test]
    fn test_topic_config_small_corpus() {
        let config = TopicConfig::for_small_corpus();
        assert_eq!(config.clustering.num_clusters, Some(5));
        assert_eq!(config.clustering.min_cluster_size, 2);
        assert_eq!(config.ctfidf.min_df, 1);
    }

    #[test]
    fn test_topic_config_large_corpus() {
        let config = TopicConfig::for_large_corpus();
        assert_eq!(config.clustering.num_clusters, Some(50));
        assert_eq!(config.clustering.checkpoint_interval, 500);
    }

    #[test]
    fn test_config_serialization() {
        let config = TopicConfig::default();
        let json = serde_json::to_string(&config).expect("serialization failed");
        let deserialized: TopicConfig =
            serde_json::from_str(&json).expect("deserialization failed");
        assert_eq!(
            config.clustering.num_clusters,
            deserialized.clustering.num_clusters
        );
    }
}
