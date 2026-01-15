//! Graph Neural Network for semantic code analysis.
//!
//! This module provides GNN-based semantic analysis for code, operating on
//! Code Property Graphs (CPGs) to detect semantic issues like:
//!
//! - Variable misuse (using wrong variable in context)
//! - Type errors (mismatched types in operations)
//! - API misuse (wrong method calls, missing error handling)
//! - Bug localization (identifying likely buggy regions)
//!
//! The architecture follows GNN-Coder (2025) and uses graph convolutions
//! over CPG representations to learn semantic patterns.

use super::cpg::{CodePropertyGraph, CpgEdgeKind, CpgNode, CpgNodeKind};
use std::collections::HashMap;

/// Configuration for GNN semantic scoring.
#[derive(Debug, Clone)]
pub struct GnnConfig {
    /// Number of GNN layers (message passing iterations)
    pub num_layers: usize,
    /// Hidden dimension size
    pub hidden_dim: usize,
    /// Dropout rate for training
    pub dropout: f64,
    /// Whether to use edge features
    pub use_edge_features: bool,
    /// Whether to use attention mechanism
    pub use_attention: bool,
    /// Embedding dimension for node features
    pub embedding_dim: usize,
}

impl Default for GnnConfig {
    fn default() -> Self {
        Self {
            num_layers: 3,
            hidden_dim: 256,
            dropout: 0.1,
            use_edge_features: true,
            use_attention: true,
            embedding_dim: 128,
        }
    }
}

/// Node feature vector for GNN input.
#[derive(Debug, Clone)]
pub struct NodeFeatures {
    /// Node index in the CPG
    pub node_idx: usize,
    /// Token/lexical features
    pub token_features: Vec<f32>,
    /// Structural features (depth, child count, etc.)
    pub structural_features: Vec<f32>,
    /// Type features (if available)
    pub type_features: Vec<f32>,
}

impl NodeFeatures {
    /// Creates features from a CPG node.
    pub fn from_cpg_node(node: &CpgNode, depth: usize, child_count: usize) -> Self {
        let mut structural = Vec::with_capacity(8);

        // Depth in AST (normalized)
        structural.push((depth as f32) / 20.0);

        // Child count (normalized)
        structural.push((child_count as f32) / 10.0);

        // Byte span (normalized)
        let span_len = (node.location.1 - node.location.0) as f32;
        structural.push(span_len / 1000.0);

        // Node kind encoding (simplified one-hot for top kinds)
        let kind_encoding = match node.kind {
            CpgNodeKind::Function => 0,
            CpgNodeKind::Variable => 1,
            CpgNodeKind::Call => 2,
            CpgNodeKind::Branch => 3,
            CpgNodeKind::Loop => 4,
            CpgNodeKind::Assignment => 5,
            CpgNodeKind::Return => 6,
            _ => 7,
        };
        structural.push(kind_encoding as f32 / 8.0);

        Self {
            node_idx: node.id,
            token_features: Vec::new(), // Populated by embedder
            structural_features: structural,
            type_features: Vec::new(), // Populated by type checker
        }
    }

    /// Returns the total feature dimension.
    pub fn feature_dim(&self) -> usize {
        self.token_features.len() + self.structural_features.len() + self.type_features.len()
    }

    /// Concatenates all features into a single vector.
    pub fn to_vector(&self) -> Vec<f32> {
        let mut v = Vec::with_capacity(self.feature_dim());
        v.extend(&self.token_features);
        v.extend(&self.structural_features);
        v.extend(&self.type_features);
        v
    }
}

/// Edge feature vector for GNN input.
#[derive(Debug, Clone)]
pub struct EdgeFeatures {
    /// Source node index
    pub source: usize,
    /// Target node index
    pub target: usize,
    /// Edge type (one-hot encoded)
    pub edge_type: Vec<f32>,
}

impl EdgeFeatures {
    /// Creates features from a CPG edge kind.
    pub fn from_edge_kind(source: usize, target: usize, kind: &CpgEdgeKind) -> Self {
        // One-hot encoding for edge types (grouped by category)
        let mut edge_type = vec![0.0; 6];
        match kind {
            // AST edges
            CpgEdgeKind::AstChild | CpgEdgeKind::AstSibling => edge_type[0] = 1.0,
            // CFG edges
            CpgEdgeKind::CfgNext
            | CpgEdgeKind::CfgTrue
            | CpgEdgeKind::CfgFalse
            | CpgEdgeKind::CfgBack
            | CpgEdgeKind::CfgException => edge_type[1] = 1.0,
            // DFG edges
            CpgEdgeKind::DfgRead
            | CpgEdgeKind::DfgWrite
            | CpgEdgeKind::DfgFlow
            | CpgEdgeKind::DfgDepends => edge_type[2] = 1.0,
            // Call graph edges
            CpgEdgeKind::Calls | CpgEdgeKind::Argument | CpgEdgeKind::Returns => edge_type[3] = 1.0,
            // Type edges
            CpgEdgeKind::HasType | CpgEdgeKind::Inherits => edge_type[4] = 1.0,
        }

        Self {
            source,
            target,
            edge_type,
        }
    }
}

/// Semantic issue detected by GNN analysis.
#[derive(Debug, Clone)]
pub struct SemanticIssue {
    /// Node index where issue was detected
    pub node_idx: usize,
    /// Issue type
    pub issue_type: IssueType,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,
    /// Suggested fix (if available)
    pub suggestion: Option<String>,
    /// Related nodes involved in the issue
    pub related_nodes: Vec<usize>,
}

/// Types of semantic issues.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IssueType {
    /// Wrong variable used in context
    VariableMisuse,
    /// Type mismatch in operation
    TypeError,
    /// Missing error handling
    MissingErrorHandling,
    /// Null/None dereference risk
    NullDereference,
    /// Unused variable/import
    UnusedBinding,
    /// API misuse (wrong method, missing call)
    ApiMisuse,
    /// Resource leak (unclosed file, connection)
    ResourceLeak,
    /// General semantic anomaly
    Anomaly,
}

/// GNN-based semantic scorer for code.
///
/// This scorer uses graph neural networks to analyze Code Property Graphs
/// and detect semantic issues in code. It supports both inference
/// (using pre-trained models) and feature extraction for training.
pub struct GnnSemanticScorer {
    config: GnnConfig,
    /// Cached node embeddings
    node_embeddings: HashMap<usize, Vec<f32>>,
}

impl GnnSemanticScorer {
    /// Creates a new GNN semantic scorer.
    pub fn new(config: GnnConfig) -> Self {
        Self {
            config,
            node_embeddings: HashMap::new(),
        }
    }

    /// Creates a scorer with default configuration.
    pub fn default_scorer() -> Self {
        Self::new(GnnConfig::default())
    }

    /// Extracts features from a Code Property Graph.
    pub fn extract_features(&self, cpg: &CodePropertyGraph) -> GnnFeatures {
        let mut node_features = Vec::new();
        let mut edge_features = Vec::new();

        // Extract node features with structural information
        let depths = cpg.compute_depths();
        let child_counts = cpg.compute_child_counts();

        for node in cpg.all_nodes() {
            let depth = depths.get(&node.id).copied().unwrap_or(0);
            let children = child_counts.get(&node.id).copied().unwrap_or(0);
            node_features.push(NodeFeatures::from_cpg_node(node, depth, children));
        }

        // Extract edge features
        for (source, target, edge) in cpg.all_edges() {
            edge_features.push(EdgeFeatures::from_edge_kind(source, target, &edge.kind));
        }

        GnnFeatures {
            node_features,
            edge_features,
            num_nodes: cpg.node_count(),
            num_edges: cpg.edge_count(),
        }
    }

    /// Scores a node for potential semantic issues.
    ///
    /// Returns a score indicating how likely the node is to have issues.
    /// Higher scores indicate more likely problems.
    pub fn score_node(&self, _cpg: &CodePropertyGraph, node_idx: usize) -> f64 {
        // Use cached embedding if available
        if let Some(embedding) = self.node_embeddings.get(&node_idx) {
            // Simple anomaly score based on embedding magnitude
            let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            // Normalize to 0-1 range (assuming typical magnitude around 1.0)
            return (magnitude / 2.0).min(1.0) as f64;
        }

        // Default score for nodes without embeddings
        0.0
    }

    /// Detects semantic issues in the CPG.
    ///
    /// This is a placeholder for the full GNN inference pipeline.
    /// In production, this would use a trained model for inference.
    ///
    /// Uses on-demand edge queries via `edges_to`/`edges_from` instead of
    /// buffering all edges in memory. For large CPGs this avoids O(E) memory
    /// overhead and multiple full edge scans.
    pub fn detect_issues(&self, cpg: &CodePropertyGraph) -> Vec<SemanticIssue> {
        let mut issues = Vec::new();

        // Build node index lookup for efficient graph queries
        let node_indices: HashMap<usize, petgraph::graph::NodeIndex> = cpg
            .all_nodes()
            .map(|n| (n.id, petgraph::graph::NodeIndex::new(n.id)))
            .collect();

        // Heuristic-based detection (placeholder for neural model)
        for node in cpg.all_nodes() {
            // Check for unused variables (simplified)
            if node.kind == CpgNodeKind::Variable {
                let node_idx = match node_indices.get(&node.id) {
                    Some(idx) => *idx,
                    None => continue,
                };

                // Query incoming edges on-demand (writes to this variable)
                let incoming_data_flow = cpg
                    .edges_to(node_idx)
                    .filter(|(_, e)| {
                        matches!(e.kind, CpgEdgeKind::DfgFlow | CpgEdgeKind::DfgWrite)
                    })
                    .count();

                // Query outgoing edges on-demand (reads from this variable)
                let outgoing_data_flow = cpg
                    .edges_from(node_idx)
                    .filter(|(_, e)| {
                        matches!(e.kind, CpgEdgeKind::DfgFlow | CpgEdgeKind::DfgRead)
                    })
                    .count();

                if incoming_data_flow > 0 && outgoing_data_flow == 0 {
                    // Variable is written but never read (potential unused)
                    issues.push(SemanticIssue {
                        node_idx: node.id,
                        issue_type: IssueType::UnusedBinding,
                        confidence: 0.6,
                        suggestion: Some("Variable may be unused".to_string()),
                        related_nodes: vec![],
                    });
                }
            }
        }

        issues
    }

    /// Computes variable misuse candidates.
    ///
    /// For each identifier, returns alternatives that might be correct
    /// based on GNN embeddings and context.
    pub fn variable_misuse_candidates(
        &self,
        cpg: &CodePropertyGraph,
        node_idx: usize,
    ) -> Vec<(String, f64)> {
        let mut node_ref = None;
        for n in cpg.all_nodes() {
            if n.id == node_idx {
                node_ref = Some(n);
                break;
            }
        }

        let node = match node_ref {
            Some(n) => n,
            None => return vec![],
        };

        // Only check variables
        if node.kind != CpgNodeKind::Variable {
            return vec![];
        }

        // Find other variables in scope as candidates
        let mut candidates = Vec::new();
        let node_name = node.name.clone().unwrap_or_default();

        for other in cpg.all_nodes() {
            if other.id == node_idx {
                continue;
            }

            if other.kind != CpgNodeKind::Variable {
                continue;
            }

            if let Some(name) = &other.name {
                if name != &node_name {
                    // Simple similarity score based on edit distance
                    // (placeholder for embedding-based similarity)
                    let score = self.compute_similarity(&node_name, name);
                    if score > 0.3 {
                        candidates.push((name.clone(), score));
                    }
                }
            }
        }

        // Sort by score descending
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(5);
        candidates
    }

    /// Computes string similarity (placeholder for embedding similarity).
    fn compute_similarity(&self, a: &str, b: &str) -> f64 {
        // Simple Jaccard similarity on character bigrams
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }

        let chars_a: Vec<char> = a.chars().collect();
        let chars_b: Vec<char> = b.chars().collect();

        let bigrams_a: std::collections::HashSet<_> = chars_a
            .windows(2)
            .map(|w| (w[0], w[1]))
            .collect();
        let bigrams_b: std::collections::HashSet<_> = chars_b
            .windows(2)
            .map(|w| (w[0], w[1]))
            .collect();

        if bigrams_a.is_empty() || bigrams_b.is_empty() {
            // Single character strings - use exact match
            return if a == b { 1.0 } else { 0.0 };
        }

        let intersection = bigrams_a.intersection(&bigrams_b).count();
        let union = bigrams_a.union(&bigrams_b).count();

        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }

    /// Returns the configuration.
    pub fn config(&self) -> &GnnConfig {
        &self.config
    }
}

/// Extracted GNN features for a Code Property Graph.
#[derive(Debug, Clone)]
pub struct GnnFeatures {
    /// Node features
    pub node_features: Vec<NodeFeatures>,
    /// Edge features
    pub edge_features: Vec<EdgeFeatures>,
    /// Total number of nodes
    pub num_nodes: usize,
    /// Total number of edges
    pub num_edges: usize,
}

impl GnnFeatures {
    /// Converts to adjacency list representation.
    pub fn to_adjacency_list(&self) -> Vec<Vec<usize>> {
        let mut adj = vec![Vec::new(); self.num_nodes];
        for edge in &self.edge_features {
            if edge.source < self.num_nodes && edge.target < self.num_nodes {
                adj[edge.source].push(edge.target);
            }
        }
        adj
    }

    /// Converts node features to a dense matrix.
    pub fn to_node_matrix(&self) -> Vec<Vec<f32>> {
        self.node_features.iter().map(|n| n.to_vector()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gnn_config_default() {
        let config = GnnConfig::default();
        assert_eq!(config.num_layers, 3);
        assert_eq!(config.hidden_dim, 256);
        assert!(config.use_attention);
    }

    #[test]
    fn test_edge_features_encoding() {
        let edge = EdgeFeatures::from_edge_kind(0, 1, &CpgEdgeKind::DfgFlow);
        assert_eq!(edge.source, 0);
        assert_eq!(edge.target, 1);
        assert_eq!(edge.edge_type[2], 1.0); // DFG is index 2
        assert_eq!(edge.edge_type.iter().filter(|&&x| x == 1.0).count(), 1);
    }

    #[test]
    fn test_similarity_computation() {
        let scorer = GnnSemanticScorer::default_scorer();

        // Similar strings
        let sim = scorer.compute_similarity("count", "counter");
        assert!(sim > 0.5);

        // Different strings
        let sim = scorer.compute_similarity("foo", "bar");
        assert!(sim < 0.3);

        // Same string
        let sim = scorer.compute_similarity("test", "test");
        assert!((sim - 1.0).abs() < 0.01);
    }
}
