//! Pattern representations for subtree mining.
//!
//! This module defines the data structures used for representing
//! trees and subtree patterns in the TreeminerD algorithm.

use std::collections::HashMap;
use std::sync::Arc;

/// A node in a flattened (depth-first encoded) tree.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FlatNode {
    /// The node label (e.g., AST node kind like "function_definition")
    pub label: Arc<str>,
    /// Depth in the tree (root = 0)
    pub depth: usize,
    /// Scope (number of -1 backtrack symbols before this node)
    pub scope: usize,
}

impl FlatNode {
    /// Create a new flat node.
    pub fn new(label: impl Into<Arc<str>>, depth: usize, scope: usize) -> Self {
        Self {
            label: label.into(),
            depth,
            scope,
        }
    }
}

/// A tree represented in depth-first encoding.
///
/// The depth-first encoding represents a tree as a sequence of nodes
/// where each node is followed by its children (recursively), then
/// a backtrack marker (-1) is implied when ascending back up.
///
/// Example tree:
/// ```text
///       A
///      / \
///     B   C
///    /
///   D
/// ```
///
/// DFS encoding: A(0) B(1) D(2) C(1)
/// With backtrack markers: A B D -1 -1 C -1 -1
#[derive(Debug, Clone)]
pub struct FlatTree {
    /// Nodes in depth-first order
    pub nodes: Vec<FlatNode>,
    /// Unique identifier for this tree (e.g., file hash)
    pub tree_id: u64,
    /// Optional metadata (e.g., file path, language)
    pub metadata: Option<TreeMetadata>,
}

/// Metadata about a source tree.
#[derive(Debug, Clone)]
pub struct TreeMetadata {
    /// Source file path
    pub path: Option<String>,
    /// Programming language
    pub language: Option<String>,
    /// Original source code
    pub source: Option<String>,
}

impl FlatTree {
    /// Create a new flat tree from nodes.
    pub fn new(nodes: Vec<FlatNode>, tree_id: u64) -> Self {
        Self {
            nodes,
            tree_id,
            metadata: None,
        }
    }

    /// Create a flat tree with metadata.
    pub fn with_metadata(nodes: Vec<FlatNode>, tree_id: u64, metadata: TreeMetadata) -> Self {
        Self {
            nodes,
            tree_id,
            metadata: Some(metadata),
        }
    }

    /// Returns the number of nodes in the tree.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns true if the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Build a flat tree from an AST node (recursive).
    pub fn from_ast_node(node: &super::super::ast::AstNode, tree_id: u64) -> Self {
        let mut nodes = Vec::new();
        Self::flatten_recursive(node, 0, &mut nodes);
        Self::new(nodes, tree_id)
    }

    fn flatten_recursive(node: &super::super::ast::AstNode, depth: usize, nodes: &mut Vec<FlatNode>) {
        // Track scope based on how many backtracks we've implied
        let scope = nodes.len();
        nodes.push(FlatNode::new(node.kind.as_str(), depth, scope));

        for child in &node.children {
            Self::flatten_recursive(child, depth + 1, nodes);
        }
    }

    /// Compute positions of each label in the tree.
    pub fn label_positions(&self) -> HashMap<Arc<str>, Vec<usize>> {
        let mut positions: HashMap<Arc<str>, Vec<usize>> = HashMap::new();
        for (i, node) in self.nodes.iter().enumerate() {
            positions
                .entry(Arc::clone(&node.label))
                .or_default()
                .push(i);
        }
        positions
    }

    /// Extract a subtree starting at the given position.
    pub fn extract_subtree(&self, start: usize) -> Option<Vec<FlatNode>> {
        if start >= self.nodes.len() {
            return None;
        }

        let start_depth = self.nodes[start].depth;
        let mut end = start + 1;

        // Find the end of the subtree (first node at same or lower depth)
        while end < self.nodes.len() && self.nodes[end].depth > start_depth {
            end += 1;
        }

        Some(self.nodes[start..end].to_vec())
    }
}

/// A node in a subtree pattern.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PatternNode {
    /// The node label
    pub label: Arc<str>,
    /// Depth in the pattern (root = 0)
    pub depth: usize,
}

impl PatternNode {
    /// Create a new pattern node.
    pub fn new(label: impl Into<Arc<str>>, depth: usize) -> Self {
        Self {
            label: label.into(),
            depth,
        }
    }

    /// Create from a flat node.
    pub fn from_flat(node: &FlatNode, base_depth: usize) -> Self {
        Self {
            label: Arc::clone(&node.label),
            depth: node.depth.saturating_sub(base_depth),
        }
    }
}

/// A discovered subtree pattern.
#[derive(Debug, Clone)]
pub struct SubtreePattern {
    /// Nodes in the pattern (depth-first order)
    pub nodes: Vec<PatternNode>,
    /// Support count (number of trees containing this pattern)
    pub support: usize,
    /// Support ratio (support / total_trees)
    pub support_ratio: f64,
    /// Tree IDs where this pattern occurs
    pub occurrences: Vec<u64>,
    /// Pattern ID (for reference)
    pub pattern_id: u64,
}

impl SubtreePattern {
    /// Create a new subtree pattern.
    pub fn new(
        nodes: Vec<PatternNode>,
        support: usize,
        total_trees: usize,
        occurrences: Vec<u64>,
        pattern_id: u64,
    ) -> Self {
        let support_ratio = if total_trees > 0 {
            support as f64 / total_trees as f64
        } else {
            0.0
        };

        Self {
            nodes,
            support,
            support_ratio,
            occurrences,
            pattern_id,
        }
    }

    /// Returns the pattern size (number of nodes).
    pub fn size(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the maximum depth in the pattern.
    pub fn max_depth(&self) -> usize {
        self.nodes.iter().map(|n| n.depth).max().unwrap_or(0)
    }

    /// Checks if this pattern is a superset of another.
    pub fn contains(&self, other: &SubtreePattern) -> bool {
        if self.nodes.len() < other.nodes.len() {
            return false;
        }

        // Simple containment check - look for subsequence
        let mut other_idx = 0;
        for self_node in &self.nodes {
            if other_idx < other.nodes.len() && self_node == &other.nodes[other_idx] {
                other_idx += 1;
            }
        }
        other_idx == other.nodes.len()
    }

    /// Convert to a human-readable string representation.
    pub fn to_string_repr(&self) -> String {
        let mut parts = Vec::new();
        for node in &self.nodes {
            let indent = "  ".repeat(node.depth);
            parts.push(format!("{}{}", indent, node.label));
        }
        parts.join("\n")
    }

    /// Get the root label of the pattern.
    pub fn root_label(&self) -> Option<&str> {
        self.nodes.first().map(|n| n.label.as_ref())
    }
}

impl PartialEq for SubtreePattern {
    fn eq(&self, other: &Self) -> bool {
        self.nodes == other.nodes
    }
}

impl Eq for SubtreePattern {}

impl std::hash::Hash for SubtreePattern {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.nodes.hash(state);
    }
}

/// Encoding utilities for pattern matching.
pub mod encoding {
    use super::*;

    /// Encode a pattern as a canonical string for hashing/comparison.
    pub fn encode_pattern(nodes: &[PatternNode]) -> String {
        let mut parts = Vec::with_capacity(nodes.len() * 2);
        for node in nodes {
            parts.push(format!("{}:{}", node.depth, node.label));
        }
        parts.join("|")
    }

    /// Decode a pattern string back to nodes.
    pub fn decode_pattern(encoded: &str) -> Vec<PatternNode> {
        encoded
            .split('|')
            .filter_map(|part| {
                let mut split = part.splitn(2, ':');
                let depth = split.next()?.parse().ok()?;
                let label = split.next()?;
                Some(PatternNode::new(label, depth))
            })
            .collect()
    }

    /// Compute a hash for a pattern.
    pub fn pattern_hash(nodes: &[PatternNode]) -> u64 {
        crate::util::hash::safe_hash(encode_pattern(nodes).as_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flat_node_creation() {
        let node = FlatNode::new("function_definition", 0, 0);
        assert_eq!(node.label.as_ref(), "function_definition");
        assert_eq!(node.depth, 0);
    }

    #[test]
    fn test_flat_tree_creation() {
        let nodes = vec![
            FlatNode::new("root", 0, 0),
            FlatNode::new("child1", 1, 1),
            FlatNode::new("child2", 1, 2),
        ];
        let tree = FlatTree::new(nodes, 1);
        assert_eq!(tree.len(), 3);
    }

    #[test]
    fn test_extract_subtree() {
        let nodes = vec![
            FlatNode::new("root", 0, 0),
            FlatNode::new("child1", 1, 1),
            FlatNode::new("grandchild", 2, 2),
            FlatNode::new("child2", 1, 3),
        ];
        let tree = FlatTree::new(nodes, 1);

        // Extract subtree rooted at child1
        let subtree = tree.extract_subtree(1).unwrap();
        assert_eq!(subtree.len(), 2);
        assert_eq!(subtree[0].label.as_ref(), "child1");
        assert_eq!(subtree[1].label.as_ref(), "grandchild");
    }

    #[test]
    fn test_pattern_encoding() {
        let nodes = vec![
            PatternNode::new("A", 0),
            PatternNode::new("B", 1),
            PatternNode::new("C", 1),
        ];

        let encoded = encoding::encode_pattern(&nodes);
        assert_eq!(encoded, "0:A|1:B|1:C");

        let decoded = encoding::decode_pattern(&encoded);
        assert_eq!(decoded, nodes);
    }

    #[test]
    fn test_subtree_pattern() {
        let nodes = vec![
            PatternNode::new("function", 0),
            PatternNode::new("params", 1),
            PatternNode::new("body", 1),
        ];

        let pattern = SubtreePattern::new(nodes, 10, 100, vec![1, 2, 3], 42);
        assert_eq!(pattern.size(), 3);
        assert_eq!(pattern.support, 10);
        assert!((pattern.support_ratio - 0.1).abs() < 1e-6);
        assert_eq!(pattern.root_label(), Some("function"));
    }
}
