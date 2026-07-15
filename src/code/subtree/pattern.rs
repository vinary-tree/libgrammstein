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

    fn flatten_recursive(
        node: &super::super::ast::AstNode,
        depth: usize,
        nodes: &mut Vec<FlatNode>,
    ) {
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
///
/// A pattern is encoded as a **self-delimiting byte key**: each node writes its
/// `depth` and its label's byte-length as LEB128 varints, followed by the raw
/// label bytes. Because the label length is written explicitly, the encoding is
/// *injective for any label content* — including AST node kinds that are literal
/// punctuation such as `"|"`, `"||"`, or `":"` (tree-sitter names anonymous nodes
/// by their literal text).
///
/// A delimiter-joined encoding (`"depth:label"` joined by `'|'`) is **not**
/// injective: a label that itself contains `'|'` collapses distinct patterns onto
/// one key — e.g. the single node `("a|1:b", 0)` and the two nodes `("a", 0)`,
/// `("b", 1)` both render `"0:a|1:b"` — silently merging their support counts.
/// No delimiter is used here, so that collision class cannot occur.
pub mod encoding {
    use super::*;

    /// Append `value` to `buf` as an unsigned LEB128 varint.
    fn write_varint(buf: &mut Vec<u8>, mut value: u64) {
        loop {
            let byte = (value & 0x7f) as u8;
            value >>= 7;
            if value == 0 {
                buf.push(byte);
                return;
            }
            buf.push(byte | 0x80);
        }
    }

    /// Read an unsigned LEB128 varint from `bytes` at `*pos`, advancing `*pos`
    /// past it. Returns `None` on a truncated or overlong (> 64-bit) varint.
    fn read_varint(bytes: &[u8], pos: &mut usize) -> Option<u64> {
        let mut result: u64 = 0;
        let mut shift = 0u32;
        loop {
            let byte = *bytes.get(*pos)?;
            *pos += 1;
            result |= ((byte & 0x7f) as u64) << shift;
            if byte & 0x80 == 0 {
                return Some(result);
            }
            shift += 7;
            if shift >= 64 {
                return None;
            }
        }
    }

    /// Encode a pattern as a canonical, injective byte key for hashing/comparison.
    ///
    /// Layout, per node: `varint(depth) · varint(label_len) · label_bytes`. The
    /// explicit length makes the stream self-delimiting, so it round-trips and
    /// never collides regardless of label content.
    pub fn encode_pattern(nodes: &[PatternNode]) -> Vec<u8> {
        // Preallocate: label bytes + up to a few varint bytes per numeric field.
        let label_bytes: usize = nodes.iter().map(|n| n.label.len()).sum();
        let mut buf = Vec::with_capacity(label_bytes + nodes.len() * 4);
        for node in nodes {
            write_varint(&mut buf, node.depth as u64);
            write_varint(&mut buf, node.label.len() as u64);
            buf.extend_from_slice(node.label.as_bytes());
        }
        buf
    }

    /// Decode a pattern byte key back to nodes — the inverse of [`encode_pattern`].
    ///
    /// Returns the nodes decoded so far and stops if `encoded` is malformed
    /// (truncated varint, length past the end of input, or non-UTF-8 label bytes),
    /// so a corrupt key never panics.
    pub fn decode_pattern(encoded: &[u8]) -> Vec<PatternNode> {
        let mut nodes = Vec::new();
        let mut pos = 0;
        while pos < encoded.len() {
            let Some(depth) = read_varint(encoded, &mut pos) else {
                break;
            };
            let Some(len) = read_varint(encoded, &mut pos) else {
                break;
            };
            let Some(end) = pos.checked_add(len as usize) else {
                break;
            };
            let Some(label_bytes) = encoded.get(pos..end) else {
                break;
            };
            let Ok(label) = std::str::from_utf8(label_bytes) else {
                break;
            };
            pos = end;
            nodes.push(PatternNode::new(label, depth as usize));
        }
        nodes
    }

    /// Compute a hash for a pattern from its injective byte encoding.
    pub fn pattern_hash(nodes: &[PatternNode]) -> u64 {
        crate::util::hash::safe_hash(&encode_pattern(nodes))
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
        // Self-delimiting layout: varint(depth) · varint(label_len) · label bytes.
        assert_eq!(
            encoded,
            vec![0x00, 0x01, b'A', 0x01, 0x01, b'B', 0x01, 0x01, b'C']
        );

        let decoded = encoding::decode_pattern(&encoded);
        assert_eq!(decoded, nodes);
    }

    /// AST node kinds may be literal punctuation (`"|"`, `"||"`, `":"`), so the
    /// encoding must round-trip such labels and, crucially, must not collide
    /// distinct patterns — the exact failure of the former `"depth:label"`+`'|'`
    /// scheme.
    #[test]
    fn test_pattern_encoding_injective_for_delimiter_labels() {
        // Labels containing the former delimiters round-trip exactly.
        for label in ["|", "||", ":", "0:A|1:B", "a|1:b"] {
            let nodes = vec![PatternNode::new(label, 3)];
            let round_tripped = encoding::decode_pattern(&encoding::encode_pattern(&nodes));
            assert_eq!(round_tripped, nodes, "label {label:?} must round-trip");
        }

        // The classic collision: one node labelled "a|1:b" versus two nodes
        // "a"(depth 0) and "b"(depth 1). Under the old scheme both encoded to
        // "0:a|1:b"; the length-prefixed encoding keeps them distinct.
        let one = vec![PatternNode::new("a|1:b", 0)];
        let two = vec![PatternNode::new("a", 0), PatternNode::new("b", 1)];
        assert_ne!(
            encoding::encode_pattern(&one),
            encoding::encode_pattern(&two),
            "distinct patterns must not share an encoding"
        );
        assert_ne!(
            encoding::pattern_hash(&one),
            encoding::pattern_hash(&two),
            "distinct patterns must not share a hash"
        );
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
