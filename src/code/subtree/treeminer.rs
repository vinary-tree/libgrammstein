//! TreeminerD algorithm implementation for frequent subtree mining.
//!
//! TreeminerD mines frequent subtrees from a forest of trees using a
//! depth-first candidate generation strategy with equivalence class pruning.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use dashmap::DashMap;
use rayon::prelude::*;

use super::pattern::{FlatNode, FlatTree, PatternNode, SubtreePattern, encoding};

/// Configuration for the TreeminerD algorithm.
#[derive(Debug, Clone)]
pub struct TreeminerConfig {
    /// Minimum support threshold (0.0 - 1.0).
    /// A pattern must appear in at least this fraction of trees.
    pub min_support: f64,
    /// Maximum pattern size (number of nodes).
    pub max_pattern_size: usize,
    /// Maximum pattern depth.
    pub max_depth: usize,
    /// Minimum pattern size to report.
    pub min_pattern_size: usize,
    /// Whether to use parallel mining.
    pub parallel: bool,
    /// Number of threads for parallel mining (0 = auto).
    pub num_threads: usize,
}

impl Default for TreeminerConfig {
    fn default() -> Self {
        Self {
            min_support: 0.1,
            max_pattern_size: 20,
            max_depth: 10,
            min_pattern_size: 2,
            parallel: true,
            num_threads: 0,
        }
    }
}

/// Result of the mining operation.
#[derive(Debug)]
pub struct MiningResult {
    /// Discovered frequent patterns
    pub patterns: Vec<SubtreePattern>,
    /// Total number of input trees
    pub num_trees: usize,
    /// Minimum support count used
    pub min_support_count: usize,
    /// Number of candidate patterns generated
    pub candidates_generated: usize,
    /// Number of patterns pruned
    pub patterns_pruned: usize,
    /// Mining time in milliseconds
    pub mining_time_ms: u64,
}

/// TreeminerD algorithm for frequent subtree mining.
///
/// The algorithm works in phases:
/// 1. Build vertical representation (label -> positions in each tree)
/// 2. Find frequent 1-subtrees (single nodes)
/// 3. Iteratively extend to find larger patterns using equivalence classes
/// 4. Prune infrequent patterns
pub struct TreeminerD {
    config: TreeminerConfig,
    pattern_id_counter: AtomicU64,
}

impl TreeminerD {
    /// Create a new TreeminerD instance with the given minimum support.
    pub fn new(min_support: f64) -> Self {
        Self {
            config: TreeminerConfig {
                min_support,
                ..Default::default()
            },
            pattern_id_counter: AtomicU64::new(0),
        }
    }

    /// Create a new TreeminerD instance with full configuration.
    pub fn with_config(config: TreeminerConfig) -> Self {
        Self {
            config,
            pattern_id_counter: AtomicU64::new(0),
        }
    }

    /// Mine frequent subtree patterns from a forest of trees.
    ///
    /// Uses iterative processing with a shared tree lookup to minimize memory overhead.
    /// The tree_map is built once and reused across all extension levels instead of
    /// being reconstructed for each level.
    pub fn mine(&self, trees: &[FlatTree]) -> MiningResult {
        let start = std::time::Instant::now();

        if trees.is_empty() {
            return MiningResult {
                patterns: vec![],
                num_trees: 0,
                min_support_count: 0,
                candidates_generated: 0,
                patterns_pruned: 0,
                mining_time_ms: 0,
            };
        }

        let num_trees = trees.len();
        let min_support_count = ((self.config.min_support * num_trees as f64).ceil() as usize).max(1);

        // Build tree lookup ONCE and reuse across all levels
        // This avoids O(T) HashMap construction per level
        let tree_map: HashMap<u64, &FlatTree> = trees.iter().map(|t| (t.tree_id, t)).collect();

        // Build vertical representation
        let vertical = self.build_vertical_representation(trees);

        // Find frequent 1-subtrees
        let frequent_1 = self.find_frequent_1_subtrees(&vertical, min_support_count, num_trees);

        let mut all_patterns: Vec<SubtreePattern> = frequent_1.clone();
        let mut candidates_generated = frequent_1.len();
        let mut patterns_pruned = 0;

        // Mine larger patterns using equivalence class extension
        let mut current_level = frequent_1;
        let mut pattern_size = 2;

        while !current_level.is_empty() && pattern_size <= self.config.max_pattern_size {
            let (next_level, generated, pruned) = if self.config.parallel {
                self.extend_patterns_parallel_with_lookup(&current_level, &tree_map, min_support_count, num_trees)
            } else {
                self.extend_patterns_with_lookup(&current_level, &tree_map, min_support_count, num_trees)
            };

            candidates_generated += generated;
            patterns_pruned += pruned;

            // Filter by size constraints in-place using retain instead of collect
            let valid_patterns: Vec<SubtreePattern> = next_level
                .into_iter()
                .filter(|p| {
                    p.size() >= self.config.min_pattern_size
                        && p.max_depth() <= self.config.max_depth
                })
                .collect();

            all_patterns.extend(valid_patterns.iter().cloned());
            current_level = valid_patterns;
            pattern_size += 1;
        }

        // Filter final patterns by size
        let patterns: Vec<SubtreePattern> = all_patterns
            .into_iter()
            .filter(|p| p.size() >= self.config.min_pattern_size)
            .collect();

        MiningResult {
            patterns,
            num_trees,
            min_support_count,
            candidates_generated,
            patterns_pruned,
            mining_time_ms: start.elapsed().as_millis() as u64,
        }
    }

    /// Build vertical representation: for each label, list of (tree_id, positions).
    fn build_vertical_representation(&self, trees: &[FlatTree]) -> HashMap<Arc<str>, Vec<(u64, Vec<usize>)>> {
        let mut vertical: HashMap<Arc<str>, Vec<(u64, Vec<usize>)>> = HashMap::new();

        for tree in trees {
            let positions = tree.label_positions();
            for (label, pos) in positions {
                vertical
                    .entry(label)
                    .or_default()
                    .push((tree.tree_id, pos));
            }
        }

        vertical
    }

    /// Find frequent 1-subtrees (single node patterns).
    fn find_frequent_1_subtrees(
        &self,
        vertical: &HashMap<Arc<str>, Vec<(u64, Vec<usize>)>>,
        min_support: usize,
        total_trees: usize,
    ) -> Vec<SubtreePattern> {
        let mut patterns = Vec::new();

        for (label, occurrences) in vertical {
            let support = occurrences.len();
            if support >= min_support {
                let tree_ids: Vec<u64> = occurrences.iter().map(|(id, _)| *id).collect();
                let pattern_id = self.next_pattern_id();

                patterns.push(SubtreePattern::new(
                    vec![PatternNode::new(Arc::clone(label), 0)],
                    support,
                    total_trees,
                    tree_ids,
                    pattern_id,
                ));
            }
        }

        patterns
    }

    /// Extend patterns by one node (sequential version).
    ///
    /// **Note:** Prefer using `extend_patterns_with_lookup` to avoid
    /// rebuilding the tree_map for each extension level.
    fn extend_patterns(
        &self,
        patterns: &[SubtreePattern],
        trees: &[FlatTree],
        min_support: usize,
        total_trees: usize,
    ) -> (Vec<SubtreePattern>, usize, usize) {
        // Build tree lookup (per-call overhead - prefer _with_lookup variant)
        let tree_map: HashMap<u64, &FlatTree> = trees.iter().map(|t| (t.tree_id, t)).collect();
        self.extend_patterns_with_lookup(patterns, &tree_map, min_support, total_trees)
    }

    /// Extend patterns by one node using a pre-built tree lookup.
    ///
    /// This avoids rebuilding the tree_map HashMap for each extension level,
    /// reducing memory churn for large tree corpora.
    fn extend_patterns_with_lookup(
        &self,
        patterns: &[SubtreePattern],
        tree_map: &HashMap<u64, &FlatTree>,
        min_support: usize,
        total_trees: usize,
    ) -> (Vec<SubtreePattern>, usize, usize) {
        let mut candidates: HashMap<String, (Vec<PatternNode>, HashSet<u64>)> = HashMap::new();
        let mut generated = 0;
        let mut pruned = 0;

        for pattern in patterns {
            // For each tree where this pattern occurs
            for &tree_id in &pattern.occurrences {
                let Some(tree) = tree_map.get(&tree_id) else {
                    continue;
                };

                // Find all occurrences of the pattern in this tree
                // and try to extend each occurrence
                let extensions = self.find_extensions(pattern, tree);

                for extension in extensions {
                    generated += 1;
                    let key = encoding::encode_pattern(&extension);

                    candidates
                        .entry(key)
                        .or_insert_with(|| (extension.clone(), HashSet::new()))
                        .1
                        .insert(tree_id);
                }
            }
        }

        // Filter by support
        let mut result = Vec::new();
        for (nodes, tree_ids) in candidates.into_values() {
            let support = tree_ids.len();
            if support >= min_support {
                let pattern_id = self.next_pattern_id();
                result.push(SubtreePattern::new(
                    nodes,
                    support,
                    total_trees,
                    tree_ids.into_iter().collect(),
                    pattern_id,
                ));
            } else {
                pruned += 1;
            }
        }

        (result, generated, pruned)
    }

    /// Extend patterns by one node (parallel version).
    ///
    /// **Note:** Prefer using `extend_patterns_parallel_with_lookup` to avoid
    /// rebuilding the tree_map for each extension level.
    fn extend_patterns_parallel(
        &self,
        patterns: &[SubtreePattern],
        trees: &[FlatTree],
        min_support: usize,
        total_trees: usize,
    ) -> (Vec<SubtreePattern>, usize, usize) {
        // Build tree lookup (per-call overhead - prefer _with_lookup variant)
        let tree_map: HashMap<u64, &FlatTree> = trees.iter().map(|t| (t.tree_id, t)).collect();
        self.extend_patterns_parallel_with_lookup(patterns, &tree_map, min_support, total_trees)
    }

    /// Extend patterns by one node in parallel using a pre-built tree lookup.
    ///
    /// This avoids rebuilding the tree_map HashMap for each extension level,
    /// reducing memory churn for large tree corpora.
    fn extend_patterns_parallel_with_lookup(
        &self,
        patterns: &[SubtreePattern],
        tree_map: &HashMap<u64, &FlatTree>,
        min_support: usize,
        total_trees: usize,
    ) -> (Vec<SubtreePattern>, usize, usize) {
        let candidates: DashMap<String, (Vec<PatternNode>, HashSet<u64>)> = DashMap::new();
        let generated = AtomicU64::new(0);

        patterns.par_iter().for_each(|pattern| {
            for &tree_id in &pattern.occurrences {
                let Some(tree) = tree_map.get(&tree_id) else {
                    continue;
                };

                let extensions = self.find_extensions(pattern, tree);

                for extension in extensions {
                    generated.fetch_add(1, Ordering::Relaxed);
                    let key = encoding::encode_pattern(&extension);

                    candidates
                        .entry(key)
                        .or_insert_with(|| (extension.clone(), HashSet::new()))
                        .1
                        .insert(tree_id);
                }
            }
        });

        // Filter by support
        let generated_count = generated.load(Ordering::Relaxed) as usize;
        let mut result = Vec::new();
        let mut pruned = 0;

        for entry in candidates.into_iter() {
            let (nodes, tree_ids) = entry.1;
            let support = tree_ids.len();
            if support >= min_support {
                let pattern_id = self.next_pattern_id();
                result.push(SubtreePattern::new(
                    nodes,
                    support,
                    total_trees,
                    tree_ids.into_iter().collect(),
                    pattern_id,
                ));
            } else {
                pruned += 1;
            }
        }

        (result, generated_count, pruned)
    }

    /// Find all valid extensions of a pattern in a tree.
    fn find_extensions(&self, pattern: &SubtreePattern, tree: &FlatTree) -> Vec<Vec<PatternNode>> {
        let mut extensions = Vec::new();

        if pattern.nodes.is_empty() || tree.nodes.is_empty() {
            return extensions;
        }

        // Find all matches of the pattern in the tree
        let matches = self.find_pattern_matches(pattern, tree);

        for match_positions in matches {
            // For each match, try to extend with adjacent nodes
            let last_match_pos = *match_positions.last().unwrap_or(&0);
            let pattern_max_depth = pattern.max_depth();

            // Try to extend as a sibling (same depth as last node in pattern)
            // or as a child (depth + 1)
            for pos in (last_match_pos + 1)..tree.nodes.len() {
                let tree_node = &tree.nodes[pos];

                // Only extend within depth limits
                let base_depth = tree.nodes[match_positions[0]].depth;
                let relative_depth = tree_node.depth.saturating_sub(base_depth);

                if relative_depth > self.config.max_depth {
                    break; // Went too deep
                }

                // Create new pattern with this extension
                let mut new_nodes = pattern.nodes.clone();
                new_nodes.push(PatternNode::new(
                    Arc::clone(&tree_node.label),
                    relative_depth,
                ));

                // Only keep if not already seen (simple dedup)
                if !extensions.contains(&new_nodes) {
                    extensions.push(new_nodes);
                }

                // Only consider immediate extensions (adjacent positions)
                // to avoid explosion
                if pos > last_match_pos + self.config.max_pattern_size {
                    break;
                }
            }
        }

        extensions
    }

    /// Find all positions where a pattern matches in a tree.
    fn find_pattern_matches(&self, pattern: &SubtreePattern, tree: &FlatTree) -> Vec<Vec<usize>> {
        let mut matches = Vec::new();

        if pattern.nodes.is_empty() {
            return matches;
        }

        let first_label = &pattern.nodes[0].label;

        // Find all potential start positions
        for (start_pos, tree_node) in tree.nodes.iter().enumerate() {
            if tree_node.label != *first_label {
                continue;
            }

            // Try to match the rest of the pattern
            let mut positions = vec![start_pos];
            let mut pattern_idx = 1;
            let base_depth = tree_node.depth;

            for (tree_pos, next_tree_node) in tree.nodes.iter().enumerate().skip(start_pos + 1) {
                if pattern_idx >= pattern.nodes.len() {
                    break;
                }

                let expected = &pattern.nodes[pattern_idx];
                let relative_depth = next_tree_node.depth.saturating_sub(base_depth);

                if next_tree_node.label == expected.label && relative_depth == expected.depth {
                    positions.push(tree_pos);
                    pattern_idx += 1;
                }

                // Stop if we've gone past the subtree
                if next_tree_node.depth <= base_depth && tree_pos > start_pos {
                    break;
                }
            }

            // Full match found
            if pattern_idx == pattern.nodes.len() {
                matches.push(positions);
            }
        }

        matches
    }

    fn next_pattern_id(&self) -> u64 {
        self.pattern_id_counter.fetch_add(1, Ordering::Relaxed)
    }
}

impl Default for TreeminerD {
    fn default() -> Self {
        Self::new(0.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_tree(labels: &[(&str, usize)], tree_id: u64) -> FlatTree {
        let nodes: Vec<FlatNode> = labels
            .iter()
            .enumerate()
            .map(|(i, (label, depth))| FlatNode::new(*label, *depth, i))
            .collect();
        FlatTree::new(nodes, tree_id)
    }

    fn miner_with_min_size(min_support: f64, min_pattern_size: usize) -> TreeminerD {
        TreeminerD::with_config(TreeminerConfig {
            min_support,
            min_pattern_size,
            ..Default::default()
        })
    }

    #[test]
    fn test_mine_single_tree() {
        // Tree: A -> B, C
        let tree = make_simple_tree(&[("A", 0), ("B", 1), ("C", 1)], 1);
        // Use min_pattern_size=1 to include single-node patterns
        let miner = miner_with_min_size(1.0, 1);

        let result = miner.mine(&[tree]);

        // Should find at least the single-node patterns
        assert!(result.patterns.iter().any(|p| p.root_label() == Some("A")));
        assert!(result.patterns.iter().any(|p| p.root_label() == Some("B")));
        assert!(result.patterns.iter().any(|p| p.root_label() == Some("C")));
    }

    #[test]
    fn test_mine_common_pattern() {
        // Two trees with common pattern A -> B
        let tree1 = make_simple_tree(&[("A", 0), ("B", 1), ("C", 1)], 1);
        let tree2 = make_simple_tree(&[("A", 0), ("B", 1), ("D", 1)], 2);

        // Use min_pattern_size=1 to include single-node patterns
        let miner = miner_with_min_size(1.0, 1);
        let result = miner.mine(&[tree1, tree2]);

        // A and B should be frequent in both
        let a_patterns: Vec<_> = result.patterns.iter().filter(|p| p.root_label() == Some("A")).collect();
        let b_patterns: Vec<_> = result.patterns.iter().filter(|p| p.root_label() == Some("B")).collect();

        assert!(!a_patterns.is_empty());
        assert!(!b_patterns.is_empty());
        assert_eq!(a_patterns[0].support, 2);
        assert_eq!(b_patterns[0].support, 2);
    }

    #[test]
    fn test_support_threshold() {
        // Three trees, only two have pattern X
        let tree1 = make_simple_tree(&[("A", 0), ("X", 1)], 1);
        let tree2 = make_simple_tree(&[("A", 0), ("X", 1)], 2);
        let tree3 = make_simple_tree(&[("A", 0), ("Y", 1)], 3);

        // With 100% support, X should not be frequent (only in 2 of 3 trees)
        let miner_100 = miner_with_min_size(1.0, 1);
        let result_100 = miner_100.mine(&[tree1.clone(), tree2.clone(), tree3.clone()]);
        let x_patterns: Vec<_> = result_100.patterns.iter().filter(|p| p.root_label() == Some("X")).collect();
        assert!(x_patterns.is_empty() || x_patterns[0].support < 3);

        // With 50% support, X should be frequent (in 2 of 3 trees = 66%)
        let miner_50 = miner_with_min_size(0.5, 1);
        let result_50 = miner_50.mine(&[tree1, tree2, tree3]);
        let x_patterns: Vec<_> = result_50.patterns.iter().filter(|p| p.root_label() == Some("X")).collect();
        assert!(!x_patterns.is_empty());
        assert!(x_patterns[0].support >= 2);
    }

    #[test]
    fn test_config_defaults() {
        let config = TreeminerConfig::default();
        assert!((config.min_support - 0.1).abs() < 1e-6);
        assert_eq!(config.max_pattern_size, 20);
        assert!(config.parallel);
    }

    #[test]
    fn test_empty_input() {
        let miner = TreeminerD::new(0.1);
        let result = miner.mine(&[]);

        assert!(result.patterns.is_empty());
        assert_eq!(result.num_trees, 0);
    }
}
