//! Dendrogram data structure for hierarchical clustering.
//!
//! The dendrogram represents the hierarchical structure produced by
//! agglomerative clustering, enabling navigation at different granularity levels.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use super::TopicId;

/// A node in the dendrogram tree.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DendrogramNode {
    /// Node identifier (corresponds to TopicId for merged clusters).
    pub id: u32,
    /// Left child (None for leaf nodes).
    pub left: Option<u32>,
    /// Right child (None for leaf nodes).
    pub right: Option<u32>,
    /// Distance at which this cluster was formed.
    pub distance: f32,
    /// Number of original points in this cluster.
    pub count: usize,
}

impl DendrogramNode {
    /// Create a leaf node.
    pub fn leaf(id: u32) -> Self {
        Self {
            id,
            left: None,
            right: None,
            distance: 0.0,
            count: 1,
        }
    }

    /// Create an internal (merged) node.
    pub fn internal(id: u32, left: u32, right: u32, distance: f32, count: usize) -> Self {
        Self {
            id,
            left: Some(left),
            right: Some(right),
            distance,
            count,
        }
    }

    /// Check if this is a leaf node.
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.left.is_none() && self.right.is_none()
    }
}

/// Dendrogram for hierarchical clustering results.
///
/// Built from a scipy-style linkage matrix where each row contains:
/// `[cluster1, cluster2, distance, count]`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Dendrogram {
    /// All nodes indexed by ID.
    nodes: HashMap<u32, DendrogramNode>,
    /// Number of original data points (leaf nodes).
    num_leaves: usize,
    /// Root node ID.
    root: Option<u32>,
}

impl Dendrogram {
    /// Create an empty dendrogram.
    pub fn new(num_leaves: usize) -> Self {
        let mut nodes = HashMap::with_capacity(2 * num_leaves - 1);

        // Create leaf nodes
        for i in 0..num_leaves {
            nodes.insert(i as u32, DendrogramNode::leaf(i as u32));
        }

        Self {
            nodes,
            num_leaves,
            root: None,
        }
    }

    /// Build dendrogram from scipy-style linkage matrix.
    ///
    /// Each entry is (cluster1, cluster2, distance, count).
    /// Clusters < num_leaves are original points; >= num_leaves are merged clusters.
    pub fn from_linkage(linkage: &[(u32, u32, f32, u32)], num_leaves: usize) -> Self {
        let mut dendro = Self::new(num_leaves);

        for (i, &(c1, c2, dist, count)) in linkage.iter().enumerate() {
            let new_id = (num_leaves + i) as u32;
            dendro.nodes.insert(
                new_id,
                DendrogramNode::internal(new_id, c1, c2, dist, count as usize),
            );
        }

        // Root is the last merged cluster
        if !linkage.is_empty() {
            dendro.root = Some((num_leaves + linkage.len() - 1) as u32);
        } else if num_leaves == 1 {
            dendro.root = Some(0);
        }

        dendro
    }

    /// Get the root node.
    pub fn root(&self) -> Option<&DendrogramNode> {
        self.root.and_then(|id| self.nodes.get(&id))
    }

    /// Get a node by ID.
    pub fn get(&self, id: u32) -> Option<&DendrogramNode> {
        self.nodes.get(&id)
    }

    /// Get the number of leaf nodes.
    pub fn num_leaves(&self) -> usize {
        self.num_leaves
    }

    /// Get the total number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Cut dendrogram at a distance threshold.
    ///
    /// Returns cluster assignments for each leaf node.
    pub fn cut_at_distance(&self, threshold: f32) -> Vec<u32> {
        let mut assignments = vec![0u32; self.num_leaves];
        let mut next_cluster = 0u32;

        if let Some(root_id) = self.root {
            self.cut_recursive(root_id, threshold, &mut assignments, &mut next_cluster);
        } else {
            // Each leaf is its own cluster
            for (i, assignment) in assignments.iter_mut().enumerate() {
                *assignment = i as u32;
            }
        }

        assignments
    }

    /// Recursive helper for cutting at distance.
    fn cut_recursive(
        &self,
        node_id: u32,
        threshold: f32,
        assignments: &mut [u32],
        next_cluster: &mut u32,
    ) {
        let Some(node) = self.nodes.get(&node_id) else {
            return;
        };

        if node.is_leaf() {
            // Assign this leaf to current cluster
            assignments[node_id as usize] = *next_cluster;
        } else if node.distance > threshold {
            // Cut here - left and right become separate clusters
            if let Some(left) = node.left {
                self.cut_recursive(left, threshold, assignments, next_cluster);
            }
            *next_cluster += 1;
            if let Some(right) = node.right {
                self.cut_recursive(right, threshold, assignments, next_cluster);
            }
        } else {
            // Don't cut - all descendants belong to same cluster
            self.assign_all_leaves(node_id, *next_cluster, assignments);
        }
    }

    /// Assign all leaf descendants to a cluster.
    fn assign_all_leaves(&self, node_id: u32, cluster: u32, assignments: &mut [u32]) {
        let Some(node) = self.nodes.get(&node_id) else {
            return;
        };

        if node.is_leaf() {
            assignments[node_id as usize] = cluster;
        } else {
            if let Some(left) = node.left {
                self.assign_all_leaves(left, cluster, assignments);
            }
            if let Some(right) = node.right {
                self.assign_all_leaves(right, cluster, assignments);
            }
        }
    }

    /// Cut dendrogram to produce exactly k clusters.
    ///
    /// Finds the distance threshold that produces k clusters.
    pub fn cut_to_k_clusters(&self, k: usize) -> Vec<u32> {
        if k >= self.num_leaves {
            // Each leaf is its own cluster
            return (0..self.num_leaves as u32).collect();
        }

        if k <= 1 {
            // All in one cluster
            return vec![0; self.num_leaves];
        }

        // Collect all internal node distances
        let mut distances: Vec<f32> = self
            .nodes
            .values()
            .filter(|n| !n.is_leaf())
            .map(|n| n.distance)
            .collect();

        distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        distances.dedup();

        // For k clusters, we want to cut at the k-1 highest-distance merges.
        // Distances are sorted ascending, so we compute threshold between
        // distances[m-k] and distances[m-k+1] where m = distances.len()
        let m = distances.len();

        if k > m + 1 {
            // More clusters requested than possible - each leaf is its own cluster
            return (0..self.num_leaves as u32).collect();
        }

        let threshold = if k == 1 {
            // All in one cluster - don't cut anything
            distances.last().copied().unwrap_or(f32::MAX) + 0.0001
        } else if k > m {
            // k clusters from m+1 leaves means cut all merges
            distances.first().copied().unwrap_or(0.0) - 0.0001
        } else {
            // Cut between distances[m-k] and distances[m-k+1]
            let lower_idx = m - k;
            let upper_idx = m - k + 1;
            (distances[lower_idx] + distances[upper_idx]) / 2.0
        };

        self.cut_at_distance(threshold)
    }

    /// Get nodes at a specific level (distance range).
    pub fn nodes_at_level(&self, min_distance: f32, max_distance: f32) -> Vec<&DendrogramNode> {
        self.nodes
            .values()
            .filter(|n| n.distance >= min_distance && n.distance < max_distance)
            .collect()
    }

    /// Get all leaf node IDs in a subtree.
    pub fn leaves_under(&self, node_id: u32) -> Vec<u32> {
        let mut leaves = Vec::new();
        self.collect_leaves(node_id, &mut leaves);
        leaves
    }

    /// Recursive helper to collect leaves.
    fn collect_leaves(&self, node_id: u32, leaves: &mut Vec<u32>) {
        let Some(node) = self.nodes.get(&node_id) else {
            return;
        };

        if node.is_leaf() {
            leaves.push(node_id);
        } else {
            if let Some(left) = node.left {
                self.collect_leaves(left, leaves);
            }
            if let Some(right) = node.right {
                self.collect_leaves(right, leaves);
            }
        }
    }

    /// Get the depth of a node (distance from root).
    pub fn depth(&self, node_id: u32) -> Option<usize> {
        self.root.map(|root_id| self.depth_from(root_id, node_id, 0))
    }

    /// Recursive helper for depth computation.
    fn depth_from(&self, current: u32, target: u32, current_depth: usize) -> usize {
        if current == target {
            return current_depth;
        }

        let Some(node) = self.nodes.get(&current) else {
            return usize::MAX;
        };

        let mut min_depth = usize::MAX;
        if let Some(left) = node.left {
            let d = self.depth_from(left, target, current_depth + 1);
            min_depth = min_depth.min(d);
        }
        if let Some(right) = node.right {
            let d = self.depth_from(right, target, current_depth + 1);
            min_depth = min_depth.min(d);
        }

        min_depth
    }

    /// Convert cluster assignments to topic IDs.
    pub fn assignments_to_topic_ids(assignments: &[u32]) -> Vec<TopicId> {
        assignments.iter().map(|&a| TopicId::new(a)).collect()
    }

    /// Get unique cluster IDs from assignments.
    pub fn unique_clusters(assignments: &[u32]) -> Vec<u32> {
        let unique: HashSet<u32> = assignments.iter().copied().collect();
        let mut sorted: Vec<u32> = unique.into_iter().collect();
        sorted.sort();
        sorted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_dendrogram() {
        let dendro = Dendrogram::new(5);
        assert_eq!(dendro.num_leaves(), 5);
        assert_eq!(dendro.num_nodes(), 5);
        assert!(dendro.root().is_none());
    }

    #[test]
    fn test_from_linkage() {
        // Simple 4-point clustering:
        // Merge 0,1 at distance 1.0 -> cluster 4
        // Merge 2,3 at distance 1.5 -> cluster 5
        // Merge 4,5 at distance 2.0 -> cluster 6 (root)
        let linkage = vec![
            (0, 1, 1.0, 2),
            (2, 3, 1.5, 2),
            (4, 5, 2.0, 4),
        ];

        let dendro = Dendrogram::from_linkage(&linkage, 4);

        assert_eq!(dendro.num_leaves(), 4);
        assert_eq!(dendro.num_nodes(), 7); // 4 leaves + 3 internal

        let root = dendro.root().expect("should have root");
        assert_eq!(root.id, 6);
        assert_eq!(root.distance, 2.0);
        assert_eq!(root.count, 4);
    }

    #[test]
    fn test_cut_at_distance() {
        let linkage = vec![
            (0, 1, 1.0, 2),
            (2, 3, 1.5, 2),
            (4, 5, 2.0, 4),
        ];
        let dendro = Dendrogram::from_linkage(&linkage, 4);

        // Cut above all merges - 1 cluster
        let assignments = dendro.cut_at_distance(3.0);
        let unique = Dendrogram::unique_clusters(&assignments);
        assert_eq!(unique.len(), 1);

        // Cut between first and second merge - 2 clusters
        let assignments = dendro.cut_at_distance(1.8);
        let unique = Dendrogram::unique_clusters(&assignments);
        assert_eq!(unique.len(), 2);

        // Cut below all merges - 4 clusters
        let assignments = dendro.cut_at_distance(0.5);
        let unique = Dendrogram::unique_clusters(&assignments);
        assert_eq!(unique.len(), 4);
    }

    #[test]
    fn test_cut_to_k_clusters() {
        let linkage = vec![
            (0, 1, 1.0, 2),
            (2, 3, 1.5, 2),
            (4, 5, 2.0, 4),
        ];
        let dendro = Dendrogram::from_linkage(&linkage, 4);

        // Cut to 2 clusters
        let assignments = dendro.cut_to_k_clusters(2);
        let unique = Dendrogram::unique_clusters(&assignments);
        assert_eq!(unique.len(), 2);

        // Verify grouping: 0,1 should be together, 2,3 should be together
        assert_eq!(assignments[0], assignments[1]);
        assert_eq!(assignments[2], assignments[3]);
        assert_ne!(assignments[0], assignments[2]);

        // Cut to 1 cluster
        let assignments = dendro.cut_to_k_clusters(1);
        let unique = Dendrogram::unique_clusters(&assignments);
        assert_eq!(unique.len(), 1);

        // Cut to 4 clusters
        let assignments = dendro.cut_to_k_clusters(4);
        let unique = Dendrogram::unique_clusters(&assignments);
        assert_eq!(unique.len(), 4);
    }

    #[test]
    fn test_leaves_under() {
        let linkage = vec![
            (0, 1, 1.0, 2),
            (2, 3, 1.5, 2),
            (4, 5, 2.0, 4),
        ];
        let dendro = Dendrogram::from_linkage(&linkage, 4);

        // Leaves under node 4 (merge of 0,1)
        let leaves = dendro.leaves_under(4);
        assert_eq!(leaves.len(), 2);
        assert!(leaves.contains(&0));
        assert!(leaves.contains(&1));

        // Leaves under root (node 6)
        let leaves = dendro.leaves_under(6);
        assert_eq!(leaves.len(), 4);
    }

    #[test]
    fn test_nodes_at_level() {
        let linkage = vec![
            (0, 1, 1.0, 2),
            (2, 3, 1.5, 2),
            (4, 5, 2.0, 4),
        ];
        let dendro = Dendrogram::from_linkage(&linkage, 4);

        // Nodes at distance 1.0-1.6
        let nodes = dendro.nodes_at_level(1.0, 1.6);
        assert_eq!(nodes.len(), 2); // Nodes 4 and 5

        // Nodes at distance 2.0+
        let nodes = dendro.nodes_at_level(2.0, f32::MAX);
        assert_eq!(nodes.len(), 1); // Only root
    }

    #[test]
    fn test_assignments_to_topic_ids() {
        let assignments = vec![0, 0, 1, 1, 2];
        let topic_ids = Dendrogram::assignments_to_topic_ids(&assignments);

        assert_eq!(topic_ids.len(), 5);
        assert_eq!(topic_ids[0], TopicId::new(0));
        assert_eq!(topic_ids[4], TopicId::new(2));
    }

    #[test]
    fn test_single_node() {
        let dendro = Dendrogram::new(1);
        assert_eq!(dendro.num_leaves(), 1);

        // Single-node dendrogram with no linkage
        let linkage: Vec<(u32, u32, f32, u32)> = vec![];
        let dendro = Dendrogram::from_linkage(&linkage, 1);
        assert_eq!(dendro.num_leaves(), 1);
    }

    #[test]
    fn test_depth() {
        let linkage = vec![
            (0, 1, 1.0, 2),
            (2, 3, 1.5, 2),
            (4, 5, 2.0, 4),
        ];
        let dendro = Dendrogram::from_linkage(&linkage, 4);

        // Root depth is 0
        assert_eq!(dendro.depth(6), Some(0));

        // Node 4 (first merge) depth is 1
        assert_eq!(dendro.depth(4), Some(1));

        // Leaf 0 depth is 2
        assert_eq!(dendro.depth(0), Some(2));
    }
}
