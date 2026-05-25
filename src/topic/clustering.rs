//! Hierarchical agglomerative clustering for topic extraction.
//!
//! This module provides lock-free parallel clustering using atomics
//! instead of mutex-based synchronization.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use rayon::prelude::*;

use super::checkpoint::{ExtractionPhase, TopicExtractionCheckpoint};
use super::config::{ClusteringConfig, LinkageMethod};
use super::dendrogram::Dendrogram;
use super::{Result, TopicError};

/// Condensed distance matrix using atomic storage for lock-free updates.
///
/// Stores the upper triangular portion of a symmetric distance matrix.
pub struct AtomicDistanceMatrix {
    /// Condensed distances stored as atomic u64 (f32 bits).
    distances: Vec<AtomicU64>,
    /// Number of original data points.
    n: usize,
}

impl AtomicDistanceMatrix {
    /// Create a new distance matrix for n points.
    pub fn new(n: usize) -> Self {
        let size = n * (n - 1) / 2;
        let distances = (0..size).map(|_| AtomicU64::new(0)).collect();
        Self { distances, n }
    }

    /// Create from a checkpoint's distance matrix.
    pub fn from_checkpoint(distances: &[f32], n: usize) -> Self {
        let atomic_distances: Vec<AtomicU64> = distances
            .iter()
            .map(|&d| AtomicU64::new((d as f64).to_bits()))
            .collect();
        Self {
            distances: atomic_distances,
            n,
        }
    }

    /// Get the condensed index for (i, j) where i < j.
    #[inline]
    pub const fn condensed_index(i: usize, j: usize, n: usize) -> usize {
        debug_assert!(i < j);
        n * i - i * (i + 1) / 2 + j - i - 1
    }

    /// Get distance between points i and j.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        if i == j {
            return 0.0;
        }
        let (i, j) = if i < j { (i, j) } else { (j, i) };
        let idx = Self::condensed_index(i, j, self.n);
        f64::from_bits(self.distances[idx].load(Ordering::Relaxed))
    }

    /// Set distance between points i and j (atomic).
    #[inline]
    pub fn set(&self, i: usize, j: usize, dist: f64) {
        if i == j {
            return;
        }
        let (i, j) = if i < j { (i, j) } else { (j, i) };
        let idx = Self::condensed_index(i, j, self.n);
        self.distances[idx].store(dist.to_bits(), Ordering::Relaxed);
    }

    /// Number of points.
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Total number of distance entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.distances.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.distances.is_empty()
    }

    /// Extract distances as a Vec for checkpointing.
    pub fn to_vec(&self) -> Vec<f32> {
        self.distances
            .iter()
            .map(|d| f64::from_bits(d.load(Ordering::Relaxed)) as f32)
            .collect()
    }
}

/// Compute cosine distance between two embeddings.
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());

    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;

    for (x, y) in a.iter().zip(b.iter()) {
        let x = *x as f64;
        let y = *y as f64;
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let denom = (norm_a.sqrt() * norm_b.sqrt()).max(1e-10);
    // Cosine distance = 1 - cosine_similarity
    (1.0 - dot / denom).max(0.0)
}

/// Compute pairwise distance matrix in parallel (lock-free).
///
/// Uses rayon for parallel computation with atomic writes.
pub fn compute_distance_matrix_parallel(
    embeddings: &[Vec<f32>],
    progress: Option<&AtomicUsize>,
) -> AtomicDistanceMatrix {
    let n = embeddings.len();
    let matrix = AtomicDistanceMatrix::new(n);
    let total_pairs = n * (n - 1) / 2;

    // Compute all pairwise distances in parallel
    (0..n).into_par_iter().for_each(|i| {
        for j in (i + 1)..n {
            let dist = cosine_distance(&embeddings[i], &embeddings[j]);
            matrix.set(i, j, dist);
        }

        // Update progress
        if let Some(prog) = progress {
            // Each row i computes (n - i - 1) pairs
            prog.fetch_add(n - i - 1, Ordering::Relaxed);
        }
    });

    // Ensure all writes are visible
    std::sync::atomic::fence(Ordering::Release);

    if let Some(prog) = progress {
        prog.store(total_pairs, Ordering::Release);
    }

    matrix
}

/// Cluster state during agglomerative clustering.
#[derive(Clone)]
pub struct ClusterState {
    /// Current cluster assignments for each original point.
    pub assignments: Vec<u32>,
    /// Size of each cluster (active clusters have size > 0).
    pub sizes: Vec<usize>,
    /// Number of active clusters.
    pub num_active: usize,
    /// Next cluster ID to assign.
    pub next_cluster_id: u32,
}

impl ClusterState {
    /// Create initial state with each point in its own cluster.
    pub fn new(n: usize) -> Self {
        Self {
            assignments: (0..n as u32).collect(),
            sizes: vec![1; n],
            num_active: n,
            next_cluster_id: n as u32,
        }
    }

    /// Merge clusters i and j into a new cluster.
    pub fn merge(&mut self, i: u32, j: u32) -> u32 {
        let new_id = self.next_cluster_id;
        self.next_cluster_id += 1;

        let size_i = self.sizes[i as usize];
        let size_j = self.sizes[j as usize];

        // Extend sizes array if needed
        while self.sizes.len() <= new_id as usize {
            self.sizes.push(0);
        }

        // Update sizes
        self.sizes[new_id as usize] = size_i + size_j;
        self.sizes[i as usize] = 0;
        self.sizes[j as usize] = 0;

        // Update assignments
        for assignment in &mut self.assignments {
            if *assignment == i || *assignment == j {
                *assignment = new_id;
            }
        }

        self.num_active -= 1;
        new_id
    }

    /// Check if cluster is active.
    #[inline]
    pub fn is_active(&self, cluster: u32) -> bool {
        (cluster as usize) < self.sizes.len() && self.sizes[cluster as usize] > 0
    }

    /// Get active cluster IDs.
    pub fn active_clusters(&self) -> Vec<u32> {
        self.sizes
            .iter()
            .enumerate()
            .filter(|(_, &size)| size > 0)
            .map(|(i, _)| i as u32)
            .collect()
    }
}

/// Active distance matrix for agglomerative clustering.
///
/// Tracks distances between active clusters, supporting efficient updates.
pub struct ActiveDistanceMatrix {
    /// Distance between cluster pairs (cluster_id, cluster_id) -> distance.
    /// Uses BTreeMap for ordered iteration.
    distances: std::collections::HashMap<(u32, u32), f64>,
    /// Minimum distance tracker.
    min_dist: f64,
    min_pair: Option<(u32, u32)>,
}

impl ActiveDistanceMatrix {
    /// Create from initial distance matrix.
    pub fn from_initial(matrix: &AtomicDistanceMatrix) -> Self {
        let n = matrix.n();
        let mut distances = std::collections::HashMap::with_capacity(n * (n - 1) / 2);
        let mut min_dist = f64::MAX;
        let mut min_pair = None;

        for i in 0..n {
            for j in (i + 1)..n {
                let dist = matrix.get(i, j);
                let key = (i as u32, j as u32);
                distances.insert(key, dist);

                if dist < min_dist {
                    min_dist = dist;
                    min_pair = Some(key);
                }
            }
        }

        Self {
            distances,
            min_dist,
            min_pair,
        }
    }

    /// Get distance between two clusters.
    pub fn get(&self, i: u32, j: u32) -> Option<f64> {
        let key = if i < j { (i, j) } else { (j, i) };
        self.distances.get(&key).copied()
    }

    /// Set distance between two clusters.
    pub fn set(&mut self, i: u32, j: u32, dist: f64) {
        let key = if i < j { (i, j) } else { (j, i) };
        self.distances.insert(key, dist);
    }

    /// Remove all distances involving a cluster.
    pub fn remove_cluster(&mut self, cluster: u32) {
        self.distances
            .retain(|&(i, j), _| i != cluster && j != cluster);
    }

    /// Find minimum distance and the cluster pair.
    pub fn find_minimum(&mut self) -> Option<(u32, u32, f64)> {
        // Recompute minimum if invalidated
        if self.min_pair.is_none() {
            self.min_dist = f64::MAX;
            for (&(i, j), &dist) in &self.distances {
                if dist < self.min_dist {
                    self.min_dist = dist;
                    self.min_pair = Some((i, j));
                }
            }
        }

        self.min_pair.map(|(i, j)| (i, j, self.min_dist))
    }

    /// Invalidate minimum cache.
    pub fn invalidate_minimum(&mut self) {
        self.min_pair = None;
    }
}

/// Compute linkage distance based on method.
#[inline]
pub fn linkage_distance(
    method: LinkageMethod,
    dist_ik: f64,
    dist_jk: f64,
    size_i: usize,
    size_j: usize,
    size_k: usize,
    dist_ij: f64,
) -> f64 {
    match method {
        LinkageMethod::Single => dist_ik.min(dist_jk),
        LinkageMethod::Complete => dist_ik.max(dist_jk),
        LinkageMethod::Average => {
            let n_i = size_i as f64;
            let n_j = size_j as f64;
            (n_i * dist_ik + n_j * dist_jk) / (n_i + n_j)
        }
        LinkageMethod::Ward => {
            // Lance-Williams formula for Ward linkage
            let n_i = size_i as f64;
            let n_j = size_j as f64;
            let n_k = size_k as f64;
            let n_total = n_i + n_j + n_k;

            ((n_i + n_k) * dist_ik + (n_j + n_k) * dist_jk - n_k * dist_ij) / n_total
        }
    }
}

/// Hierarchical agglomerative clustering result.
#[derive(Clone, Debug)]
pub struct ClusteringResult {
    /// Linkage matrix: (cluster1, cluster2, distance, size).
    pub linkage: Vec<(u32, u32, f32, u32)>,
    /// Dendrogram built from linkage.
    pub dendrogram: Dendrogram,
    /// Final cluster assignments.
    pub assignments: Vec<u32>,
    /// Number of original data points.
    pub num_points: usize,
}

/// Hierarchical agglomerative clustering.
pub struct HierarchicalClustering {
    config: ClusteringConfig,
}

impl HierarchicalClustering {
    /// Create a new clustering instance.
    pub fn new(config: ClusteringConfig) -> Self {
        Self { config }
    }

    /// Perform clustering on embeddings.
    ///
    /// Returns linkage matrix and dendrogram.
    pub fn cluster(&self, embeddings: &[Vec<f32>]) -> Result<ClusteringResult> {
        let n = embeddings.len();

        if n < 2 {
            return Err(TopicError::ClusteringError(
                "Need at least 2 points for clustering".to_string(),
            ));
        }

        // Compute distance matrix in parallel
        let progress = if self.config.verbose {
            Some(AtomicUsize::new(0))
        } else {
            None
        };

        let dist_matrix = compute_distance_matrix_parallel(embeddings, progress.as_ref());

        // Perform agglomerative clustering
        self.cluster_from_distances(&dist_matrix)
    }

    /// Cluster from precomputed distance matrix.
    pub fn cluster_from_distances(
        &self,
        dist_matrix: &AtomicDistanceMatrix,
    ) -> Result<ClusteringResult> {
        let n = dist_matrix.n();

        if n < 2 {
            return Err(TopicError::ClusteringError(
                "Need at least 2 points for clustering".to_string(),
            ));
        }

        // Initialize cluster state
        let mut state = ClusterState::new(n);
        let mut active_distances = ActiveDistanceMatrix::from_initial(dist_matrix);
        let mut linkage: Vec<(u32, u32, f32, u32)> = Vec::with_capacity(n - 1);

        // Perform n-1 merges
        for _ in 0..(n - 1) {
            // Find minimum distance pair
            let Some((i, j, dist)) = active_distances.find_minimum() else {
                break;
            };

            let size_i = state.sizes[i as usize];
            let size_j = state.sizes[j as usize];

            // Record merge
            linkage.push((i, j, dist as f32, (size_i + size_j) as u32));

            // Compute new distances BEFORE removing old clusters
            // Collect distances from old clusters to other active clusters
            let matrix_n = dist_matrix.n();
            let mut new_distances: Vec<(u32, f64)> = Vec::new();

            // Get other active clusters (excluding i and j which will be merged)
            let other_clusters: Vec<u32> = state
                .active_clusters()
                .into_iter()
                .filter(|&k| k != i && k != j)
                .collect();

            for k in &other_clusters {
                let k = *k;
                // Get distances from i and j to k
                let dist_ik = active_distances
                    .get(i, k)
                    .or_else(|| {
                        if (i as usize) < matrix_n && (k as usize) < matrix_n {
                            Some(dist_matrix.get(i as usize, k as usize))
                        } else {
                            None
                        }
                    })
                    .unwrap_or(f64::MAX);
                let dist_jk = active_distances
                    .get(j, k)
                    .or_else(|| {
                        if (j as usize) < matrix_n && (k as usize) < matrix_n {
                            Some(dist_matrix.get(j as usize, k as usize))
                        } else {
                            None
                        }
                    })
                    .unwrap_or(f64::MAX);

                let size_k = state.sizes[k as usize];

                let new_dist = linkage_distance(
                    self.config.linkage,
                    dist_ik,
                    dist_jk,
                    size_i,
                    size_j,
                    size_k,
                    dist,
                );

                new_distances.push((k, new_dist));
            }

            // Create new cluster
            let new_cluster = state.merge(i, j);

            // Remove old cluster distances
            active_distances.remove_cluster(i);
            active_distances.remove_cluster(j);
            active_distances.invalidate_minimum();

            // Add new distances
            for (k, new_dist) in new_distances {
                active_distances.set(new_cluster, k, new_dist);
            }
        }

        // Build dendrogram from linkage
        let dendrogram = Dendrogram::from_linkage(&linkage, n);

        // Get final assignments based on config
        let assignments = if let Some(k) = self.config.num_clusters {
            dendrogram.cut_to_k_clusters(k)
        } else if let Some(threshold) = self.config.distance_threshold {
            dendrogram.cut_at_distance(threshold)
        } else {
            // Default: each point in its own cluster
            (0..n as u32).collect()
        };

        Ok(ClusteringResult {
            linkage,
            dendrogram,
            assignments,
            num_points: n,
        })
    }

    /// Resume clustering from a checkpoint.
    pub fn cluster_from_checkpoint(
        &self,
        embeddings: &[Vec<f32>],
        checkpoint: &TopicExtractionCheckpoint,
    ) -> Result<ClusteringResult> {
        let n = embeddings.len();

        if n != checkpoint.num_documents {
            return Err(TopicError::ClusteringError(format!(
                "Document count mismatch: expected {}, got {}",
                checkpoint.num_documents, n
            )));
        }

        match checkpoint.phase {
            ExtractionPhase::DistanceMatrix => {
                // Start from scratch
                self.cluster(embeddings)
            }
            ExtractionPhase::Clustering => {
                // Resume from distance matrix
                if let Some(ref distances) = checkpoint.distance_matrix {
                    let dist_matrix = AtomicDistanceMatrix::from_checkpoint(distances, n);
                    self.resume_from_linkage(&dist_matrix, &checkpoint.linkage_matrix, n)
                } else {
                    // Recompute distance matrix
                    self.cluster(embeddings)
                }
            }
            _ => {
                // Clustering already complete, reconstruct from linkage
                let dendrogram = Dendrogram::from_linkage(&checkpoint.linkage_matrix, n);
                let assignments = checkpoint.cluster_assignments.clone();
                Ok(ClusteringResult {
                    linkage: checkpoint.linkage_matrix.clone(),
                    dendrogram,
                    assignments,
                    num_points: n,
                })
            }
        }
    }

    /// Resume clustering from partial linkage.
    fn resume_from_linkage(
        &self,
        dist_matrix: &AtomicDistanceMatrix,
        partial_linkage: &[(u32, u32, f32, u32)],
        n: usize,
    ) -> Result<ClusteringResult> {
        // Reconstruct state from partial linkage
        let mut state = ClusterState::new(n);
        let mut linkage = partial_linkage.to_vec();

        // Apply existing merges
        for &(i, j, _, _) in partial_linkage {
            state.merge(i, j);
        }

        // Continue from here
        let mut active_distances = ActiveDistanceMatrix::from_initial(dist_matrix);

        // Remove merged clusters from active distances
        for &(i, j, _, _) in partial_linkage {
            active_distances.remove_cluster(i);
            active_distances.remove_cluster(j);
        }
        active_distances.invalidate_minimum();

        // Continue merging
        while state.num_active > 1 {
            let Some((i, j, dist)) = active_distances.find_minimum() else {
                break;
            };

            let size_i = state.sizes[i as usize];
            let size_j = state.sizes[j as usize];

            linkage.push((i, j, dist as f32, (size_i + size_j) as u32));

            // Compute new distances BEFORE removing old clusters
            let matrix_n = dist_matrix.n();
            let mut new_distances: Vec<(u32, f64)> = Vec::new();

            let other_clusters: Vec<u32> = state
                .active_clusters()
                .into_iter()
                .filter(|&k| k != i && k != j)
                .collect();

            for k in &other_clusters {
                let k = *k;
                let dist_ik = active_distances
                    .get(i, k)
                    .or_else(|| {
                        if (i as usize) < matrix_n && (k as usize) < matrix_n {
                            Some(dist_matrix.get(i as usize, k as usize))
                        } else {
                            None
                        }
                    })
                    .unwrap_or(f64::MAX);
                let dist_jk = active_distances
                    .get(j, k)
                    .or_else(|| {
                        if (j as usize) < matrix_n && (k as usize) < matrix_n {
                            Some(dist_matrix.get(j as usize, k as usize))
                        } else {
                            None
                        }
                    })
                    .unwrap_or(f64::MAX);

                let size_k = state.sizes[k as usize];

                let new_dist = linkage_distance(
                    self.config.linkage,
                    dist_ik,
                    dist_jk,
                    size_i,
                    size_j,
                    size_k,
                    dist,
                );

                new_distances.push((k, new_dist));
            }

            let new_cluster = state.merge(i, j);

            active_distances.remove_cluster(i);
            active_distances.remove_cluster(j);
            active_distances.invalidate_minimum();

            for (k, new_dist) in new_distances {
                active_distances.set(new_cluster, k, new_dist);
            }
        }

        let dendrogram = Dendrogram::from_linkage(&linkage, n);

        let assignments = if let Some(k) = self.config.num_clusters {
            dendrogram.cut_to_k_clusters(k)
        } else if let Some(threshold) = self.config.distance_threshold {
            dendrogram.cut_at_distance(threshold)
        } else {
            (0..n as u32).collect()
        };

        Ok(ClusteringResult {
            linkage,
            dendrogram,
            assignments,
            num_points: n,
        })
    }

    /// Get current configuration.
    pub fn config(&self) -> &ClusteringConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_distance() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];

        // Orthogonal vectors have cosine distance 1
        let dist = cosine_distance(&a, &b);
        assert!((dist - 1.0).abs() < 1e-6);

        // Same vector has cosine distance 0
        let dist = cosine_distance(&a, &a);
        assert!(dist.abs() < 1e-6);

        // Opposite vectors have cosine distance 2
        let c = vec![-1.0, 0.0, 0.0];
        let dist = cosine_distance(&a, &c);
        assert!((dist - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_atomic_distance_matrix() {
        let matrix = AtomicDistanceMatrix::new(4);

        matrix.set(0, 1, 0.5);
        matrix.set(0, 2, 0.8);
        matrix.set(1, 2, 0.3);

        assert!((matrix.get(0, 1) - 0.5).abs() < 1e-6);
        assert!((matrix.get(1, 0) - 0.5).abs() < 1e-6); // Symmetric
        assert!((matrix.get(1, 2) - 0.3).abs() < 1e-6);
        assert!(matrix.get(0, 0).abs() < 1e-6); // Diagonal is 0
    }

    #[test]
    fn test_condensed_index() {
        // For n=4, condensed indices should be:
        // (0,1)=0, (0,2)=1, (0,3)=2, (1,2)=3, (1,3)=4, (2,3)=5
        assert_eq!(AtomicDistanceMatrix::condensed_index(0, 1, 4), 0);
        assert_eq!(AtomicDistanceMatrix::condensed_index(0, 2, 4), 1);
        assert_eq!(AtomicDistanceMatrix::condensed_index(0, 3, 4), 2);
        assert_eq!(AtomicDistanceMatrix::condensed_index(1, 2, 4), 3);
        assert_eq!(AtomicDistanceMatrix::condensed_index(1, 3, 4), 4);
        assert_eq!(AtomicDistanceMatrix::condensed_index(2, 3, 4), 5);
    }

    #[test]
    fn test_compute_distance_matrix_parallel() {
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
        ];

        let matrix = compute_distance_matrix_parallel(&embeddings, None);

        assert_eq!(matrix.n(), 4);
        assert_eq!(matrix.len(), 6); // 4*3/2

        // Check some distances
        assert!((matrix.get(0, 1) - 1.0).abs() < 1e-6); // Orthogonal
        assert!(matrix.get(0, 3) < 1.0); // [1,0,0] and [1,1,0] have angle < 90
    }

    #[test]
    fn test_cluster_state() {
        let mut state = ClusterState::new(4);

        assert_eq!(state.num_active, 4);
        assert!(state.is_active(0));
        assert!(state.is_active(3));

        // Merge clusters 0 and 1
        let new_id = state.merge(0, 1);
        assert_eq!(new_id, 4);
        assert_eq!(state.num_active, 3);
        assert!(!state.is_active(0));
        assert!(!state.is_active(1));
        assert!(state.is_active(4));

        // Check assignments
        assert_eq!(state.assignments[0], 4);
        assert_eq!(state.assignments[1], 4);
        assert_eq!(state.assignments[2], 2);
        assert_eq!(state.assignments[3], 3);
    }

    #[test]
    fn test_linkage_methods() {
        let dist_ik = 1.0;
        let dist_jk = 2.0;
        let size_i = 2;
        let size_j = 3;
        let size_k = 4;
        let dist_ij = 0.5;

        // Single linkage: min
        let single = linkage_distance(
            LinkageMethod::Single,
            dist_ik,
            dist_jk,
            size_i,
            size_j,
            size_k,
            dist_ij,
        );
        assert!((single - 1.0).abs() < 1e-6);

        // Complete linkage: max
        let complete = linkage_distance(
            LinkageMethod::Complete,
            dist_ik,
            dist_jk,
            size_i,
            size_j,
            size_k,
            dist_ij,
        );
        assert!((complete - 2.0).abs() < 1e-6);

        // Average linkage: weighted average
        let average = linkage_distance(
            LinkageMethod::Average,
            dist_ik,
            dist_jk,
            size_i,
            size_j,
            size_k,
            dist_ij,
        );
        let expected_avg = (2.0 * 1.0 + 3.0 * 2.0) / 5.0;
        assert!((average - expected_avg).abs() < 1e-6);
    }

    #[test]
    fn test_hierarchical_clustering() {
        let embeddings = vec![
            vec![1.0, 0.0],
            vec![1.1, 0.0],
            vec![0.0, 1.0],
            vec![0.0, 1.1],
        ];

        let config = ClusteringConfig {
            num_clusters: Some(2),
            linkage: LinkageMethod::Single,
            ..Default::default()
        };

        let clustering = HierarchicalClustering::new(config);
        let result = clustering.cluster(&embeddings).expect("clustering failed");

        // Should have n-1 = 3 merges
        assert_eq!(result.linkage.len(), 3);
        assert_eq!(result.num_points, 4);

        // With 2 clusters, points 0,1 should be together and 2,3 should be together
        assert_eq!(result.assignments[0], result.assignments[1]);
        assert_eq!(result.assignments[2], result.assignments[3]);
        assert_ne!(result.assignments[0], result.assignments[2]);
    }

    #[test]
    fn test_clustering_single_linkage() {
        let embeddings = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![2.0, 0.0]];

        let config = ClusteringConfig {
            linkage: LinkageMethod::Single,
            ..Default::default()
        };

        let clustering = HierarchicalClustering::new(config);
        let result = clustering.cluster(&embeddings).expect("clustering failed");

        // Single linkage should merge closest pairs first
        // Points are at distances 1.0 and 1.0
        assert_eq!(result.linkage.len(), 2);

        // First merge should be between adjacent points
        let (c1, c2, _, _) = result.linkage[0];
        assert!(
            (c1 == 0 && c2 == 1) || (c1 == 1 && c2 == 2),
            "First merge should be adjacent: ({}, {})",
            c1,
            c2
        );
    }

    #[test]
    fn test_distance_matrix_to_vec() {
        let matrix = AtomicDistanceMatrix::new(3);
        matrix.set(0, 1, 0.5);
        matrix.set(0, 2, 0.8);
        matrix.set(1, 2, 0.3);

        let vec = matrix.to_vec();
        assert_eq!(vec.len(), 3); // 3*2/2

        // Verify roundtrip
        let matrix2 = AtomicDistanceMatrix::from_checkpoint(&vec, 3);
        assert!((matrix2.get(0, 1) - 0.5).abs() < 1e-6);
        assert!((matrix2.get(0, 2) - 0.8).abs() < 1e-6);
        assert!((matrix2.get(1, 2) - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_active_distance_matrix() {
        let initial = AtomicDistanceMatrix::new(4);
        initial.set(0, 1, 0.5);
        initial.set(0, 2, 0.8);
        initial.set(0, 3, 1.0);
        initial.set(1, 2, 0.3);
        initial.set(1, 3, 0.9);
        initial.set(2, 3, 0.7);

        let mut active = ActiveDistanceMatrix::from_initial(&initial);

        // Minimum should be (1, 2, 0.3)
        let min = active.find_minimum().expect("should have minimum");
        assert_eq!(min.0, 1);
        assert_eq!(min.1, 2);
        assert!((min.2 - 0.3).abs() < 1e-6);

        // Remove cluster 1
        active.remove_cluster(1);
        active.invalidate_minimum();

        // New minimum should not involve cluster 1
        let min = active.find_minimum().expect("should have minimum");
        assert_ne!(min.0, 1);
        assert_ne!(min.1, 1);
    }
}
