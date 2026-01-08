# Hierarchical Agglomerative Clustering

The topic module uses hierarchical agglomerative clustering (HAC) to group documents into topics.

## What is HAC?

HAC builds a hierarchy of clusters by iteratively merging the most similar clusters:

```
Initial: Each document is its own cluster
         [A] [B] [C] [D] [E]

Step 1:  Merge most similar pair (A, B)
         [A,B] [C] [D] [E]

Step 2:  Merge most similar pair (D, E)
         [A,B] [C] [D,E]

Step 3:  Merge most similar pair (A,B) and (C)
         [A,B,C] [D,E]

Step 4:  Merge final clusters
         [A,B,C,D,E]

Dendrogram:
              ┌───────────────┐
         ┌────┴────┐          │
      ┌──┴──┐      │       ┌──┴──┐
      A     B      C       D     E
```

## Linkage Methods

The linkage method determines how cluster distances are computed:

### Ward Linkage (Default)

Minimizes within-cluster variance:

```
Distance = Δ(variance) when merging clusters

Ward produces compact, spherical clusters.
Best for: General-purpose topic modeling
```

### Complete Linkage

Maximum distance between cluster members:

```
Distance(A, B) = max(d(a, b)) for a ∈ A, b ∈ B

Complete produces tight, uniform clusters.
Best for: Outlier-sensitive clustering
```

### Average Linkage

Mean distance between cluster members:

```
Distance(A, B) = mean(d(a, b)) for a ∈ A, b ∈ B

Average balances between single and complete.
Best for: Balanced cluster sizes
```

### Single Linkage

Minimum distance between cluster members:

```
Distance(A, B) = min(d(a, b)) for a ∈ A, b ∈ B

Single can produce elongated "chaining" clusters.
Best for: Detecting connected components
```

## Algorithm

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 1: Distance Matrix Computation                                      │
│                                                                          │
│   For n documents with 768-dim embeddings:                              │
│   D[i,j] = 1 - cosine_similarity(embed[i], embed[j])                   │
│                                                                          │
│   Parallel computation with rayon                                        │
│   Memory: n × (n-1) / 2 floats (upper triangle)                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 2: Nearest Neighbor Chain Algorithm                                 │
│                                                                          │
│   1. Push arbitrary point onto chain                                    │
│   2. Find nearest neighbor of chain tip                                 │
│   3. If reciprocal (mutual) nearest neighbor:                           │
│      - Merge the pair                                                   │
│      - Record in linkage matrix                                         │
│      - Pop both from chain                                              │
│   4. Else push neighbor onto chain                                      │
│   5. Repeat until single cluster                                        │
│                                                                          │
│   Complexity: O(n²) time, O(n) space (for chain)                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 3: Cluster Assignment                                               │
│                                                                          │
│   Cut dendrogram at k clusters or distance threshold                    │
│   Assign each document to a cluster label                               │
│                                                                          │
│   Lock-free assignment using atomics                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

## Usage

### Basic Clustering

```rust
use libgrammstein::topic::{HierarchicalClustering, LinkageMethod};

let clustering = HierarchicalClustering::new(LinkageMethod::Ward);

// Cluster embeddings into 10 groups
let (labels, dendrogram) = clustering.fit(&embeddings, 10)?;

for (i, label) in labels.iter().enumerate() {
    println!("Document {} → Cluster {}", i, label);
}
```

### With Configuration

```rust
use libgrammstein::topic::{ClusteringConfig, LinkageMethod};

let config = ClusteringConfig {
    linkage: LinkageMethod::Complete,
    distance_threshold: None,  // Use num_clusters instead
};

let clustering = HierarchicalClustering::with_config(config);
let (labels, dendrogram) = clustering.fit(&embeddings, num_clusters)?;
```

### Cut by Distance

```rust
// Cluster with distance threshold instead of fixed k
let config = ClusteringConfig {
    linkage: LinkageMethod::Ward,
    distance_threshold: Some(0.5),  // Merge until distance > 0.5
};

let clustering = HierarchicalClustering::with_config(config);
let (labels, dendrogram) = clustering.fit_auto(&embeddings)?;
```

## Distance Computation

### Cosine Distance

For normalized embeddings (dot product = cosine similarity):

```rust
// Distance = 1 - cosine_similarity
let distance = 1.0 - dot_product(&embed_a, &embed_b);
```

### Parallel Distance Matrix

```rust
use rayon::prelude::*;

// Compute upper triangle in parallel
let distances: Vec<f32> = (0..n)
    .into_par_iter()
    .flat_map(|i| {
        (i+1..n).map(move |j| {
            1.0 - dot_product(&embeddings[i], &embeddings[j])
        }).collect::<Vec<_>>()
    })
    .collect();
```

## Lock-Free Algorithm

The clustering uses atomic operations for thread safety:

### Cluster Assignment

```rust
use std::sync::atomic::{AtomicU32, Ordering};

// Cluster labels as atomics
let labels: Vec<AtomicU32> = (0..n)
    .map(|i| AtomicU32::new(i as u32))
    .collect();

// Merge clusters atomically
fn merge_clusters(labels: &[AtomicU32], from: u32, to: u32) {
    labels.par_iter().for_each(|label| {
        let current = label.load(Ordering::Relaxed);
        if current == from {
            label.store(to, Ordering::Relaxed);
        }
    });
}
```

### Union-Find with Path Compression

```rust
// Find root with path compression
fn find(parent: &[AtomicU32], mut x: u32) -> u32 {
    while parent[x as usize].load(Ordering::Relaxed) != x {
        let p = parent[x as usize].load(Ordering::Relaxed);
        let gp = parent[p as usize].load(Ordering::Relaxed);
        parent[x as usize].store(gp, Ordering::Relaxed);  // Path compression
        x = gp;
    }
    x
}
```

## Performance

### Time Complexity

| Operation | Complexity |
|-----------|------------|
| Distance matrix | O(n² × d) |
| HAC algorithm | O(n²) |
| Label assignment | O(n) |
| Total | O(n² × d) |

### Space Complexity

| Component | Memory |
|-----------|--------|
| Distance matrix | O(n²) |
| Linkage matrix | O(n) |
| Labels | O(n) |
| Total | O(n²) |

### Practical Limits

| Documents | Distance Matrix | Time (approx) |
|-----------|-----------------|---------------|
| 1,000 | 4 MB | < 1s |
| 10,000 | 400 MB | ~10s |
| 100,000 | 40 GB | ~20min |

## Linkage Matrix Format

Compatible with scipy's linkage format:

```
Row i: [cluster_a, cluster_b, distance, size]

cluster_a: First cluster merged (index < n = leaf, >= n = previous merge)
cluster_b: Second cluster merged
distance: Distance at merge
size: Total documents in merged cluster

Example for n=5 documents:
Row 0: [0, 1, 0.05, 2]  → Merge docs 0,1 at distance 0.05
Row 1: [3, 4, 0.10, 2]  → Merge docs 3,4 at distance 0.10
Row 2: [5, 2, 0.20, 3]  → Merge cluster 5 (row 0) with doc 2
Row 3: [6, 7, 0.50, 5]  → Merge clusters 6,7 (rows 1,2)
```

## Best Practices

### 1. Choose Linkage by Use Case

```rust
// Compact, similar-sized topics
let clustering = HierarchicalClustering::new(LinkageMethod::Ward);

// Tight, well-separated topics
let clustering = HierarchicalClustering::new(LinkageMethod::Complete);

// Balanced approach
let clustering = HierarchicalClustering::new(LinkageMethod::Average);
```

### 2. Pre-normalize Embeddings

```rust
// Ensure embeddings are unit normalized
let normalized: Vec<Vec<f32>> = embeddings
    .iter()
    .map(|e| normalize(e))
    .collect();
```

### 3. Use Dendrogram for Topic Count Selection

```rust
// Visualize merge distances
let dendrogram = model.dendrogram();
for (i, merge) in dendrogram.iter().enumerate() {
    println!("Merge {}: distance = {:.4}", i, merge.distance);
}

// Look for "elbow" in distances
```

### 4. Consider Memory for Large Datasets

```rust
// For > 50k documents, consider:
// 1. Sampling for initial clustering
// 2. Approximate nearest neighbors
// 3. Incremental clustering
```

## See Also

- [Overview](overview.md) - Topic module introduction
- [Dendrogram](dendrogram.md) - Hierarchy navigation
- [c-TF-IDF](ctfidf.md) - Keyword extraction
