# Dendrogram

The dendrogram represents the hierarchical clustering structure, enabling flexible topic count selection.

## What is a Dendrogram?

A dendrogram visualizes hierarchical clustering as a tree:

```
Distance
│
│                                 ┌────────────────┐
0.8 ├─────────────────────────────┤                │
│                          ┌──────┘                │
0.6 ├──────────────────────┤                       │
│                   ┌──────┘                       │
0.4 ├───────────────┤                              │
│            ┌──────┘                              │
0.2 ├────────┤                              ┌──────┘
│     ┌──────┘                       ┌──────┘
0.0 ──┼──────┼──────┼──────┼──────┼──────┼──────┼──────
     Doc0   Doc1   Doc2   Doc3   Doc4   Doc5   Doc6

Reading the dendrogram:
- Y-axis: Distance at which clusters merge
- X-axis: Individual documents (leaves)
- Horizontal lines: Cluster merges
- Height of merge: Dissimilarity between merged clusters
```

## Scipy-Style Linkage Matrix

The dendrogram is stored as a linkage matrix compatible with scipy:

```
Linkage Matrix Format:
┌─────────────────────────────────────────────────────────────────────────┐
│ Row i: [cluster_a, cluster_b, distance, size]                           │
│                                                                          │
│ cluster_a, cluster_b: Indices of merged clusters                        │
│   - If index < n: It's a leaf (original document)                       │
│   - If index >= n: It's a previous merge (index - n = row number)       │
│ distance: Distance at which merge occurred                              │
│ size: Total number of documents in merged cluster                       │
└─────────────────────────────────────────────────────────────────────────┘

Example for n=5 documents:
┌────────┬───────────┬───────────┬──────────┬──────┐
│ Row    │ Cluster A │ Cluster B │ Distance │ Size │
├────────┼───────────┼───────────┼──────────┼──────┤
│ 0      │ 0         │ 1         │ 0.05     │ 2    │
│ 1      │ 3         │ 4         │ 0.10     │ 2    │
│ 2      │ 5         │ 2         │ 0.20     │ 3    │
│ 3      │ 6         │ 7         │ 0.50     │ 5    │
└────────┴───────────┴───────────┴──────────┴──────┘

Row 0: Merge documents 0 and 1 → cluster 5
Row 1: Merge documents 3 and 4 → cluster 6
Row 2: Merge cluster 5 (docs 0,1) with doc 2 → cluster 7
Row 3: Merge clusters 6 and 7 → final cluster (all docs)
```

## Dendrogram Structure

```rust
use libgrammstein::topic::Dendrogram;

pub struct Dendrogram {
    /// Linkage matrix (n-1 rows for n documents)
    linkage: Vec<LinkageRow>,

    /// Number of original documents
    n_samples: usize,
}

pub struct LinkageRow {
    pub cluster_a: u32,   // First cluster merged
    pub cluster_b: u32,   // Second cluster merged
    pub distance: f32,    // Distance at merge
    pub size: u32,        // Combined cluster size
}
```

## Creating a Dendrogram

Dendrograms are created by hierarchical clustering:

```rust
use libgrammstein::topic::{HierarchicalClustering, LinkageMethod};

let clustering = HierarchicalClustering::new(LinkageMethod::Ward);
let (labels, dendrogram) = clustering.fit(&embeddings, num_clusters)?;
```

## Cutting the Dendrogram

### By Number of Clusters

```rust
// Cut to get exactly k clusters
let k = 10;
let labels = dendrogram.cut_tree(k);

// labels: Vec<u32> with cluster assignment for each document
for (doc, cluster) in labels.iter().enumerate() {
    println!("Document {} → Cluster {}", doc, cluster);
}
```

### By Distance Threshold

```rust
// Cut at distance threshold
let threshold = 0.5;
let labels = dendrogram.cut_by_distance(threshold);

// All merges below threshold are kept
// Merges at or above threshold define cluster boundaries
```

## Navigating the Hierarchy

### Iterate Merges

```rust
// Iterate from first merge to last
for (i, merge) in dendrogram.iter().enumerate() {
    println!(
        "Merge {}: clusters {},{} at distance {:.4} (size {})",
        i, merge.cluster_a, merge.cluster_b, merge.distance, merge.size
    );
}
```

### Find Merge Distances

```rust
// Get all merge distances
let distances: Vec<f32> = dendrogram.iter()
    .map(|m| m.distance)
    .collect();

// Find "elbow" for optimal k
let mut diffs: Vec<f32> = distances
    .windows(2)
    .map(|w| w[1] - w[0])
    .collect();

// Large jump suggests natural cluster boundary
let max_jump_idx = diffs.iter()
    .enumerate()
    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
    .map(|(i, _)| i);
```

### Get Cluster at Level

```rust
// Get clusters at specific number of merges
fn clusters_at_level(dendrogram: &Dendrogram, level: usize) -> Vec<u32> {
    // level = 0: n clusters (no merges)
    // level = n-1: 1 cluster (all merged)
    let n_clusters = dendrogram.n_samples() - level;
    dendrogram.cut_tree(n_clusters)
}
```

## Topic Count Selection

### Method 1: Fixed k

```rust
// User specifies number of topics
let config = TopicConfig {
    num_topics: Some(20),
    ..Default::default()
};
```

### Method 2: Distance Threshold

```rust
// Automatic k based on distance
let threshold = 0.3;  // Merge similar clusters only
let labels = dendrogram.cut_by_distance(threshold);
let num_topics = labels.iter().collect::<HashSet<_>>().len();
```

### Method 3: Elbow Method

```rust
// Find natural break in merge distances
fn find_elbow(dendrogram: &Dendrogram) -> usize {
    let distances: Vec<f32> = dendrogram.iter()
        .map(|m| m.distance)
        .collect();

    // Compute second derivative (acceleration)
    let mut max_accel = 0.0;
    let mut elbow_idx = distances.len() / 2;

    for i in 1..distances.len()-1 {
        let accel = (distances[i+1] - distances[i])
                  - (distances[i] - distances[i-1]);
        if accel > max_accel {
            max_accel = accel;
            elbow_idx = i;
        }
    }

    // Number of clusters = n - elbow_idx
    dendrogram.n_samples() - elbow_idx
}
```

### Method 4: Silhouette Score

```rust
// Evaluate cluster quality at different k
fn optimal_k(embeddings: &[Vec<f32>], dendrogram: &Dendrogram) -> usize {
    let mut best_k = 2;
    let mut best_score = f32::NEG_INFINITY;

    for k in 2..=20 {
        let labels = dendrogram.cut_tree(k);
        let score = silhouette_score(embeddings, &labels);
        if score > best_score {
            best_score = score;
            best_k = k;
        }
    }

    best_k
}
```

## Visualization (ASCII)

```rust
fn print_dendrogram_ascii(dendrogram: &Dendrogram, width: usize) {
    let n = dendrogram.n_samples();
    let max_dist = dendrogram.iter().map(|m| m.distance).fold(0.0, f32::max);

    println!("Distance");
    println!("│");

    for level in (0..10).rev() {
        let threshold = max_dist * level as f32 / 10.0;
        let labels = dendrogram.cut_by_distance(threshold);
        let n_clusters = labels.iter().collect::<HashSet<_>>().len();

        print!("{:.2} ├", threshold);
        // Draw cluster boundaries
        let mut prev_cluster = labels[0];
        for label in &labels {
            if *label != prev_cluster {
                print!("│");
                prev_cluster = *label;
            } else {
                print!("─");
            }
        }
        println!(" ({} clusters)", n_clusters);
    }
}
```

## Persistence

The dendrogram is stored with the topic model:

```rust
// Save with index
index.extract_topics(&config, &embeddings, &texts)?;
index.save("./index")?;

// Load includes dendrogram
let index = RagIndex::load("./index")?;
if let Some(model) = index.topic_model() {
    let dendrogram = model.dendrogram();
    println!("Dendrogram with {} merges", dendrogram.len());
}
```

## Properties

```rust
// Number of original documents
let n = dendrogram.n_samples();

// Number of merges (n - 1)
let n_merges = dendrogram.len();

// Check if empty
let is_empty = dendrogram.is_empty();

// Get specific merge
let merge = &dendrogram[5];  // 6th merge
```

## See Also

- [Overview](overview.md) - Topic module introduction
- [Clustering](clustering.md) - Hierarchical clustering algorithm
- [c-TF-IDF](ctfidf.md) - Keyword extraction
