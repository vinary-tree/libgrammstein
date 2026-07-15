# Hierarchical Agglomerative Clustering

The topic module groups documents with **hierarchical agglomerative clustering (HAC)**: start
with every document in its own cluster, then repeatedly merge the two closest clusters until one
remains, recording each merge. The full sequence of merges is a *dendrogram* that can be cut at
any granularity. This document derives the distance model, the **Lance-Williams** linkage update
[[1]](#references) that libgrammstein uses for all four linkage methods, and the exact merge loop
the code runs.

> **Scope.** Source of truth:
> [`src/topic/clustering.rs`](../../../src/topic/clustering.rs) and
> [`src/topic/config.rs`](../../../src/topic/config.rs). The result of clustering is a
> [`Dendrogram`](dendrogram.md); the keywords that label each cluster come from
> [c-TF-IDF](ctfidf.md). For the surrounding pipeline see the [Topic Overview](overview.md).

## Notation

| Symbol | Meaning |
|---|---|
| $`n`$ | number of documents (initial singleton clusters) |
| $`v_i`$ | the embedding vector of document $`i`$ |
| $`C_i, C_j, C_k`$ | clusters |
| $`n_i`$ | size (document count) of cluster $`C_i`$ |
| $`d(C_i, C_j)`$ | linkage distance between clusters $`C_i`$ and $`C_j`$ |
| $`d_{ij}`$ | shorthand for $`d(C_i, C_j)`$ |
| $`C_i \cup C_j`$ | the cluster formed by merging $`C_i`$ and $`C_j`$ |
| $`\lVert v \rVert`$ | Euclidean norm of vector $`v`$ |

## Distance model: cosine distance

Documents are compared by the angle between their embeddings, not their magnitude. The **cosine
similarity** of two vectors and the derived **cosine distance** are

```math
\begin{array}{lr}
\displaystyle \cos(v_i, v_j) = \frac{v_i \cdot v_j}{\lVert v_i \rVert\,\lVert v_j \rVert},
\qquad
d(v_i, v_j) = \bigl[\,1 - \cos(v_i, v_j)\,\bigr]^{+} & \text{(C1)}
\end{array}
```

where $`[x]^{+} = \max(x, 0)`$. The implementation
([`cosine_distance`](../../../src/topic/clustering.rs)) accumulates the dot product and both norms
in a single pass, floors the denominator at $`10^{-10}`$ to avoid division by zero, and clamps the
result into $`[0, 2]`$. Identical directions give distance $`0`$; orthogonal vectors give $`1`$;
opposed vectors give $`2`$.

## The initial distance matrix

The $`\binom{n}{2}`$ pairwise distances of $`(\mathrm{C1})`$ are precomputed once and stored as
the **upper triangle** of a symmetric matrix. libgrammstein packs them into a *condensed* array —
the same layout SciPy uses — indexed by

```math
\begin{array}{lr}
\displaystyle \mathrm{idx}(i, j) = n\,i - \frac{i\,(i+1)}{2} + j - i - 1, \qquad i < j & \text{(C2)}
\end{array}
```

Each cell is an `AtomicU64` holding the bit pattern of the `f64` distance, so the matrix can be
filled in parallel with no locks
([`compute_distance_matrix_parallel`](../../../src/topic/clustering.rs)): row $`i`$ is a `rayon`
task that writes cells $`(i, i{+}1), \dots, (i, n{-}1)`$, and a release fence publishes every write
before the agglomeration reads them.

## Linkage: the Lance-Williams update

When two clusters $`C_i`$ and $`C_j`$ merge, the distance from the new cluster
$`C_i \cup C_j`$ to every other cluster $`C_k`$ must be recomputed. Rather than revisit the
underlying points, the **Lance-Williams recurrence** expresses the new distance as a fixed linear
combination of distances that are already known [[1]](#references):

```math
\begin{array}{lr}
\displaystyle d(C_i \cup C_j,\, C_k) =
\alpha_i\,d_{ik} + \alpha_j\,d_{jk} + \beta\,d_{ij} + \gamma\,\lvert d_{ik} - d_{jk} \rvert & \text{(C3)}
\end{array}
```

The four linkage methods of
[`LinkageMethod`](../../../src/topic/config.rs) are exactly four choices of the coefficients
$`(\alpha_i, \alpha_j, \beta, \gamma)`$:

| Method | $`\alpha_i`$ | $`\alpha_j`$ | $`\beta`$ | $`\gamma`$ | Closed form used in code |
|---|---|---|---|---|---|
| Single | $`\tfrac12`$ | $`\tfrac12`$ | $`0`$ | $`-\tfrac12`$ | $`\min(d_{ik}, d_{jk})`$ |
| Complete | $`\tfrac12`$ | $`\tfrac12`$ | $`0`$ | $`+\tfrac12`$ | $`\max(d_{ik}, d_{jk})`$ |
| Average (UPGMA) | $`\tfrac{n_i}{n_i+n_j}`$ | $`\tfrac{n_j}{n_i+n_j}`$ | $`0`$ | $`0`$ | $`\tfrac{n_i d_{ik} + n_j d_{jk}}{n_i + n_j}`$ |
| Ward | $`\tfrac{n_i+n_k}{n_i+n_j+n_k}`$ | $`\tfrac{n_j+n_k}{n_i+n_j+n_k}`$ | $`\tfrac{-n_k}{n_i+n_j+n_k}`$ | $`0`$ | see $`(\mathrm{C4})`$ |

[`linkage_distance`](../../../src/topic/clustering.rs) computes single and complete linkage
directly as $`\min`$ and $`\max`$ (the $`\gamma = \pm\tfrac12`$ forms of $`(\mathrm{C3})`$ reduce
to exactly these), and evaluates average and Ward from their closed forms. The Ward update in full
is

```math
\begin{array}{lr}
\displaystyle d(C_i \cup C_j,\, C_k) =
\frac{(n_i + n_k)\,d_{ik} + (n_j + n_k)\,d_{jk} - n_k\,d_{ij}}{n_i + n_j + n_k} & \text{(C4)}
\end{array}
```

**Choosing a method.** Ward (the default) minimizes the increase in within-cluster variance and
yields compact, similarly-sized topics — the right bias for most corpora. Complete linkage also
favors tight clusters but is more sensitive to outliers; average linkage sits between the
extremes; single linkage can *chain* — a bridge of near-duplicate documents will fuse two
otherwise-distinct topics — which is occasionally useful for connected-component discovery but
rarely for topics.

## The merge loop, literately

The following mirrors
[`HierarchicalClustering::cluster_from_distances`](../../../src/topic/clustering.rs). The active
clusters and their pairwise distances live in an
[`ActiveDistanceMatrix`](../../../src/topic/clustering.rs) (a hash map keyed by cluster-id pairs,
with a cached global minimum); the membership bookkeeping lives in a
[`ClusterState`](../../../src/topic/clustering.rs) (`assignments`, `sizes`, `num_active`, and the
`next_cluster_id` counter).

```
function cluster(embeddings):
    D <- compute_distance_matrix_parallel(embeddings)   ▸ (C1), (C2); atomic, rayon
    return cluster_from_distances(D)

function cluster_from_distances(D):
    n     <- D.n()
    state <- ClusterState::new(n)          ▸ n singletons, ids 0..n, next id = n
    active <- ActiveDistanceMatrix::from_initial(D)      ▸ copy pairs, seed the min
    linkage <- empty list                  ▸ will hold n-1 merges

    repeat n - 1 times:
        (i, j, d) <- active.find_minimum()               ▸ closest active pair
        if none: break
        s_i <- state.sizes[i];  s_j <- state.sizes[j]
        append (i, j, d, s_i + s_j) to linkage           ▸ SciPy-style merge row
        ⟨Recompute distances to every other active cluster⟩
        new <- state.merge(i, j)           ▸ deactivate i, j; activate id = next_cluster_id++
        active.remove_cluster(i);  active.remove_cluster(j);  active.invalidate_minimum()
        for (k, d_new) in updated:  active.set(new, k, d_new)

    dendro <- Dendrogram::from_linkage(linkage, n)
    return ClusteringResult { linkage, dendrogram: dendro, assignments: ⟨cut⟩, num_points: n }

⟨Recompute distances to every other active cluster⟩ ≡
    updated <- empty list
    for k in state.active_clusters() where k != i and k != j:
        d_ik <- active.get(i, k)  else  D.get(i, k)      ▸ fall back to the base matrix
        d_jk <- active.get(j, k)  else  D.get(j, k)
        d_new <- linkage_distance(method, d_ik, d_jk, s_i, s_j, state.sizes[k], d)   ▸ (C3)
        append (k, d_new) to updated

⟨cut⟩ ≡                                     ▸ how the flat labels are derived
    if config.num_clusters == Some(k):       return dendro.cut_to_k_clusters(k)
    else if config.distance_threshold == Some(t): return dendro.cut_at_distance(t)
    else:                                    return 0..n           ▸ every point its own cluster
```

**The merge counter and cluster ids.** `ClusterState::merge` assigns the merged cluster the id
`next_cluster_id`, which starts at $`n`$ and increments by one per merge. Because the loop runs
$`n-1`$ times, the ids of merged clusters are exactly $`n, n{+}1, \dots, 2n{-}2`$ — the same
convention SciPy's linkage matrix uses, and the convention
[`Dendrogram::from_linkage`](../../../src/topic/dendrogram.rs) expects.

## The linkage matrix

`ClusteringResult::linkage` is a `Vec<(u32, u32, f32, u32)>`; row $`r`$ is
$`(a, b, d, s)`$ meaning "cluster $`a`$ merged with cluster $`b`$ at distance $`d`$, forming a
cluster of $`s`$ documents with id $`n + r`$." An id below $`n`$ is an original document (a leaf);
an id at or above $`n`$ refers to the merge that produced it.

```
Example for n = 5 documents:
row 0: (0, 1, 0.05, 2)   →  merge docs 0,1                → cluster 5
row 1: (3, 4, 0.10, 2)   →  merge docs 3,4                → cluster 6
row 2: (5, 2, 0.20, 3)   →  merge cluster 5 with doc 2    → cluster 7
row 3: (6, 7, 0.50, 5)   →  merge clusters 6,7 (all docs) → cluster 8 (root)
```

## Usage

`HierarchicalClustering::new` takes a
[`ClusteringConfig`](../../../src/topic/config.rs); `cluster` returns a
[`ClusteringResult`](../../../src/topic/clustering.rs) bundling the linkage matrix, the
dendrogram, the flat assignments, and the point count.

```rust
use libgrammstein::topic::{ClusteringConfig, HierarchicalClustering, LinkageMethod};

let config = ClusteringConfig {
    num_clusters: Some(10),          // cut the dendrogram to 10 clusters
    linkage: LinkageMethod::Ward,
    ..Default::default()
};

let clustering = HierarchicalClustering::new(config);
let result = clustering.cluster(&embeddings)?;   // embeddings: &[Vec<f32>]

println!("{} merges over {} points", result.linkage.len(), result.num_points);
for (doc, &cluster) in result.assignments.iter().enumerate() {
    println!("document {doc} -> cluster {cluster}");
}
# Ok::<(), libgrammstein::topic::TopicError>(())
```

To cut by a distance threshold instead of a fixed $`k`$, leave `num_clusters` as `None` and set
`distance_threshold`:

```rust
use libgrammstein::topic::ClusteringConfig;

let config = ClusteringConfig::with_distance_threshold(0.5); // merges with d > 0.5 stay split
```

Fewer than two points returns
[`TopicError::ClusteringError`](../../../src/topic/mod.rs).

## Complexity

| Stage | Time | Space |
|---|---|---|
| Distance matrix | $`O(n^2 \cdot D)`$ for $`D`$-dimensional embeddings | $`O(n^2)`$ condensed cells |
| Agglomeration | $`O(n^3)`$ worst case | $`O(n^2)`$ active distances |
| Cut to labels | $`O(n)`$ | $`O(n)`$ |

The agglomeration is the naive Lance-Williams scheme: each of the $`n-1`$ merges invalidates the
cached minimum, so the next `find_minimum` rescans the remaining active pairs — $`O(n^2)`$ per
merge in the worst case, hence $`O(n^3)`$ overall. This is comfortably fast for the few-thousand
document corpora topic extraction targets; the $`O(n^2)`$ distance matrix dominates memory, so for
very large collections cluster a representative sample. (The implementation does **not** use the
nearest-neighbor-chain optimization or a union-find structure — it selects the true global minimum
each step, which keeps every linkage method exact.)

## References

1. G. N. Lance & W. T. Williams (1967). *A general theory of classificatory sorting strategies:
   1. Hierarchical systems.* The Computer Journal 9(4), 373–380.
   [doi:10.1093/comjnl/9.4.373](https://doi.org/10.1093/comjnl/9.4.373)
2. J. H. Ward Jr. (1963). *Hierarchical grouping to optimize an objective function.* Journal of
   the American Statistical Association 58(301), 236–244.
   [doi:10.1080/01621459.1963.10500845](https://doi.org/10.1080/01621459.1963.10500845)
3. D. Müllner (2011). *Modern hierarchical, agglomerative clustering algorithms.* arXiv:1109.2378.
   [arxiv.org/abs/1109.2378](https://arxiv.org/abs/1109.2378)

## See also

- [Topic Overview](overview.md) — the end-to-end pipeline
- [Dendrogram](dendrogram.md) — the merge tree this stage produces, and how to cut it
- [c-TF-IDF](ctfidf.md) — labeling the clusters with keywords
