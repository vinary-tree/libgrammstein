# TreeminerD Algorithm

TreeminerD is a depth-first frequent subtree mining algorithm that discovers common tree patterns using equivalence class extensions.

## Algorithm Overview

TreeminerD works in phases:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TreeminerD Phases                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Phase 1: Build Vertical Representation                                  │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  For each unique label L:                                        │    │
│  │    vertical[L] = [(tree_id, [positions where L appears])]       │    │
│  │                                                                  │    │
│  │  Example: label "function"                                       │    │
│  │    Tree 1: positions [0, 15, 42]                                │    │
│  │    Tree 2: positions [0, 8]                                     │    │
│  │    Tree 3: positions [0]                                        │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                               │                                          │
│                               ▼                                          │
│  Phase 2: Mine Frequent 1-Subtrees                                       │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  For each label with support >= min_support:                     │    │
│  │    Create single-node pattern                                    │    │
│  │                                                                  │    │
│  │  "function": support=3, ratio=100%  ✓                           │    │
│  │  "class":    support=2, ratio=67%   ✓ (if threshold < 67%)      │    │
│  │  "async":    support=1, ratio=33%   ✗ (if threshold > 33%)      │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                               │                                          │
│                               ▼                                          │
│  Phase 3: Extend Patterns (k → k+1)                                      │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  For each frequent k-pattern:                                    │    │
│  │    Find all valid extensions in trees where pattern occurs       │    │
│  │    Group extensions by canonical encoding                        │    │
│  │    Keep those meeting support threshold                          │    │
│  │                                                                  │    │
│  │  Pattern [A, B]:  Try extending with C, D, E...                 │    │
│  │    [A, B, C]: support=5 ✓                                       │    │
│  │    [A, B, D]: support=2 ✗ (below threshold)                     │    │
│  │    [A, B, E]: support=4 ✓                                       │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                               │                                          │
│                               ▼                                          │
│  Phase 4: Repeat until no more extensions                                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Depth-First String Encoding

The algorithm encodes patterns using depth values:

```
Pattern Tree:          Encoding:
      A                [A:0, B:1, D:2, C:1]
     / \
    B   C              Canonical String: "0:A|1:B|2:D|1:C"
   /
  D
```

This encoding is:
- **Canonical**: Same tree always produces same string
- **Compact**: O(n) space for n nodes
- **Comparable**: String equality = tree equality

## Configuration

### TreeminerConfig

```rust
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
            min_support: 0.1,        // 10%
            max_pattern_size: 20,
            max_depth: 10,
            min_pattern_size: 2,
            parallel: true,
            num_threads: 0,          // Auto-detect
        }
    }
}
```

### Configuration Guidelines

| Parameter | Low Value | High Value | Trade-off |
|-----------|-----------|------------|-----------|
| min_support | 0.01 | 0.5 | More patterns vs faster mining |
| max_pattern_size | 5 | 50 | Smaller patterns vs more detail |
| max_depth | 3 | 20 | Shallow patterns vs deep structures |
| min_pattern_size | 1 | 5 | Include singletons vs meaningful patterns |

## Creating a Miner

### Simple Construction

```rust
use libgrammstein::code::subtree::TreeminerD;

// With default settings and 10% minimum support
let miner = TreeminerD::new(0.1);

// With default configuration
let miner = TreeminerD::default();  // Also 10% support
```

### Custom Configuration

```rust
use libgrammstein::code::subtree::{TreeminerD, TreeminerConfig};

let config = TreeminerConfig {
    min_support: 0.05,       // 5% of trees
    max_pattern_size: 30,    // Up to 30 nodes
    max_depth: 15,           // Up to 15 levels deep
    min_pattern_size: 3,     // At least 3 nodes
    parallel: true,          // Use parallel mining
    num_threads: 8,          // Use 8 threads
};

let miner = TreeminerD::with_config(config);
```

## Mining Results

### MiningResult Structure

```rust
pub struct MiningResult {
    /// Discovered frequent patterns
    pub patterns: Vec<SubtreePattern>,

    /// Total number of input trees
    pub num_trees: usize,

    /// Minimum support count used (ceil(min_support * num_trees))
    pub min_support_count: usize,

    /// Number of candidate patterns generated
    pub candidates_generated: usize,

    /// Number of patterns pruned (below threshold)
    pub patterns_pruned: usize,

    /// Mining time in milliseconds
    pub mining_time_ms: u64,
}
```

### Working with Results

```rust
let result = miner.mine(&trees);

// Summary statistics
println!("Found {} patterns from {} trees in {}ms",
    result.patterns.len(),
    result.num_trees,
    result.mining_time_ms
);

println!("Generated {} candidates, pruned {}",
    result.candidates_generated,
    result.patterns_pruned
);

// Most frequent patterns
let mut by_support = result.patterns.clone();
by_support.sort_by(|a, b| b.support.cmp(&a.support));

println!("Top 5 patterns:");
for pattern in by_support.iter().take(5) {
    println!("  Support: {} ({:.1}%)",
        pattern.support,
        pattern.support_ratio * 100.0
    );
    println!("  Structure:\n{}", pattern.to_string_repr());
}

// Largest patterns
let mut by_size = result.patterns.clone();
by_size.sort_by(|a, b| b.size().cmp(&a.size()));

println!("Largest pattern ({} nodes):", by_size[0].size());
println!("{}", by_size[0].to_string_repr());
```

## Pattern Representation

### SubtreePattern API

```rust
impl SubtreePattern {
    /// Number of nodes in the pattern
    pub fn size(&self) -> usize;

    /// Maximum depth of any node
    pub fn max_depth(&self) -> usize;

    /// Check if this pattern contains another
    pub fn contains(&self, other: &SubtreePattern) -> bool;

    /// Human-readable tree representation
    pub fn to_string_repr(&self) -> String;

    /// Get the root node's label
    pub fn root_label(&self) -> Option<&str>;
}
```

### Pattern Visualization

```rust
let pattern = &result.patterns[0];

// String representation
println!("{}", pattern.to_string_repr());
// Output:
// function_definition
//   parameters
//     identifier
//   block
//     return_statement

// Access individual nodes
for node in &pattern.nodes {
    let indent = "  ".repeat(node.depth);
    println!("{}[{}] {}", indent, node.depth, node.label);
}
```

## Pattern Encoding

Patterns are encoded for efficient storage and comparison:

```rust
use libgrammstein::code::subtree::pattern::encoding;

// Encode pattern to string
let encoded = encoding::encode_pattern(&pattern.nodes);
// "0:function_definition|1:parameters|2:identifier|1:block|2:return_statement"

// Decode back to nodes
let decoded = encoding::decode_pattern(&encoded);

// Compute hash for pattern
let hash = encoding::pattern_hash(&pattern.nodes);
```

## Examples

### Finding Common Function Structures

```rust
use libgrammstein::code::subtree::{TreeminerD, TreeminerConfig, FlatTree, FlatNode};

// Parse source files to flat trees
let trees: Vec<FlatTree> = source_files
    .iter()
    .enumerate()
    .map(|(i, source)| {
        let ast = parse_to_ast(source);
        FlatTree::from_ast_node(&ast, i as u64)
    })
    .collect();

// Mine with settings for function-level patterns
let miner = TreeminerD::with_config(TreeminerConfig {
    min_support: 0.1,
    min_pattern_size: 3,
    max_depth: 5,
    ..Default::default()
});

let result = miner.mine(&trees);

// Find function-related patterns
let function_patterns: Vec<_> = result.patterns
    .iter()
    .filter(|p| p.root_label() == Some("function_definition"))
    .collect();

println!("Found {} function patterns", function_patterns.len());
```

### Clone Detection Pipeline

```rust
fn detect_clones(trees: &[FlatTree]) -> Vec<CloneGroup> {
    // Mine with low support to catch clones
    let miner = TreeminerD::with_config(TreeminerConfig {
        min_support: 0.01,      // 1% - even 2 occurrences in 200 trees
        min_pattern_size: 8,    // Substantial size for true clones
        max_pattern_size: 50,   // Allow large clones
        max_depth: 15,
        ..Default::default()
    });

    let result = miner.mine(trees);

    // Group patterns by similarity
    let mut clone_groups = Vec::new();

    for pattern in &result.patterns {
        // Clones should be specific (not too common) but duplicated
        if pattern.support >= 2 && pattern.support_ratio < 0.1 {
            clone_groups.push(CloneGroup {
                pattern: pattern.clone(),
                locations: pattern.occurrences.clone(),
            });
        }
    }

    // Sort by pattern size (larger = more significant clone)
    clone_groups.sort_by(|a, b| b.pattern.size().cmp(&a.pattern.size()));

    clone_groups
}

struct CloneGroup {
    pattern: SubtreePattern,
    locations: Vec<u64>,
}
```

### Incremental Mining

For streaming or interactive use, maintain a pattern cache:

```rust
struct IncrementalMiner {
    miner: TreeminerD,
    all_trees: Vec<FlatTree>,
    cached_patterns: Vec<SubtreePattern>,
}

impl IncrementalMiner {
    fn add_tree(&mut self, tree: FlatTree) {
        self.all_trees.push(tree);

        // Re-mine periodically or when threshold reached
        if self.all_trees.len() % 100 == 0 {
            let result = self.miner.mine(&self.all_trees);
            self.cached_patterns = result.patterns;
        }
    }

    fn get_patterns(&self) -> &[SubtreePattern] {
        &self.cached_patterns
    }
}
```

### Pattern Filtering and Analysis

```rust
fn analyze_patterns(result: &MiningResult) {
    // Group by root label
    let mut by_root: HashMap<String, Vec<&SubtreePattern>> = HashMap::new();
    for pattern in &result.patterns {
        if let Some(root) = pattern.root_label() {
            by_root.entry(root.to_string()).or_default().push(pattern);
        }
    }

    println!("Patterns by root node:");
    for (root, patterns) in by_root.iter() {
        println!("  {}: {} patterns", root, patterns.len());
    }

    // Find patterns at different depth levels
    let shallow: Vec<_> = result.patterns.iter()
        .filter(|p| p.max_depth() <= 2)
        .collect();

    let deep: Vec<_> = result.patterns.iter()
        .filter(|p| p.max_depth() >= 5)
        .collect();

    println!("Shallow patterns (depth <= 2): {}", shallow.len());
    println!("Deep patterns (depth >= 5): {}", deep.len());

    // Identify most specific (largest, least common)
    let mut specificity: Vec<_> = result.patterns.iter()
        .map(|p| (p, p.size() as f64 / (p.support_ratio + 0.01)))
        .collect();

    specificity.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("Most specific patterns:");
    for (pattern, score) in specificity.iter().take(5) {
        println!("  Score {:.2}: {} nodes, {:.1}% support",
            score, pattern.size(), pattern.support_ratio * 100.0);
    }
}
```

## Parallel vs Sequential Mining

The algorithm supports both parallel and sequential execution:

```rust
// Parallel (default) - uses rayon and DashMap
let parallel_config = TreeminerConfig {
    parallel: true,
    num_threads: 0,  // Auto-detect CPU count
    ..Default::default()
};

// Sequential - single-threaded, deterministic order
let sequential_config = TreeminerConfig {
    parallel: false,
    ..Default::default()
};
```

### When to Use Each

**Use Parallel (default):**
- Large datasets (1000+ trees)
- Multi-core systems
- Batch processing

**Use Sequential:**
- Debugging
- Small datasets
- Reproducible ordering required
- Memory-constrained systems

## Performance Tuning

### Memory Usage

```
Memory ≈ (num_trees × avg_nodes × 100 bytes) + (num_patterns × pattern_size × 50 bytes)
```

For 10,000 trees with 50 nodes average:
- Trees: ~50MB
- Patterns (1000 × 10 nodes): ~500KB
- Total: ~50-60MB

### Speed Optimization

1. **Increase min_support** - fewer candidates to explore
2. **Decrease max_pattern_size** - limits extension iterations
3. **Pre-filter trees** - remove small/trivial trees
4. **Use parallel mining** - scales with CPU cores

### Benchmark Example

```rust
use std::time::Instant;

fn benchmark_mining(trees: &[FlatTree]) {
    let configs = vec![
        ("10% support", TreeminerConfig { min_support: 0.1, ..Default::default() }),
        ("5% support", TreeminerConfig { min_support: 0.05, ..Default::default() }),
        ("1% support", TreeminerConfig { min_support: 0.01, ..Default::default() }),
    ];

    for (name, config) in configs {
        let miner = TreeminerD::with_config(config);

        let start = Instant::now();
        let result = miner.mine(trees);
        let elapsed = start.elapsed();

        println!("{}: {} patterns in {:?}",
            name, result.patterns.len(), elapsed);
    }
}
```

## Algorithm Complexity

| Phase | Time Complexity | Space Complexity |
|-------|-----------------|------------------|
| Vertical representation | O(n × m) | O(L × T) |
| 1-subtree mining | O(L) | O(L) |
| Pattern extension (per level) | O(P × T × E) | O(C) |
| Total (k levels) | O(k × P × T × E) | O(P + T) |

Where:
- n = number of trees
- m = average nodes per tree
- L = number of unique labels
- T = trees containing current pattern
- P = patterns at current level
- E = average extensions per pattern
- C = candidate patterns generated
- k = maximum pattern size

## See Also

- [Overview](overview.md) - Subtree mining introduction
- [Code Embeddings](../code-embeddings/overview.md) - Neural code representations
- [Paradigm Detection](../paradigm/overview.md) - Programming paradigm analysis
