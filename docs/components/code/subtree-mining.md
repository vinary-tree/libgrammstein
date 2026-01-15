# Subtree Mining with TreeminerD

Discover frequent subtree patterns in code ASTs using the TreeminerD algorithm for idiom detection, clone finding, and pattern-based analysis.

## Overview

The subtree mining module provides:

- **TreeminerD algorithm**: Efficient frequent subtree mining
- **Flat tree encoding**: Depth-first tree representation
- **Pattern discovery**: Find common code structures
- **Parallel mining**: Multi-threaded pattern extraction

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        TreeminerD Pipeline                       │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                     Input: AST Forest                       │ │
│  │                                                             │ │
│  │  ParsedCode[1] ──► FlatTree[1]                             │ │
│  │  ParsedCode[2] ──► FlatTree[2]                             │ │
│  │  ParsedCode[n] ──► FlatTree[n]                             │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼ Build vertical representation     │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                   Vertical Index                            │ │
│  │                                                             │ │
│  │  "function_definition" → [(tree1, [0,5]), (tree2, [0])]   │ │
│  │  "identifier" → [(tree1, [1,6]), (tree2, [1,3])]          │ │
│  │  "parameters" → [(tree1, [2]), (tree2, [2])]              │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼ Mine frequent patterns            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                 Equivalence Class Extension                 │ │
│  │                                                             │ │
│  │  1-patterns: {A}, {B}, {C}, ...                           │ │
│  │  2-patterns: {A→B}, {A→C}, {B→D}, ...                     │ │
│  │  k-patterns: extend with sibling/child nodes              │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │               Output: SubtreePattern[]                      │ │
│  │                                                             │ │
│  │  Pattern 1: function → params → body (support: 85%)        │ │
│  │  Pattern 2: if → condition → block (support: 72%)          │ │
│  │  Pattern 3: for → init → condition → update (support: 45%) │ │
│  └────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

## TreeminerConfig

Configuration for the TreeminerD algorithm:

```rust
pub struct TreeminerConfig {
    /// Minimum support threshold (0.0 - 1.0)
    pub min_support: f64,
    /// Maximum pattern size (number of nodes)
    pub max_pattern_size: usize,
    /// Maximum pattern depth
    pub max_depth: usize,
    /// Minimum pattern size to report
    pub min_pattern_size: usize,
    /// Whether to use parallel mining
    pub parallel: bool,
    /// Number of threads (0 = auto)
    pub num_threads: usize,
}
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_support` | 0.1 | Minimum fraction of trees containing pattern |
| `max_pattern_size` | 20 | Maximum nodes in a pattern |
| `max_depth` | 10 | Maximum pattern depth |
| `min_pattern_size` | 2 | Minimum nodes to report |
| `parallel` | true | Enable parallel mining |
| `num_threads` | 0 | Thread count (0 = auto) |

### Creating Configuration

```rust
use libgrammstein::code::subtree::TreeminerConfig;

// Default configuration
let config = TreeminerConfig::default();

// Custom configuration
let config = TreeminerConfig {
    min_support: 0.05,       // 5% minimum support
    max_pattern_size: 30,    // Larger patterns
    max_depth: 15,           // Deeper structures
    min_pattern_size: 3,     // At least 3 nodes
    parallel: true,
    num_threads: 8,          // 8 threads
};
```

## FlatNode

A node in a depth-first encoded tree:

```rust
pub struct FlatNode {
    /// Node label (e.g., "function_definition")
    pub label: Arc<str>,
    /// Depth in tree (root = 0)
    pub depth: usize,
    /// Scope (position in encoding)
    pub scope: usize,
}
```

### Creating FlatNodes

```rust
use libgrammstein::code::subtree::FlatNode;

let node = FlatNode::new("function_definition", 0, 0);
println!("Label: {}", node.label);  // function_definition
println!("Depth: {}", node.depth);  // 0

// Create child node
let child = FlatNode::new("identifier", 1, 1);
```

## FlatTree

A tree represented in depth-first encoding:

```rust
pub struct FlatTree {
    /// Nodes in depth-first order
    pub nodes: Vec<FlatNode>,
    /// Unique tree identifier
    pub tree_id: u64,
    /// Optional metadata
    pub metadata: Option<TreeMetadata>,
}
```

### Depth-First Encoding

The encoding represents a tree as a sequence of nodes in depth-first order:

```
       A              DFS encoding: A(0) B(1) D(2) C(1)
      / \
     B   C            With implicit backtrack markers:
    /                 A B D -1 -1 C -1 -1
   D
```

### Creating FlatTrees

```rust
use libgrammstein::code::subtree::{FlatTree, FlatNode, TreeMetadata};

// Create from nodes
let nodes = vec![
    FlatNode::new("function", 0, 0),
    FlatNode::new("params", 1, 1),
    FlatNode::new("body", 1, 2),
    FlatNode::new("return", 2, 3),
];
let tree = FlatTree::new(nodes, 1);

// With metadata
let metadata = TreeMetadata {
    path: Some("src/main.rs".to_string()),
    language: Some("rust".to_string()),
    source: None,
};
let tree = FlatTree::with_metadata(nodes, 1, metadata);

// From AST node
let ast_node = parse_code("def foo(): pass")?;
let tree = FlatTree::from_ast_node(&ast_node, 42);
```

### FlatTree Operations

```rust
// Tree properties
println!("Nodes: {}", tree.len());
println!("Empty: {}", tree.is_empty());

// Get label positions
let positions = tree.label_positions();
for (label, pos) in positions {
    println!("{} at positions: {:?}", label, pos);
}

// Extract subtree starting at position
if let Some(subtree) = tree.extract_subtree(1) {
    println!("Subtree has {} nodes", subtree.len());
}
```

## PatternNode

A node in a subtree pattern:

```rust
pub struct PatternNode {
    /// Node label
    pub label: Arc<str>,
    /// Depth in pattern (root = 0)
    pub depth: usize,
}
```

### Creating PatternNodes

```rust
use libgrammstein::code::subtree::PatternNode;

// Direct creation
let node = PatternNode::new("if_statement", 0);

// From FlatNode with relative depth
let flat_node = FlatNode::new("block", 3, 5);
let pattern_node = PatternNode::from_flat(&flat_node, 2);  // base_depth=2
// pattern_node.depth = 3 - 2 = 1
```

## SubtreePattern

A discovered frequent subtree pattern:

```rust
pub struct SubtreePattern {
    /// Nodes in depth-first order
    pub nodes: Vec<PatternNode>,
    /// Support count (trees containing pattern)
    pub support: usize,
    /// Support ratio (support / total_trees)
    pub support_ratio: f64,
    /// Tree IDs where pattern occurs
    pub occurrences: Vec<u64>,
    /// Pattern ID
    pub pattern_id: u64,
}
```

### Pattern Properties

```rust
let pattern: SubtreePattern = /* ... */;

// Size and depth
println!("Size: {} nodes", pattern.size());
println!("Max depth: {}", pattern.max_depth());

// Support
println!("Support: {} trees", pattern.support);
println!("Support ratio: {:.1}%", pattern.support_ratio * 100.0);

// Root label
if let Some(root) = pattern.root_label() {
    println!("Root: {}", root);
}

// Human-readable representation
println!("Pattern:\n{}", pattern.to_string_repr());
// Output:
// function
//   params
//   body
```

### Pattern Containment

```rust
// Check if one pattern contains another
let larger_pattern: SubtreePattern = /* A -> B -> C */;
let smaller_pattern: SubtreePattern = /* A -> B */;

if larger_pattern.contains(&smaller_pattern) {
    println!("Larger pattern is a superset");
}
```

## TreeminerD

Main algorithm for frequent subtree mining:

```rust
pub struct TreeminerD {
    config: TreeminerConfig,
    // ...
}
```

### Creating a Miner

```rust
use libgrammstein::code::subtree::{TreeminerD, TreeminerConfig};

// With minimum support
let miner = TreeminerD::new(0.1);  // 10% support

// With full configuration
let config = TreeminerConfig {
    min_support: 0.05,
    max_pattern_size: 25,
    parallel: true,
    ..Default::default()
};
let miner = TreeminerD::with_config(config);
```

### Mining Patterns

```rust
let miner = TreeminerD::new(0.1);
let trees: Vec<FlatTree> = prepare_trees(&sources);

let result = miner.mine(&trees);

println!("Mining Statistics:");
println!("  Trees: {}", result.num_trees);
println!("  Min support count: {}", result.min_support_count);
println!("  Patterns found: {}", result.patterns.len());
println!("  Candidates generated: {}", result.candidates_generated);
println!("  Patterns pruned: {}", result.patterns_pruned);
println!("  Time: {}ms", result.mining_time_ms);

// Access patterns
for pattern in &result.patterns {
    println!(
        "Pattern {} (support: {:.1}%):\n{}",
        pattern.pattern_id,
        pattern.support_ratio * 100.0,
        pattern.to_string_repr()
    );
}
```

## MiningResult

Result of a mining operation:

```rust
pub struct MiningResult {
    /// Discovered frequent patterns
    pub patterns: Vec<SubtreePattern>,
    /// Total input trees
    pub num_trees: usize,
    /// Minimum support count used
    pub min_support_count: usize,
    /// Candidates generated
    pub candidates_generated: usize,
    /// Patterns pruned (below support)
    pub patterns_pruned: usize,
    /// Mining time in milliseconds
    pub mining_time_ms: u64,
}
```

## Pattern Encoding

Utilities for encoding and comparing patterns:

```rust
use libgrammstein::code::subtree::encoding;

let nodes = vec![
    PatternNode::new("A", 0),
    PatternNode::new("B", 1),
    PatternNode::new("C", 1),
];

// Encode to string
let encoded = encoding::encode_pattern(&nodes);
println!("Encoded: {}", encoded);  // "0:A|1:B|1:C"

// Decode from string
let decoded = encoding::decode_pattern(&encoded);
assert_eq!(decoded, nodes);

// Compute hash
let hash = encoding::pattern_hash(&nodes);
println!("Hash: {}", hash);
```

## Algorithm Details

### Phase 1: Vertical Representation

Build an index mapping labels to their positions:

```rust
// For each label, track (tree_id, [positions])
// "function" → [(tree1, [0, 5]), (tree2, [0])]
// "params"   → [(tree1, [1]), (tree2, [1])]
```

### Phase 2: Frequent 1-Subtrees

Find single-node patterns meeting support threshold:

```rust
// Count occurrences across trees
// "function" appears in 95% of trees → frequent
// "generator" appears in 2% of trees → pruned
```

### Phase 3: Equivalence Class Extension

Iteratively extend patterns:

```rust
// Start with frequent 1-patterns: {function}, {params}, ...
// Generate 2-patterns: {function→params}, {function→body}, ...
// Continue until no more frequent patterns found
```

### Phase 4: Support Filtering

Prune patterns below threshold:

```rust
// Keep: function→params (support: 80%)
// Prune: function→params→generator (support: 3%)
```

## Integration Example

Complete workflow for mining code idioms:

```rust
use libgrammstein::code::{
    CodeParser,
    subtree::{TreeminerD, TreeminerConfig, FlatTree}
};

fn mine_code_patterns(sources: &[&str]) -> Result<Vec<SubtreePattern>, Box<dyn Error>> {
    // Parse sources to ASTs
    let parser = CodeParser::<Python>::new()?;
    let mut trees = Vec::with_capacity(sources.len());

    for (i, source) in sources.iter().enumerate() {
        let parsed = parser.parse(source)?;
        if let Some(ast) = &parsed.ast {
            trees.push(FlatTree::from_ast_node(ast, i as u64));
        }
    }

    // Configure miner
    let config = TreeminerConfig {
        min_support: 0.1,        // 10% minimum
        max_pattern_size: 15,    // Up to 15 nodes
        max_depth: 8,            // Max depth 8
        min_pattern_size: 3,     // At least 3 nodes
        parallel: true,
        num_threads: 0,          // Auto-detect
    };

    let miner = TreeminerD::with_config(config);
    let result = miner.mine(&trees);

    println!("Found {} patterns in {}ms",
        result.patterns.len(),
        result.mining_time_ms
    );

    // Sort by support
    let mut patterns = result.patterns;
    patterns.sort_by(|a, b| {
        b.support_ratio.partial_cmp(&a.support_ratio).unwrap()
    });

    Ok(patterns)
}

fn main() -> Result<(), Box<dyn Error>> {
    let sources = load_python_files("./src")?;
    let patterns = mine_code_patterns(&sources)?;

    println!("\nTop 10 Code Patterns:");
    for (i, pattern) in patterns.iter().take(10).enumerate() {
        println!(
            "\n{}. {} ({:.1}% of files):",
            i + 1,
            pattern.root_label().unwrap_or("unknown"),
            pattern.support_ratio * 100.0
        );
        println!("{}", pattern.to_string_repr());
    }

    Ok(())
}
```

## Applications

### Idiom Discovery

Find common coding patterns:

```rust
let config = TreeminerConfig {
    min_support: 0.3,  // Common patterns (30%+)
    min_pattern_size: 4,
    ..Default::default()
};

let miner = TreeminerD::with_config(config);
let result = miner.mine(&project_trees);

// Filter to specific structures
let function_idioms: Vec<_> = result.patterns
    .iter()
    .filter(|p| p.root_label() == Some("function_definition"))
    .collect();
```

### Clone Detection

Find duplicated code structures:

```rust
let config = TreeminerConfig {
    min_support: 0.02,  // Rare patterns (2%+)
    min_pattern_size: 10,  // Larger structures
    ..Default::default()
};

let miner = TreeminerD::with_config(config);
let result = miner.mine(&all_trees);

// Patterns occurring in few files might be clones
let potential_clones: Vec<_> = result.patterns
    .iter()
    .filter(|p| p.support >= 2 && p.support <= 5)
    .filter(|p| p.size() >= 15)  // Substantial size
    .collect();
```

### Pattern-Based Completion

Build suggestions from frequent patterns:

```rust
// Mine common continuations
let result = miner.mine(&corpus_trees);

// Index by prefix
let mut completions: HashMap<String, Vec<&SubtreePattern>> = HashMap::new();
for pattern in &result.patterns {
    if let Some(root) = pattern.root_label() {
        completions.entry(root.to_string())
            .or_default()
            .push(pattern);
    }
}

// Suggest completions
fn suggest(context: &str, completions: &HashMap<String, Vec<&SubtreePattern>>) -> Vec<String> {
    completions.get(context)
        .map(|patterns| {
            patterns.iter()
                .map(|p| p.to_string_repr())
                .collect()
        })
        .unwrap_or_default()
}
```

## Performance

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Build vertical | O(n × m) | n = trees, m = avg nodes |
| Find 1-patterns | O(l) | l = unique labels |
| Extend patterns | O(p × t × e) | p = patterns, t = trees, e = extensions |
| Total mining | O(k × n × m) | k = max pattern size |

### Memory Usage

```
Vertical index: O(l × n) where l = labels, n = trees
Patterns: O(p × s) where p = patterns, s = avg size
Working set: O(current_level_size)
```

### Optimization Tips

1. **Use parallelism**: Enable `parallel: true` for large datasets
2. **Limit depth/size**: Reduce `max_depth` and `max_pattern_size` for speed
3. **Higher support**: Increase `min_support` to reduce search space
4. **Batch processing**: Process files in batches for memory efficiency

## Thread Safety

`TreeminerD` is thread-safe:

```rust
use std::sync::Arc;

let miner = Arc::new(TreeminerD::new(0.1));

// Mining uses internal parallelism (rayon)
let result = miner.mine(&trees);

// Safe to share across threads
let miner_clone = Arc::clone(&miner);
std::thread::spawn(move || {
    let other_result = miner_clone.mine(&other_trees);
});
```

## See Also

- [AST](ast.md) - Tree-sitter parsing
- [CPG](cpg.md) - Code Property Graphs
- [Pipeline](pipeline.md) - End-to-end workflow
- [Semantic Corrector](correctors/semantic.md) - Pattern-based correction
