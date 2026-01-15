# Subtree Mining

libgrammstein provides frequent subtree mining for discovering common code patterns in AST forests using the TreeminerD algorithm.

## Why Subtree Mining?

Source code ASTs contain recurring structural patterns:
- Common idioms (`if-else-return`, `for-each-accumulate`)
- Design patterns (factory methods, builder chains)
- Language-specific constructs
- Copy-pasted code clones

Subtree mining discovers these patterns automatically by finding frequently occurring tree structures across a codebase.

## Key Concepts

### Depth-First Tree Encoding

Trees are encoded as depth-first sequences for efficient pattern matching:

```
Original Tree:              Depth-First Encoding:
      A                     A(depth=0)
     / \                    B(depth=1)
    B   C                   D(depth=2)
   /                        C(depth=1)
  D

Sequence: [A:0, B:1, D:2, C:1]
```

The depth value indicates nesting level, allowing pattern matching without explicit parent pointers.

### Support and Frequency

A pattern's **support** is the number of trees containing it:

```
Tree 1:    Tree 2:    Tree 3:
   A          A          X
  / \        / \        / \
 B   C      B   D      B   Y

Pattern [A, B] appears in Tree 1 and Tree 2
  → Support = 2
  → Support Ratio = 2/3 = 0.667 (67%)
```

**Minimum support** filters out rare patterns, focusing on truly common structures.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Subtree Mining Pipeline                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │ Source Code  │───▶│  Tree-sitter │───▶│     FlatTree Forest      │  │
│  │   (files)    │    │    Parser    │    │  (depth-first encoded)   │  │
│  └──────────────┘    └──────────────┘    └───────────┬──────────────┘  │
│                                                       │                  │
│                                                       ▼                  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                       TreeminerD Algorithm                        │  │
│  │  ┌────────────────┐  ┌───────────────┐  ┌──────────────────────┐ │  │
│  │  │  Build Vertical │  │ Mine 1-Trees  │  │  Extend Patterns     │ │  │
│  │  │  Representation │─▶│ (single nodes)│─▶│  (equivalence class) │ │  │
│  │  └────────────────┘  └───────────────┘  └──────────────────────┘ │  │
│  │                                                   │               │  │
│  │                                                   ▼               │  │
│  │                              ┌─────────────────────────────────┐  │  │
│  │                              │    Prune Infrequent Patterns    │  │  │
│  │                              └─────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                         │                               │
│                                         ▼                               │
│                              ┌─────────────────────────────────┐       │
│                              │      SubtreePattern Results     │       │
│                              │  • Pattern nodes                │       │
│                              │  • Support count/ratio          │       │
│                              │  • Occurrence locations         │       │
│                              └─────────────────────────────────┘       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```rust
use libgrammstein::code::subtree::{TreeminerD, FlatTree, FlatNode};

// Create flat trees from AST nodes
let tree1 = FlatTree::new(vec![
    FlatNode::new("function_definition", 0, 0),
    FlatNode::new("parameters", 1, 1),
    FlatNode::new("block", 1, 2),
    FlatNode::new("return_statement", 2, 3),
], 1);

let tree2 = FlatTree::new(vec![
    FlatNode::new("function_definition", 0, 0),
    FlatNode::new("parameters", 1, 1),
    FlatNode::new("block", 1, 2),
    FlatNode::new("if_statement", 2, 3),
], 2);

// Mine patterns with 50% minimum support
let miner = TreeminerD::new(0.5);
let result = miner.mine(&[tree1, tree2]);

// Print discovered patterns
for pattern in &result.patterns {
    println!("Pattern (support={}): {:?}",
        pattern.support,
        pattern.nodes.iter().map(|n| n.label.as_ref()).collect::<Vec<_>>()
    );
}
```

## Use Cases

### 1. Code Clone Detection

Find duplicated code structures across a codebase:

```rust
let miner = TreeminerD::with_config(TreeminerConfig {
    min_support: 0.01,  // 1% - find even rare clones
    min_pattern_size: 5, // At least 5 nodes to be meaningful
    ..Default::default()
});

let result = miner.mine(&project_trees);

// Large patterns with low support may indicate copy-paste clones
let clones: Vec<_> = result.patterns.iter()
    .filter(|p| p.size() >= 10 && p.support_ratio < 0.05)
    .collect();
```

### 2. Idiom Discovery

Discover common coding patterns:

```rust
let miner = TreeminerD::with_config(TreeminerConfig {
    min_support: 0.2,   // 20% - common patterns
    min_pattern_size: 3,
    max_pattern_size: 15,
    ..Default::default()
});

let result = miner.mine(&project_trees);

// High-support patterns are likely idioms
for pattern in result.patterns.iter().filter(|p| p.support_ratio > 0.5) {
    println!("Common idiom:\n{}", pattern.to_string_repr());
}
```

### 3. Pattern-Based Code Completion

Use discovered patterns to suggest completions:

```rust
// Mine patterns from a large corpus
let patterns = miner.mine(&corpus_trees).patterns;

// Given partial code, find matching patterns
fn suggest_completions(
    partial_tree: &FlatTree,
    patterns: &[SubtreePattern],
) -> Vec<&SubtreePattern> {
    patterns.iter()
        .filter(|p| is_prefix_of(partial_tree, p))
        .collect()
}
```

### 4. Design Pattern Detection

Find structural patterns like factories or builders:

```rust
// Mine with settings tuned for design patterns
let miner = TreeminerD::with_config(TreeminerConfig {
    min_support: 0.05,
    min_pattern_size: 4,
    max_depth: 8,
    ..Default::default()
});

let result = miner.mine(&project_trees);

// Look for patterns matching known design pattern structures
let factory_candidates = result.patterns.iter()
    .filter(|p| matches_factory_structure(p))
    .collect::<Vec<_>>();
```

## Core Types

### FlatTree

A tree encoded as a depth-first sequence:

```rust
pub struct FlatTree {
    /// Nodes in depth-first order
    pub nodes: Vec<FlatNode>,
    /// Unique identifier for this tree
    pub tree_id: u64,
    /// Optional metadata (file path, language, source)
    pub metadata: Option<TreeMetadata>,
}
```

### FlatNode

A single node in the flat encoding:

```rust
pub struct FlatNode {
    /// The node label (AST kind like "function_definition")
    pub label: Arc<str>,
    /// Depth in the tree (root = 0)
    pub depth: usize,
    /// Scope identifier for pattern matching
    pub scope: usize,
}
```

### SubtreePattern

A discovered frequent pattern:

```rust
pub struct SubtreePattern {
    /// Pattern nodes in depth-first order
    pub nodes: Vec<PatternNode>,
    /// Number of trees containing this pattern
    pub support: usize,
    /// Fraction of trees (support / total)
    pub support_ratio: f64,
    /// Tree IDs where this pattern appears
    pub occurrences: Vec<u64>,
    /// Unique pattern identifier
    pub pattern_id: u64,
}
```

## Integration with Tree-sitter

Convert tree-sitter ASTs to FlatTrees:

```rust
use libgrammstein::code::ast::AstNode;
use libgrammstein::code::subtree::FlatTree;

// From tree-sitter node
let flat_tree = FlatTree::from_ast_node(&ast_node, file_hash);

// Or build manually from tree-sitter
fn tree_sitter_to_flat(node: tree_sitter::Node, tree_id: u64) -> FlatTree {
    let mut nodes = Vec::new();
    flatten_ts_node(&node, 0, &mut nodes);
    FlatTree::new(nodes, tree_id)
}

fn flatten_ts_node(node: &tree_sitter::Node, depth: usize, out: &mut Vec<FlatNode>) {
    out.push(FlatNode::new(node.kind(), depth, out.len()));
    for child in node.children(&mut node.walk()) {
        flatten_ts_node(&child, depth + 1, out);
    }
}
```

## Performance Considerations

| Metric | Typical Value |
|--------|---------------|
| Trees (1000) | ~100ms |
| Trees (10000) | ~2-5s |
| Trees (100000) | ~30-60s |
| Memory per tree | ~1KB average |

### Optimization Tips

1. **Use parallel mining** (default) for large datasets
2. **Increase min_support** to reduce search space
3. **Limit max_pattern_size** for faster iteration
4. **Pre-filter AST nodes** to remove uninteresting nodes (whitespace, comments)

## See Also

- [TreeminerD Algorithm](treeminer-d.md) - Detailed algorithm documentation
- [Code Embeddings](../code-embeddings/overview.md) - Neural code representations
- [Paradigm Detection](../paradigm/overview.md) - Programming paradigm mining
