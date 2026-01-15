# Graph Neural Networks for Code

Graph Neural Networks (GNNs) analyze code structure through Code Property Graphs to detect semantic issues and patterns.

## Overview

The GNN module provides:

- **Feature extraction**: Node and edge features from CPGs
- **Semantic scoring**: Detect anomalies and semantic issues
- **Variable misuse detection**: Find wrong variable usage
- **Pattern analysis**: Identify buggy code patterns

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    GnnSemanticScorer                             │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                   Feature Extraction                        │ │
│  │                                                             │ │
│  │  CPG Nodes ──► NodeFeatures (structural, token, type)      │ │
│  │  CPG Edges ──► EdgeFeatures (edge type one-hot)            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │               Message Passing Layers                        │ │
│  │                                                             │ │
│  │  For each layer l = 1..L:                                  │ │
│  │    h_v^l = σ(W^l · AGG({h_u^{l-1} : u ∈ N(v)}) + b^l)     │ │
│  │                                                             │ │
│  │  Layer 1 ──► Layer 2 ──► Layer 3 ──► Node Embeddings      │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │               Semantic Issue Detection                      │ │
│  │                                                             │ │
│  │  • Variable misuse scoring                                 │ │
│  │  • Unused binding detection                                │ │
│  │  • Type error identification                               │ │
│  │  • Anomaly scoring                                         │ │
│  └────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

## GnnConfig

Configuration for the GNN semantic scorer:

```rust
pub struct GnnConfig {
    /// Number of GNN layers (message passing iterations)
    pub num_layers: usize,
    /// Hidden dimension size
    pub hidden_dim: usize,
    /// Dropout rate for training
    pub dropout: f64,
    /// Whether to use edge features
    pub use_edge_features: bool,
    /// Whether to use attention mechanism
    pub use_attention: bool,
    /// Embedding dimension for node features
    pub embedding_dim: usize,
}
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_layers` | 3 | Message passing iterations |
| `hidden_dim` | 256 | Hidden layer dimension |
| `dropout` | 0.1 | Dropout rate during training |
| `use_edge_features` | true | Include edge type information |
| `use_attention` | true | Use attention mechanism |
| `embedding_dim` | 128 | Node feature embedding size |

### Creating Configuration

```rust
use libgrammstein::code::GnnConfig;

// Default configuration
let config = GnnConfig::default();

// Custom configuration
let config = GnnConfig {
    num_layers: 4,          // More layers for complex patterns
    hidden_dim: 512,        // Larger hidden dimension
    dropout: 0.2,           // Higher dropout
    use_edge_features: true,
    use_attention: true,
    embedding_dim: 256,
};
```

## NodeFeatures

Feature vectors extracted from CPG nodes:

```rust
pub struct NodeFeatures {
    /// Node index in the CPG
    pub node_idx: usize,
    /// Token/lexical features
    pub token_features: Vec<f32>,
    /// Structural features (depth, child count, etc.)
    pub structural_features: Vec<f32>,
    /// Type features (if available)
    pub type_features: Vec<f32>,
}
```

### Structural Features

```rust
// Features extracted from CPG node
impl NodeFeatures {
    pub fn from_cpg_node(node: &CpgNode, depth: usize, child_count: usize) -> Self {
        let mut structural = Vec::with_capacity(8);

        // Normalized depth in AST (0.0 - 1.0)
        structural.push((depth as f32) / 20.0);

        // Normalized child count
        structural.push((child_count as f32) / 10.0);

        // Byte span (normalized)
        let span_len = (node.location.1 - node.location.0) as f32;
        structural.push(span_len / 1000.0);

        // Node kind encoding (0.0 - 1.0)
        let kind_encoding = match node.kind {
            CpgNodeKind::Function   => 0,
            CpgNodeKind::Variable   => 1,
            CpgNodeKind::Call       => 2,
            CpgNodeKind::Branch     => 3,
            CpgNodeKind::Loop       => 4,
            CpgNodeKind::Assignment => 5,
            CpgNodeKind::Return     => 6,
            _                       => 7,
        };
        structural.push(kind_encoding as f32 / 8.0);

        Self {
            node_idx: node.id,
            token_features: Vec::new(),
            structural_features: structural,
            type_features: Vec::new(),
        }
    }
}
```

### Feature Operations

```rust
let features = NodeFeatures::from_cpg_node(&node, depth, child_count);

// Get total feature dimension
let dim = features.feature_dim();

// Concatenate all features into a single vector
let feature_vec = features.to_vector();
```

## EdgeFeatures

Feature vectors for CPG edges:

```rust
pub struct EdgeFeatures {
    /// Source node index
    pub source: usize,
    /// Target node index
    pub target: usize,
    /// Edge type (one-hot encoded)
    pub edge_type: Vec<f32>,
}
```

### Edge Type Encoding

```rust
impl EdgeFeatures {
    pub fn from_edge_kind(source: usize, target: usize, kind: &CpgEdgeKind) -> Self {
        // One-hot encoding for edge types (6 categories)
        let mut edge_type = vec![0.0; 6];

        match kind {
            // AST edges (index 0)
            CpgEdgeKind::AstChild | CpgEdgeKind::AstSibling => edge_type[0] = 1.0,

            // CFG edges (index 1)
            CpgEdgeKind::CfgNext | CpgEdgeKind::CfgTrue |
            CpgEdgeKind::CfgFalse | CpgEdgeKind::CfgBack |
            CpgEdgeKind::CfgException => edge_type[1] = 1.0,

            // DFG edges (index 2)
            CpgEdgeKind::DfgRead | CpgEdgeKind::DfgWrite |
            CpgEdgeKind::DfgFlow | CpgEdgeKind::DfgDepends => edge_type[2] = 1.0,

            // Call graph edges (index 3)
            CpgEdgeKind::Calls | CpgEdgeKind::Argument |
            CpgEdgeKind::Returns => edge_type[3] = 1.0,

            // Type edges (index 4)
            CpgEdgeKind::HasType | CpgEdgeKind::Inherits => edge_type[4] = 1.0,
        }

        Self { source, target, edge_type }
    }
}
```

## GnnFeatures

Complete feature set extracted from a CPG:

```rust
pub struct GnnFeatures {
    /// Node features
    pub node_features: Vec<NodeFeatures>,
    /// Edge features
    pub edge_features: Vec<EdgeFeatures>,
    /// Total number of nodes
    pub num_nodes: usize,
    /// Total number of edges
    pub num_edges: usize,
}
```

### Feature Extraction

```rust
use libgrammstein::code::{GnnSemanticScorer, GnnConfig, CodePropertyGraph};

let scorer = GnnSemanticScorer::new(GnnConfig::default());
let features = scorer.extract_features(&cpg);

println!("Nodes: {}", features.num_nodes);
println!("Edges: {}", features.num_edges);
```

### Feature Representations

```rust
// Convert to adjacency list for graph processing
let adj_list = features.to_adjacency_list();

// Convert node features to dense matrix
let node_matrix = features.to_node_matrix();
```

## IssueType

Types of semantic issues detected:

```rust
pub enum IssueType {
    /// Wrong variable used in context
    VariableMisuse,
    /// Type mismatch in operation
    TypeError,
    /// Missing error handling
    MissingErrorHandling,
    /// Null/None dereference risk
    NullDereference,
    /// Unused variable/import
    UnusedBinding,
    /// API misuse (wrong method, missing call)
    ApiMisuse,
    /// Resource leak (unclosed file, connection)
    ResourceLeak,
    /// General semantic anomaly
    Anomaly,
}
```

### Issue Type Examples

| Issue Type | Example | Detection |
|------------|---------|-----------|
| `VariableMisuse` | `return resutl` instead of `result` | Name similarity + data flow |
| `TypeError` | `int + string` | Type analysis |
| `MissingErrorHandling` | Unchecked result | Exception flow |
| `NullDereference` | `obj.method()` when obj may be None | Null propagation |
| `UnusedBinding` | Variable defined but never read | Data flow analysis |
| `ApiMisuse` | Wrong method arguments | API pattern matching |
| `ResourceLeak` | File opened but not closed | Resource tracking |
| `Anomaly` | Unusual code pattern | Statistical deviation |

## SemanticIssue

Detected semantic issue with context:

```rust
pub struct SemanticIssue {
    /// Node index where issue was detected
    pub node_idx: usize,
    /// Issue type
    pub issue_type: IssueType,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,
    /// Suggested fix (if available)
    pub suggestion: Option<String>,
    /// Related nodes involved in the issue
    pub related_nodes: Vec<usize>,
}
```

## GnnSemanticScorer

Main interface for GNN-based semantic analysis:

```rust
pub struct GnnSemanticScorer {
    config: GnnConfig,
    node_embeddings: HashMap<usize, Vec<f32>>,
}
```

### Creating a Scorer

```rust
use libgrammstein::code::{GnnSemanticScorer, GnnConfig};

// With default configuration
let scorer = GnnSemanticScorer::default_scorer();

// With custom configuration
let config = GnnConfig {
    num_layers: 4,
    hidden_dim: 512,
    ..Default::default()
};
let scorer = GnnSemanticScorer::new(config);
```

### Detecting Issues

```rust
use libgrammstein::code::{CodeParser, CodePropertyGraph, Python};
use std::sync::Arc;

let python = Arc::new(Python::new());
let mut parser = CodeParser::new(python.clone()).unwrap();
let scorer = GnnSemanticScorer::default_scorer();

let source = r#"
def process(data):
    result = []
    for item in data:
        total += item.value  # Error: 'total' not defined
    return result            # Warning: 'total' never used
"#;

let parsed = parser.parse(source).unwrap();
let cpg = CodePropertyGraph::from_parsed_code(&parsed);

// Detect semantic issues
let issues = scorer.detect_issues(&cpg);

for issue in &issues {
    println!("Issue at node {}: {:?}", issue.node_idx, issue.issue_type);
    println!("  Confidence: {:.2}", issue.confidence);
    if let Some(suggestion) = &issue.suggestion {
        println!("  Suggestion: {}", suggestion);
    }
}
```

### Scoring Individual Nodes

```rust
// Score a specific node for potential issues
let score = scorer.score_node(&cpg, node_idx);
println!("Anomaly score: {:.2}", score);  // Higher = more likely problematic
```

### Variable Misuse Detection

```rust
// Find alternative variables that might be correct
let candidates = scorer.variable_misuse_candidates(&cpg, node_idx);

for (name, score) in &candidates {
    println!("  {} (similarity: {:.2})", name, score);
}
// Output:
//   result (similarity: 0.85)
//   results (similarity: 0.65)
```

## Name Similarity

The scorer uses Jaccard similarity on character bigrams:

```rust
fn compute_similarity(&self, a: &str, b: &str) -> f64 {
    // Extract character bigrams
    let bigrams_a: HashSet<_> = a.chars().collect::<Vec<_>>()
        .windows(2)
        .map(|w| (w[0], w[1]))
        .collect();

    let bigrams_b: HashSet<_> = b.chars().collect::<Vec<_>>()
        .windows(2)
        .map(|w| (w[0], w[1]))
        .collect();

    // Jaccard similarity
    let intersection = bigrams_a.intersection(&bigrams_b).count();
    let union = bigrams_a.union(&bigrams_b).count();

    intersection as f64 / union as f64
}
```

### Similarity Examples

| String A | String B | Similarity |
|----------|----------|------------|
| `count` | `counter` | ~0.67 |
| `result` | `resutl` | ~0.80 |
| `foo` | `bar` | ~0.00 |
| `test` | `test` | 1.00 |

## Integration Example

Complete semantic analysis workflow:

```rust
use libgrammstein::code::{
    CodeParser, CodePropertyGraph, GnnSemanticScorer, GnnConfig,
    IssueType, Python
};
use std::sync::Arc;

fn analyze_code_semantics(source: &str) -> Vec<String> {
    let python = Arc::new(Python::new());
    let mut parser = CodeParser::new(python.clone()).unwrap();

    // Parse source
    let parsed = match parser.parse(source) {
        Ok(p) => p,
        Err(_) => return vec!["Failed to parse".to_string()],
    };

    // Build CPG
    let cpg = CodePropertyGraph::from_parsed_code(&parsed);

    // Configure scorer
    let config = GnnConfig {
        num_layers: 3,
        use_attention: true,
        ..Default::default()
    };
    let scorer = GnnSemanticScorer::new(config);

    // Extract features for analysis
    let features = scorer.extract_features(&cpg);
    println!("Analyzing {} nodes, {} edges",
        features.num_nodes, features.num_edges);

    // Detect issues
    let issues = scorer.detect_issues(&cpg);

    // Format results
    let mut messages = Vec::new();
    for issue in &issues {
        let msg = match issue.issue_type {
            IssueType::VariableMisuse => {
                let candidates = scorer.variable_misuse_candidates(&cpg, issue.node_idx);
                let suggestions: Vec<_> = candidates.iter()
                    .take(3)
                    .map(|(n, _)| n.as_str())
                    .collect();
                format!("Variable misuse at node {}: did you mean {:?}?",
                    issue.node_idx, suggestions)
            }
            IssueType::UnusedBinding => {
                format!("Unused binding at node {} (confidence: {:.0}%)",
                    issue.node_idx, issue.confidence * 100.0)
            }
            IssueType::TypeError => {
                format!("Type error at node {}: {}",
                    issue.node_idx,
                    issue.suggestion.as_deref().unwrap_or("type mismatch"))
            }
            _ => {
                format!("{:?} at node {} (confidence: {:.0}%)",
                    issue.issue_type, issue.node_idx, issue.confidence * 100.0)
            }
        };
        messages.push(msg);
    }

    messages
}

let source = r#"
def calculate(x, y):
    total = x + y
    return totla  # Typo
"#;

let issues = analyze_code_semantics(source);
for issue in issues {
    println!("  {}", issue);
}
```

## Unused Binding Detection

The scorer detects variables written but never read:

```rust
// Simplified detection logic
for node in cpg.all_nodes() {
    if node.kind == CpgNodeKind::Variable {
        // Count incoming writes
        let writes = edges.iter()
            .filter(|(_, t, e)| *t == node.id && matches!(e.kind,
                CpgEdgeKind::DfgFlow | CpgEdgeKind::DfgWrite))
            .count();

        // Count outgoing reads
        let reads = edges.iter()
            .filter(|(s, _, e)| *s == node.id && matches!(e.kind,
                CpgEdgeKind::DfgFlow | CpgEdgeKind::DfgRead))
            .count();

        // Variable written but never read
        if writes > 0 && reads == 0 {
            issues.push(SemanticIssue {
                node_idx: node.id,
                issue_type: IssueType::UnusedBinding,
                confidence: 0.6,
                suggestion: Some("Variable may be unused".to_string()),
                related_nodes: vec![],
            });
        }
    }
}
```

## Performance

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Feature extraction | O(n + e) | n = nodes, e = edges |
| Issue detection | O(n × e) | Quadratic in worst case |
| Similarity computation | O(len²) | Bigram comparison |
| Variable candidates | O(v) | v = variables in scope |

### Optimization Tips

1. **Limit scope**: Focus on error regions
2. **Cache embeddings**: Reuse computed embeddings
3. **Batch processing**: Process multiple nodes together
4. **Prune edges**: Use relevant edge types only

## Thread Safety

`GnnSemanticScorer` is `Send + Sync` for read-only operations:

```rust
use std::sync::Arc;

let scorer = Arc::new(GnnSemanticScorer::default_scorer());

// Safe to share across threads
let results: Vec<_> = cpgs.par_iter()
    .map(|cpg| scorer.detect_issues(cpg))
    .collect();
```

## See Also

- [CPG](cpg.md) - Code Property Graph structure
- [Semantic Corrector](correctors/semantic.md) - Using GNN for corrections
- [Embeddings](embeddings.md) - Code embedding models
- [Pipeline](pipeline.md) - End-to-end workflow
