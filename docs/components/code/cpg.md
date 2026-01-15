# Code Property Graph (CPG)

The CPG module provides construction and analysis of Code Property Graphs, a unified representation combining AST, CFG, and DFG for semantic code analysis.

## What is a Code Property Graph?

A Code Property Graph combines three representations into a single graph structure:

1. **Abstract Syntax Tree (AST)**: Syntactic structure of the code
2. **Control Flow Graph (CFG)**: Possible execution paths
3. **Data Flow Graph (DFG)**: Variable dependencies and data movement

This unified representation enables powerful semantic analysis for:

- Variable misuse detection
- Type error identification
- Dead code detection
- Vulnerability analysis
- Program understanding

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     Code Property Graph                          │
│                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │      AST        │  │      CFG        │  │      DFG        │  │
│  │                 │  │                 │  │                 │  │
│  │  • AstChild     │  │  • CfgNext      │  │  • DfgRead      │  │
│  │  • AstSibling   │  │  • CfgTrue      │  │  • DfgWrite     │  │
│  │                 │  │  • CfgFalse     │  │  • DfgFlow      │  │
│  │                 │  │  • CfgBack      │  │  • DfgDepends   │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                  Unified Graph (petgraph)                    ││
│  │                                                               ││
│  │   Nodes: CpgNode (Function, Variable, Call, Branch, ...)     ││
│  │   Edges: CpgEdge (AST, CFG, DFG, Call, Type relationships)   ││
│  └─────────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────┘
```

## Key Types

| Type | Description |
|------|-------------|
| `CodePropertyGraph` | Main graph container |
| `CpgNode` | Node in the graph |
| `CpgNodeKind` | Classification of nodes |
| `CpgEdge` | Edge between nodes |
| `CpgEdgeKind` | Type of relationship |

## CpgNode

A node in the Code Property Graph:

```rust
pub struct CpgNode {
    /// Unique identifier for this node
    pub id: usize,
    /// The kind of node (e.g., "function", "variable", "call")
    pub kind: CpgNodeKind,
    /// The text/name of the node
    pub name: Option<String>,
    /// Source location (start_byte, end_byte)
    pub location: (usize, usize),
    /// Line and column
    pub position: (usize, usize),
    /// AST node kind from tree-sitter
    pub ast_kind: String,
    /// Additional properties
    pub properties: HashMap<String, String>,
}
```

## CpgNodeKind

Classification of nodes in the graph:

```rust
pub enum CpgNodeKind {
    Function,     // Function/method definition
    Parameter,    // Function parameter
    Variable,     // Local variable
    Type,         // Type reference
    Call,         // Function/method call
    Return,       // Return statement
    Branch,       // Control flow branch (if, match)
    Loop,         // Loop construct (for, while)
    Assignment,   // Assignment expression
    BinaryOp,     // Binary operation
    UnaryOp,      // Unary operation
    Literal,      // Literal value
    FieldAccess,  // Field access
    IndexAccess,  // Index access
    Block,        // Block/scope
    Import,       // Import/use statement
    Other,        // Other/unknown
}
```

## CpgEdge

An edge representing a relationship:

```rust
pub struct CpgEdge {
    /// The type of relationship
    pub kind: CpgEdgeKind,
    /// Optional label for the edge
    pub label: Option<String>,
}
```

## CpgEdgeKind

Types of relationships in the graph:

```rust
pub enum CpgEdgeKind {
    // AST edges
    AstChild,     // Parent-child in AST
    AstSibling,   // Next sibling in AST

    // CFG edges
    CfgNext,      // Sequential execution
    CfgTrue,      // True branch
    CfgFalse,     // False branch
    CfgBack,      // Loop back edge
    CfgException, // Exception path

    // DFG edges
    DfgRead,      // Value is read
    DfgWrite,     // Value is written
    DfgFlow,      // Value flows from source to target
    DfgDepends,   // Data dependency

    // Call graph edges
    Calls,        // Function call relationship
    Argument,     // Argument passing
    Returns,      // Return value

    // Type edges
    HasType,      // Has type
    Inherits,     // Type inheritance
}
```

## Building a CPG

### From Parsed Code

```rust
use libgrammstein::code::{CodeParser, CodePropertyGraph, Python};
use std::sync::Arc;

let python = Arc::new(Python::new());
let mut parser = CodeParser::new(python)?;

let source = r#"
def calculate_total(items):
    total = 0
    for item in items:
        total += item.price
    return total
"#;

let parsed = parser.parse(source)?;
let cpg = CodePropertyGraph::from_parsed_code(&parsed);

println!("Nodes: {}", cpg.node_count());
println!("Edges: {}", cpg.edge_count());
```

### Manual Construction

```rust
use libgrammstein::code::{
    CodePropertyGraph, CpgNode, CpgEdge, CpgNodeKind, CpgEdgeKind
};
use std::collections::HashMap;

let mut cpg = CodePropertyGraph::new();

// Add a function node
let func_node = CpgNode {
    id: 0,
    kind: CpgNodeKind::Function,
    name: Some("calculate".to_string()),
    location: (0, 50),
    position: (1, 0),
    ast_kind: "function_definition".to_string(),
    properties: HashMap::new(),
};
let func_idx = cpg.add_node(func_node);

// Add a variable node
let var_node = CpgNode {
    id: 1,
    kind: CpgNodeKind::Variable,
    name: Some("total".to_string()),
    location: (20, 25),
    position: (2, 4),
    ast_kind: "identifier".to_string(),
    properties: HashMap::new(),
};
let var_idx = cpg.add_node(var_node);

// Connect with AST edge
cpg.add_edge(func_idx, var_idx, CpgEdge {
    kind: CpgEdgeKind::AstChild,
    label: None,
});
```

## Querying the CPG

### Finding Nodes by Kind

```rust
// Find all function definitions
let functions = cpg.find_by_kind(CpgNodeKind::Function);
for idx in functions {
    if let Some(node) = cpg.node(idx) {
        println!("Function: {:?}", node.name);
    }
}

// Find all variables
let variables = cpg.find_by_kind(CpgNodeKind::Variable);

// Find all function calls
let calls = cpg.find_by_kind(CpgNodeKind::Call);
```

### Finding Nodes by Location

```rust
// Find node at specific source location
if let Some(idx) = cpg.node_at_location(20, 25) {
    let node = cpg.node(idx).unwrap();
    println!("Found node: {:?} at bytes 20-25", node.name);
}
```

### Variable Definitions

```rust
// Find all definitions of variable "total"
let defs = cpg.variable_definitions("total");
for def_idx in defs {
    let node = cpg.node(def_idx).unwrap();
    println!("'total' defined at line {}", node.position.0 + 1);
}
```

## Control Flow Analysis

### CFG Traversal

```rust
// Get CFG successors (where can execution go from here?)
let successors = cpg.cfg_successors(node_idx);
for succ in successors {
    let node = cpg.node(succ).unwrap();
    println!("  -> {:?}", node.kind);
}

// Get CFG predecessors (where did execution come from?)
let predecessors = cpg.cfg_predecessors(node_idx);
```

### Entry and Exit Points

```rust
// Get function entry point
if let Some(entry) = cpg.entry() {
    let node = cpg.node(entry).unwrap();
    println!("Entry point: {:?}", node.name);
}

// Get all exit points (return statements, end of function)
for exit in cpg.exits() {
    let node = cpg.node(*exit).unwrap();
    println!("Exit at line {}", node.position.0 + 1);
}
```

## Data Flow Analysis

### Tracking Variable Usage

```rust
// Get all reads and writes for a variable
let flows = cpg.data_flow_for_variable("total");
for (node_idx, edge_kind) in flows {
    let node = cpg.node(node_idx).unwrap();
    match edge_kind {
        CpgEdgeKind::DfgRead => {
            println!("'total' read at line {}", node.position.0 + 1);
        }
        CpgEdgeKind::DfgWrite => {
            println!("'total' written at line {}", node.position.0 + 1);
        }
        CpgEdgeKind::DfgFlow => {
            println!("'total' flows to {:?}", node.kind);
        }
        _ => {}
    }
}
```

### Edge Traversal

```rust
// Get all edges from a node
for (target_idx, edge) in cpg.edges_from(node_idx) {
    let target = cpg.node(target_idx).unwrap();
    println!("{:?} -> {:?}: {:?}", node.name, target.name, edge.kind);
}

// Get all edges to a node
for (source_idx, edge) in cpg.edges_to(node_idx) {
    let source = cpg.node(source_idx).unwrap();
    println!("{:?} <- {:?}: {:?}", node.name, source.name, edge.kind);
}
```

## Analysis Utilities

### Computing Node Depths

```rust
// Compute AST depth for each node
let depths = cpg.compute_depths();
for (node_id, depth) in depths {
    println!("Node {} at depth {}", node_id, depth);
}
```

### Computing Child Counts

```rust
// Count children for each node
let child_counts = cpg.compute_child_counts();
for (node_id, count) in child_counts {
    if count > 5 {
        println!("Node {} has {} children (complex)", node_id, count);
    }
}
```

### Iterating All Nodes and Edges

```rust
// Iterate all nodes
for node in cpg.all_nodes() {
    println!("Node {}: {:?} - {:?}", node.id, node.kind, node.name);
}

// Iterate all edges
for (source_id, target_id, edge) in cpg.all_edges() {
    println!("{} -> {}: {:?}", source_id, target_id, edge.kind);
}
```

## Use Cases

### Variable Misuse Detection

```rust
fn find_undefined_variables(cpg: &CodePropertyGraph) -> Vec<String> {
    let mut undefined = Vec::new();

    for idx in cpg.find_by_kind(CpgNodeKind::Variable) {
        let node = cpg.node(idx).unwrap();
        if let Some(name) = &node.name {
            // Check if there's a definition before this use
            let defs = cpg.variable_definitions(name);
            if defs.is_empty() {
                undefined.push(name.clone());
            }
        }
    }

    undefined
}
```

### Dead Code Detection

```rust
fn find_unreachable_code(cpg: &CodePropertyGraph) -> Vec<usize> {
    let mut unreachable = Vec::new();

    // Find nodes with no CFG predecessors (except entry)
    for (idx, node) in cpg.nodes() {
        if cpg.cfg_predecessors(idx).is_empty()
            && cpg.entry() != Some(idx)
            && node.kind != CpgNodeKind::Function
        {
            unreachable.push(node.id);
        }
    }

    unreachable
}
```

### Unused Variable Detection

```rust
fn find_unused_variables(cpg: &CodePropertyGraph) -> Vec<String> {
    let mut unused = Vec::new();

    for idx in cpg.find_by_kind(CpgNodeKind::Variable) {
        let node = cpg.node(idx).unwrap();
        if let Some(name) = &node.name {
            // Check if variable has any reads
            let flows = cpg.data_flow_for_variable(name);
            let has_reads = flows.iter().any(|(_, kind)| *kind == CpgEdgeKind::DfgRead);
            if !has_reads {
                unused.push(name.clone());
            }
        }
    }

    unused
}
```

## Integration with Semantic Corrector

The CPG powers the semantic corrector for detecting issues:

```rust
use libgrammstein::code::{
    SemanticCorrector, CodePropertyGraph, Python
};
use std::sync::Arc;

let python = Arc::new(Python::new());
let corrector = SemanticCorrector::with_defaults(python);

// The corrector uses CPG internally for analysis
let corrections = corrector.correct_token(&token, &context);

for correction in corrections {
    if correction.kind == CorrectionKind::VariableMisuse {
        println!(
            "Variable misuse: {} -> {} (from CPG analysis)",
            correction.original,
            correction.replacement
        );
    }
}
```

## Performance

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Build from AST | O(n + e) | n = nodes, e = edges |
| Find by kind | O(n) | Linear scan |
| Node at location | O(1) | HashMap lookup |
| Variable definitions | O(1) | HashMap lookup |
| CFG successors | O(d) | d = out-degree |
| Compute depths | O(n + e) | BFS traversal |

## Thread Safety

`CodePropertyGraph` is not `Sync` but can be built and then shared:

```rust
use std::sync::Arc;

// Build CPG
let cpg = CodePropertyGraph::from_parsed_code(&parsed);

// Wrap for sharing (read-only)
let cpg = Arc::new(cpg);

// Share across threads for analysis
let cpg1 = Arc::clone(&cpg);
let cpg2 = Arc::clone(&cpg);
```

## See Also

- [AST](ast.md) - Tree-sitter parsing (input to CPG)
- [GNN](gnn.md) - Graph neural networks on CPG
- [Semantic Corrector](correctors/semantic.md) - CPG-based correction
- [Pipeline](pipeline.md) - End-to-end integration
