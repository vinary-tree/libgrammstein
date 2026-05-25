//! Code Property Graph construction and analysis.
//!
//! A Code Property Graph (CPG) combines three representations:
//! 1. Abstract Syntax Tree (AST) - syntactic structure
//! 2. Control Flow Graph (CFG) - execution paths
//! 3. Data Flow Graph (DFG) - variable dependencies
//!
//! This unified representation enables semantic analysis for:
//! - Variable misuse detection
//! - Type error identification
//! - Dead code detection
//! - Vulnerability analysis

use super::ast::{AstNode, ParsedCode};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use std::collections::HashMap;

/// A node in the Code Property Graph.
#[derive(Debug, Clone)]
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

/// Classification of CPG nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CpgNodeKind {
    /// Function/method definition
    Function,
    /// Parameter in a function signature
    Parameter,
    /// Local variable
    Variable,
    /// Type reference
    Type,
    /// Function/method call
    Call,
    /// Return statement
    Return,
    /// Control flow branch (if, match, etc.)
    Branch,
    /// Loop construct
    Loop,
    /// Assignment expression
    Assignment,
    /// Binary operation
    BinaryOp,
    /// Unary operation
    UnaryOp,
    /// Literal value
    Literal,
    /// Field access
    FieldAccess,
    /// Index access
    IndexAccess,
    /// Block/scope
    Block,
    /// Import/use statement
    Import,
    /// Other/unknown
    Other,
}

/// An edge in the Code Property Graph.
#[derive(Debug, Clone)]
pub struct CpgEdge {
    /// The type of relationship
    pub kind: CpgEdgeKind,
    /// Optional label for the edge
    pub label: Option<String>,
}

/// Types of edges in the CPG.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CpgEdgeKind {
    // AST edges
    /// Parent-child relationship in AST
    AstChild,
    /// Next sibling in AST
    AstSibling,

    // CFG edges
    /// Control flow: sequential execution
    CfgNext,
    /// Control flow: true branch
    CfgTrue,
    /// Control flow: false branch
    CfgFalse,
    /// Control flow: loop back edge
    CfgBack,
    /// Control flow: exception/error path
    CfgException,

    // DFG edges
    /// Data flow: value is read
    DfgRead,
    /// Data flow: value is written/defined
    DfgWrite,
    /// Data flow: value flows from source to target
    DfgFlow,
    /// Data dependency
    DfgDepends,

    // Call graph edges
    /// Function call relationship
    Calls,
    /// Argument passing
    Argument,
    /// Return value
    Returns,

    // Type edges
    /// Has type
    HasType,
    /// Type inheritance/implementation
    Inherits,
}

/// A Code Property Graph combining AST, CFG, and DFG.
pub struct CodePropertyGraph {
    /// The underlying graph structure
    graph: DiGraph<CpgNode, CpgEdge>,
    /// Map from source locations to node indices
    location_map: HashMap<(usize, usize), NodeIndex>,
    /// Map from variable names to their definition nodes
    variable_defs: HashMap<String, Vec<NodeIndex>>,
    /// Entry point node (e.g., function entry)
    entry_node: Option<NodeIndex>,
    /// Exit nodes (return points, end of function)
    exit_nodes: Vec<NodeIndex>,
}

impl CodePropertyGraph {
    /// Creates an empty CPG.
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            location_map: HashMap::new(),
            variable_defs: HashMap::new(),
            entry_node: None,
            exit_nodes: Vec::new(),
        }
    }

    /// Builds a CPG from parsed code.
    pub fn from_parsed_code(parsed: &ParsedCode) -> Self {
        let mut cpg = Self::new();
        let ast_node = AstNode::from_ts_node(parsed.root(), &parsed.source);
        cpg.build_from_ast(&ast_node, None);
        cpg.build_cfg();
        cpg.build_dfg();
        cpg
    }

    /// Adds a node to the graph.
    pub fn add_node(&mut self, node: CpgNode) -> NodeIndex {
        let location = node.location;
        let idx = self.graph.add_node(node);
        self.location_map.insert(location, idx);
        idx
    }

    /// Adds an edge between two nodes.
    pub fn add_edge(&mut self, from: NodeIndex, to: NodeIndex, edge: CpgEdge) {
        self.graph.add_edge(from, to, edge);
    }

    /// Returns the node at the given index.
    pub fn node(&self, idx: NodeIndex) -> Option<&CpgNode> {
        self.graph.node_weight(idx)
    }

    /// Returns all nodes in the graph.
    pub fn nodes(&self) -> impl Iterator<Item = (NodeIndex, &CpgNode)> {
        self.graph
            .node_indices()
            .filter_map(move |idx| self.graph.node_weight(idx).map(|n| (idx, n)))
    }

    /// Returns all edges from a node.
    pub fn edges_from(&self, idx: NodeIndex) -> impl Iterator<Item = (NodeIndex, &CpgEdge)> + '_ {
        self.graph.edges(idx).map(|e| (e.target(), e.weight()))
    }

    /// Returns all incoming edges to a node.
    pub fn edges_to(&self, idx: NodeIndex) -> impl Iterator<Item = (NodeIndex, &CpgEdge)> + '_ {
        self.graph
            .edges_directed(idx, petgraph::Direction::Incoming)
            .map(|e| (e.source(), e.weight()))
    }

    /// Finds nodes by kind.
    pub fn find_by_kind(&self, kind: CpgNodeKind) -> Vec<NodeIndex> {
        self.graph
            .node_indices()
            .filter(|&idx| {
                self.graph
                    .node_weight(idx)
                    .map(|n| n.kind == kind)
                    .unwrap_or(false)
            })
            .collect()
    }

    /// Finds the node at a specific source location.
    pub fn node_at_location(&self, start: usize, end: usize) -> Option<NodeIndex> {
        self.location_map.get(&(start, end)).copied()
    }

    /// Returns all variable definitions with a given name.
    pub fn variable_definitions(&self, name: &str) -> Vec<NodeIndex> {
        self.variable_defs.get(name).cloned().unwrap_or_default()
    }

    /// Returns all data flow edges (reads and writes) for a variable.
    pub fn data_flow_for_variable(&self, name: &str) -> Vec<(NodeIndex, CpgEdgeKind)> {
        let mut flows = Vec::new();
        for &def_idx in self.variable_defs.get(name).unwrap_or(&Vec::new()) {
            for edge in self.graph.edges(def_idx) {
                match edge.weight().kind {
                    CpgEdgeKind::DfgRead | CpgEdgeKind::DfgWrite | CpgEdgeKind::DfgFlow => {
                        flows.push((edge.target(), edge.weight().kind));
                    }
                    _ => {}
                }
            }
        }
        flows
    }

    /// Returns control flow successors of a node.
    pub fn cfg_successors(&self, idx: NodeIndex) -> Vec<NodeIndex> {
        self.graph
            .edges(idx)
            .filter(|e| {
                matches!(
                    e.weight().kind,
                    CpgEdgeKind::CfgNext
                        | CpgEdgeKind::CfgTrue
                        | CpgEdgeKind::CfgFalse
                        | CpgEdgeKind::CfgBack
                )
            })
            .map(|e| e.target())
            .collect()
    }

    /// Returns control flow predecessors of a node.
    pub fn cfg_predecessors(&self, idx: NodeIndex) -> Vec<NodeIndex> {
        self.graph
            .edges_directed(idx, petgraph::Direction::Incoming)
            .filter(|e| {
                matches!(
                    e.weight().kind,
                    CpgEdgeKind::CfgNext
                        | CpgEdgeKind::CfgTrue
                        | CpgEdgeKind::CfgFalse
                        | CpgEdgeKind::CfgBack
                )
            })
            .map(|e| e.source())
            .collect()
    }

    /// Returns the entry node of the graph.
    pub fn entry(&self) -> Option<NodeIndex> {
        self.entry_node
    }

    /// Returns all exit nodes.
    pub fn exits(&self) -> &[NodeIndex] {
        &self.exit_nodes
    }

    /// Returns the number of nodes.
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Returns the number of edges.
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Computes the depth of each node in the AST.
    pub fn compute_depths(&self) -> HashMap<usize, usize> {
        let mut depths = HashMap::new();

        // BFS from entry or first node
        let start = self.entry_node.or_else(|| self.graph.node_indices().next());
        if let Some(start_idx) = start {
            let mut queue = std::collections::VecDeque::new();
            queue.push_back((start_idx, 0usize));
            depths.insert(
                self.graph.node_weight(start_idx).map(|n| n.id).unwrap_or(0),
                0,
            );

            while let Some((idx, depth)) = queue.pop_front() {
                for edge in self.graph.edges(idx) {
                    if edge.weight().kind == CpgEdgeKind::AstChild {
                        let target = edge.target();
                        if let Some(node) = self.graph.node_weight(target) {
                            if !depths.contains_key(&node.id) {
                                depths.insert(node.id, depth + 1);
                                queue.push_back((target, depth + 1));
                            }
                        }
                    }
                }
            }
        }

        depths
    }

    /// Computes the number of children for each node.
    pub fn compute_child_counts(&self) -> HashMap<usize, usize> {
        let mut counts = HashMap::new();

        for idx in self.graph.node_indices() {
            let count = self
                .graph
                .edges(idx)
                .filter(|e| e.weight().kind == CpgEdgeKind::AstChild)
                .count();
            if let Some(node) = self.graph.node_weight(idx) {
                counts.insert(node.id, count);
            }
        }

        counts
    }

    /// Returns an iterator over all edges with source and target indices.
    pub fn all_edges(&self) -> impl Iterator<Item = (usize, usize, &CpgEdge)> + '_ {
        self.graph.edge_references().map(|e| {
            let source_id = self
                .graph
                .node_weight(e.source())
                .map(|n| n.id)
                .unwrap_or(0);
            let target_id = self
                .graph
                .node_weight(e.target())
                .map(|n| n.id)
                .unwrap_or(0);
            (source_id, target_id, e.weight())
        })
    }

    /// Returns an iterator over all node references.
    pub fn all_nodes(&self) -> impl Iterator<Item = &CpgNode> + '_ {
        self.graph
            .node_indices()
            .filter_map(move |idx| self.graph.node_weight(idx))
    }

    // Private methods for building the graph

    fn build_from_ast(&mut self, ast: &AstNode, parent: Option<NodeIndex>) -> NodeIndex {
        let kind = self.classify_ast_node(&ast.kind);
        let node = CpgNode {
            id: self.graph.node_count(),
            kind,
            name: ast.text.clone(),
            location: (ast.start_byte, ast.end_byte),
            position: ast.start_position,
            ast_kind: ast.kind.clone(),
            properties: HashMap::new(),
        };

        let idx = self.add_node(node);

        // Track variable definitions
        if kind == CpgNodeKind::Variable || kind == CpgNodeKind::Parameter {
            if let Some(name) = &ast.text {
                self.variable_defs
                    .entry(name.clone())
                    .or_default()
                    .push(idx);
            }
        }

        // Connect to parent
        if let Some(parent_idx) = parent {
            self.add_edge(
                parent_idx,
                idx,
                CpgEdge {
                    kind: CpgEdgeKind::AstChild,
                    label: None,
                },
            );
        }

        // Process children
        let mut prev_child: Option<NodeIndex> = None;
        for child in &ast.children {
            let child_idx = self.build_from_ast(child, Some(idx));

            // Connect siblings
            if let Some(prev) = prev_child {
                self.add_edge(
                    prev,
                    child_idx,
                    CpgEdge {
                        kind: CpgEdgeKind::AstSibling,
                        label: None,
                    },
                );
            }
            prev_child = Some(child_idx);
        }

        idx
    }

    fn classify_ast_node(&self, kind: &str) -> CpgNodeKind {
        match kind {
            // Common patterns across languages
            k if k.contains("function") || k.contains("method") => CpgNodeKind::Function,
            k if k.contains("parameter") || k.contains("param") => CpgNodeKind::Parameter,
            k if k.contains("variable") || k.contains("binding") || k.contains("let") => {
                CpgNodeKind::Variable
            }
            k if k.contains("type") && !k.contains("typeof") => CpgNodeKind::Type,
            k if k.contains("call") || k.contains("invoke") => CpgNodeKind::Call,
            k if k.contains("return") => CpgNodeKind::Return,
            k if k.contains("if") || k.contains("match") || k.contains("switch") => {
                CpgNodeKind::Branch
            }
            k if k.contains("for") || k.contains("while") || k.contains("loop") => {
                CpgNodeKind::Loop
            }
            k if k.contains("assignment") || k == "=" => CpgNodeKind::Assignment,
            k if k.contains("binary") => CpgNodeKind::BinaryOp,
            k if k.contains("unary") => CpgNodeKind::UnaryOp,
            k if k.contains("literal") || k.contains("string") || k.contains("number") => {
                CpgNodeKind::Literal
            }
            k if k.contains("field") || k.contains("member") => CpgNodeKind::FieldAccess,
            k if k.contains("index") || k.contains("subscript") => CpgNodeKind::IndexAccess,
            k if k.contains("block") || k.contains("body") => CpgNodeKind::Block,
            k if k.contains("import") || k.contains("use") || k.contains("include") => {
                CpgNodeKind::Import
            }
            _ => CpgNodeKind::Other,
        }
    }

    fn build_cfg(&mut self) {
        // Build control flow edges
        // This is a simplified implementation - a full CFG builder would need
        // language-specific logic for proper control flow analysis

        let nodes: Vec<NodeIndex> = self.graph.node_indices().collect();

        for i in 0..nodes.len() {
            let idx = nodes[i];
            if let Some(node) = self.graph.node_weight(idx) {
                match node.kind {
                    CpgNodeKind::Function => {
                        if self.entry_node.is_none() {
                            self.entry_node = Some(idx);
                        }
                    }
                    CpgNodeKind::Return => {
                        self.exit_nodes.push(idx);
                    }
                    CpgNodeKind::Branch => {
                        // Find true and false branches in children
                        let children: Vec<NodeIndex> = self
                            .graph
                            .edges(idx)
                            .filter(|e| e.weight().kind == CpgEdgeKind::AstChild)
                            .map(|e| e.target())
                            .collect();

                        if children.len() >= 2 {
                            self.add_edge(
                                idx,
                                children[0],
                                CpgEdge {
                                    kind: CpgEdgeKind::CfgTrue,
                                    label: Some("true".to_string()),
                                },
                            );
                        }
                        if children.len() >= 3 {
                            self.add_edge(
                                idx,
                                children[1],
                                CpgEdge {
                                    kind: CpgEdgeKind::CfgFalse,
                                    label: Some("false".to_string()),
                                },
                            );
                        }
                    }
                    CpgNodeKind::Loop => {
                        // Add back edge placeholder
                        // Full implementation would track loop body and create proper back edges
                    }
                    _ => {}
                }
            }
        }
    }

    fn build_dfg(&mut self) {
        // Build data flow edges
        // Track variable reads and writes

        let nodes: Vec<NodeIndex> = self.graph.node_indices().collect();

        for idx in nodes {
            if let Some(node) = self.graph.node_weight(idx).cloned() {
                match node.kind {
                    CpgNodeKind::Assignment => {
                        // Find the target (left side) and source (right side)
                        let children: Vec<NodeIndex> = self
                            .graph
                            .edges(idx)
                            .filter(|e| e.weight().kind == CpgEdgeKind::AstChild)
                            .map(|e| e.target())
                            .collect();

                        if children.len() >= 2 {
                            // Left side is written to
                            self.add_edge(
                                idx,
                                children[0],
                                CpgEdge {
                                    kind: CpgEdgeKind::DfgWrite,
                                    label: None,
                                },
                            );
                            // Right side is read from
                            self.add_edge(
                                idx,
                                children[1],
                                CpgEdge {
                                    kind: CpgEdgeKind::DfgRead,
                                    label: None,
                                },
                            );
                        }
                    }
                    CpgNodeKind::Variable => {
                        // Check if this variable reference has a definition
                        if let Some(name) = &node.name {
                            // Clone to avoid borrow conflict with self.add_edge
                            let defs = self.variable_defs.get(name).cloned().unwrap_or_default();
                            for def_idx in defs {
                                if def_idx != idx {
                                    self.add_edge(
                                        def_idx,
                                        idx,
                                        CpgEdge {
                                            kind: CpgEdgeKind::DfgFlow,
                                            label: None,
                                        },
                                    );
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

impl Default for CodePropertyGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpg_basic_operations() {
        let mut cpg = CodePropertyGraph::new();

        let node1 = CpgNode {
            id: 0,
            kind: CpgNodeKind::Function,
            name: Some("foo".to_string()),
            location: (0, 10),
            position: (0, 0),
            ast_kind: "function_definition".to_string(),
            properties: HashMap::new(),
        };

        let node2 = CpgNode {
            id: 1,
            kind: CpgNodeKind::Variable,
            name: Some("x".to_string()),
            location: (11, 15),
            position: (1, 0),
            ast_kind: "identifier".to_string(),
            properties: HashMap::new(),
        };

        let idx1 = cpg.add_node(node1);
        let idx2 = cpg.add_node(node2);

        cpg.add_edge(
            idx1,
            idx2,
            CpgEdge {
                kind: CpgEdgeKind::AstChild,
                label: None,
            },
        );

        assert_eq!(cpg.node_count(), 2);
        assert_eq!(cpg.edge_count(), 1);

        let functions = cpg.find_by_kind(CpgNodeKind::Function);
        assert_eq!(functions.len(), 1);
    }
}
