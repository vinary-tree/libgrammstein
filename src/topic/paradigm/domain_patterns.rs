//! Domain-specific pattern catalogs for Rholang and MeTTa.
//!
//! This module provides pattern detection specific to F1R3FLY.io's
//! domain-specific languages:
//!
//! - **Rholang**: A concurrent process calculus for smart contracts
//! - **MeTTa**: A meta-language for hypergraph-based AI
//!
//! # Rholang Patterns
//!
//! Rholang is a reflective, higher-order process calculus based on
//! the rho-calculus. Key patterns include:
//!
//! - Process composition (|, ;)
//! - Channel operations (!, *, @)
//! - Contracts and registry patterns
//! - Concurrency idioms (join patterns, barriers)
//!
//! # MeTTa Patterns
//!
//! MeTTa is a declarative language for hypergraph-based AI systems.
//! Key patterns include:
//!
//! - Atom patterns (Symbol, Expression, Grounded)
//! - Type patterns
//! - Inference patterns
//! - Knowledge representation

use std::collections::HashMap;
use std::sync::Arc;

#[cfg(feature = "serde-extras")]
use serde::{Deserialize, Serialize};


/// Rholang-specific pattern categories.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde-extras", derive(Serialize, Deserialize))]
pub enum RholangPatternCategory {
    /// Process composition patterns (|, ;).
    ProcessComposition,
    /// Channel operations (send !, receive for/select).
    ChannelOperation,
    /// Replication and persistence (!pattern).
    Replication,
    /// Contract definitions.
    Contract,
    /// Registry and lookup patterns.
    Registry,
    /// Join patterns (multi-channel synchronization).
    JoinPattern,
    /// Barrier and synchronization patterns.
    Synchronization,
    /// Continuation passing patterns.
    Continuation,
    /// Data marshalling (serialization/deserialization).
    DataMarshalling,
    /// Access control and capability patterns.
    AccessControl,
    /// State management patterns.
    StateManagement,
    /// Name reflection (@ and *).
    NameReflection,
}

impl RholangPatternCategory {
    /// Get the description of this category.
    pub fn description(&self) -> &'static str {
        match self {
            Self::ProcessComposition => "Process composition with | (parallel) or ; (sequential)",
            Self::ChannelOperation => "Channel send (!) and receive (for) operations",
            Self::Replication => "Persistent processes with ! prefix",
            Self::Contract => "Contract definitions with contract keyword",
            Self::Registry => "Registry lookup and insertion patterns",
            Self::JoinPattern => "Multi-channel join synchronization",
            Self::Synchronization => "Barrier and synchronization primitives",
            Self::Continuation => "Continuation passing style patterns",
            Self::DataMarshalling => "Data serialization and deserialization",
            Self::AccessControl => "Capability-based access control",
            Self::StateManagement => "Mutable state management patterns",
            Self::NameReflection => "Name quote (@) and dereference (*) operations",
        }
    }
}

/// A Rholang pattern definition.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde-extras", derive(Serialize, Deserialize))]
pub struct RholangPattern {
    /// Pattern name.
    pub name: String,
    /// Pattern category.
    pub category: RholangPatternCategory,
    /// Token sequence patterns (multiple variants).
    pub patterns: Vec<Vec<Arc<str>>>,
    /// Description of the pattern.
    pub description: String,
    /// Example code snippet.
    pub example: Option<String>,
    /// Weight/importance of this pattern.
    pub weight: f64,
    /// Whether this is an anti-pattern.
    pub is_antipattern: bool,
}

impl RholangPattern {
    /// Create a new Rholang pattern.
    pub fn new(
        name: impl Into<String>,
        category: RholangPatternCategory,
        description: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            category,
            patterns: Vec::new(),
            description: description.into(),
            example: None,
            weight: 0.5,
            is_antipattern: false,
        }
    }

    /// Add a pattern variant.
    pub fn with_pattern(mut self, pattern: &[&str]) -> Self {
        self.patterns.push(pattern.iter().map(|&s| Arc::from(s)).collect());
        self
    }

    /// Set the example.
    pub fn with_example(mut self, example: impl Into<String>) -> Self {
        self.example = Some(example.into());
        self
    }

    /// Set the weight.
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }

    /// Mark as anti-pattern.
    pub fn antipattern(mut self) -> Self {
        self.is_antipattern = true;
        self
    }
}

/// MeTTa-specific pattern categories.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde-extras", derive(Serialize, Deserialize))]
pub enum MettaPatternCategory {
    /// Atom definitions (Symbol, Expression, Variable).
    AtomDefinition,
    /// Type definitions and type patterns.
    TypePattern,
    /// Function definitions.
    FunctionDefinition,
    /// Pattern matching with match keyword.
    PatternMatching,
    /// Unification patterns.
    Unification,
    /// Inference rules and reasoning.
    InferenceRule,
    /// Knowledge base operations.
    KnowledgeBase,
    /// Query patterns.
    Query,
    /// Module and import patterns.
    Module,
    /// Grounded atom patterns.
    GroundedAtom,
    /// Space operations.
    SpaceOperation,
    /// Reduction rules.
    ReductionRule,
}

impl MettaPatternCategory {
    /// Get the description of this category.
    pub fn description(&self) -> &'static str {
        match self {
            Self::AtomDefinition => "Atom definitions (Symbol, Expression, Variable)",
            Self::TypePattern => "Type definitions and type-level patterns",
            Self::FunctionDefinition => "Function definitions with = or ->",
            Self::PatternMatching => "Pattern matching with match keyword",
            Self::Unification => "Unification patterns",
            Self::InferenceRule => "Inference rules for reasoning",
            Self::KnowledgeBase => "Knowledge base add/remove operations",
            Self::Query => "Query and retrieval patterns",
            Self::Module => "Module definitions and imports",
            Self::GroundedAtom => "Grounded (external) atom patterns",
            Self::SpaceOperation => "Space operations (add, remove, query)",
            Self::ReductionRule => "Reduction rules for term rewriting",
        }
    }
}

/// A MeTTa pattern definition.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde-extras", derive(Serialize, Deserialize))]
pub struct MettaPattern {
    /// Pattern name.
    pub name: String,
    /// Pattern category.
    pub category: MettaPatternCategory,
    /// Token sequence patterns (multiple variants).
    pub patterns: Vec<Vec<Arc<str>>>,
    /// Description of the pattern.
    pub description: String,
    /// Example code snippet.
    pub example: Option<String>,
    /// Weight/importance of this pattern.
    pub weight: f64,
    /// Whether this is an anti-pattern.
    pub is_antipattern: bool,
}

impl MettaPattern {
    /// Create a new MeTTa pattern.
    pub fn new(
        name: impl Into<String>,
        category: MettaPatternCategory,
        description: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            category,
            patterns: Vec::new(),
            description: description.into(),
            example: None,
            weight: 0.5,
            is_antipattern: false,
        }
    }

    /// Add a pattern variant.
    pub fn with_pattern(mut self, pattern: &[&str]) -> Self {
        self.patterns.push(pattern.iter().map(|&s| Arc::from(s)).collect());
        self
    }

    /// Set the example.
    pub fn with_example(mut self, example: impl Into<String>) -> Self {
        self.example = Some(example.into());
        self
    }

    /// Set the weight.
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }

    /// Mark as anti-pattern.
    pub fn antipattern(mut self) -> Self {
        self.is_antipattern = true;
        self
    }
}

/// Catalog of Rholang patterns.
#[derive(Clone, Debug)]
pub struct RholangPatternCatalog {
    patterns: Vec<RholangPattern>,
    by_category: HashMap<RholangPatternCategory, Vec<usize>>,
}

impl Default for RholangPatternCatalog {
    fn default() -> Self {
        Self::new()
    }
}

impl RholangPatternCatalog {
    /// Create a new catalog with built-in patterns.
    pub fn new() -> Self {
        let mut catalog = Self {
            patterns: Vec::new(),
            by_category: HashMap::new(),
        };

        catalog.add_builtin_patterns();
        catalog
    }

    /// Get all patterns.
    pub fn patterns(&self) -> &[RholangPattern] {
        &self.patterns
    }

    /// Get patterns by category.
    pub fn by_category(&self, category: RholangPatternCategory) -> Vec<&RholangPattern> {
        self.by_category
            .get(&category)
            .map(|indices| indices.iter().map(|&i| &self.patterns[i]).collect())
            .unwrap_or_default()
    }

    /// Add a custom pattern.
    pub fn add_pattern(&mut self, pattern: RholangPattern) {
        let idx = self.patterns.len();
        self.by_category
            .entry(pattern.category)
            .or_default()
            .push(idx);
        self.patterns.push(pattern);
    }

    fn add_builtin_patterns(&mut self) {
        // === Process Composition ===
        self.add_pattern(
            RholangPattern::new(
                "parallel_composition",
                RholangPatternCategory::ProcessComposition,
                "Parallel process composition with |",
            )
            .with_pattern(&["|"])
            .with_example("P | Q")
            .with_weight(0.8),
        );

        // === Channel Operations ===
        self.add_pattern(
            RholangPattern::new(
                "channel_send",
                RholangPatternCategory::ChannelOperation,
                "Send data on a channel",
            )
            .with_pattern(&["!"])
            .with_pattern(&["!", "("])
            .with_example("channel!(data)")
            .with_weight(0.9),
        );

        self.add_pattern(
            RholangPattern::new(
                "channel_receive",
                RholangPatternCategory::ChannelOperation,
                "Receive data from a channel using for comprehension",
            )
            .with_pattern(&["for", "(", "@"])
            .with_pattern(&["for", "("])
            .with_example("for (@data <- channel) { ... }")
            .with_weight(0.9),
        );

        self.add_pattern(
            RholangPattern::new(
                "select_receive",
                RholangPatternCategory::ChannelOperation,
                "Select from multiple channels",
            )
            .with_pattern(&["select", "{"])
            .with_example("select { case @x <- ch1 => ... }")
            .with_weight(0.7),
        );

        // === Contracts ===
        self.add_pattern(
            RholangPattern::new(
                "contract_definition",
                RholangPatternCategory::Contract,
                "Define a persistent contract",
            )
            .with_pattern(&["contract"])
            .with_pattern(&["contract", "("])
            .with_example("contract foo(@arg, ret) = { ... }")
            .with_weight(0.95),
        );

        // === Name Reflection ===
        self.add_pattern(
            RholangPattern::new(
                "name_quote",
                RholangPatternCategory::NameReflection,
                "Quote a process as a name with @",
            )
            .with_pattern(&["@"])
            .with_example("@P")
            .with_weight(0.7),
        );

        self.add_pattern(
            RholangPattern::new(
                "name_dereference",
                RholangPatternCategory::NameReflection,
                "Dereference a name as a process with *",
            )
            .with_pattern(&["*"])
            .with_example("*channel")
            .with_weight(0.7),
        );

        // === Registry ===
        self.add_pattern(
            RholangPattern::new(
                "registry_lookup",
                RholangPatternCategory::Registry,
                "Look up a URI in the registry",
            )
            .with_pattern(&["rho", ":", "registry", ":", "lookup"])
            .with_pattern(&["registry", "!", "(", "`"])
            .with_example("new lookup(`rho:registry:lookup`) in { ... }")
            .with_weight(0.8),
        );

        self.add_pattern(
            RholangPattern::new(
                "registry_insert",
                RholangPatternCategory::Registry,
                "Insert into the registry",
            )
            .with_pattern(&["rho", ":", "registry", ":", "insertArbitrary"])
            .with_example("new insertArbitrary(`rho:registry:insertArbitrary`) in { ... }")
            .with_weight(0.8),
        );

        // === New Channel Creation ===
        self.add_pattern(
            RholangPattern::new(
                "new_channel",
                RholangPatternCategory::ChannelOperation,
                "Create new unforgeable channels",
            )
            .with_pattern(&["new"])
            .with_pattern(&["new", "in", "{"])
            .with_example("new channel in { ... }")
            .with_weight(0.85),
        );

        // === Join Patterns ===
        self.add_pattern(
            RholangPattern::new(
                "join_receive",
                RholangPatternCategory::JoinPattern,
                "Synchronize on multiple channels",
            )
            .with_pattern(&["for", "(", ";"])
            .with_pattern(&["for", "(", "@", "<", "-"])
            .with_example("for (@x <- ch1; @y <- ch2) { ... }")
            .with_weight(0.85),
        );

        // === State Management ===
        self.add_pattern(
            RholangPattern::new(
                "state_cell",
                RholangPatternCategory::StateManagement,
                "Mutable state cell pattern",
            )
            .with_pattern(&["for", "(", "@", "state", "<", "-"])
            .with_example("for (@state <- cell) { cell!(newState) }")
            .with_weight(0.75),
        );

        // === Synchronization ===
        self.add_pattern(
            RholangPattern::new(
                "ack_pattern",
                RholangPatternCategory::Synchronization,
                "Acknowledgment pattern for synchronization",
            )
            .with_pattern(&["ack", "!", "("])
            .with_pattern(&["ret", "!", "("])
            .with_example("ack!(Nil)")
            .with_weight(0.7),
        );

        // === Replication ===
        self.add_pattern(
            RholangPattern::new(
                "persistent_receive",
                RholangPatternCategory::Replication,
                "Persistent (replicated) receive",
            )
            .with_pattern(&["contract"])
            .with_example("contract service(@request, response) = { ... }")
            .with_weight(0.85),
        );
    }
}

/// Catalog of MeTTa patterns.
#[derive(Clone, Debug)]
pub struct MettaPatternCatalog {
    patterns: Vec<MettaPattern>,
    by_category: HashMap<MettaPatternCategory, Vec<usize>>,
}

impl Default for MettaPatternCatalog {
    fn default() -> Self {
        Self::new()
    }
}

impl MettaPatternCatalog {
    /// Create a new catalog with built-in patterns.
    pub fn new() -> Self {
        let mut catalog = Self {
            patterns: Vec::new(),
            by_category: HashMap::new(),
        };

        catalog.add_builtin_patterns();
        catalog
    }

    /// Get all patterns.
    pub fn patterns(&self) -> &[MettaPattern] {
        &self.patterns
    }

    /// Get patterns by category.
    pub fn by_category(&self, category: MettaPatternCategory) -> Vec<&MettaPattern> {
        self.by_category
            .get(&category)
            .map(|indices| indices.iter().map(|&i| &self.patterns[i]).collect())
            .unwrap_or_default()
    }

    /// Add a custom pattern.
    pub fn add_pattern(&mut self, pattern: MettaPattern) {
        let idx = self.patterns.len();
        self.by_category
            .entry(pattern.category)
            .or_default()
            .push(idx);
        self.patterns.push(pattern);
    }

    fn add_builtin_patterns(&mut self) {
        // === Atom Definitions ===
        self.add_pattern(
            MettaPattern::new(
                "symbol_definition",
                MettaPatternCategory::AtomDefinition,
                "Define a symbolic atom",
            )
            .with_pattern(&["(", ":", "Symbol", ")"])
            .with_example("(: foo Symbol)")
            .with_weight(0.7),
        );

        self.add_pattern(
            MettaPattern::new(
                "expression_atom",
                MettaPatternCategory::AtomDefinition,
                "Expression atom (nested structure)",
            )
            .with_pattern(&["(", "("])
            .with_example("((foo bar) baz)")
            .with_weight(0.6),
        );

        self.add_pattern(
            MettaPattern::new(
                "variable_atom",
                MettaPatternCategory::AtomDefinition,
                "Variable atom with $ prefix",
            )
            .with_pattern(&["$"])
            .with_example("$x")
            .with_weight(0.8),
        );

        // === Type Patterns ===
        self.add_pattern(
            MettaPattern::new(
                "type_declaration",
                MettaPatternCategory::TypePattern,
                "Declare the type of an atom",
            )
            .with_pattern(&["(", ":", "->", ")"])
            .with_pattern(&["(", ":"])
            .with_example("(: add (-> Number Number Number))")
            .with_weight(0.85),
        );

        self.add_pattern(
            MettaPattern::new(
                "function_type",
                MettaPatternCategory::TypePattern,
                "Function type signature",
            )
            .with_pattern(&["(", "->"])
            .with_example("(-> A B)")
            .with_weight(0.8),
        );

        // === Function Definitions ===
        self.add_pattern(
            MettaPattern::new(
                "function_definition",
                MettaPatternCategory::FunctionDefinition,
                "Define a function with =",
            )
            .with_pattern(&["(", "="])
            .with_example("(= (add $x $y) (+ $x $y))")
            .with_weight(0.9),
        );

        self.add_pattern(
            MettaPattern::new(
                "pattern_function",
                MettaPatternCategory::FunctionDefinition,
                "Function with pattern matching",
            )
            .with_pattern(&["(", "=", "("])
            .with_example("(= (fib 0) 1) (= (fib 1) 1) (= (fib $n) ...)")
            .with_weight(0.85),
        );

        // === Pattern Matching ===
        self.add_pattern(
            MettaPattern::new(
                "match_expression",
                MettaPatternCategory::PatternMatching,
                "Match expression for pattern matching",
            )
            .with_pattern(&["(", "match"])
            .with_example("(match &self (foo $x) $x)")
            .with_weight(0.9),
        );

        self.add_pattern(
            MettaPattern::new(
                "case_match",
                MettaPatternCategory::PatternMatching,
                "Case-based pattern matching",
            )
            .with_pattern(&["(", "case"])
            .with_example("(case $x ((Nil) ...) ((Cons $h $t) ...))")
            .with_weight(0.85),
        );

        // === Unification ===
        self.add_pattern(
            MettaPattern::new(
                "unify_expression",
                MettaPatternCategory::Unification,
                "Unification expression",
            )
            .with_pattern(&["(", "unify"])
            .with_example("(unify $x $y ...)")
            .with_weight(0.8),
        );

        // === Inference Rules ===
        self.add_pattern(
            MettaPattern::new(
                "inference_rule",
                MettaPatternCategory::InferenceRule,
                "Inference rule definition",
            )
            .with_pattern(&["(", ":-"])
            .with_example("(:- (conclusion) (premise1) (premise2))")
            .with_weight(0.85),
        );

        self.add_pattern(
            MettaPattern::new(
                "chain_rule",
                MettaPatternCategory::InferenceRule,
                "Chain rule for inference",
            )
            .with_pattern(&["(", "chain"])
            .with_example("(chain ... ... ...)")
            .with_weight(0.7),
        );

        // === Knowledge Base Operations ===
        self.add_pattern(
            MettaPattern::new(
                "add_atom",
                MettaPatternCategory::KnowledgeBase,
                "Add atom to knowledge base",
            )
            .with_pattern(&["(", "add-atom"])
            .with_pattern(&["!", "("])
            .with_example("!(add-atom &self (foo bar))")
            .with_weight(0.8),
        );

        self.add_pattern(
            MettaPattern::new(
                "remove_atom",
                MettaPatternCategory::KnowledgeBase,
                "Remove atom from knowledge base",
            )
            .with_pattern(&["(", "remove-atom"])
            .with_example("!(remove-atom &self (foo bar))")
            .with_weight(0.75),
        );

        self.add_pattern(
            MettaPattern::new(
                "get_atoms",
                MettaPatternCategory::KnowledgeBase,
                "Get atoms from knowledge base",
            )
            .with_pattern(&["(", "get-atoms"])
            .with_example("!(get-atoms &self)")
            .with_weight(0.7),
        );

        // === Query Patterns ===
        self.add_pattern(
            MettaPattern::new(
                "query_space",
                MettaPatternCategory::Query,
                "Query the space/atomspace",
            )
            .with_pattern(&["(", "match", "&"])
            .with_example("!(match &self (foo $x) $x)")
            .with_weight(0.85),
        );

        // === Space Operations ===
        self.add_pattern(
            MettaPattern::new(
                "self_reference",
                MettaPatternCategory::SpaceOperation,
                "Reference to the current space",
            )
            .with_pattern(&["&self"])
            .with_example("&self")
            .with_weight(0.9),
        );

        self.add_pattern(
            MettaPattern::new(
                "space_bind",
                MettaPatternCategory::SpaceOperation,
                "Bind a value in a space",
            )
            .with_pattern(&["(", "bind!", ")"])
            .with_example("!(bind! &self foo (bar baz))")
            .with_weight(0.75),
        );

        // === Grounded Atoms ===
        self.add_pattern(
            MettaPattern::new(
                "grounded_symbol",
                MettaPatternCategory::GroundedAtom,
                "Grounded (external) symbol",
            )
            .with_pattern(&["@"])
            .with_example("@python-func")
            .with_weight(0.7),
        );

        // === Reduction Rules ===
        self.add_pattern(
            MettaPattern::new(
                "reduction_rule",
                MettaPatternCategory::ReductionRule,
                "Term reduction rule",
            )
            .with_pattern(&["(", "=", "("])
            .with_example("(= (inc $x) (+ $x 1))")
            .with_weight(0.8),
        );

        // === Module Patterns ===
        self.add_pattern(
            MettaPattern::new(
                "import_module",
                MettaPatternCategory::Module,
                "Import a module",
            )
            .with_pattern(&["!", "(", "import!"])
            .with_example("!(import! &self stdlib)")
            .with_weight(0.7),
        );

        // === Command Patterns ===
        self.add_pattern(
            MettaPattern::new(
                "command_prefix",
                MettaPatternCategory::Query,
                "Command execution prefix",
            )
            .with_pattern(&["!"])
            .with_example("!(foo)")
            .with_weight(0.6),
        );
    }
}

/// Domain-specific pattern detector for F1R3FLY.io languages.
#[derive(Clone, Debug)]
pub struct DomainPatternDetector {
    rholang_catalog: RholangPatternCatalog,
    metta_catalog: MettaPatternCatalog,
}

impl Default for DomainPatternDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl DomainPatternDetector {
    /// Create a new detector with default catalogs.
    pub fn new() -> Self {
        Self {
            rholang_catalog: RholangPatternCatalog::new(),
            metta_catalog: MettaPatternCatalog::new(),
        }
    }

    /// Get the Rholang catalog.
    pub fn rholang_catalog(&self) -> &RholangPatternCatalog {
        &self.rholang_catalog
    }

    /// Get the MeTTa catalog.
    pub fn metta_catalog(&self) -> &MettaPatternCatalog {
        &self.metta_catalog
    }

    /// Detect Rholang patterns in code.
    pub fn detect_rholang_patterns(&self, tokens: &[&str]) -> Vec<RholangPatternMatch> {
        let mut matches = Vec::new();

        for pattern in self.rholang_catalog.patterns() {
            for (pos, _) in tokens.iter().enumerate() {
                if let Some(matched_len) = self.try_match_pattern(tokens, pos, pattern.patterns.as_slice()) {
                    matches.push(RholangPatternMatch {
                        pattern_name: pattern.name.clone(),
                        category: pattern.category,
                        position: pos,
                        length: matched_len,
                        weight: pattern.weight,
                    });
                }
            }
        }

        matches
    }

    /// Detect MeTTa patterns in code.
    pub fn detect_metta_patterns(&self, tokens: &[&str]) -> Vec<MettaPatternMatch> {
        let mut matches = Vec::new();

        for pattern in self.metta_catalog.patterns() {
            for (pos, _) in tokens.iter().enumerate() {
                if let Some(matched_len) = self.try_match_pattern(tokens, pos, pattern.patterns.as_slice()) {
                    matches.push(MettaPatternMatch {
                        pattern_name: pattern.name.clone(),
                        category: pattern.category,
                        position: pos,
                        length: matched_len,
                        weight: pattern.weight,
                    });
                }
            }
        }

        matches
    }

    /// Try to match any pattern variant at the given position.
    fn try_match_pattern(&self, tokens: &[&str], pos: usize, variants: &[Vec<Arc<str>>]) -> Option<usize> {
        for variant in variants {
            if pos + variant.len() <= tokens.len() {
                let matches = variant.iter()
                    .enumerate()
                    .all(|(i, p)| tokens[pos + i].to_lowercase() == p.as_ref());

                if matches {
                    return Some(variant.len());
                }
            }
        }
        None
    }
}

/// A matched Rholang pattern.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde-extras", derive(Serialize, Deserialize))]
pub struct RholangPatternMatch {
    /// Name of the matched pattern.
    pub pattern_name: String,
    /// Category of the pattern.
    pub category: RholangPatternCategory,
    /// Position in the token stream.
    pub position: usize,
    /// Length of the match.
    pub length: usize,
    /// Weight of the match.
    pub weight: f64,
}

/// A matched MeTTa pattern.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde-extras", derive(Serialize, Deserialize))]
pub struct MettaPatternMatch {
    /// Name of the matched pattern.
    pub pattern_name: String,
    /// Category of the pattern.
    pub category: MettaPatternCategory,
    /// Position in the token stream.
    pub position: usize,
    /// Length of the match.
    pub length: usize,
    /// Weight of the match.
    pub weight: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rholang_catalog_creation() {
        let catalog = RholangPatternCatalog::new();
        assert!(!catalog.patterns().is_empty());
    }

    #[test]
    fn test_rholang_patterns_by_category() {
        let catalog = RholangPatternCatalog::new();
        let channel_patterns = catalog.by_category(RholangPatternCategory::ChannelOperation);
        assert!(!channel_patterns.is_empty());
    }

    #[test]
    fn test_metta_catalog_creation() {
        let catalog = MettaPatternCatalog::new();
        assert!(!catalog.patterns().is_empty());
    }

    #[test]
    fn test_metta_patterns_by_category() {
        let catalog = MettaPatternCatalog::new();
        let type_patterns = catalog.by_category(MettaPatternCategory::TypePattern);
        assert!(!type_patterns.is_empty());
    }

    #[test]
    fn test_domain_detector_creation() {
        let detector = DomainPatternDetector::new();
        assert!(!detector.rholang_catalog().patterns().is_empty());
        assert!(!detector.metta_catalog().patterns().is_empty());
    }

    #[test]
    fn test_detect_rholang_channel_send() {
        let detector = DomainPatternDetector::new();
        let tokens = vec!["channel", "!", "(", "data", ")"];
        let matches = detector.detect_rholang_patterns(&tokens);

        // Should detect send pattern
        let send_match = matches.iter()
            .find(|m| m.pattern_name == "channel_send");
        assert!(send_match.is_some());
    }

    #[test]
    fn test_detect_rholang_contract() {
        let detector = DomainPatternDetector::new();
        let tokens = vec!["contract", "foo", "(", "@", "arg", ",", "ret", ")"];
        let matches = detector.detect_rholang_patterns(&tokens);

        // Should detect contract pattern
        let contract_match = matches.iter()
            .find(|m| m.pattern_name == "contract_definition");
        assert!(contract_match.is_some());
    }

    #[test]
    fn test_detect_metta_function_def() {
        let detector = DomainPatternDetector::new();
        let tokens = vec!["(", "=", "(", "foo", "$x", ")", "(", "bar", "$x", ")", ")"];
        let matches = detector.detect_metta_patterns(&tokens);

        // Should detect function definition pattern
        let func_match = matches.iter()
            .find(|m| m.pattern_name == "function_definition");
        assert!(func_match.is_some());
    }

    #[test]
    fn test_detect_metta_type_decl() {
        let detector = DomainPatternDetector::new();
        let tokens = vec!["(", ":", "foo", "(", "->", "A", "B", ")", ")"];
        let matches = detector.detect_metta_patterns(&tokens);

        // Should detect type declaration pattern
        let type_match = matches.iter()
            .find(|m| m.pattern_name == "type_declaration" || m.pattern_name == "function_type");
        assert!(type_match.is_some());
    }

    #[test]
    fn test_detect_metta_variable() {
        let detector = DomainPatternDetector::new();
        let tokens = vec!["(", "foo", "$", "x", ")"];
        let matches = detector.detect_metta_patterns(&tokens);

        // Should detect variable pattern
        let var_match = matches.iter()
            .find(|m| m.pattern_name == "variable_atom");
        assert!(var_match.is_some());
    }

    #[test]
    fn test_rholang_pattern_categories() {
        assert_eq!(
            RholangPatternCategory::ProcessComposition.description(),
            "Process composition with | (parallel) or ; (sequential)"
        );
        assert_eq!(
            RholangPatternCategory::ChannelOperation.description(),
            "Channel send (!) and receive (for) operations"
        );
    }

    #[test]
    fn test_metta_pattern_categories() {
        assert_eq!(
            MettaPatternCategory::AtomDefinition.description(),
            "Atom definitions (Symbol, Expression, Variable)"
        );
        assert_eq!(
            MettaPatternCategory::PatternMatching.description(),
            "Pattern matching with match keyword"
        );
    }

    #[test]
    fn test_custom_pattern() {
        let mut catalog = RholangPatternCatalog::new();
        let initial_count = catalog.patterns().len();

        catalog.add_pattern(
            RholangPattern::new(
                "custom_pattern",
                RholangPatternCategory::StateManagement,
                "A custom state pattern",
            )
            .with_pattern(&["custom", "state"])
            .with_weight(0.6),
        );

        assert_eq!(catalog.patterns().len(), initial_count + 1);
    }
}
