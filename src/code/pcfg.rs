//! Probabilistic Context-Free Grammar (PCFG) for programming languages.
//!
//! This module provides:
//! - PCFG training from parsed code corpora
//! - Grammar rule probability estimation
//! - WFST export for integration with lling-llang
//!
//! Since programming languages have known formal grammars, PCFGs can be used
//! to constrain correction candidates to syntactically valid outputs.

use super::ast::{AstNode, ParsedCode};
use super::language::CodeLanguage;
use std::collections::HashMap;
use std::hash::Hash;

/// A production rule in the grammar.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Production {
    /// Left-hand side (non-terminal)
    pub lhs: String,
    /// Right-hand side (sequence of symbols)
    pub rhs: Vec<Symbol>,
}

impl Production {
    /// Creates a new production rule.
    pub fn new(lhs: impl Into<String>, rhs: Vec<Symbol>) -> Self {
        Self {
            lhs: lhs.into(),
            rhs,
        }
    }

    /// Returns true if this is an epsilon (empty) production.
    pub fn is_epsilon(&self) -> bool {
        self.rhs.is_empty()
    }

    /// Returns the arity (number of RHS symbols) of this production.
    pub fn arity(&self) -> usize {
        self.rhs.len()
    }
}

impl std::fmt::Display for Production {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ->", self.lhs)?;
        for sym in &self.rhs {
            write!(f, " {}", sym)?;
        }
        Ok(())
    }
}

/// A symbol in the grammar (terminal or non-terminal).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Symbol {
    /// Non-terminal symbol (e.g., "expression", "statement")
    NonTerminal(String),
    /// Terminal symbol (actual token, e.g., "if", "+", identifier)
    Terminal(String),
}

impl Symbol {
    /// Creates a non-terminal symbol.
    pub fn non_terminal(s: impl Into<String>) -> Self {
        Symbol::NonTerminal(s.into())
    }

    /// Creates a terminal symbol.
    pub fn terminal(s: impl Into<String>) -> Self {
        Symbol::Terminal(s.into())
    }

    /// Returns true if this is a non-terminal.
    pub fn is_non_terminal(&self) -> bool {
        matches!(self, Symbol::NonTerminal(_))
    }

    /// Returns true if this is a terminal.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Symbol::Terminal(_))
    }

    /// Returns the symbol name.
    pub fn name(&self) -> &str {
        match self {
            Symbol::NonTerminal(s) | Symbol::Terminal(s) => s,
        }
    }
}

impl std::fmt::Display for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Symbol::NonTerminal(s) => write!(f, "<{}>", s),
            Symbol::Terminal(s) => write!(f, "'{}'", s),
        }
    }
}

/// A weighted context-free grammar.
///
/// Each production rule has an associated weight (unnormalized) or probability.
/// Weights can be converted to probabilities by normalizing over all rules
/// with the same LHS.
#[derive(Debug, Clone)]
pub struct WeightedCFG {
    /// Production rules with their weights
    rules: HashMap<Production, f64>,
    /// Rules indexed by LHS for efficient lookup
    rules_by_lhs: HashMap<String, Vec<(Production, f64)>>,
    /// Start symbol
    start_symbol: String,
    /// Total weight for each non-terminal (for normalization)
    lhs_totals: HashMap<String, f64>,
}

impl WeightedCFG {
    /// Creates a new weighted CFG with the given start symbol.
    pub fn new(start_symbol: impl Into<String>) -> Self {
        Self {
            rules: HashMap::new(),
            rules_by_lhs: HashMap::new(),
            start_symbol: start_symbol.into(),
            lhs_totals: HashMap::new(),
        }
    }

    /// Adds a production rule with the given weight.
    pub fn add_rule(&mut self, production: Production, weight: f64) {
        let lhs = production.lhs.clone();

        // Update total weight for this LHS
        *self.lhs_totals.entry(lhs.clone()).or_insert(0.0) += weight;

        // Store the rule
        *self.rules.entry(production.clone()).or_insert(0.0) += weight;

        // Index by LHS
        self.rules_by_lhs
            .entry(lhs)
            .or_default()
            .push((production, weight));
    }

    /// Returns the weight of a production rule.
    pub fn weight(&self, production: &Production) -> f64 {
        self.rules.get(production).copied().unwrap_or(0.0)
    }

    /// Returns the probability of a production rule (normalized).
    pub fn probability(&self, production: &Production) -> f64 {
        let weight = self.weight(production);
        let total = self.lhs_totals.get(&production.lhs).copied().unwrap_or(1.0);
        if total > 0.0 {
            weight / total
        } else {
            0.0
        }
    }

    /// Returns the log probability of a production rule.
    pub fn log_probability(&self, production: &Production) -> f64 {
        let prob = self.probability(production);
        if prob > 0.0 {
            prob.ln()
        } else {
            f64::NEG_INFINITY
        }
    }

    /// Returns all rules with the given LHS.
    pub fn rules_for(&self, lhs: &str) -> Vec<(&Production, f64)> {
        self.rules_by_lhs
            .get(lhs)
            .map(|rules| rules.iter().map(|(p, w)| (p, *w)).collect())
            .unwrap_or_default()
    }

    /// Returns the start symbol.
    pub fn start_symbol(&self) -> &str {
        &self.start_symbol
    }

    /// Returns all non-terminals in the grammar.
    pub fn non_terminals(&self) -> impl Iterator<Item = &str> {
        self.rules_by_lhs.keys().map(|s| s.as_str())
    }

    /// Returns all terminals in the grammar.
    pub fn terminals(&self) -> impl Iterator<Item = &str> {
        self.rules
            .keys()
            .flat_map(|p| p.rhs.iter())
            .filter_map(|s| match s {
                Symbol::Terminal(t) => Some(t.as_str()),
                _ => None,
            })
    }

    /// Returns the number of production rules.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Iterates over all production rules and their weights.
    pub fn iter_rules(&self) -> impl Iterator<Item = (&Production, &f64)> {
        self.rules.iter()
    }

    /// Returns all production rules.
    pub fn rules(&self) -> &HashMap<Production, f64> {
        &self.rules
    }

    /// Normalizes all weights to probabilities.
    pub fn normalize(&mut self) {
        let mut normalized_rules = HashMap::new();

        for (production, weight) in &self.rules {
            let total = self.lhs_totals.get(&production.lhs).copied().unwrap_or(1.0);
            let prob = if total > 0.0 { weight / total } else { 0.0 };
            normalized_rules.insert(production.clone(), prob);
        }

        self.rules = normalized_rules;

        // Reset totals to 1.0 for all LHS
        for total in self.lhs_totals.values_mut() {
            *total = 1.0;
        }

        // Rebuild rules_by_lhs
        self.rules_by_lhs.clear();
        for (production, weight) in &self.rules {
            self.rules_by_lhs
                .entry(production.lhs.clone())
                .or_default()
                .push((production.clone(), *weight));
        }
    }
}

/// Trainer for building PCFGs from parsed code corpora.
pub struct PcfgTrainer<'a, L: CodeLanguage> {
    language: &'a L,
    rule_counts: HashMap<Production, u64>,
    start_symbol: String,
}

impl<'a, L: CodeLanguage> PcfgTrainer<'a, L> {
    /// Creates a new PCFG trainer for the given language.
    pub fn new(language: &'a L) -> Self {
        Self {
            language,
            rule_counts: HashMap::new(),
            start_symbol: "source_file".to_string(),
        }
    }

    /// Sets the start symbol for the grammar.
    pub fn with_start_symbol(mut self, symbol: impl Into<String>) -> Self {
        self.start_symbol = symbol.into();
        self
    }

    /// Returns the language used to parse training inputs.
    pub fn language(&self) -> &L {
        self.language
    }

    /// Trains the PCFG from a single parsed file.
    pub fn train_from_parsed(&mut self, parsed: &ParsedCode) {
        let ast = AstNode::from_ts_node(parsed.root(), &parsed.source);
        self.extract_rules(&ast);
    }

    /// Trains the PCFG from multiple parsed files.
    pub fn train_from_parsed_iter<'b, I>(&mut self, parsed_iter: I)
    where
        I: Iterator<Item = &'b ParsedCode>,
    {
        for parsed in parsed_iter {
            self.train_from_parsed(parsed);
        }
    }

    /// Extracts production rules from an AST node recursively.
    fn extract_rules(&mut self, node: &AstNode) {
        // Skip error nodes
        if node.is_error || node.is_missing {
            return;
        }

        // Only create rules for named nodes (non-terminals)
        if node.is_named && !node.children.is_empty() {
            let lhs = node.kind.clone();
            let rhs: Vec<Symbol> = node
                .children
                .iter()
                .filter(|c| c.is_named) // Only named children
                .map(|c| {
                    if c.children.is_empty() && c.text.is_some() {
                        // Leaf node with text -> terminal
                        Symbol::Terminal(c.kind.clone())
                    } else {
                        // Non-leaf -> non-terminal
                        Symbol::NonTerminal(c.kind.clone())
                    }
                })
                .collect();

            if !rhs.is_empty() {
                let production = Production::new(lhs, rhs);
                *self.rule_counts.entry(production).or_insert(0) += 1;
            }
        }

        // Recurse into children
        for child in &node.children {
            self.extract_rules(child);
        }
    }

    /// Converts accumulated counts to a weighted CFG.
    pub fn to_weighted_cfg(&self) -> WeightedCFG {
        let mut cfg = WeightedCFG::new(self.start_symbol.clone());

        for (production, count) in &self.rule_counts {
            cfg.add_rule(production.clone(), *count as f64);
        }

        cfg
    }

    /// Returns the rule counts for inspection.
    pub fn rule_counts(&self) -> &HashMap<Production, u64> {
        &self.rule_counts
    }

    /// Returns the number of unique rules observed.
    pub fn unique_rule_count(&self) -> usize {
        self.rule_counts.len()
    }

    /// Returns the total number of rule instances observed.
    pub fn total_rule_count(&self) -> u64 {
        self.rule_counts.values().sum()
    }

    /// Clears all accumulated counts.
    pub fn clear(&mut self) {
        self.rule_counts.clear();
    }
}

/// Configuration for WFST export of a PCFG.
#[derive(Debug, Clone)]
pub struct PcfgWfstConfig {
    /// Whether to include epsilon transitions for optional rules
    pub include_epsilon: bool,
    /// Minimum probability threshold for rules (rules below this are excluded)
    pub min_probability: f64,
    /// Maximum number of rules to include
    pub max_rules: Option<usize>,
}

impl Default for PcfgWfstConfig {
    fn default() -> Self {
        Self {
            include_epsilon: true,
            min_probability: 1e-10,
            max_rules: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_production_display() {
        let prod = Production::new(
            "expr",
            vec![
                Symbol::NonTerminal("term".to_string()),
                Symbol::Terminal("+".to_string()),
                Symbol::NonTerminal("expr".to_string()),
            ],
        );

        assert_eq!(format!("{}", prod), "expr -> <term> '+' <expr>");
    }

    #[test]
    fn test_weighted_cfg_probability() {
        let mut cfg = WeightedCFG::new("S");

        // Add rules: S -> A (weight 3), S -> B (weight 1)
        cfg.add_rule(
            Production::new("S", vec![Symbol::NonTerminal("A".to_string())]),
            3.0,
        );
        cfg.add_rule(
            Production::new("S", vec![Symbol::NonTerminal("B".to_string())]),
            1.0,
        );

        let prob_a = cfg.probability(&Production::new(
            "S",
            vec![Symbol::NonTerminal("A".to_string())],
        ));
        let prob_b = cfg.probability(&Production::new(
            "S",
            vec![Symbol::NonTerminal("B".to_string())],
        ));

        assert!((prob_a - 0.75).abs() < 1e-6);
        assert!((prob_b - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_symbol_types() {
        let nt = Symbol::non_terminal("expr");
        let t = Symbol::terminal("+");

        assert!(nt.is_non_terminal());
        assert!(!nt.is_terminal());
        assert!(!t.is_non_terminal());
        assert!(t.is_terminal());
        assert_eq!(nt.name(), "expr");
        assert_eq!(t.name(), "+");
    }
}
