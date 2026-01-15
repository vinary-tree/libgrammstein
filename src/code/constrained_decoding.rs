//! Grammar-constrained decoding for code generation.
//!
//! This module provides infrastructure for constraining neural model outputs
//! to syntactically valid code according to a formal grammar. It supports:
//!
//! - **Token masking**: Disabling invalid tokens during decoding
//! - **Earley parsing**: Incremental parsing for validity checking
//! - **PCFG integration**: Using weighted grammars for scoring
//!
//! The approach follows LLGUIDANCE (2025) and XGrammar for efficient
//! grammar-constrained generation.

use super::pcfg::{Symbol, WeightedCFG};
use std::collections::{HashMap, HashSet};

/// Configuration for grammar-constrained decoding.
#[derive(Debug, Clone)]
pub struct ConstrainedDecodingConfig {
    /// Maximum lookahead for token validity checking
    pub max_lookahead: usize,
    /// Whether to cache parse states
    pub cache_states: bool,
    /// Minimum probability for grammar rules
    pub min_rule_probability: f64,
    /// Whether to allow partial matches
    pub allow_partial: bool,
}

impl Default for ConstrainedDecodingConfig {
    fn default() -> Self {
        Self {
            max_lookahead: 3,
            cache_states: true,
            min_rule_probability: 1e-10,
            allow_partial: true,
        }
    }
}

/// State in the Earley parser.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EarleyState {
    /// Index of the rule in the grammar
    pub rule_idx: usize,
    /// Position in the RHS of the rule (dot position)
    pub dot_pos: usize,
    /// Starting position in the input
    pub start_pos: usize,
}

impl EarleyState {
    /// Creates a new Earley state.
    pub fn new(rule_idx: usize, dot_pos: usize, start_pos: usize) -> Self {
        Self {
            rule_idx,
            dot_pos,
            start_pos,
        }
    }

    /// Returns true if the state is complete (dot at end).
    pub fn is_complete(&self, rhs_len: usize) -> bool {
        self.dot_pos >= rhs_len
    }
}

/// Earley parser for incremental grammar checking.
pub struct EarleyParser {
    grammar: WeightedCFG,
    /// Index from non-terminal to rules that produce it
    rules_by_lhs: HashMap<String, Vec<usize>>,
    /// Rule definitions (lhs, rhs, weight)
    rules: Vec<(String, Vec<Symbol>, f64)>,
}

impl EarleyParser {
    /// Creates a new Earley parser from a weighted CFG.
    pub fn new(grammar: WeightedCFG) -> Self {
        let mut rules_by_lhs: HashMap<String, Vec<usize>> = HashMap::new();
        let mut rules = Vec::new();

        for (production, &weight) in grammar.iter_rules() {
            let rule_idx = rules.len();
            rules_by_lhs
                .entry(production.lhs.clone())
                .or_default()
                .push(rule_idx);
            rules.push((production.lhs.clone(), production.rhs.clone(), weight));
        }

        Self {
            grammar,
            rules_by_lhs,
            rules,
        }
    }

    /// Returns the start symbol.
    pub fn start_symbol(&self) -> &str {
        self.grammar.start_symbol()
    }

    /// Returns rules that produce the given non-terminal.
    pub fn rules_for(&self, nt: &str) -> impl Iterator<Item = usize> + '_ {
        self.rules_by_lhs
            .get(nt)
            .map(|v| v.iter().copied())
            .into_iter()
            .flatten()
    }

    /// Returns the rule at the given index.
    pub fn rule(&self, idx: usize) -> Option<(&str, &[Symbol], f64)> {
        self.rules
            .get(idx)
            .map(|(lhs, rhs, w)| (lhs.as_str(), rhs.as_slice(), *w))
    }
}

/// Chart entry for Earley parsing.
///
/// Uses Vec-based storage for each position to support efficient index-based
/// iteration without cloning states during chart completion.
#[derive(Debug, Clone, Default)]
pub struct EarleyChart {
    /// Ordered lists of states at each position (Vec for index-based access)
    states: Vec<Vec<EarleyState>>,
    /// Sets for O(1) membership checking (deduplication)
    seen: Vec<HashSet<EarleyState>>,
}

impl EarleyChart {
    /// Creates a new chart with the given number of positions.
    pub fn new(size: usize) -> Self {
        Self {
            states: vec![Vec::new(); size + 1],
            seen: vec![HashSet::new(); size + 1],
        }
    }

    /// Adds a state to the chart at the given position.
    /// Returns true if the state was newly added.
    pub fn add(&mut self, pos: usize, state: EarleyState) -> bool {
        if pos < self.states.len() {
            if self.seen[pos].insert(state.clone()) {
                self.states[pos].push(state);
                true
            } else {
                false
            }
        } else {
            false
        }
    }

    /// Returns states at the given position.
    pub fn states_at(&self, pos: usize) -> impl Iterator<Item = &EarleyState> {
        self.states
            .get(pos)
            .map(|s| s.iter())
            .into_iter()
            .flatten()
    }

    /// Returns the number of states at the given position.
    /// Used for index-based iteration without cloning.
    pub fn state_count_at(&self, pos: usize) -> usize {
        self.states.get(pos).map(|s| s.len()).unwrap_or(0)
    }

    /// Returns a reference to a state by index.
    /// Used for index-based iteration without cloning.
    pub fn state_at_index(&self, pos: usize, idx: usize) -> Option<&EarleyState> {
        self.states.get(pos).and_then(|s| s.get(idx))
    }

    /// Returns the number of positions.
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Returns true if the chart is empty.
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }
}

/// Grammar constraint checker for token validity.
pub struct GrammarConstraint {
    parser: EarleyParser,
    config: ConstrainedDecodingConfig,
    /// Current parse chart
    chart: EarleyChart,
    /// Current position in the input
    position: usize,
    /// Cached valid next tokens
    valid_tokens_cache: Option<HashSet<String>>,
}

impl GrammarConstraint {
    /// Creates a new grammar constraint checker.
    pub fn new(grammar: WeightedCFG, config: ConstrainedDecodingConfig) -> Self {
        Self {
            parser: EarleyParser::new(grammar),
            config,
            chart: EarleyChart::new(0),
            position: 0,
            valid_tokens_cache: None,
        }
    }

    /// Creates a constraint checker with default configuration.
    pub fn with_default_config(grammar: WeightedCFG) -> Self {
        Self::new(grammar, ConstrainedDecodingConfig::default())
    }

    /// Resets the parser state.
    pub fn reset(&mut self) {
        self.chart = EarleyChart::new(0);
        self.position = 0;
        self.valid_tokens_cache = None;
        self.initialize();
    }

    /// Initializes the parser with the start symbol.
    pub fn initialize(&mut self) {
        self.chart = EarleyChart::new(self.config.max_lookahead * 2);
        let start = self.parser.start_symbol().to_string();

        // Add initial states for all rules producing the start symbol
        for rule_idx in self.parser.rules_for(&start) {
            self.chart.add(0, EarleyState::new(rule_idx, 0, 0));
        }

        // Complete the initial chart
        self.complete_chart(0);
    }

    /// Completes the chart at the given position (prediction and completion).
    ///
    /// Uses index-based iteration to avoid cloning states. New states are
    /// appended to the Vec, and we iterate by index until we've processed
    /// all states (including newly added ones).
    fn complete_chart(&mut self, pos: usize) {
        // Index-based iteration: process states by index, new states are appended
        // This avoids cloning since we access by index and the Vec grows in place
        let mut state_idx = 0;

        while state_idx < self.chart.state_count_at(pos) {
            // Get state data by index (read-only reference)
            let (rule_idx, dot_pos, start_pos) = {
                let state = match self.chart.state_at_index(pos, state_idx) {
                    Some(s) => s,
                    None => {
                        state_idx += 1;
                        continue;
                    }
                };
                (state.rule_idx, state.dot_pos, state.start_pos)
            };

            if let Some((_, rhs, _)) = self.parser.rule(rule_idx) {
                if dot_pos < rhs.len() {
                    // Prediction: if dot is before a non-terminal
                    if let Symbol::NonTerminal(nt) = &rhs[dot_pos] {
                        for pred_rule_idx in self.parser.rules_for(nt) {
                            // Adding new states appends to the Vec; they'll be processed
                            // in subsequent iterations of this loop
                            self.chart.add(pos, EarleyState::new(pred_rule_idx, 0, pos));
                        }
                    }
                } else {
                    // Completion: if dot is at end
                    let (lhs, _, _) = self.parser.rule(rule_idx).expect("rule exists");
                    let completed_nt = lhs.to_string();

                    // Find states waiting for this non-terminal by index
                    let waiting_count = self.chart.state_count_at(start_pos);
                    for waiting_idx in 0..waiting_count {
                        let (w_rule_idx, w_dot_pos, w_start_pos) = {
                            let waiting_state = match self.chart.state_at_index(start_pos, waiting_idx) {
                                Some(s) => s,
                                None => continue,
                            };
                            (waiting_state.rule_idx, waiting_state.dot_pos, waiting_state.start_pos)
                        };

                        if let Some((_, w_rhs, _)) = self.parser.rule(w_rule_idx) {
                            if w_dot_pos < w_rhs.len() {
                                if let Symbol::NonTerminal(nt) = &w_rhs[w_dot_pos] {
                                    if *nt == completed_nt {
                                        let new_state = EarleyState::new(
                                            w_rule_idx,
                                            w_dot_pos + 1,
                                            w_start_pos,
                                        );
                                        self.chart.add(pos, new_state);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            state_idx += 1;
        }
    }

    /// Checks if a terminal token is valid at the current position.
    pub fn is_valid_token(&self, token: &str) -> bool {
        // Check cache first
        if let Some(valid) = &self.valid_tokens_cache {
            return valid.contains(token);
        }

        // Check if any state expects this terminal
        for state in self.chart.states_at(self.position) {
            if let Some((_, rhs, _)) = self.parser.rule(state.rule_idx) {
                if state.dot_pos < rhs.len() {
                    if let Symbol::Terminal(t) = &rhs[state.dot_pos] {
                        if t == token {
                            return true;
                        }
                    }
                }
            }
        }

        false
    }

    /// Returns all valid tokens at the current position.
    pub fn valid_tokens(&mut self) -> HashSet<String> {
        if let Some(cached) = &self.valid_tokens_cache {
            return cached.clone();
        }

        let mut valid = HashSet::new();

        for state in self.chart.states_at(self.position) {
            if let Some((_, rhs, _)) = self.parser.rule(state.rule_idx) {
                if state.dot_pos < rhs.len() {
                    if let Symbol::Terminal(t) = &rhs[state.dot_pos] {
                        valid.insert(t.clone());
                    }
                }
            }
        }

        if self.config.cache_states {
            self.valid_tokens_cache = Some(valid.clone());
        }

        valid
    }

    /// Advances the parser state by scanning a terminal.
    ///
    /// Uses index-based iteration to avoid cloning states.
    pub fn advance(&mut self, token: &str) -> bool {
        if !self.is_valid_token(token) {
            return false;
        }

        let current_pos = self.position;
        let next_pos = current_pos + 1;

        // Ensure chart has space
        if next_pos >= self.chart.len() {
            return false;
        }

        // Scan: advance states that expect this terminal using index-based iteration
        let state_count = self.chart.state_count_at(current_pos);
        for state_idx in 0..state_count {
            let (rule_idx, dot_pos, start_pos) = {
                let state = match self.chart.state_at_index(current_pos, state_idx) {
                    Some(s) => s,
                    None => continue,
                };
                (state.rule_idx, state.dot_pos, state.start_pos)
            };

            if let Some((_, rhs, _)) = self.parser.rule(rule_idx) {
                if dot_pos < rhs.len() {
                    if let Symbol::Terminal(t) = &rhs[dot_pos] {
                        if t == token {
                            let new_state = EarleyState::new(
                                rule_idx,
                                dot_pos + 1,
                                start_pos,
                            );
                            self.chart.add(next_pos, new_state);
                        }
                    }
                }
            }
        }

        self.position = next_pos;
        self.valid_tokens_cache = None;
        self.complete_chart(next_pos);

        true
    }

    /// Checks if the parse can be completed from the current state.
    pub fn can_complete(&self) -> bool {
        // Check if any state at the current position is complete
        // and started at position 0 with the start symbol
        let start = self.parser.start_symbol();

        for state in self.chart.states_at(self.position) {
            if let Some((lhs, rhs, _)) = self.parser.rule(state.rule_idx) {
                if lhs == start && state.start_pos == 0 && state.dot_pos >= rhs.len() {
                    return true;
                }
            }
        }

        false
    }

    /// Returns the current position.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Returns the grammar.
    pub fn grammar(&self) -> &WeightedCFG {
        &self.parser.grammar
    }
}

/// Token mask for constrained decoding.
///
/// A mask indicating which tokens are valid at a given position.
#[derive(Debug, Clone)]
pub struct TokenMask {
    /// Token indices that are allowed
    allowed: HashSet<usize>,
    /// Total vocabulary size
    vocab_size: usize,
}

impl TokenMask {
    /// Creates a mask allowing all tokens.
    pub fn allow_all(vocab_size: usize) -> Self {
        Self {
            allowed: (0..vocab_size).collect(),
            vocab_size,
        }
    }

    /// Creates a mask allowing specific tokens.
    pub fn from_allowed(allowed: HashSet<usize>, vocab_size: usize) -> Self {
        Self { allowed, vocab_size }
    }

    /// Checks if a token index is allowed.
    pub fn is_allowed(&self, idx: usize) -> bool {
        self.allowed.contains(&idx)
    }

    /// Returns allowed token indices.
    pub fn allowed_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.allowed.iter().copied()
    }

    /// Returns the number of allowed tokens.
    pub fn count_allowed(&self) -> usize {
        self.allowed.len()
    }

    /// Converts to a boolean vector.
    pub fn to_bool_vec(&self) -> Vec<bool> {
        let mut v = vec![false; self.vocab_size];
        for &idx in &self.allowed {
            if idx < self.vocab_size {
                v[idx] = true;
            }
        }
        v
    }

    /// Applies the mask to logits (sets disallowed to negative infinity).
    pub fn apply_to_logits(&self, logits: &mut [f32]) {
        for (i, logit) in logits.iter_mut().enumerate() {
            if !self.allowed.contains(&i) {
                *logit = f32::NEG_INFINITY;
            }
        }
    }
}

/// Vocabulary mapping for constrained decoding.
pub struct DecodingVocabulary {
    token_to_idx: HashMap<String, usize>,
    idx_to_token: Vec<String>,
}

impl DecodingVocabulary {
    /// Creates a new vocabulary.
    pub fn new() -> Self {
        Self {
            token_to_idx: HashMap::new(),
            idx_to_token: Vec::new(),
        }
    }

    /// Adds a token to the vocabulary.
    pub fn add_token(&mut self, token: impl Into<String>) -> usize {
        let token = token.into();
        if let Some(&idx) = self.token_to_idx.get(&token) {
            return idx;
        }
        let idx = self.idx_to_token.len();
        self.idx_to_token.push(token.clone());
        self.token_to_idx.insert(token, idx);
        idx
    }

    /// Gets the index for a token.
    pub fn get_idx(&self, token: &str) -> Option<usize> {
        self.token_to_idx.get(token).copied()
    }

    /// Gets the token for an index.
    pub fn get_token(&self, idx: usize) -> Option<&str> {
        self.idx_to_token.get(idx).map(|s| s.as_str())
    }

    /// Returns the vocabulary size.
    pub fn len(&self) -> usize {
        self.idx_to_token.len()
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.idx_to_token.is_empty()
    }

    /// Creates a token mask from valid tokens.
    pub fn create_mask(&self, valid_tokens: &HashSet<String>) -> TokenMask {
        let allowed: HashSet<usize> = valid_tokens
            .iter()
            .filter_map(|t| self.get_idx(t))
            .collect();
        TokenMask::from_allowed(allowed, self.len())
    }
}

impl Default for DecodingVocabulary {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code::pcfg::Production;

    fn create_simple_grammar() -> WeightedCFG {
        let mut cfg = WeightedCFG::new("S");

        // S -> A B
        cfg.add_rule(
            Production::new(
                "S",
                vec![
                    Symbol::NonTerminal("A".to_string()),
                    Symbol::NonTerminal("B".to_string()),
                ],
            ),
            1.0,
        );

        // A -> "a"
        cfg.add_rule(
            Production::new("A", vec![Symbol::Terminal("a".to_string())]),
            1.0,
        );

        // B -> "b"
        cfg.add_rule(
            Production::new("B", vec![Symbol::Terminal("b".to_string())]),
            1.0,
        );

        cfg
    }

    #[test]
    fn test_earley_parser_creation() {
        let grammar = create_simple_grammar();
        let parser = EarleyParser::new(grammar);

        assert_eq!(parser.start_symbol(), "S");
        assert_eq!(parser.rules_for("S").count(), 1);
        assert_eq!(parser.rules_for("A").count(), 1);
        assert_eq!(parser.rules_for("B").count(), 1);
    }

    #[test]
    fn test_grammar_constraint_basic() {
        let grammar = create_simple_grammar();
        let mut constraint = GrammarConstraint::with_default_config(grammar);
        constraint.reset();

        // At position 0, "a" should be valid
        let valid = constraint.valid_tokens();
        assert!(valid.contains("a"));
        assert!(!valid.contains("b"));

        // Advance with "a"
        assert!(constraint.advance("a"));

        // Now "b" should be valid
        let valid = constraint.valid_tokens();
        assert!(valid.contains("b"));
        assert!(!valid.contains("a"));

        // Advance with "b"
        assert!(constraint.advance("b"));

        // Parse should be complete
        assert!(constraint.can_complete());
    }

    #[test]
    fn test_token_mask() {
        let mut allowed = HashSet::new();
        allowed.insert(1);
        allowed.insert(3);
        allowed.insert(5);

        let mask = TokenMask::from_allowed(allowed, 10);

        assert!(mask.is_allowed(1));
        assert!(mask.is_allowed(3));
        assert!(mask.is_allowed(5));
        assert!(!mask.is_allowed(0));
        assert!(!mask.is_allowed(2));

        let bool_vec = mask.to_bool_vec();
        assert_eq!(bool_vec.len(), 10);
        assert!(bool_vec[1]);
        assert!(!bool_vec[0]);
    }

    #[test]
    fn test_decoding_vocabulary() {
        let mut vocab = DecodingVocabulary::new();
        let idx_a = vocab.add_token("a");
        let idx_b = vocab.add_token("b");
        let idx_a2 = vocab.add_token("a"); // Duplicate

        assert_eq!(idx_a, idx_a2);
        assert_ne!(idx_a, idx_b);
        assert_eq!(vocab.get_token(idx_a), Some("a"));
        assert_eq!(vocab.get_idx("b"), Some(idx_b));
        assert_eq!(vocab.len(), 2);
    }
}
