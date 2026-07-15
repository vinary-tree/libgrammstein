//! WFST export for probabilistic context-free grammars (PCFGs).
//!
//! Since CFGs are strictly more expressive than finite-state automata,
//! this module provides several approximation strategies:
//!
//! 1. **Finite-depth unrolling**: Unroll the grammar to a fixed depth
//! 2. **Regular approximation**: Approximate with a regular grammar
//! 3. **Local scoring**: Use local rule probabilities for scoring
//!
//! For exact grammar constraints, use the grammar-constrained decoding
//! module which uses a pushdown automaton.

#[cfg(feature = "lling-llang-integration")]
use lling_llang::semiring::Semiring;
#[cfg(feature = "lling-llang-integration")]
use lling_llang::wfst::{MutableWfst, StateId, VectorWfst, Wfst};

use super::pcfg::{Production, Symbol, WeightedCFG};
use crate::integration::wfst_export::FromLogProb;
use std::collections::HashMap;

/// Configuration for PCFG to WFST export.
#[derive(Debug, Clone)]
pub struct PcfgWfstConfig {
    /// Maximum depth to unroll the grammar
    pub max_depth: usize,
    /// Minimum probability threshold for rules
    pub min_probability: f64,
    /// Whether to include backoff transitions
    pub include_backoff: bool,
    /// Maximum number of states to create
    pub max_states: usize,
}

impl Default for PcfgWfstConfig {
    fn default() -> Self {
        Self {
            max_depth: 5,
            min_probability: 1e-10,
            include_backoff: true,
            max_states: 100_000,
        }
    }
}

/// Symbol ID for WFST labels.
pub type SymbolId = u32;

/// Vocabulary mapping symbols to IDs.
#[derive(Debug, Clone)]
pub struct SymbolVocabulary {
    symbol_to_id: HashMap<String, SymbolId>,
    id_to_symbol: Vec<String>,
}

impl SymbolVocabulary {
    /// Creates a new empty vocabulary.
    pub fn new() -> Self {
        let mut vocab = Self {
            symbol_to_id: HashMap::new(),
            id_to_symbol: Vec::new(),
        };
        // Reserve ID 0 for epsilon
        vocab.add_symbol("<eps>");
        vocab
    }

    /// Adds a symbol to the vocabulary.
    pub fn add_symbol(&mut self, symbol: &str) -> SymbolId {
        if let Some(&id) = self.symbol_to_id.get(symbol) {
            return id;
        }
        let id = self.id_to_symbol.len() as SymbolId;
        self.id_to_symbol.push(symbol.to_string());
        self.symbol_to_id.insert(symbol.to_string(), id);
        id
    }

    /// Gets the ID for a symbol.
    pub fn get_id(&self, symbol: &str) -> Option<SymbolId> {
        self.symbol_to_id.get(symbol).copied()
    }

    /// Gets the symbol for an ID.
    pub fn get_symbol(&self, id: SymbolId) -> Option<&str> {
        self.id_to_symbol.get(id as usize).map(|s| s.as_str())
    }

    /// Returns the number of symbols.
    pub fn len(&self) -> usize {
        self.id_to_symbol.len()
    }

    /// Returns true if the vocabulary is empty.
    pub fn is_empty(&self) -> bool {
        self.id_to_symbol.is_empty()
    }

    /// Iterates over all symbols.
    pub fn iter(&self) -> impl Iterator<Item = (&str, SymbolId)> {
        self.id_to_symbol
            .iter()
            .enumerate()
            .map(|(id, sym)| (sym.as_str(), id as SymbolId))
    }
}

impl Default for SymbolVocabulary {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating WFST from PCFG.
///
/// This creates a finite-state approximation of the PCFG by unrolling
/// to a fixed depth. The resulting WFST can be used for local scoring
/// in conjunction with exact grammar-constrained decoding.
#[cfg(feature = "lling-llang-integration")]
pub struct PcfgWfstBuilder<W: Semiring + FromLogProb> {
    grammar: WeightedCFG,
    config: PcfgWfstConfig,
    vocabulary: SymbolVocabulary,
    wfst: VectorWfst<SymbolId, W>,
    /// Maps (non-terminal, depth) to state ID
    state_map: HashMap<(String, usize), StateId>,
}

#[cfg(feature = "lling-llang-integration")]
impl<W: Semiring + FromLogProb> PcfgWfstBuilder<W> {
    /// Creates a new PCFG WFST builder.
    pub fn new(grammar: WeightedCFG, config: PcfgWfstConfig) -> Self {
        let mut vocabulary = SymbolVocabulary::new();

        // Add all terminals to vocabulary
        for terminal in grammar.terminals() {
            vocabulary.add_symbol(terminal);
        }

        // Add all non-terminals to vocabulary (for debugging/tracing)
        for nt in grammar.non_terminals() {
            vocabulary.add_symbol(nt);
        }

        let wfst = VectorWfst::new();

        Self {
            grammar,
            config,
            vocabulary,
            wfst,
            state_map: HashMap::new(),
        }
    }

    /// Builds the WFST from the grammar.
    pub fn build(mut self) -> (VectorWfst<SymbolId, W>, SymbolVocabulary) {
        // Create start state
        let start = self.wfst.add_state();
        self.wfst.set_start(start);
        self.wfst.set_final(start, W::one());

        // Get start symbol
        let start_symbol = self.grammar.start_symbol().to_string();

        // Unroll from start symbol
        self.unroll_symbol(&start_symbol, start, 0);

        (self.wfst, self.vocabulary)
    }

    /// Recursively unrolls a non-terminal symbol.
    fn unroll_symbol(&mut self, symbol: &str, from_state: StateId, depth: usize) {
        if depth >= self.config.max_depth {
            return;
        }

        if self.wfst.num_states() >= self.config.max_states {
            return;
        }

        // Get all rules for this non-terminal and collect them to avoid borrow conflict
        let rules: Vec<_> = self
            .grammar
            .rules_for(symbol)
            .into_iter()
            .filter(|(_, prob)| *prob >= self.config.min_probability)
            .map(|(production, prob)| (production.clone(), prob))
            .collect();

        for (production, prob) in rules {
            let weight = W::from_log_prob(prob.ln());

            // Process the RHS of the production
            self.unroll_production(&production, from_state, weight, depth);
        }
    }

    /// Unrolls a single production rule.
    fn unroll_production(
        &mut self,
        production: &Production,
        from_state: StateId,
        weight: W,
        depth: usize,
    ) {
        if production.rhs.is_empty() {
            // Epsilon production - just add weight to final
            return;
        }

        let mut current_state = from_state;
        let rhs_len = production.rhs.len();

        for (i, symbol) in production.rhs.iter().enumerate() {
            let is_last = i == rhs_len - 1;

            match symbol {
                Symbol::Terminal(term) => {
                    // Create transition for terminal
                    let term_id = self.vocabulary.get_id(term).unwrap_or(0);

                    let next_state = if is_last {
                        // Last symbol - can go to any final state
                        let state = self.wfst.add_state();
                        self.wfst.set_final(state, W::one());
                        state
                    } else {
                        self.wfst.add_state()
                    };

                    let arc_weight = if i == 0 { weight } else { W::one() };

                    self.wfst.add_arc(
                        current_state,
                        Some(term_id),
                        Some(term_id),
                        next_state,
                        arc_weight,
                    );

                    current_state = next_state;
                }
                Symbol::NonTerminal(nt) => {
                    // Recursively unroll non-terminal
                    let state_key = (nt.clone(), depth + 1);

                    let next_state = if let Some(&state) = self.state_map.get(&state_key) {
                        state
                    } else {
                        let state = self.wfst.add_state();
                        self.wfst.set_final(state, W::one());
                        self.state_map.insert(state_key, state);
                        state
                    };

                    // Add epsilon transition to non-terminal state
                    let arc_weight = if i == 0 { weight } else { W::one() };
                    self.wfst.add_epsilon(current_state, next_state, arc_weight);

                    // Recursively unroll
                    self.unroll_symbol(nt, next_state, depth + 1);

                    current_state = next_state;
                }
            }
        }
    }
}

/// Extension trait for WeightedCFG to provide WFST export.
#[cfg(feature = "lling-llang-integration")]
pub trait PcfgWfstExport {
    /// Export the PCFG as a finite-state approximation.
    ///
    /// # Type Parameters
    ///
    /// * `W` - The semiring weight type
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration for the export
    fn to_wfst<W>(&self, config: PcfgWfstConfig) -> (VectorWfst<SymbolId, W>, SymbolVocabulary)
    where
        W: Semiring + FromLogProb;

    /// Export with default configuration.
    fn to_wfst_default<W>(&self) -> (VectorWfst<SymbolId, W>, SymbolVocabulary)
    where
        W: Semiring + FromLogProb,
    {
        self.to_wfst(PcfgWfstConfig::default())
    }
}

#[cfg(feature = "lling-llang-integration")]
impl PcfgWfstExport for WeightedCFG {
    fn to_wfst<W>(&self, config: PcfgWfstConfig) -> (VectorWfst<SymbolId, W>, SymbolVocabulary)
    where
        W: Semiring + FromLogProb,
    {
        let builder = PcfgWfstBuilder::new(self.clone(), config);
        builder.build()
    }
}

/// Local scoring using PCFG rule probabilities.
///
/// This provides a simpler interface for scoring sequences using
/// PCFG rule probabilities without building a full WFST.
pub struct PcfgScorer {
    grammar: WeightedCFG,
}

impl PcfgScorer {
    /// Creates a new PCFG scorer.
    pub fn new(grammar: WeightedCFG) -> Self {
        Self { grammar }
    }

    /// Scores a production rule.
    pub fn score_rule(&self, production: &Production) -> f64 {
        self.grammar.log_probability(production)
    }

    /// Scores a sequence of rules (parse).
    pub fn score_parse(&self, productions: &[Production]) -> f64 {
        productions.iter().map(|p| self.score_rule(p)).sum()
    }

    /// Returns the probability of a terminal given a non-terminal.
    pub fn terminal_probability(&self, non_terminal: &str, terminal: &str) -> f64 {
        let rules = self.grammar.rules_for(non_terminal);

        for (production, _) in rules {
            if production.rhs.len() == 1 {
                if let Symbol::Terminal(t) = &production.rhs[0] {
                    if t == terminal {
                        return self.grammar.probability(production);
                    }
                }
            }
        }

        0.0
    }

    /// Returns the grammar.
    pub fn grammar(&self) -> &WeightedCFG {
        &self.grammar
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_grammar() -> WeightedCFG {
        let mut cfg = WeightedCFG::new("S");

        // S -> NP VP (weight 1.0)
        cfg.add_rule(
            Production::new(
                "S",
                vec![
                    Symbol::NonTerminal("NP".to_string()),
                    Symbol::NonTerminal("VP".to_string()),
                ],
            ),
            1.0,
        );

        // NP -> Det N (weight 0.6)
        cfg.add_rule(
            Production::new(
                "NP",
                vec![
                    Symbol::NonTerminal("Det".to_string()),
                    Symbol::NonTerminal("N".to_string()),
                ],
            ),
            0.6,
        );

        // NP -> N (weight 0.4)
        cfg.add_rule(
            Production::new("NP", vec![Symbol::NonTerminal("N".to_string())]),
            0.4,
        );

        // VP -> V NP (weight 0.7)
        cfg.add_rule(
            Production::new(
                "VP",
                vec![
                    Symbol::NonTerminal("V".to_string()),
                    Symbol::NonTerminal("NP".to_string()),
                ],
            ),
            0.7,
        );

        // VP -> V (weight 0.3)
        cfg.add_rule(
            Production::new("VP", vec![Symbol::NonTerminal("V".to_string())]),
            0.3,
        );

        // Det -> "the" (weight 0.6)
        cfg.add_rule(
            Production::new("Det", vec![Symbol::Terminal("the".to_string())]),
            0.6,
        );

        // Det -> "a" (weight 0.4)
        cfg.add_rule(
            Production::new("Det", vec![Symbol::Terminal("a".to_string())]),
            0.4,
        );

        // N -> "cat" (weight 0.5)
        cfg.add_rule(
            Production::new("N", vec![Symbol::Terminal("cat".to_string())]),
            0.5,
        );

        // N -> "dog" (weight 0.5)
        cfg.add_rule(
            Production::new("N", vec![Symbol::Terminal("dog".to_string())]),
            0.5,
        );

        // V -> "runs" (weight 0.5)
        cfg.add_rule(
            Production::new("V", vec![Symbol::Terminal("runs".to_string())]),
            0.5,
        );

        // V -> "sees" (weight 0.5)
        cfg.add_rule(
            Production::new("V", vec![Symbol::Terminal("sees".to_string())]),
            0.5,
        );

        cfg
    }

    #[test]
    fn test_symbol_vocabulary() {
        let mut vocab = SymbolVocabulary::new();
        let id1 = vocab.add_symbol("hello");
        let id2 = vocab.add_symbol("world");
        let id3 = vocab.add_symbol("hello"); // Duplicate

        assert_eq!(id1, id3);
        assert_ne!(id1, id2);
        assert_eq!(vocab.get_symbol(id1), Some("hello"));
        assert_eq!(vocab.get_id("world"), Some(id2));
    }

    #[test]
    fn test_pcfg_scorer() {
        let grammar = create_test_grammar();
        let scorer = PcfgScorer::new(grammar);

        // Test terminal probability
        let prob = scorer.terminal_probability("Det", "the");
        assert!((prob - 0.6).abs() < 1e-6);

        let prob = scorer.terminal_probability("N", "cat");
        assert!((prob - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_pcfg_scorer_parse() {
        let grammar = create_test_grammar();
        let scorer = PcfgScorer::new(grammar);

        let parse = vec![
            Production::new(
                "S",
                vec![
                    Symbol::NonTerminal("NP".to_string()),
                    Symbol::NonTerminal("VP".to_string()),
                ],
            ),
            Production::new("NP", vec![Symbol::NonTerminal("N".to_string())]),
            Production::new("N", vec![Symbol::Terminal("cat".to_string())]),
        ];

        let score = scorer.score_parse(&parse);
        // Score should be sum of log probabilities
        assert!(score < 0.0); // Log probs are negative
    }
}
