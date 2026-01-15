# WFST Export for PCFGs

Export Probabilistic Context-Free Grammars as Weighted Finite-State Transducers for integration with lling-llang.

## Overview

Since CFGs are strictly more expressive than finite-state automata, this module provides approximation strategies:

- **Finite-depth unrolling**: Unroll grammar to fixed depth
- **Regular approximation**: Approximate with regular grammar
- **Local scoring**: Use rule probabilities for scoring

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     PcfgWfstBuilder                              │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    WeightedCFG                              │ │
│  │                                                             │ │
│  │  S -> NP VP [0.8]                                          │ │
│  │  NP -> Det N [0.6]                                         │ │
│  │  VP -> V NP [0.7]                                          │ │
│  │  ...                                                       │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼ Unroll to depth                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                   VectorWfst<W>                             │ │
│  │                                                             │ │
│  │    ┌───┐  "the"  ┌───┐  "cat"  ┌───┐  "runs"  ┌───┐       │ │
│  │    │ 0 │────────►│ 1 │────────►│ 2 │─────────►│ 3 │       │ │
│  │    └───┘         └───┘         └───┘          └───┘       │ │
│  │      │             │             │              │          │ │
│  │      └─────────────┴─────────────┴──────────────┘          │ │
│  │                  ε-transitions                              │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                  SymbolVocabulary                           │ │
│  │                                                             │ │
│  │  <eps> → 0, "the" → 1, "cat" → 2, "runs" → 3, ...         │ │
│  └────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

## PcfgWfstConfig

Configuration for PCFG to WFST export:

```rust
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
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_depth` | 5 | Unrolling depth limit |
| `min_probability` | 1e-10 | Filter low-probability rules |
| `include_backoff` | true | Add backoff transitions |
| `max_states` | 100,000 | State count limit |

### Creating Configuration

```rust
use libgrammstein::code::wfst_export::PcfgWfstConfig;

// Default configuration
let config = PcfgWfstConfig::default();

// Custom configuration
let config = PcfgWfstConfig {
    max_depth: 3,           // Shallow unrolling
    min_probability: 0.001, // Higher threshold
    include_backoff: true,
    max_states: 50_000,     // Smaller FST
};
```

## SymbolVocabulary

Maps symbols to integer IDs for WFST labels:

```rust
pub struct SymbolVocabulary {
    symbol_to_id: HashMap<String, SymbolId>,
    id_to_symbol: Vec<String>,
}
```

### Creating a Vocabulary

```rust
use libgrammstein::code::wfst_export::SymbolVocabulary;

let mut vocab = SymbolVocabulary::new();

// ID 0 is reserved for epsilon (<eps>)
assert_eq!(vocab.get_id("<eps>"), Some(0));

// Add symbols
let id_the = vocab.add_symbol("the");
let id_cat = vocab.add_symbol("cat");
let id_runs = vocab.add_symbol("runs");

println!("'the' has ID {}", id_the);  // 1
println!("'cat' has ID {}", id_cat);  // 2
```

### Vocabulary Operations

```rust
// Lookup by symbol
let id = vocab.get_id("the");
assert_eq!(id, Some(1));

// Lookup by ID
let symbol = vocab.get_symbol(1);
assert_eq!(symbol, Some("the"));

// Size and emptiness
println!("Vocabulary size: {}", vocab.len());
println!("Is empty: {}", vocab.is_empty());

// Iterate over all symbols
for (symbol, id) in vocab.iter() {
    println!("{} -> {}", symbol, id);
}
```

## PcfgWfstBuilder

Builds WFST from PCFG (requires `lling-llang-integration` feature):

```rust
#[cfg(feature = "lling-llang-integration")]
pub struct PcfgWfstBuilder<W: Semiring + FromLogProb> {
    grammar: WeightedCFG,
    config: PcfgWfstConfig,
    vocabulary: SymbolVocabulary,
    wfst: VectorWfst<SymbolId, W>,
    state_map: HashMap<(String, usize), StateId>,
}
```

### Building a WFST

```rust
#[cfg(feature = "lling-llang-integration")]
use libgrammstein::code::wfst_export::{PcfgWfstBuilder, PcfgWfstConfig};
use lling_llang::semiring::TropicalWeight;

let grammar = build_grammar();  // Your WeightedCFG
let config = PcfgWfstConfig::default();

let builder = PcfgWfstBuilder::<TropicalWeight>::new(grammar, config);
let (wfst, vocab) = builder.build();

println!("WFST has {} states", wfst.num_states());
println!("Vocabulary has {} symbols", vocab.len());
```

### Using the PcfgWfstExport Trait

```rust
#[cfg(feature = "lling-llang-integration")]
use libgrammstein::code::wfst_export::PcfgWfstExport;
use lling_llang::semiring::LogWeight;

let grammar = build_grammar();

// Export with custom config
let config = PcfgWfstConfig {
    max_depth: 4,
    min_probability: 0.01,
    ..Default::default()
};
let (wfst, vocab) = grammar.to_wfst::<LogWeight>(config);

// Export with default config
let (wfst, vocab) = grammar.to_wfst_default::<LogWeight>();
```

## PcfgScorer

Simple scoring interface using PCFG probabilities:

```rust
pub struct PcfgScorer {
    grammar: WeightedCFG,
}
```

### Creating a Scorer

```rust
use libgrammstein::code::wfst_export::PcfgScorer;

let grammar = build_grammar();
let scorer = PcfgScorer::new(grammar);
```

### Scoring Rules

```rust
use libgrammstein::code::{Production, Symbol};

// Score a single production
let production = Production::new("NP", vec![
    Symbol::Terminal("the".to_string()),
    Symbol::NonTerminal("N".to_string()),
]);
let log_prob = scorer.score_rule(&production);
println!("Log P(NP -> 'the' N) = {:.3}", log_prob);
```

### Scoring Parses

```rust
// Score a sequence of productions (derivation)
let parse = vec![
    Production::new("S", vec![
        Symbol::NonTerminal("NP".to_string()),
        Symbol::NonTerminal("VP".to_string()),
    ]),
    Production::new("NP", vec![
        Symbol::NonTerminal("N".to_string()),
    ]),
    Production::new("N", vec![
        Symbol::Terminal("cat".to_string()),
    ]),
];

// Sum of log probabilities
let parse_score = scorer.score_parse(&parse);
println!("Parse score: {:.3}", parse_score);
```

### Terminal Probability

```rust
// Get probability of terminal given non-terminal
let prob = scorer.terminal_probability("Det", "the");
println!("P(Det -> 'the') = {:.2}", prob);  // e.g., 0.60

let prob = scorer.terminal_probability("N", "cat");
println!("P(N -> 'cat') = {:.2}", prob);    // e.g., 0.50
```

## Approximation Strategies

### Finite-Depth Unrolling

Unroll the grammar to a fixed depth, creating states for each (non-terminal, depth) pair:

```
Depth 0: S → NP VP
Depth 1: NP → Det N, VP → V NP
Depth 2: Det → "the", N → "cat", V → "runs"
...

States: (S, 0), (NP, 1), (VP, 1), (Det, 2), (N, 2), (V, 2), ...
```

### Epsilon Transitions

Non-terminals become epsilon transitions to sub-states:

```
State (S, 0):
  ε → State (NP, 1) [weight from S → NP VP]
  After NP, ε → State (VP, 1)
```

### Terminal Transitions

Terminals become labeled transitions:

```
State (Det, 2):
  "the" → Final [weight from Det → "the"]
  "a" → Final [weight from Det → "a"]
```

## Integration Example

Complete example exporting grammar and using for scoring:

```rust
use libgrammstein::code::{
    WeightedCFG, Production, Symbol,
    wfst_export::{PcfgWfstConfig, PcfgScorer, SymbolVocabulary}
};

#[cfg(feature = "lling-llang-integration")]
use libgrammstein::code::wfst_export::PcfgWfstExport;

fn build_nlp_grammar() -> WeightedCFG {
    let mut cfg = WeightedCFG::new("S");

    // S -> NP VP
    cfg.add_rule(
        Production::new("S", vec![
            Symbol::NonTerminal("NP".to_string()),
            Symbol::NonTerminal("VP".to_string()),
        ]),
        1.0,
    );

    // NP -> Det N (0.6) | N (0.4)
    cfg.add_rule(
        Production::new("NP", vec![
            Symbol::NonTerminal("Det".to_string()),
            Symbol::NonTerminal("N".to_string()),
        ]),
        0.6,
    );
    cfg.add_rule(
        Production::new("NP", vec![
            Symbol::NonTerminal("N".to_string()),
        ]),
        0.4,
    );

    // VP -> V NP (0.7) | V (0.3)
    cfg.add_rule(
        Production::new("VP", vec![
            Symbol::NonTerminal("V".to_string()),
            Symbol::NonTerminal("NP".to_string()),
        ]),
        0.7,
    );
    cfg.add_rule(
        Production::new("VP", vec![
            Symbol::NonTerminal("V".to_string()),
        ]),
        0.3,
    );

    // Terminals
    cfg.add_rule(Production::new("Det", vec![Symbol::Terminal("the".to_string())]), 0.6);
    cfg.add_rule(Production::new("Det", vec![Symbol::Terminal("a".to_string())]), 0.4);
    cfg.add_rule(Production::new("N", vec![Symbol::Terminal("cat".to_string())]), 0.5);
    cfg.add_rule(Production::new("N", vec![Symbol::Terminal("dog".to_string())]), 0.5);
    cfg.add_rule(Production::new("V", vec![Symbol::Terminal("runs".to_string())]), 0.5);
    cfg.add_rule(Production::new("V", vec![Symbol::Terminal("sees".to_string())]), 0.5);

    cfg
}

fn main() {
    let grammar = build_nlp_grammar();

    // Use PcfgScorer for simple scoring
    let scorer = PcfgScorer::new(grammar.clone());

    // Score "the cat runs"
    let parse = vec![
        Production::new("S", vec![
            Symbol::NonTerminal("NP".to_string()),
            Symbol::NonTerminal("VP".to_string()),
        ]),
        Production::new("NP", vec![
            Symbol::NonTerminal("Det".to_string()),
            Symbol::NonTerminal("N".to_string()),
        ]),
        Production::new("Det", vec![Symbol::Terminal("the".to_string())]),
        Production::new("N", vec![Symbol::Terminal("cat".to_string())]),
        Production::new("VP", vec![Symbol::NonTerminal("V".to_string())]),
        Production::new("V", vec![Symbol::Terminal("runs".to_string())]),
    ];

    let score = scorer.score_parse(&parse);
    println!("Parse score for 'the cat runs': {:.3}", score);

    // Export to WFST (if feature enabled)
    #[cfg(feature = "lling-llang-integration")]
    {
        use lling_llang::semiring::LogWeight;

        let config = PcfgWfstConfig {
            max_depth: 3,
            ..Default::default()
        };

        let (wfst, vocab) = grammar.to_wfst::<LogWeight>(config);
        println!("\nWFST Statistics:");
        println!("  States: {}", wfst.num_states());
        println!("  Vocabulary: {} symbols", vocab.len());
    }
}
```

## Semiring Weights

The WFST builder supports different semiring weight types:

### TropicalWeight

For finding best path (Viterbi):

```rust
#[cfg(feature = "lling-llang-integration")]
use lling_llang::semiring::TropicalWeight;

let (wfst, vocab) = grammar.to_wfst::<TropicalWeight>(config);
// Weights are -log probabilities
// ⊕ = min, ⊗ = +
```

### LogWeight

For summing probabilities (forward/backward):

```rust
#[cfg(feature = "lling-llang-integration")]
use lling_llang::semiring::LogWeight;

let (wfst, vocab) = grammar.to_wfst::<LogWeight>(config);
// Weights are -log probabilities
// ⊕ = log-add, ⊗ = +
```

## Limitations

1. **Approximation**: WFST cannot represent full CFG
2. **Depth limit**: Deep recursion requires higher depth
3. **State explosion**: Large grammars create many states
4. **Memory**: May consume significant memory

### When to Use

| Use Case | Recommendation |
|----------|----------------|
| Exact CFG parsing | Use `GrammarConstraint` (Earley) |
| Local scoring | Use `PcfgScorer` |
| Integration with FST tools | Use WFST export |
| Memory constrained | Use lower `max_depth` |

## Performance

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Build WFST | O(d^g) | d = depth, g = grammar branching |
| Score rule | O(1) | HashMap lookup |
| Score parse | O(p) | p = parse length |
| Vocabulary lookup | O(1) | HashMap |

### Memory Usage

```
States ≈ O(N × D) where N = non-terminals, D = max_depth
Arcs ≈ O(S × T) where S = states, T = average transitions
```

## Feature Flag

WFST export requires the `lling-llang-integration` feature:

```toml
[dependencies]
libgrammstein = { version = "0.1", features = ["code", "lling-llang-integration"] }
```

Without this feature, only `PcfgScorer` and `SymbolVocabulary` are available.

## See Also

- [PCFG](pcfg.md) - Probabilistic grammars
- [Constrained Decoding](constrained-decoding.md) - Earley-based validation
- [Grammar Corrector](correctors/grammar.md) - Grammar-based correction
- [lling-llang Integration](../../integration/lling-llang.md) - WFST library
