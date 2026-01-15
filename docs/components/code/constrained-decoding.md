# Grammar-Constrained Decoding

Grammar-constrained decoding ensures neural model outputs are syntactically valid according to formal grammar rules using Earley parsing.

## Overview

The constrained decoding module provides:

- **Token validity checking**: Verify tokens against grammar constraints
- **Earley parsing**: Incremental parsing for validity checking
- **Token masking**: Disable invalid tokens during generation
- **Vocabulary management**: Map tokens to grammar symbols

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    GrammarConstraint                             │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    EarleyParser                             │ │
│  │                                                             │ │
│  │  WeightedCFG ──► Rule Index ──► Incremental Parsing        │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    EarleyChart                              │ │
│  │                                                             │ │
│  │  Position 0: {S → • NP VP, NP → • Det N, NP → • N, ...}   │ │
│  │  Position 1: {NP → Det • N, ...}                           │ │
│  │  Position 2: {NP → Det N •, S → NP • VP, ...}             │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    TokenMask                                │ │
│  │                                                             │ │
│  │  Valid tokens at current position: {"if", "while", "for"}  │ │
│  │  Apply to logits: invalid tokens → -∞                      │ │
│  └────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

## ConstrainedDecodingConfig

Configuration for grammar-constrained decoding:

```rust
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
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_lookahead` | 3 | Tokens to look ahead for validity |
| `cache_states` | true | Cache valid token sets |
| `min_rule_probability` | 1e-10 | Filter low-probability rules |
| `allow_partial` | true | Accept incomplete parses |

### Creating Configuration

```rust
use libgrammstein::code::ConstrainedDecodingConfig;

// Default configuration
let config = ConstrainedDecodingConfig::default();

// Custom configuration
let config = ConstrainedDecodingConfig {
    max_lookahead: 5,
    cache_states: true,
    min_rule_probability: 0.001,
    allow_partial: false,
};
```

## EarleyParser

Earley parser for incremental grammar checking:

```rust
pub struct EarleyParser {
    grammar: WeightedCFG,
    rules_by_lhs: HashMap<String, Vec<usize>>,
    rules: Vec<(String, Vec<Symbol>, f64)>,
}
```

### Creating a Parser

```rust
use libgrammstein::code::{EarleyParser, WeightedCFG};

let grammar = build_grammar();  // Your WeightedCFG
let parser = EarleyParser::new(grammar);

// Get start symbol
println!("Start: {}", parser.start_symbol());

// Get rules for a non-terminal
for rule_idx in parser.rules_for("expr") {
    if let Some((lhs, rhs, weight)) = parser.rule(rule_idx) {
        println!("{} -> {:?} [w={:.2}]", lhs, rhs, weight);
    }
}
```

## EarleyState

A state in the Earley parser (dotted rule):

```rust
pub struct EarleyState {
    /// Index of the rule in the grammar
    pub rule_idx: usize,
    /// Position in the RHS (dot position)
    pub dot_pos: usize,
    /// Starting position in the input
    pub start_pos: usize,
}
```

### State Representation

```
Rule: S -> NP VP

State at different dot positions:
  S -> • NP VP    (dot_pos = 0, expecting NP)
  S -> NP • VP    (dot_pos = 1, expecting VP)
  S -> NP VP •    (dot_pos = 2, complete)
```

### State Methods

```rust
let state = EarleyState::new(rule_idx, dot_pos, start_pos);

// Check if state is complete (dot at end)
if state.is_complete(rhs_length) {
    println!("State complete: rule {} finished", state.rule_idx);
}
```

## EarleyChart

Chart data structure for Earley parsing:

```rust
pub struct EarleyChart {
    /// Sets of states at each position
    sets: Vec<HashSet<EarleyState>>,
}
```

### Chart Operations

```rust
use libgrammstein::code::EarleyChart;

// Create chart with capacity for 10 positions
let mut chart = EarleyChart::new(10);

// Add states
chart.add(0, EarleyState::new(0, 0, 0));
chart.add(1, EarleyState::new(1, 1, 0));

// Query states at position
for state in chart.states_at(0) {
    println!("State at pos 0: rule {}", state.rule_idx);
}

// Chart size
println!("Positions: {}", chart.len());
```

## GrammarConstraint

Main interface for grammar-constrained token validation:

```rust
pub struct GrammarConstraint {
    parser: EarleyParser,
    config: ConstrainedDecodingConfig,
    chart: EarleyChart,
    position: usize,
    valid_tokens_cache: Option<HashSet<String>>,
}
```

### Creating a Constraint

```rust
use libgrammstein::code::{GrammarConstraint, WeightedCFG, ConstrainedDecodingConfig};

let grammar = build_grammar();

// With custom config
let config = ConstrainedDecodingConfig::default();
let constraint = GrammarConstraint::new(grammar, config);

// With default config
let constraint = GrammarConstraint::with_default_config(grammar);
```

### Initialization

```rust
let mut constraint = GrammarConstraint::with_default_config(grammar);

// Initialize parser with start symbol
constraint.reset();  // Also calls initialize()

// Or initialize explicitly
constraint.initialize();
```

### Checking Token Validity

```rust
// Check if a specific token is valid
if constraint.is_valid_token("if") {
    println!("'if' is valid at position {}", constraint.position());
}

// Get all valid tokens at current position
let valid = constraint.valid_tokens();
println!("Valid tokens: {:?}", valid);
```

### Advancing the Parser

```rust
// Try to advance with a token
if constraint.advance("if") {
    println!("Advanced to position {}", constraint.position());

    // Check what's valid next
    let next_valid = constraint.valid_tokens();
    println!("Now valid: {:?}", next_valid);
} else {
    println!("Token 'if' is not valid here");
}
```

### Checking Completion

```rust
// Check if we can complete from current state
if constraint.can_complete() {
    println!("Parse is complete!");
} else {
    println!("Parse incomplete, need more tokens");
}
```

## TokenMask

Mask for constraining model output:

```rust
pub struct TokenMask {
    /// Token indices that are allowed
    allowed: HashSet<usize>,
    /// Total vocabulary size
    vocab_size: usize,
}
```

### Creating Masks

```rust
use libgrammstein::code::TokenMask;

// Allow all tokens
let mask = TokenMask::allow_all(50000);

// Allow specific tokens
let allowed = vec![1, 5, 10, 15].into_iter().collect();
let mask = TokenMask::from_allowed(allowed, 50000);
```

### Using Masks

```rust
// Check if token is allowed
if mask.is_allowed(5) {
    println!("Token 5 is allowed");
}

// Get allowed indices
for idx in mask.allowed_indices() {
    println!("Allowed: {}", idx);
}

// Count allowed tokens
println!("Allowed count: {}", mask.count_allowed());

// Convert to boolean vector
let bool_vec = mask.to_bool_vec();
```

### Applying to Logits

```rust
let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];

// Apply mask - sets disallowed tokens to -infinity
mask.apply_to_logits(&mut logits);

// Now disallowed tokens have -inf, won't be selected
for (i, &logit) in logits.iter().enumerate() {
    if logit.is_finite() {
        println!("Token {} has logit {:.2}", i, logit);
    }
}
```

## DecodingVocabulary

Vocabulary mapping for constrained decoding:

```rust
pub struct DecodingVocabulary {
    token_to_idx: HashMap<String, usize>,
    idx_to_token: Vec<String>,
}
```

### Building a Vocabulary

```rust
use libgrammstein::code::DecodingVocabulary;

let mut vocab = DecodingVocabulary::new();

// Add tokens
let idx_if = vocab.add_token("if");
let idx_else = vocab.add_token("else");
let idx_while = vocab.add_token("while");

// Lookup by token
let idx = vocab.get_idx("if");
assert_eq!(idx, Some(idx_if));

// Lookup by index
let token = vocab.get_token(idx_if);
assert_eq!(token, Some("if"));

// Vocabulary size
println!("Vocab size: {}", vocab.len());
```

### Creating Masks from Vocabulary

```rust
let mut valid_tokens = HashSet::new();
valid_tokens.insert("if".to_string());
valid_tokens.insert("while".to_string());

// Create mask from valid token strings
let mask = vocab.create_mask(&valid_tokens);

// Mask only allows tokens that are both:
// 1. In valid_tokens
// 2. In the vocabulary
```

## Integration Example

Complete example of grammar-constrained decoding:

```rust
use libgrammstein::code::{
    GrammarConstraint, DecodingVocabulary, TokenMask,
    WeightedCFG, Production, Symbol, ConstrainedDecodingConfig
};

fn constrained_decode(grammar: WeightedCFG, vocab: &DecodingVocabulary) {
    let config = ConstrainedDecodingConfig {
        max_lookahead: 5,
        cache_states: true,
        ..Default::default()
    };

    let mut constraint = GrammarConstraint::new(grammar, config);
    constraint.reset();

    let mut generated = Vec::new();

    // Simulated generation loop
    loop {
        // Get valid tokens according to grammar
        let valid_tokens = constraint.valid_tokens();

        if valid_tokens.is_empty() {
            println!("No valid tokens - generation stuck");
            break;
        }

        // Create mask for model
        let mask = vocab.create_mask(&valid_tokens);

        // In real usage, apply mask to model logits:
        // model_logits = get_model_logits(context);
        // mask.apply_to_logits(&mut model_logits);
        // next_token_idx = sample(model_logits);
        // next_token = vocab.get_token(next_token_idx);

        // For this example, just pick first valid token
        let next_token = valid_tokens.iter().next().unwrap().clone();

        // Advance parser
        if !constraint.advance(&next_token) {
            println!("Failed to advance with: {}", next_token);
            break;
        }

        generated.push(next_token);
        println!("Generated: {:?}", generated);

        // Check if complete
        if constraint.can_complete() {
            println!("Generation complete!");
            break;
        }

        // Safety limit
        if generated.len() > 100 {
            println!("Reached length limit");
            break;
        }
    }
}

// Build a simple grammar
fn build_simple_grammar() -> WeightedCFG {
    let mut cfg = WeightedCFG::new("S");

    // S -> A B
    cfg.add_rule(
        Production::new("S", vec![
            Symbol::NonTerminal("A".to_string()),
            Symbol::NonTerminal("B".to_string()),
        ]),
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

fn main() {
    let grammar = build_simple_grammar();

    let mut vocab = DecodingVocabulary::new();
    vocab.add_token("a");
    vocab.add_token("b");

    constrained_decode(grammar, &vocab);
    // Output:
    // Generated: ["a"]
    // Generated: ["a", "b"]
    // Generation complete!
}
```

## Earley Algorithm

The parser implements the classic Earley algorithm with three operations:

### Prediction

When a state expects a non-terminal, add initial states for all rules producing it:

```rust
// State: S -> • NP VP (expecting NP)
// Add all NP rules: NP -> • Det N, NP -> • N, etc.
fn predict(&mut self, state: &EarleyState, pos: usize) {
    if let Some(Symbol::NonTerminal(nt)) = self.next_symbol(state) {
        for rule_idx in self.parser.rules_for(&nt) {
            self.chart.add(pos, EarleyState::new(rule_idx, 0, pos));
        }
    }
}
```

### Scanning

When a state expects a terminal that matches input, advance the dot:

```rust
// State: NP -> Det • N (expecting N)
// Input at pos: "cat" (matches N terminal)
// Add: NP -> Det N • at pos+1
fn scan(&mut self, token: &str, pos: usize) {
    for state in self.chart.states_at(pos) {
        if let Some(Symbol::Terminal(t)) = self.next_symbol(&state) {
            if t == token {
                let new_state = EarleyState::new(
                    state.rule_idx,
                    state.dot_pos + 1,
                    state.start_pos,
                );
                self.chart.add(pos + 1, new_state);
            }
        }
    }
}
```

### Completion

When a state is complete, advance states that were waiting for it:

```rust
// Complete state: NP -> Det N • (started at pos 0, complete at pos 2)
// Find: S -> • NP VP at pos 0
// Add: S -> NP • VP at pos 2
fn complete(&mut self, state: &EarleyState, pos: usize) {
    if !state.is_complete() { return; }

    let completed_nt = self.lhs_of(state);

    for waiting in self.chart.states_at(state.start_pos) {
        if let Some(Symbol::NonTerminal(nt)) = self.next_symbol(&waiting) {
            if nt == completed_nt {
                let new_state = EarleyState::new(
                    waiting.rule_idx,
                    waiting.dot_pos + 1,
                    waiting.start_pos,
                );
                self.chart.add(pos, new_state);
            }
        }
    }
}
```

## Performance

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Initialize | O(g) | g = grammar size |
| Valid tokens | O(s) | s = states at position |
| Advance | O(s × g) | Prediction + completion |
| Apply mask | O(v) | v = vocabulary size |

### Earley Complexity

- **Best case**: O(n) for unambiguous grammars
- **Typical**: O(n²) for most programming languages
- **Worst case**: O(n³) for highly ambiguous grammars

Where n = number of tokens parsed.

## Thread Safety

`GrammarConstraint` is not `Send` or `Sync` due to mutable state. For parallel generation, create separate instances:

```rust
let grammar = Arc::new(build_grammar());

// Each thread gets its own constraint
let handles: Vec<_> = (0..4).map(|_| {
    let grammar = Arc::clone(&grammar);
    std::thread::spawn(move || {
        let mut constraint = GrammarConstraint::with_default_config(
            (*grammar).clone()
        );
        constraint.reset();
        // Use constraint...
    })
}).collect();
```

## See Also

- [PCFG](pcfg.md) - Probabilistic grammars
- [WFST Export](wfst-export.md) - FST approximation
- [Grammar Corrector](correctors/grammar.md) - Grammar-based correction
- [Pipeline](pipeline.md) - End-to-end workflow
