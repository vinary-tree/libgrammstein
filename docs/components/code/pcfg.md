# Probabilistic Context-Free Grammars

Probabilistic Context-Free Grammars (PCFGs) provide formal grammar representations with weighted production rules for syntax validation and scoring.

## Overview

The PCFG module provides:

- **Production rules**: Grammar rules with left-hand and right-hand sides
- **Weighted grammars**: Probability distributions over derivations
- **Grammar training**: Learn rule probabilities from parsed code
- **Grammar-constrained decoding**: Ensure syntactic validity of outputs

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        WeightedCFG                               │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                  Production Rules                           │ │
│  │                                                             │ │
│  │  stmt -> "if" "(" expr ")" stmt          [p=0.30]          │ │
│  │  stmt -> "while" "(" expr ")" stmt       [p=0.20]          │ │
│  │  stmt -> "return" expr ";"               [p=0.30]          │ │
│  │  stmt -> expr ";"                        [p=0.20]          │ │
│  │  expr -> identifier                      [p=0.50]          │ │
│  │  expr -> literal                         [p=0.50]          │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              Probability Calculation                        │ │
│  │                                                             │ │
│  │  P(production) = weight(production) / Σ weight(lhs=X)      │ │
│  │                                                             │ │
│  │  Normalization ensures probabilities sum to 1 per LHS      │ │
│  └────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                       PcfgTrainer                                │
│                                                                  │
│  Parsed AST ──► Extract Productions ──► Count Rules ──► CFG     │
└──────────────────────────────────────────────────────────────────┘
```

## Symbol

Symbols represent grammar elements (terminals and non-terminals):

```rust
pub enum Symbol {
    /// Non-terminal symbol (e.g., "expression", "statement")
    NonTerminal(String),
    /// Terminal symbol (actual token, e.g., "if", "+", identifier)
    Terminal(String),
}
```

### Creating Symbols

```rust
use libgrammstein::code::Symbol;

// Non-terminal (grammar category)
let expr = Symbol::non_terminal("expr");
let stmt = Symbol::non_terminal("statement");

// Terminal (actual token)
let plus = Symbol::terminal("+");
let keyword = Symbol::terminal("if");

// Checking symbol type
assert!(expr.is_non_terminal());
assert!(plus.is_terminal());

// Get symbol name
assert_eq!(expr.name(), "expr");
assert_eq!(plus.name(), "+");
```

### Display Format

```rust
// Non-terminals are displayed with angle brackets
let nt = Symbol::non_terminal("expr");
println!("{}", nt);  // Output: <expr>

// Terminals are displayed with quotes
let t = Symbol::terminal("+");
println!("{}", t);   // Output: '+'
```

## Production

A production rule maps a non-terminal to a sequence of symbols:

```rust
pub struct Production {
    /// Left-hand side (non-terminal)
    pub lhs: String,
    /// Right-hand side (sequence of symbols)
    pub rhs: Vec<Symbol>,
}
```

### Creating Productions

```rust
use libgrammstein::code::{Production, Symbol};

// Simple production: expr -> identifier
let prod1 = Production::new(
    "expr",
    vec![Symbol::Terminal("identifier".to_string())],
);

// Compound production: expr -> expr "+" term
let prod2 = Production::new(
    "expr",
    vec![
        Symbol::NonTerminal("expr".to_string()),
        Symbol::Terminal("+".to_string()),
        Symbol::NonTerminal("term".to_string()),
    ],
);

// Epsilon production (empty RHS)
let epsilon = Production::new("optional", vec![]);
assert!(epsilon.is_epsilon());

// Production arity
assert_eq!(prod2.arity(), 3);
```

### Display Format

```rust
let prod = Production::new(
    "expr",
    vec![
        Symbol::NonTerminal("term".to_string()),
        Symbol::Terminal("+".to_string()),
        Symbol::NonTerminal("expr".to_string()),
    ],
);

println!("{}", prod);
// Output: expr -> <term> '+' <expr>
```

## WeightedCFG

A weighted context-free grammar with probability distributions:

```rust
pub struct WeightedCFG {
    /// Production rules with their weights
    rules: HashMap<Production, f64>,
    /// Start symbol
    start_symbol: String,
    // ... indexing structures
}
```

### Creating a Grammar

```rust
use libgrammstein::code::{WeightedCFG, Production, Symbol};

// Create grammar with start symbol
let mut cfg = WeightedCFG::new("S");

// Add production rules with weights
cfg.add_rule(
    Production::new("S", vec![
        Symbol::NonTerminal("NP".to_string()),
        Symbol::NonTerminal("VP".to_string()),
    ]),
    1.0,
);

cfg.add_rule(
    Production::new("NP", vec![
        Symbol::Terminal("the".to_string()),
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
```

### Querying the Grammar

```rust
// Get rules for a non-terminal
let np_rules = cfg.rules_for("NP");
for (production, weight) in np_rules {
    println!("{} [weight: {:.2}]", production, weight);
}

// Get probability (normalized)
let production = Production::new("NP", vec![
    Symbol::Terminal("the".to_string()),
    Symbol::NonTerminal("N".to_string()),
]);
let prob = cfg.probability(&production);
println!("P(NP -> 'the' <N>) = {:.2}", prob);  // 0.60

// Get log probability
let log_prob = cfg.log_probability(&production);
println!("log P = {:.3}", log_prob);  // -0.511

// Iterate over all rules
for (production, weight) in cfg.iter_rules() {
    let prob = cfg.probability(production);
    println!("{} [prob: {:.2}]", production, prob);
}
```

### Grammar Properties

```rust
// Start symbol
let start = cfg.start_symbol();
println!("Start: {}", start);

// Number of rules
println!("Rules: {}", cfg.rule_count());

// Get all non-terminals
for nt in cfg.non_terminals() {
    println!("Non-terminal: {}", nt);
}

// Get all terminals
for t in cfg.terminals() {
    println!("Terminal: {}", t);
}
```

### Normalizing Weights

Weights can be normalized to ensure they sum to 1.0 for each LHS:

```rust
let mut cfg = WeightedCFG::new("S");

// Add rules with counts (not probabilities)
cfg.add_rule(rule_a.clone(), 75.0);  // Seen 75 times
cfg.add_rule(rule_b.clone(), 25.0);  // Seen 25 times

// Before normalization
println!("Weight A: {}", cfg.weight(&rule_a));  // 75.0
println!("Prob A: {}", cfg.probability(&rule_a));  // 0.75

// Normalize to convert weights to probabilities
cfg.normalize();

// After normalization, weights are probabilities
println!("Weight A: {}", cfg.weight(&rule_a));  // 0.75
```

## PcfgTrainer

Train PCFGs from parsed code corpora:

```rust
pub struct PcfgTrainer<'a, L: CodeLanguage> {
    language: &'a L,
    rule_counts: HashMap<Production, u64>,
    start_symbol: String,
}
```

### Training from Code

```rust
use libgrammstein::code::{PcfgTrainer, CodeParser, Python};
use std::sync::Arc;

let python = Arc::new(Python::new());
let mut parser = CodeParser::new(python.clone()).unwrap();
let mut trainer = PcfgTrainer::new(&*python);

// Parse source files
let sources = vec![
    "def foo(x): return x + 1",
    "def bar(a, b): return a * b",
    "class MyClass: pass",
];

for source in &sources {
    let parsed = parser.parse(source).unwrap();
    trainer.train_from_parsed(&parsed);
}

// Convert to weighted CFG
let cfg = trainer.to_weighted_cfg();

println!("Unique rules: {}", trainer.unique_rule_count());
println!("Total instances: {}", trainer.total_rule_count());
```

### Batch Training

```rust
// Train from iterator of parsed files
let parsed_files: Vec<ParsedCode> = /* load files */;
trainer.train_from_parsed_iter(parsed_files.iter());

// Build the CFG
let cfg = trainer.to_weighted_cfg();
```

### Custom Start Symbol

```rust
// Use custom start symbol instead of "source_file"
let trainer = PcfgTrainer::new(&*python)
    .with_start_symbol("function_definition");
```

### Inspecting Training Progress

```rust
// Get rule counts
for (production, count) in trainer.rule_counts() {
    println!("{}: {} occurrences", production, count);
}

// Clear and retrain
trainer.clear();
```

## Rule Extraction

The trainer extracts production rules from AST nodes:

```
Source: def foo(x): return x + 1

AST:
  function_definition
    ├── "def"
    ├── identifier: "foo"
    ├── parameters
    │   └── identifier: "x"
    └── return_statement
        └── binary_operator
            ├── identifier: "x"
            ├── "+"
            └── integer: "1"

Extracted Rules:
  function_definition -> identifier parameters return_statement
  parameters -> identifier
  return_statement -> binary_operator
  binary_operator -> identifier "+" integer
```

### Rule Filtering

Only named AST nodes generate rules:

```rust
fn extract_rules(&mut self, node: &AstNode) {
    // Skip error nodes
    if node.is_error || node.is_missing {
        return;
    }

    // Only create rules for named nodes with children
    if node.is_named && !node.children.is_empty() {
        let lhs = node.kind.clone();
        let rhs: Vec<Symbol> = node.children
            .iter()
            .filter(|c| c.is_named)
            .map(|c| /* ... */)
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
```

## PcfgWfstConfig

Configuration for WFST export (for integration with lling-llang):

```rust
pub struct PcfgWfstConfig {
    /// Whether to include epsilon transitions
    pub include_epsilon: bool,
    /// Minimum probability threshold
    pub min_probability: f64,
    /// Maximum number of rules to include
    pub max_rules: Option<usize>,
}
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `include_epsilon` | true | Include epsilon transitions for optional rules |
| `min_probability` | 1e-10 | Filter rules below this probability |
| `max_rules` | None | Limit total rules (None = no limit) |

## Integration Example

Complete example training and using a PCFG:

```rust
use libgrammstein::code::{
    PcfgTrainer, WeightedCFG, Production, Symbol,
    CodeParser, Python
};
use std::sync::Arc;

fn train_python_grammar(sources: &[&str]) -> WeightedCFG {
    let python = Arc::new(Python::new());
    let mut parser = CodeParser::new(python.clone()).unwrap();
    let mut trainer = PcfgTrainer::new(&*python);

    for source in sources {
        if let Ok(parsed) = parser.parse(source) {
            // Only train on error-free parses
            if !parsed.has_errors {
                trainer.train_from_parsed(&parsed);
            }
        }
    }

    let mut cfg = trainer.to_weighted_cfg();
    cfg.normalize();
    cfg
}

fn main() {
    let corpus = vec![
        "def add(a, b): return a + b",
        "def sub(a, b): return a - b",
        "def mul(a, b): return a * b",
        "x = 42",
        "y = x + 1",
    ];

    let cfg = train_python_grammar(&corpus);

    println!("Trained grammar with {} rules", cfg.rule_count());
    println!("Start symbol: {}", cfg.start_symbol());

    // Find most probable rules for function definitions
    let rules = cfg.rules_for("function_definition");
    println!("\nFunction definition rules:");
    for (prod, _) in rules {
        let prob = cfg.probability(prod);
        if prob > 0.01 {
            println!("  {} [p={:.3}]", prod, prob);
        }
    }
}
```

## Building a Grammar Manually

For simple languages or testing:

```rust
use libgrammstein::code::{WeightedCFG, Production, Symbol};

// Simple expression grammar
fn build_expr_grammar() -> WeightedCFG {
    let mut cfg = WeightedCFG::new("expr");

    // expr -> expr "+" term
    cfg.add_rule(
        Production::new("expr", vec![
            Symbol::NonTerminal("expr".to_string()),
            Symbol::Terminal("+".to_string()),
            Symbol::NonTerminal("term".to_string()),
        ]),
        0.3,
    );

    // expr -> expr "-" term
    cfg.add_rule(
        Production::new("expr", vec![
            Symbol::NonTerminal("expr".to_string()),
            Symbol::Terminal("-".to_string()),
            Symbol::NonTerminal("term".to_string()),
        ]),
        0.2,
    );

    // expr -> term
    cfg.add_rule(
        Production::new("expr", vec![
            Symbol::NonTerminal("term".to_string()),
        ]),
        0.5,
    );

    // term -> "(" expr ")"
    cfg.add_rule(
        Production::new("term", vec![
            Symbol::Terminal("(".to_string()),
            Symbol::NonTerminal("expr".to_string()),
            Symbol::Terminal(")".to_string()),
        ]),
        0.3,
    );

    // term -> NUMBER
    cfg.add_rule(
        Production::new("term", vec![
            Symbol::Terminal("NUMBER".to_string()),
        ]),
        0.5,
    );

    // term -> IDENTIFIER
    cfg.add_rule(
        Production::new("term", vec![
            Symbol::Terminal("IDENTIFIER".to_string()),
        ]),
        0.2,
    );

    cfg
}
```

## Performance

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Add rule | O(1) amortized | HashMap insertion |
| Get probability | O(1) | Lookup and division |
| Rules for LHS | O(1) | Pre-indexed |
| Train from AST | O(n) | n = AST nodes |
| Normalize | O(r) | r = number of rules |

### Memory Usage

The grammar stores each unique production once. For a language like Python with ~100 AST node types and average arity 3:

```
Storage ≈ O(n × a) where n = node types, a = average arity
Typical: ~500 rules × 50 bytes = ~25 KB
```

## Thread Safety

`WeightedCFG` is `Send + Sync` and can be safely shared:

```rust
use std::sync::Arc;

let cfg = Arc::new(train_grammar(corpus));

// Share across threads
let cfg_clone = Arc::clone(&cfg);
std::thread::spawn(move || {
    let prob = cfg_clone.probability(&some_rule);
    println!("P = {}", prob);
});
```

Note: `PcfgTrainer` requires mutable access during training.

## See Also

- [Grammar Corrector](correctors/grammar.md) - Using PCFGs for correction
- [Constrained Decoding](constrained-decoding.md) - Grammar-constrained generation
- [WFST Export](wfst-export.md) - WFST approximation for PCFGs
- [Pipeline](pipeline.md) - End-to-end workflow
