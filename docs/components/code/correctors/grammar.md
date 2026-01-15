# Grammar Corrector

The grammar corrector uses Probabilistic Context-Free Grammars (PCFG) and Earley parsing to suggest syntactically valid corrections based on grammar constraints.

## Overview

The `GrammarCorrector` provides:

- **Grammar validation**: Check token sequences against PCFG rules
- **Completion suggestions**: Valid next tokens from parse state
- **Syntax error detection**: Identify invalid token positions
- **Insertion/deletion suggestions**: Fix missing or extra tokens

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      GrammarCorrector                            │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    WeightedCFG                               ││
│  │  ┌─────────────────────────────────────────────────────────┐││
│  │  │ Production Rules with Probabilities                      │││
│  │  │                                                          │││
│  │  │ stmt → "if" "(" expr ")" stmt       [0.30]              │││
│  │  │ stmt → "while" "(" expr ")" stmt    [0.20]              │││
│  │  │ stmt → "return" expr ";"            [0.30]              │││
│  │  │ stmt → expr ";"                     [0.20]              │││
│  │  │ expr → identifier                   [0.50]              │││
│  │  │ expr → literal                      [0.50]              │││
│  │  └─────────────────────────────────────────────────────────┘││
│  └─────────────────────────────────────────────────────────────┘│
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    EarleyParser                              ││
│  │                                                              ││
│  │  • Incremental parsing                                       ││
│  │  • Valid next token computation                              ││
│  │  • Error recovery                                            ││
│  └─────────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────┘
```

## GrammarCorrectorConfig

Configuration options for the grammar corrector:

```rust
pub struct GrammarCorrectorConfig {
    /// Maximum candidates per error (default: 5)
    pub max_candidates: usize,
    /// Minimum rule probability to consider (default: 0.01)
    pub min_rule_probability: f64,
    /// Whether to suggest insertions (default: true)
    pub suggest_insertions: bool,
    /// Whether to suggest deletions (default: true)
    pub suggest_deletions: bool,
    /// Maximum lookahead for completions (default: 3)
    pub max_lookahead: usize,
    /// Base confidence for grammar corrections (default: 0.8)
    pub base_confidence: f64,
}
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_candidates` | 5 | Maximum suggestions per error |
| `min_rule_probability` | 0.01 | Filter low-probability rules |
| `suggest_insertions` | true | Suggest missing tokens |
| `suggest_deletions` | true | Suggest removing tokens |
| `max_lookahead` | 3 | Tokens to look ahead |
| `base_confidence` | 0.8 | Starting confidence score |

## Creating a Grammar Corrector

### With a Trained Grammar

```rust
use libgrammstein::code::{
    GrammarCorrector, GrammarCorrectorConfig, Python,
    WeightedCFG, Production, Symbol
};
use std::sync::Arc;

// Create a grammar
let mut grammar = WeightedCFG::new("stmt");

grammar.add_rule(
    Production::new("stmt", vec![
        Symbol::Terminal("if".to_string()),
        Symbol::Terminal("(".to_string()),
        Symbol::NonTerminal("expr".to_string()),
        Symbol::Terminal(")".to_string()),
        Symbol::NonTerminal("stmt".to_string()),
    ]),
    0.3, // 30% probability
);

grammar.add_rule(
    Production::new("stmt", vec![
        Symbol::Terminal("return".to_string()),
        Symbol::NonTerminal("expr".to_string()),
        Symbol::Terminal(";".to_string()),
    ]),
    0.4,
);

// Create corrector
let python = Arc::new(Python::new());
let corrector = GrammarCorrector::with_defaults(python, grammar);
```

### With Custom Configuration

```rust
let config = GrammarCorrectorConfig {
    max_candidates: 10,
    min_rule_probability: 0.05,
    suggest_insertions: true,
    suggest_deletions: false,  // Only insertions
    max_lookahead: 5,
    base_confidence: 0.9,
};

let corrector = GrammarCorrector::new(python, grammar, config);
```

## Grammar Constraint

The corrector creates a `GrammarConstraint` for validation:

```rust
// Create a constraint for checking validity
let constraint = corrector.create_constraint();

// Check if a token is valid at current position
if constraint.is_valid_token("if") {
    println!("'if' is valid here");
}

// Advance the parser state
constraint.advance("if");

// Get all valid tokens at current position
let valid = constraint.valid_tokens();
println!("Valid next tokens: {:?}", valid);
```

## Valid Next Tokens

Query valid tokens given a token history:

```rust
// After "if", what tokens are valid?
let valid = corrector.valid_next_tokens(&["if"]);
// Returns: {"("} - an opening paren is expected

// After "if (", what's valid?
let valid = corrector.valid_next_tokens(&["if", "("]);
// Returns: expressions - identifiers, literals, etc.

// After "if ( x )", what's valid?
let valid = corrector.valid_next_tokens(&["if", "(", "x", ")"]);
// Returns: statement starters - "{", another if, etc.
```

## Suggesting Completions

Get ranked completion suggestions:

```rust
// Get top 5 completions after "return"
let completions = corrector.suggest_completions(&["return"], 5);

for (token, probability) in &completions {
    println!("  {} (prob: {:.2})", token, probability);
}
// Output:
//   x (prob: 0.50)
//   y (prob: 0.50)
```

## Syntax Error Detection

Find syntax errors in a token sequence:

```rust
let tokens = vec!["fi", "(", "x", ")", "return", "x", ";"];
//                 ^^ "fi" is not a valid keyword

let errors = corrector.find_syntax_errors(&tokens);

for error in &errors {
    println!("Position {}: {}", error.position, error.message());
}
// Output: Position 0: Invalid token 'fi', expected one of: 'if', 'while', 'return'
```

### SyntaxErrorType

Types of syntax errors detected:

```rust
pub enum SyntaxErrorType {
    InvalidToken,      // Token not valid at this position
    UnexpectedToken,   // No valid tokens possible
    MissingToken,      // Expected token not present
    UnclosedDelimiter, // Missing closing bracket/paren
}
```

### SyntaxError Structure

```rust
pub struct SyntaxError {
    /// Position in the token stream
    pub position: usize,
    /// The problematic token
    pub token: String,
    /// Valid tokens expected at this position
    pub expected: HashSet<String>,
    /// Type of error
    pub error_type: SyntaxErrorType,
}
```

## Correction Types

The grammar corrector generates three types of corrections:

### Insertions

For missing tokens (e.g., missing semicolon):

```rust
// Source: "return x"  (missing ";")
// Correction: Insert ";" after "x"

let correction = Correction::new(
    CorrectionKind::Insertion,
    byte_position,
    byte_position,  // Same start/end for insertion
    "",             // Nothing being replaced
    ";",            // Token to insert
)
.with_source(CorrectionSource::Grammar)
.with_context("Expected token at position N");
```

### Deletions

For extra tokens:

```rust
// Source: "return return x;"  (duplicate "return")
// Correction: Delete first or second "return"

let correction = Correction::new(
    CorrectionKind::Deletion,
    start_byte,
    end_byte,
    "return",  // Token to remove
    "",        // Replace with nothing
)
.with_source(CorrectionSource::Grammar)
.with_context("Unexpected token");
```

### Replacements

For wrong tokens that should be something else:

```rust
// Source: "fi (x) return x;"  ("fi" instead of "if")
// Correction: Replace "fi" with "if"

let correction = Correction::new(
    CorrectionKind::Replacement,
    start_byte,
    end_byte,
    "fi",   // Original
    "if",   // Replacement
)
.with_source(CorrectionSource::Grammar)
.with_context("Grammar suggests 'if'");
```

## Confidence Calculation

Grammar corrections compute confidence based on:

1. **Base confidence**: Configurable starting point (default 0.8)
2. **Rule probability**: Higher probability rules score better
3. **String similarity**: For replacements, similar tokens score higher

```rust
// Insertion confidence
confidence = base_confidence × rule_probability

// Deletion confidence (lower due to destructive nature)
confidence = base_confidence × 0.7

// Replacement confidence
confidence = base_confidence × similarity × (0.5 + 0.5 × rule_probability)
```

## String Similarity

The corrector uses Jaccard similarity on character bigrams:

```rust
// Compare "fi" to "if"
// Bigrams of "fi": {('f', 'i')}
// Bigrams of "if": {('i', 'f')}
// Intersection: {} (empty)
// Union: {('f', 'i'), ('i', 'f')}
// Similarity: 0 / 2 = 0.0

// Compare "whiel" to "while"
// Bigrams of "whiel": {('w','h'), ('h','i'), ('i','e'), ('e','l')}
// Bigrams of "while": {('w','h'), ('h','i'), ('i','l'), ('l','e')}
// Intersection: {('w','h'), ('h','i')}
// Union: 6 unique bigrams
// Similarity: 2 / 6 ≈ 0.33
```

Replacements with similarity below 0.3 are filtered out.

## Integration Example

Complete example using the grammar corrector:

```rust
use libgrammstein::code::{
    GrammarCorrector, WeightedCFG, Production, Symbol,
    Python, CodeCorrector, CodeToken, TokenContext, TokenType
};
use std::sync::Arc;

// Build a simple expression grammar
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

    // expr -> term
    cfg.add_rule(
        Production::new("expr", vec![
            Symbol::NonTerminal("term".to_string()),
        ]),
        0.7,
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
        0.7,
    );

    cfg
}

fn main() {
    let python = Arc::new(Python::new());
    let grammar = build_expr_grammar();
    let corrector = GrammarCorrector::with_defaults(python, grammar);

    // Check what's valid at start
    let valid_start = corrector.valid_next_tokens(&[]);
    println!("Valid start tokens: {:?}", valid_start);

    // Check what's valid after "("
    let valid_after_paren = corrector.valid_next_tokens(&["("]);
    println!("After '(': {:?}", valid_after_paren);

    // Find syntax errors
    let tokens = vec!["(", "NUMBER", "+", ")"];  // Missing second operand
    let errors = corrector.find_syntax_errors(&tokens);

    for error in errors {
        println!("Error: {}", error.message());
    }
}
```

## Working with PCFG

### Accessing the Grammar

```rust
let grammar = corrector.grammar();

// Iterate over rules
for (production, normalized) in grammar.iter_rules() {
    let prob = grammar.probability(production);
    println!("{} -> {:?} (prob: {:.2})",
        production.lhs,
        production.rhs,
        prob
    );
}
```

### Token Probability

Estimate the probability of a token in context:

```rust
// Internal method used for ranking
let prob = corrector.token_probability("return", &["if", "(", "x", ")"]);
```

## Limitations

1. **Context window**: Grammar checking requires token history
2. **Ambiguous grammars**: May produce multiple valid suggestions
3. **Training data**: Quality depends on grammar training
4. **Performance**: Earley parsing is O(n³) worst case

## Performance

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Create constraint | O(1) | Initializes parser state |
| Valid next tokens | O(g) | g = grammar size |
| Find syntax errors | O(n × g) | n = token count |
| Suggest completions | O(g) | Filtered by probability |

## Thread Safety

`GrammarCorrector` is `Send + Sync` when its language type is:

```rust
use std::sync::Arc;

let corrector = Arc::new(GrammarCorrector::with_defaults(python, grammar));

// Safe to share across threads for reading
let corrector_clone = Arc::clone(&corrector);
std::thread::spawn(move || {
    let valid = corrector_clone.valid_next_tokens(&["if"]);
});
```

## See Also

- [Correctors Overview](overview.md) - Architecture and comparison
- [Lexical Corrector](lexical.md) - Fuzzy matching
- [Semantic Corrector](semantic.md) - CPG/GNN analysis
- [Ensemble Corrector](ensemble.md) - Multi-source aggregation
- [PCFG](../pcfg.md) - Probabilistic grammars
- [Constrained Decoding](../constrained-decoding.md) - Grammar constraints
