# LaTeX Tokenizer

The LaTeX tokenizer provides mode-aware tokenization for LaTeX documents, essential for accurate n-gram scoring and neural processing.

## Overview

Unlike plain text tokenization, LaTeX requires understanding of:
- Command syntax (`\command{arg}`)
- Math mode delimiters (`$`, `$$`, `\[`, `\]`)
- Environment boundaries (`\begin{...}`, `\end{...}`)
- Special characters (`%`, `&`, `#`)

## Token Types

```rust
pub enum LaTeXTokenKind {
    /// LaTeX command: \alpha, \frac, \textbf
    Command,
    /// Environment name: equation, align, theorem
    Environment,
    /// Opening delimiter: {, [, (
    OpenDelim,
    /// Closing delimiter: }, ], )
    CloseDelim,
    /// Math mode: $, $$, \[, \]
    MathDelim,
    /// Regular text
    Text,
    /// Whitespace
    Whitespace,
    /// Comment (% to end of line)
    Comment,
    /// Parameter: #1, #2
    Parameter,
    /// Special character: &, ~
    Special,
}
```

## Mode Detection

The tokenizer tracks the current mode to enable mode-specific processing:

```rust
pub enum TokenMode {
    /// Regular text mode
    Text,
    /// Inline math ($...$)
    InlineMath,
    /// Display math ($$...$$, \[...\])
    DisplayMath,
    /// Inside a command
    Command,
    /// Inside an environment argument
    EnvArg,
}
```

## Basic Usage

```rust
use libgrammstein::latex::{LaTeXTokenizer, LaTeXToken};

let tokenizer = LaTeXTokenizer::new();
let input = r"The equation $\alpha + \beta = \gamma$ is fundamental.";

let tokens = tokenizer.tokenize(input);

for token in &tokens {
    println!("{:?}: '{}' (mode: {:?})",
        token.kind, token.text, token.mode);
}
```

Output:

```
Text: 'The equation ' (mode: Text)
MathDelim: '$' (mode: Text)
Command: '\alpha' (mode: InlineMath)
Text: ' + ' (mode: InlineMath)
Command: '\beta' (mode: InlineMath)
Text: ' = ' (mode: InlineMath)
Command: '\gamma' (mode: InlineMath)
MathDelim: '$' (mode: InlineMath)
Text: ' is fundamental.' (mode: Text)
```

## Streaming Tokenization

For large documents, use the streaming iterator:

```rust
let tokenizer = LaTeXTokenizer::new();
let mut stream = tokenizer.stream(input);

while let Some(token) = stream.next() {
    process_token(token);
}
```

## Token Structure

```rust
pub struct LaTeXToken {
    /// The token text
    pub text: String,
    /// Token kind
    pub kind: LaTeXTokenKind,
    /// Current mode
    pub mode: TokenMode,
    /// Start position in source
    pub start: usize,
    /// End position in source
    pub end: usize,
    /// Depth (for nested structures)
    pub depth: usize,
}
```

## Command Parsing

Commands are parsed with their structure:

```rust
let input = r"\frac{1}{2}";
let tokens = tokenizer.tokenize(input);

// Produces:
// Command: '\frac'
// OpenDelim: '{'
// Text: '1'
// CloseDelim: '}'
// OpenDelim: '{'
// Text: '2'
// CloseDelim: '}'
```

## Environment Recognition

Environments are tracked for proper mode handling:

```rust
let input = r"\begin{equation}
    x^2 + y^2 = z^2
\end{equation}";

let tokens = tokenizer.tokenize(input);

// Mode changes:
// \begin{equation} -> switches to DisplayMath
// x^2 + y^2 = z^2 -> tokenized in DisplayMath mode
// \end{equation} -> switches back to Text
```

## Math Mode Detection

The tokenizer correctly handles different math delimiters:

| Delimiter | Mode | Matching |
|-----------|------|----------|
| `$...$` | InlineMath | Single dollar |
| `$$...$$` | DisplayMath | Double dollar |
| `\(...\)` | InlineMath | Escaped parens |
| `\[...\]` | DisplayMath | Escaped brackets |
| `\begin{equation}` | DisplayMath | Environment |
| `\begin{align}` | DisplayMath | Environment |

## Configuration

```rust
pub struct TokenizerConfig {
    /// Preserve whitespace tokens
    pub preserve_whitespace: bool,
    /// Preserve comment tokens
    pub preserve_comments: bool,
    /// Track nesting depth
    pub track_depth: bool,
    /// Maximum nesting depth (0 = unlimited)
    pub max_depth: usize,
}

let config = TokenizerConfig {
    preserve_whitespace: false,
    preserve_comments: false,
    ..Default::default()
};

let tokenizer = LaTeXTokenizer::with_config(config);
```

## Token Normalization

For n-gram training, tokens can be normalized:

```rust
impl LaTeXToken {
    /// Normalize token for n-gram
    pub fn normalize(&self) -> String {
        match self.kind {
            // Commands: lowercase, remove backslash
            LaTeXTokenKind::Command => self.text[1..].to_lowercase(),
            // Text: lowercase
            LaTeXTokenKind::Text => self.text.to_lowercase(),
            // Others: as-is
            _ => self.text.clone(),
        }
    }
}
```

## Special Handling

### Verbatim Content

Content inside `\verb` or verbatim environments is passed through:

```rust
let input = r"\verb|x = y|";
// Produces single verbatim token, not parsed internally
```

### Ligatures and Special Characters

```rust
// TeX ligatures
"``" -> OpenQuote
"''" -> CloseQuote
"--" -> EnDash
"---" -> EmDash
```

### Escape Sequences

```rust
// Special characters
r"\%" -> PercentLiteral
r"\$" -> DollarLiteral
r"\&" -> AmpersandLiteral
r"\#" -> HashLiteral
```

## Integration with N-gram Model

The tokenizer output feeds directly into the n-gram scorer:

```rust
let tokens = tokenizer.tokenize(document);
let ngram_input: Vec<String> = tokens
    .iter()
    .filter(|t| t.kind != LaTeXTokenKind::Whitespace)
    .map(|t| t.normalize())
    .collect();

let score = ngram_model.score(&ngram_input)?;
```

## Performance

| Document Size | Tokenization Time |
|---------------|-------------------|
| 1 KB | 0.05ms |
| 10 KB | 0.3ms |
| 100 KB | 2.5ms |
| 1 MB | 25ms |

## Error Recovery

The tokenizer handles malformed input gracefully:

```rust
// Unclosed math
let input = r"$\alpha + \beta";  // Missing closing $
// Tokenizer completes with implicit close

// Mismatched braces
let input = r"\frac{1}{2";  // Missing closing }
// Tokenizer marks as error but continues
```

## Related

- [N-gram Models](./ngram.md): Using tokens for scoring
- [Embeddings](./embedding.md): Token embeddings
- [Overview](./overview.md): Module architecture
