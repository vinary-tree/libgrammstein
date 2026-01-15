# Correction Pipeline

The correction pipeline provides an end-to-end workflow for analyzing and correcting source code, integrating parsing, tokenization, semantic analysis, and multi-source correction.

## Overview

The `CorrectionPipeline` orchestrates:

1. **Parse**: Tree-sitter parsing with error recovery
2. **Tokenize**: Extract tokens with context information
3. **Analyze**: Build CPG for semantic analysis (optional)
4. **Correct**: Apply ensemble of correctors
5. **Rank**: Combine and rank correction candidates

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                       CorrectionPipeline                             │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                          Input: Source Code                     │ │
│  └───────────────────────────────┬────────────────────────────────┘ │
│                                  │                                   │
│                                  ▼                                   │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ Phase 1: Parse                                                  │ │
│  │ ┌──────────────────────────────────────────────────────────┐   │ │
│  │ │ CodeParser (Tree-sitter) → ParsedCode + Error Regions    │   │ │
│  │ └──────────────────────────────────────────────────────────┘   │ │
│  └───────────────────────────────┬────────────────────────────────┘ │
│                                  │                                   │
│                                  ▼                                   │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ Phase 2: Tokenize                                               │ │
│  │ ┌──────────────────────────────────────────────────────────┐   │ │
│  │ │ CodeTokenizer → Tokens with Context                       │   │ │
│  │ └──────────────────────────────────────────────────────────┘   │ │
│  └───────────────────────────────┬────────────────────────────────┘ │
│                                  │                                   │
│                                  ▼                                   │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ Phase 3: Analyze (optional)                                     │ │
│  │ ┌──────────────────────────────────────────────────────────┐   │ │
│  │ │ CodePropertyGraph (AST + CFG + DFG)                       │   │ │
│  │ └──────────────────────────────────────────────────────────┘   │ │
│  └───────────────────────────────┬────────────────────────────────┘ │
│                                  │                                   │
│                                  ▼                                   │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ Phase 4: Correct                                                │ │
│  │ ┌──────────────────────────────────────────────────────────┐   │ │
│  │ │ EnsembleCorrector (Lexical + Grammar + Semantic)          │   │ │
│  │ └──────────────────────────────────────────────────────────┘   │ │
│  └───────────────────────────────┬────────────────────────────────┘ │
│                                  │                                   │
│                                  ▼                                   │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ Phase 5: Rank & Filter                                          │ │
│  │ ┌──────────────────────────────────────────────────────────┐   │ │
│  │ │ Deduplicate → Sort by confidence → Truncate              │   │ │
│  │ └──────────────────────────────────────────────────────────┘   │ │
│  └───────────────────────────────┬────────────────────────────────┘ │
│                                  │                                   │
│                                  ▼                                   │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    Output: AnalysisResult                       │ │
│  │  • source       • corrections   • diagnostics                   │ │
│  │  • tokens       • error_count   • has_parse_errors              │ │
│  └────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

## PipelineConfig

Configuration options for the correction pipeline:

```rust
pub struct PipelineConfig {
    /// Maximum corrections to return per file (default: 50)
    pub max_corrections: usize,
    /// Minimum confidence threshold (default: 0.3)
    pub min_confidence: f64,
    /// Whether to include diagnostic messages (default: true)
    pub include_diagnostics: bool,
    /// Threshold for auto-applying fixes (default: None)
    pub auto_apply_threshold: Option<f64>,
    /// Whether to do full semantic analysis (default: true)
    pub full_semantic_analysis: bool,
}
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_corrections` | 50 | Maximum suggestions per file |
| `min_confidence` | 0.3 | Filter low-confidence corrections |
| `include_diagnostics` | true | Generate diagnostic messages |
| `auto_apply_threshold` | None | Auto-apply above this confidence |
| `full_semantic_analysis` | true | Build CPG for semantic checks |

## Creating a Pipeline

### Basic Creation

```rust
use libgrammstein::code::{CorrectionPipeline, PipelineConfig, Python};
use std::sync::Arc;

let python = Arc::new(Python::new());

// With default configuration
let pipeline = CorrectionPipeline::with_defaults(python.clone(), None)?;

// With grammar for syntax checking
let grammar = build_python_grammar();
let pipeline = CorrectionPipeline::with_defaults(python, Some(grammar))?;
```

### Custom Configuration

```rust
let config = PipelineConfig {
    max_corrections: 20,
    min_confidence: 0.5,
    include_diagnostics: true,
    auto_apply_threshold: Some(0.9),  // Auto-apply very confident fixes
    full_semantic_analysis: true,
};

let pipeline = CorrectionPipeline::new(python, Some(grammar), config)?;
```

### Minimal Pipeline

For fast analysis without semantic checks:

```rust
// Lexical-only, no CPG construction
let pipeline = CorrectionPipeline::minimal(python)?;
```

## AnalysisResult

The result of analyzing source code:

```rust
pub struct AnalysisResult {
    /// Original source code
    pub source: String,
    /// Whether parsing produced any errors
    pub has_parse_errors: bool,
    /// Number of parse errors found
    pub error_count: usize,
    /// Tokens extracted from source
    pub tokens: Vec<CodeToken>,
    /// Ranked corrections
    pub corrections: CorrectionCandidates,
    /// Diagnostic messages
    pub diagnostics: Vec<Diagnostic>,
}
```

### Accessing Results

```rust
let result = pipeline.analyze(source)?;

// Check for parse errors
if result.has_parse_errors {
    println!("Found {} parse errors", result.error_count);
}

// Get best correction
if let Some(best) = result.corrections.best() {
    println!("Top suggestion: {} → {} ({:.0}%)",
        best.original, best.replacement, best.confidence * 100.0);
}

// Iterate all corrections
for correction in result.corrections.ranked() {
    println!("  {} → {} ({:.2})",
        correction.original, correction.replacement, correction.confidence);
}

// Access diagnostics
for diagnostic in &result.diagnostics {
    println!("[{:?}] Line {}: {}",
        diagnostic.severity, diagnostic.line + 1, diagnostic.message);
}
```

## Diagnostic

Diagnostic messages from analysis:

```rust
pub struct Diagnostic {
    /// Severity level
    pub severity: DiagnosticSeverity,
    /// Message text
    pub message: String,
    /// Start byte offset
    pub start_byte: usize,
    /// End byte offset
    pub end_byte: usize,
    /// Line number (0-indexed)
    pub line: usize,
    /// Column number (0-indexed)
    pub column: usize,
}
```

### DiagnosticSeverity

```rust
pub enum DiagnosticSeverity {
    Error,    // Prevents compilation/execution
    Warning,  // Potential issues
    Info,     // Informational
    Hint,     // Suggestions for improvement
}
```

### Diagnostic Examples

```rust
// Parse error diagnostic
Diagnostic {
    severity: DiagnosticSeverity::Error,
    message: "Syntax error: ERROR 'retrun'",
    start_byte: 20,
    end_byte: 26,
    line: 1,
    column: 4,
}

// Correction hint diagnostic
Diagnostic {
    severity: DiagnosticSeverity::Hint,
    message: "Consider: retrun -> return",
    start_byte: 20,
    end_byte: 26,
    line: 1,
    column: 4,
}
```

## Analyzing Code

### Basic Analysis

```rust
let mut pipeline = CorrectionPipeline::with_defaults(python, None)?;

let source = r#"
def calculate(x, y):
    retrun x + y
"#;

let result = pipeline.analyze(source)?;

println!("Parse errors: {}", result.error_count);
println!("Corrections available: {}", result.corrections.len());
```

### With Project Context

Add project-specific identifiers for better corrections:

```rust
let mut pipeline = CorrectionPipeline::with_defaults(python, None)?;

// Add project identifiers to the corrector
pipeline.corrector_mut().add_identifiers(&[
    "calculateTotal",
    "processUserData",
    "handleNetworkError",
]);

// Register known variables
pipeline.corrector_mut().register_variables(&[
    ("userCount".to_string(), Some("int".to_string())),
    ("userName".to_string(), Some("string".to_string())),
]);

let result = pipeline.analyze(source)?;
```

## Applying Corrections

### Apply All Corrections

```rust
let result = pipeline.analyze(source)?;

// Get all high-confidence corrections
let corrections: Vec<_> = result.corrections.ranked()
    .iter()
    .filter(|c| c.confidence >= 0.7)
    .cloned()
    .collect();

// Apply to source
let fixed_source = pipeline.apply_corrections(source, &corrections);
println!("Fixed:\n{}", fixed_source);
```

### Apply Best Correction Only

```rust
let result = pipeline.analyze(source)?;

if let Some(best) = result.corrections.best() {
    let fixed = pipeline.apply_corrections(source, &[best.clone()]);
    println!("After applying best fix:\n{}", fixed);
}
```

### Apply Corrections Above Threshold

```rust
let config = PipelineConfig {
    auto_apply_threshold: Some(0.9),
    ..Default::default()
};

let mut pipeline = CorrectionPipeline::new(python, None, config)?;
let result = pipeline.analyze(source)?;

// Get auto-applicable corrections
let auto_apply: Vec<_> = result.corrections.ranked()
    .iter()
    .filter(|c| c.confidence >= 0.9)
    .cloned()
    .collect();

if !auto_apply.is_empty() {
    let fixed = pipeline.apply_corrections(source, &auto_apply);
    println!("Auto-fixed {} issues", auto_apply.len());
}
```

## Pipeline Phases

### Phase 1: Parse

The pipeline uses `CodeParser` with tree-sitter for error-tolerant parsing:

```rust
// Internal: Phase 1
let parsed = self.parser.parse(source)?;

// Access errors
for error in parsed.errors() {
    println!("Error at line {}: {}", error.start_position.0, error.text);
}
```

### Phase 2: Tokenize

Extract tokens with context information:

```rust
// Internal: Phase 2
let tokenizer = CodeTokenizer::new(&*self.language);
let tokens = tokenizer.tokenize(&parsed.tree, &parsed.source);
```

### Phase 3: Analyze (CPG)

Build Code Property Graph for semantic analysis:

```rust
// Internal: Phase 3 (if full_semantic_analysis is true)
let cpg = CodePropertyGraph::from_parsed_code(&parsed);
```

### Phase 4: Correct

Apply ensemble corrector to tokens:

```rust
// Internal: Phase 4
for token in &tokens {
    let context = TokenContext::new(token.token_type);
    let corrections = self.corrector.correct_token(token, &context);
    all_corrections.extend(corrections);
}

// Semantic corrections from CPG
if let Some(ref cpg) = cpg {
    let semantic = self.corrector.analyze_full(&parsed, cpg);
    all_corrections.extend(semantic);
}
```

### Phase 5: Rank

Filter, deduplicate, and sort corrections:

```rust
// Internal: Phase 5
corrections.retain(|c| c.confidence >= self.config.min_confidence);
corrections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

// Deduplicate by (position, replacement)
let mut seen = HashSet::new();
corrections.retain(|c| {
    let key = (c.start_byte, c.end_byte, c.replacement.clone());
    seen.insert(key)
});

corrections.truncate(self.config.max_corrections);
```

## Error Handling

### PipelineError

Errors that can occur during pipeline execution:

```rust
pub enum PipelineError {
    ParseError(String),      // Tree-sitter parsing failed
    TokenizeError(String),   // Tokenization failed
    CpgError(String),        // CPG construction failed
    CorrectionError(String), // Correction failed
    IoError(std::io::Error), // I/O error
}
```

### Handling Errors

```rust
use libgrammstein::code::PipelineError;

match pipeline.analyze(source) {
    Ok(result) => {
        println!("Analysis complete: {} corrections", result.corrections.len());
    }
    Err(PipelineError::ParseError(msg)) => {
        eprintln!("Parse failed: {}", msg);
    }
    Err(PipelineError::CpgError(msg)) => {
        eprintln!("CPG construction failed: {}", msg);
    }
    Err(e) => {
        eprintln!("Pipeline error: {}", e);
    }
}
```

## Integration Example

Complete example using the pipeline:

```rust
use libgrammstein::code::{
    CorrectionPipeline, PipelineConfig, Python, DiagnosticSeverity
};
use std::sync::Arc;

fn analyze_and_fix(source: &str) -> Result<String, Box<dyn std::error::Error>> {
    let python = Arc::new(Python::new());

    // Create pipeline with custom config
    let config = PipelineConfig {
        max_corrections: 20,
        min_confidence: 0.5,
        include_diagnostics: true,
        auto_apply_threshold: Some(0.85),
        full_semantic_analysis: true,
    };

    let mut pipeline = CorrectionPipeline::new(python, None, config)?;

    // Add project context
    pipeline.corrector_mut().add_identifiers(&[
        "calculate_total", "process_data", "handle_error"
    ]);

    // Analyze
    let result = pipeline.analyze(source)?;

    // Report diagnostics
    println!("=== Diagnostics ===");
    for diag in &result.diagnostics {
        let prefix = match diag.severity {
            DiagnosticSeverity::Error => "ERROR",
            DiagnosticSeverity::Warning => "WARN",
            DiagnosticSeverity::Info => "INFO",
            DiagnosticSeverity::Hint => "HINT",
        };
        println!("[{}] Line {}: {}", prefix, diag.line + 1, diag.message);
    }

    // Report corrections
    println!("\n=== Corrections ===");
    for correction in result.corrections.ranked() {
        println!("  {} → {} (confidence: {:.0}%)",
            correction.original,
            correction.replacement,
            correction.confidence * 100.0
        );
    }

    // Apply high-confidence corrections
    let to_apply: Vec<_> = result.corrections.ranked()
        .iter()
        .filter(|c| c.confidence >= 0.85)
        .cloned()
        .collect();

    let fixed = pipeline.apply_corrections(source, &to_apply);

    println!("\n=== Fixed Source ===");
    println!("{}", fixed);

    Ok(fixed)
}

fn main() {
    let source = r#"
def calculate(x, y):
    reuslt = x + y
    retrun reuslt
"#;

    match analyze_and_fix(source) {
        Ok(fixed) => println!("Success!"),
        Err(e) => eprintln!("Error: {}", e),
    }
}
```

## Performance Considerations

### Phase Timing

| Phase | Complexity | Notes |
|-------|------------|-------|
| Parse | O(n) | Linear in source length |
| Tokenize | O(t) | t = number of tokens |
| CPG Build | O(n + e) | n = nodes, e = edges |
| Correct | O(t × c) | c = correction candidates |
| Rank | O(m log m) | m = total corrections |

### Optimization Tips

1. **Use `minimal()` for speed**: Skip CPG construction
2. **Increase `min_confidence`**: Reduce candidate processing
3. **Decrease `max_corrections`**: Limit sorting overhead
4. **Disable `full_semantic_analysis`**: Skip CPG when not needed

### Minimal vs Full Pipeline

| Feature | Minimal | Full |
|---------|---------|------|
| Lexical corrections | Yes | Yes |
| Grammar corrections | No | With grammar |
| Semantic corrections | No | Yes |
| CPG construction | No | Yes |
| Speed | Fast | Slower |
| Accuracy | Good | Best |

## Thread Safety

The pipeline is not `Sync` due to mutable parser state, but can be used across threads with proper synchronization:

```rust
use std::sync::Mutex;

let pipeline = Mutex::new(CorrectionPipeline::with_defaults(python, None)?);

// Lock for analysis
{
    let mut p = pipeline.lock().unwrap();
    let result = p.analyze(source)?;
}
```

For parallel analysis of multiple files, create separate pipeline instances:

```rust
use rayon::prelude::*;

let sources: Vec<&str> = vec![...];

let results: Vec<_> = sources.par_iter()
    .map(|source| {
        let python = Arc::new(Python::new());
        let mut pipeline = CorrectionPipeline::minimal(python).unwrap();
        pipeline.analyze(source)
    })
    .collect();
```

## See Also

- [Correctors Overview](correctors/overview.md) - Correction architecture
- [Ensemble Corrector](correctors/ensemble.md) - Multi-source correction
- [AST](ast.md) - Tree-sitter parsing
- [Tokenizer](tokenizer.md) - Token extraction
- [CPG](cpg.md) - Code Property Graphs
- [Correction Framework](correction.md) - Correction types
