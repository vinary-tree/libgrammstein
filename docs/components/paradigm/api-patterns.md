# API Pattern Mining

libgrammstein includes an API pattern mining system that discovers common sequences of API calls using the PrefixSpan algorithm.

## What is API Pattern Mining?

API pattern mining identifies frequently occurring sequences of function or method calls in codebases. These patterns reveal:

- Common usage patterns for libraries and frameworks
- Idiomatic code sequences
- Potential API design issues
- Opportunities for abstraction

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    API Pattern Mining Pipeline                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Source Code                                                            │
│       │                                                                  │
│       ▼                                                                  │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  1. Sequence Extraction                                          │   │
│   │     • Parse function bodies                                      │   │
│   │     • Extract API call sequences                                 │   │
│   │     • Build sequence database                                    │   │
│   └───────────────────────────────┬─────────────────────────────────┘   │
│                                   │                                      │
│                                   ▼                                      │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  Sequence Database                                               │   │
│   │  ["db.connect", "db.query", "db.close"]                         │   │
│   │  ["db.connect", "db.beginTransaction", "db.query", "db.commit"] │   │
│   │  ["fs.open", "fs.read", "fs.close"]                             │   │
│   │  ...                                                             │   │
│   └───────────────────────────────┬─────────────────────────────────┘   │
│                                   │                                      │
│                                   ▼                                      │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  2. PrefixSpan Mining                                            │   │
│   │     • Find frequent subsequences                                 │   │
│   │     • Apply minimum support threshold                            │   │
│   │     • Grow patterns prefix by prefix                             │   │
│   └───────────────────────────────┬─────────────────────────────────┘   │
│                                   │                                      │
│                                   ▼                                      │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  Frequent Patterns                                               │   │
│   │  ["db.connect", "db.query"] (support: 0.85)                     │   │
│   │  ["db.beginTransaction", "db.commit"] (support: 0.72)           │   │
│   │  ["db.connect", ..., "db.close"] (support: 0.68)                │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Core Types

### ApiPatternMiner

The main mining interface:

```rust
pub struct ApiPatternMiner {
    config: ApiPatternConfig,
}

impl ApiPatternMiner {
    /// Create a new miner with configuration
    pub fn new(config: ApiPatternConfig) -> Self;

    /// Mine patterns from a sequence database
    pub fn mine(&self, sequences: &[Vec<String>]) -> Vec<ApiPattern>;
}
```

### ApiPatternConfig

Configuration for the mining process:

```rust
pub struct ApiPatternConfig {
    /// Minimum support threshold (0.0 to 1.0)
    /// Patterns must appear in at least this fraction of sequences
    pub min_support: f64,

    /// Maximum pattern length
    pub max_length: usize,

    /// Minimum pattern length
    pub min_length: usize,

    /// Whether to allow gaps in patterns
    pub allow_gaps: bool,

    /// Maximum gap size (if gaps allowed)
    pub max_gap: usize,
}

impl Default for ApiPatternConfig {
    fn default() -> Self {
        Self {
            min_support: 0.1,    // 10% of sequences
            max_length: 10,
            min_length: 2,
            allow_gaps: true,
            max_gap: 3,
        }
    }
}
```

### ApiPattern

A discovered frequent pattern:

```rust
pub struct ApiPattern {
    /// The sequence of API calls
    pub sequence: Vec<String>,

    /// Support: fraction of sequences containing this pattern
    pub support: f64,

    /// Absolute count of occurrences
    pub count: usize,

    /// Positions where pattern occurs (sequence index, start position)
    pub occurrences: Vec<(usize, usize)>,
}
```

## Quick Start

### Basic Pattern Mining

```rust
use libgrammstein::topic::paradigm::{ApiPatternMiner, ApiPatternConfig};

// Create miner with default configuration
let miner = ApiPatternMiner::new(ApiPatternConfig::default());

// Build sequence database from code analysis
let sequences = vec![
    vec!["db.connect", "db.query", "db.close"].into_iter().map(String::from).collect(),
    vec!["db.connect", "db.beginTransaction", "db.query", "db.commit", "db.close"].into_iter().map(String::from).collect(),
    vec!["db.connect", "db.query", "db.query", "db.close"].into_iter().map(String::from).collect(),
    vec!["fs.open", "fs.read", "fs.close"].into_iter().map(String::from).collect(),
];

// Mine frequent patterns
let patterns = miner.mine(&sequences);

for pattern in patterns {
    println!("Pattern: {:?}", pattern.sequence);
    println!("  Support: {:.1}%", pattern.support * 100.0);
    println!("  Count: {}", pattern.count);
}
```

Output:
```
Pattern: ["db.connect", "db.close"]
  Support: 75.0%
  Count: 3

Pattern: ["db.connect", "db.query"]
  Support: 75.0%
  Count: 3

Pattern: ["db.connect", "db.query", "db.close"]
  Support: 75.0%
  Count: 3
```

### Extracting Sequences from Code

```rust
use libcpg::{CodePropertyGraph, TreeSitterCpgBuilder, Language};

fn extract_api_sequences(cpg: &CodePropertyGraph) -> Vec<Vec<String>> {
    let mut sequences = Vec::new();

    for func in cpg.functions() {
        let mut calls = Vec::new();

        // Get all call nodes in function
        for node_id in cpg.ast_descendants(func.id()) {
            if let Some(node) = cpg.node(node_id) {
                if matches!(node.kind(), CpgNodeKind::Call) {
                    if let Some(name) = node.name() {
                        calls.push(name.to_string());
                    }
                }
            }
        }

        if calls.len() >= 2 {
            sequences.push(calls);
        }
    }

    sequences
}

// Usage
let builder = TreeSitterCpgBuilder::new();
let cpg = builder.build(source_code, Language::Rust)?;
let sequences = extract_api_sequences(&cpg);
let patterns = miner.mine(&sequences);
```

## The PrefixSpan Algorithm

PrefixSpan (Prefix-projected Sequential pattern mining) efficiently finds frequent subsequences by:

1. **Finding frequent items**: Scan database for items meeting min_support
2. **Prefix projection**: For each frequent item, project the database
3. **Recursive mining**: Mine projected databases for extensions
4. **Pattern growth**: Grow patterns prefix by prefix

### Algorithm Walkthrough

```
Initial Database:
  S1: [a, b, c, d]
  S2: [a, c, d]
  S3: [a, b, d]
  S4: [b, c, d]

Step 1: Find frequent 1-sequences (min_support = 0.5)
  a: 3/4 = 0.75 ✓
  b: 3/4 = 0.75 ✓
  c: 3/4 = 0.75 ✓
  d: 4/4 = 1.00 ✓

Step 2: Project database by prefix 'a'
  S1|a: [b, c, d]  (suffix after first 'a')
  S2|a: [c, d]
  S3|a: [b, d]

Step 3: Mine projected database for prefix 'a'
  Find frequent items in S|a: b(2/3), c(2/3), d(3/3)
  Pattern [a, d] has support 3/4 = 0.75

Step 4: Continue recursively...
  [a, b, d]: support 2/4 = 0.50 ✓
  [a, c, d]: support 2/4 = 0.50 ✓
```

### Implementation Details

```rust
impl ApiPatternMiner {
    pub fn mine(&self, sequences: &[Vec<String>]) -> Vec<ApiPattern> {
        let n = sequences.len();
        if n == 0 {
            return Vec::new();
        }

        let min_count = (n as f64 * self.config.min_support).ceil() as usize;
        let mut patterns = Vec::new();

        // Find frequent 1-sequences
        let freq_items = self.find_frequent_items(sequences, min_count);

        // Mine patterns starting from each frequent item
        for item in freq_items {
            let prefix = vec![item.clone()];
            let projected = self.project_database(sequences, &prefix);

            if projected.len() >= min_count {
                patterns.push(ApiPattern {
                    sequence: prefix.clone(),
                    support: projected.len() as f64 / n as f64,
                    count: projected.len(),
                    occurrences: projected,
                });

                // Recursively extend prefix
                self.extend_pattern(
                    sequences,
                    &prefix,
                    &projected,
                    min_count,
                    &mut patterns,
                );
            }
        }

        patterns
    }

    fn extend_pattern(
        &self,
        sequences: &[Vec<String>],
        prefix: &[String],
        projected: &[(usize, usize)],
        min_count: usize,
        patterns: &mut Vec<ApiPattern>,
    ) {
        if prefix.len() >= self.config.max_length {
            return;
        }

        // Find frequent extensions
        let extensions = self.find_extensions(sequences, projected);

        for (item, new_projected) in extensions {
            if new_projected.len() >= min_count {
                let mut new_prefix = prefix.to_vec();
                new_prefix.push(item);

                patterns.push(ApiPattern {
                    sequence: new_prefix.clone(),
                    support: new_projected.len() as f64 / sequences.len() as f64,
                    count: new_projected.len(),
                    occurrences: new_projected.clone(),
                });

                // Continue extending
                self.extend_pattern(
                    sequences,
                    &new_prefix,
                    &new_projected,
                    min_count,
                    patterns,
                );
            }
        }
    }
}
```

## Configuration Options

### Support Threshold

The minimum fraction of sequences that must contain a pattern:

```rust
// High support: common patterns only
let config = ApiPatternConfig {
    min_support: 0.5,  // Pattern must appear in 50% of sequences
    ..Default::default()
};

// Low support: rare patterns too
let config = ApiPatternConfig {
    min_support: 0.05, // Pattern in 5% of sequences
    ..Default::default()
};
```

### Pattern Length

Control the size of discovered patterns:

```rust
let config = ApiPatternConfig {
    min_length: 3,  // At least 3 calls
    max_length: 8,  // At most 8 calls
    ..Default::default()
};
```

### Gap Handling

Allow non-contiguous patterns:

```rust
// Contiguous only: [a, b, c] matches "a, b, c" but not "a, x, b, c"
let config = ApiPatternConfig {
    allow_gaps: false,
    ..Default::default()
};

// Allow gaps: [a, b, c] matches "a, x, b, y, z, c"
let config = ApiPatternConfig {
    allow_gaps: true,
    max_gap: 2,  // At most 2 items between pattern elements
    ..Default::default()
};
```

## Use Cases

### Library Usage Analysis

Discover how developers use a library:

```rust
fn analyze_library_usage(codebase: &[SourceFile], library: &str) -> Vec<ApiPattern> {
    let miner = ApiPatternMiner::new(ApiPatternConfig {
        min_support: 0.1,
        min_length: 2,
        max_length: 6,
        ..Default::default()
    });

    let sequences: Vec<Vec<String>> = codebase.iter()
        .flat_map(|file| extract_api_sequences(&file.cpg))
        .filter(|seq| seq.iter().any(|call| call.starts_with(library)))
        .collect();

    miner.mine(&sequences)
}

// Usage
let patterns = analyze_library_usage(&codebase, "React.");
for pattern in patterns {
    println!("{:?} (used in {:.0}% of components)",
             pattern.sequence, pattern.support * 100.0);
}
```

### Anti-Pattern Detection

Find common but problematic patterns:

```rust
// Known anti-patterns
let anti_patterns = vec![
    vec!["db.query", "db.query"],  // Multiple queries without transaction
    vec!["file.open", "file.read"],  // No close after open
];

fn detect_anti_patterns(
    mined: &[ApiPattern],
    anti_patterns: &[Vec<&str>],
) -> Vec<(&ApiPattern, &[&str])> {
    mined.iter()
        .filter_map(|pattern| {
            for anti in anti_patterns {
                if is_subsequence(anti, &pattern.sequence) {
                    return Some((pattern, anti.as_slice()));
                }
            }
            None
        })
        .collect()
}
```

### Framework Idiom Discovery

Learn idiomatic patterns from well-written code:

```rust
fn discover_idioms(exemplar_code: &[SourceFile]) -> Vec<ApiPattern> {
    let miner = ApiPatternMiner::new(ApiPatternConfig {
        min_support: 0.3,  // Common in exemplar code
        min_length: 3,
        ..Default::default()
    });

    let sequences = exemplar_code.iter()
        .flat_map(|f| extract_api_sequences(&f.cpg))
        .collect::<Vec<_>>();

    miner.mine(&sequences)
}

// Document discovered idioms
for pattern in discover_idioms(&exemplar_code) {
    println!("Idiom: {}", pattern.sequence.join(" -> "));
    println!("Usage: {:.0}% of exemplar code", pattern.support * 100.0);
}
```

### API Evolution Tracking

Track how API usage changes across versions:

```rust
fn compare_api_usage(
    old_code: &[SourceFile],
    new_code: &[SourceFile],
) -> ApiEvolution {
    let miner = ApiPatternMiner::new(ApiPatternConfig::default());

    let old_patterns = miner.mine(&extract_all_sequences(old_code));
    let new_patterns = miner.mine(&extract_all_sequences(new_code));

    let old_set: HashSet<_> = old_patterns.iter()
        .map(|p| &p.sequence)
        .collect();
    let new_set: HashSet<_> = new_patterns.iter()
        .map(|p| &p.sequence)
        .collect();

    ApiEvolution {
        deprecated: old_set.difference(&new_set).cloned().collect(),
        new_patterns: new_set.difference(&old_set).cloned().collect(),
        stable: old_set.intersection(&new_set).cloned().collect(),
    }
}
```

## Performance Considerations

### Sequence Database Size

Mining time increases with database size:

```rust
// For large codebases, sample or partition
fn sample_sequences(sequences: &[Vec<String>], sample_rate: f64) -> Vec<Vec<String>> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    sequences.iter()
        .filter(|_| rng.gen::<f64>() < sample_rate)
        .cloned()
        .collect()
}
```

### Pattern Explosion

Low support thresholds can produce many patterns:

```rust
// Start with high support, lower if needed
let mut config = ApiPatternConfig {
    min_support: 0.5,
    ..Default::default()
};

let patterns = miner.mine(&sequences);

if patterns.len() < 10 {
    config.min_support = 0.2;
    let patterns = miner.mine(&sequences);
}
```

### Memory Usage

Projected databases can be large:

```rust
// Use indices instead of copying sequences
struct ProjectedDb {
    original: Arc<Vec<Vec<String>>>,
    indices: Vec<(usize, usize)>,  // (sequence_idx, position)
}
```

## See Also

- [Overview](overview.md) - Paradigm detection introduction
- [Detection](detection.md) - Paradigm detector usage
- [Indicators](indicators.md) - Indicator types and categories
- [Domain Patterns](domain-patterns.md) - Rholang and MeTTa patterns
