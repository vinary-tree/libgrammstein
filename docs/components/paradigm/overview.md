# Paradigm Detection Overview

libgrammstein provides tools for detecting programming paradigms in source code through pattern matching and indicator analysis.

## What is Paradigm Detection?

Programming paradigms are fundamental styles of organizing and writing code. Different paradigms emphasize different programming constructs and patterns. libgrammstein can identify which paradigms are present in a codebase and to what degree.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Paradigm Detection Pipeline                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Source Code                                                            │
│       │                                                                  │
│       ▼                                                                  │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                  Pattern Matching Engine                         │   │
│   │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────┐        │   │
│   │  │   OOP   │  │   FP    │  │Reactive │  │ Procedural  │        │   │
│   │  │Patterns │  │Patterns │  │Patterns │  │  Patterns   │        │   │
│   │  └────┬────┘  └────┬────┘  └────┬────┘  └──────┬──────┘        │   │
│   │       └────────────┴────────────┴──────────────┘               │   │
│   └───────────────────────────────┬─────────────────────────────────┘   │
│                                   │                                      │
│                                   ▼                                      │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                  Indicator Extraction                            │   │
│   │  • Match location and category                                   │   │
│   │  • Paradigm attribution                                          │   │
│   │  • Confidence scoring                                            │   │
│   └───────────────────────────────┬─────────────────────────────────┘   │
│                                   │                                      │
│                                   ▼                                      │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                  Paradigm Profile                                │   │
│   │  • OOP Score: 0.45                                               │   │
│   │  • FP Score: 0.35                                                │   │
│   │  • Reactive Score: 0.15                                          │   │
│   │  • Procedural Score: 0.05                                        │   │
│   │  • Dominant: ObjectOriented                                      │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Supported Paradigms

libgrammstein detects five paradigm classifications:

| Paradigm | Description | Key Indicators |
|----------|-------------|----------------|
| **ObjectOriented** | Organizes code around objects and classes | Classes, inheritance, encapsulation |
| **Functional** | Emphasizes pure functions and immutability | Higher-order functions, map/filter/reduce |
| **Reactive** | Event-driven with observable streams | Observables, subscribers, event handlers |
| **Procedural** | Sequential execution with procedures | Functions, loops, mutable state |
| **Mixed** | Multiple paradigms in balance | No dominant style |

## Core Types

### Paradigm

The fundamental paradigm classification:

```rust
pub enum Paradigm {
    ObjectOriented,
    Functional,
    Reactive,
    Procedural,
    Mixed,
}
```

### ParadigmProfile

Captures the paradigm composition of analyzed code:

```rust
pub struct ParadigmProfile {
    /// OOP paradigm score (0.0 to 1.0)
    pub oop_score: f64,
    /// Functional paradigm score (0.0 to 1.0)
    pub fp_score: f64,
    /// Reactive paradigm score (0.0 to 1.0)
    pub reactive_score: f64,
    /// Procedural paradigm score (0.0 to 1.0)
    pub procedural_score: f64,
    /// Individual indicator matches
    pub indicators: Vec<ParadigmIndicator>,
}
```

## Quick Start

### Basic Paradigm Detection

```rust
use libgrammstein::topic::paradigm::{ParadigmDetector, ParadigmConfig, Paradigm};

// Create detector with default configuration
let detector = ParadigmDetector::new(ParadigmConfig::default());

// Analyze source code
let code = r#"
class UserService {
    constructor(private repository: UserRepository) {}

    async getUser(id: string): Promise<User> {
        return this.repository.findById(id);
    }
}
"#;

let profile = detector.analyze(code);

// Check paradigm scores
println!("OOP Score: {:.2}", profile.oop_score);
println!("FP Score: {:.2}", profile.fp_score);

// Get dominant paradigm
match profile.dominant_paradigm() {
    Paradigm::ObjectOriented => println!("This code follows OOP principles"),
    Paradigm::Functional => println!("This code follows FP principles"),
    Paradigm::Mixed => println!("This code uses multiple paradigms"),
    _ => {}
}
```

### Examining Indicators

```rust
// Get detailed indicators
for indicator in &profile.indicators {
    println!("Found {} pattern at position {}",
             indicator.category, indicator.start);
}

// Filter by paradigm
let oop_indicators: Vec<_> = profile.indicators.iter()
    .filter(|i| i.paradigm() == Paradigm::ObjectOriented)
    .collect();

println!("Found {} OOP indicators", oop_indicators.len());
```

## Paradigm Characteristics

### Object-Oriented Programming (OOP)

OOP organizes code around objects that combine data and behavior:

```rust
// High OOP indicators
class Rectangle {
    private width: number;
    private height: number;

    constructor(width: number, height: number) {
        this.width = width;
        this.height = height;
    }

    getArea(): number {
        return this.width * this.height;
    }
}

class Square extends Rectangle {
    constructor(side: number) {
        super(side, side);
    }
}
```

Key indicators detected:
- Class definitions
- Inheritance (`extends`, `implements`)
- Encapsulation (`private`, `protected`, `public`)
- Constructor patterns
- Method definitions

### Functional Programming (FP)

FP emphasizes pure functions and data transformations:

```rust
// High FP indicators
const numbers = [1, 2, 3, 4, 5];

const doubled = numbers.map(x => x * 2);
const evens = numbers.filter(x => x % 2 === 0);
const sum = numbers.reduce((a, b) => a + b, 0);

const compose = (f, g) => x => f(g(x));
const pipe = (...fns) => x => fns.reduce((v, f) => f(v), x);
```

Key indicators detected:
- Higher-order functions (`map`, `filter`, `reduce`)
- Lambda expressions
- Function composition
- Immutability patterns
- Currying

### Reactive Programming

Reactive code responds to data streams and changes:

```rust
// High Reactive indicators
const clicks$ = fromEvent(button, 'click');

clicks$
    .pipe(
        debounceTime(300),
        map(event => event.target.value),
        filter(value => value.length > 2)
    )
    .subscribe(value => console.log(value));
```

Key indicators detected:
- Observable patterns
- Subscription handling
- Event streams
- Reactive operators

### Procedural Programming

Procedural code follows sequential execution:

```rust
// High Procedural indicators
fn calculate_total(items: &[Item]) -> f64 {
    let mut total = 0.0;

    for item in items {
        if item.is_taxable {
            total += item.price * 1.1;
        } else {
            total += item.price;
        }
    }

    total
}
```

Key indicators detected:
- Mutable variables
- Sequential loops
- Control flow statements
- Function calls without objects

## Use Cases

### Code Quality Analysis

Identify paradigm consistency across a codebase:

```rust
fn analyze_codebase(files: &[SourceFile]) -> ParadigmReport {
    let detector = ParadigmDetector::new(ParadigmConfig::default());

    let profiles: Vec<_> = files.iter()
        .map(|f| detector.analyze(&f.content))
        .collect();

    // Calculate average paradigm distribution
    let avg_oop = profiles.iter().map(|p| p.oop_score).sum::<f64>() / profiles.len() as f64;
    let avg_fp = profiles.iter().map(|p| p.fp_score).sum::<f64>() / profiles.len() as f64;

    // Identify files with mixed paradigms
    let mixed_files: Vec<_> = files.iter()
        .zip(&profiles)
        .filter(|(_, p)| p.dominant_paradigm() == Paradigm::Mixed)
        .collect();

    ParadigmReport { avg_oop, avg_fp, mixed_files }
}
```

### Style Guide Enforcement

Ensure code follows team conventions:

```rust
fn check_paradigm_compliance(code: &str, expected: Paradigm) -> ComplianceResult {
    let profile = detector.analyze(code);

    let is_compliant = match expected {
        Paradigm::ObjectOriented => profile.oop_score > 0.7,
        Paradigm::Functional => profile.fp_score > 0.7,
        _ => true,
    };

    ComplianceResult {
        is_compliant,
        dominant: profile.dominant_paradigm(),
        scores: profile,
    }
}
```

### Educational Tools

Help learners understand paradigm differences:

```rust
fn explain_paradigm(code: &str) -> String {
    let profile = detector.analyze(code);

    let mut explanation = String::new();

    if profile.oop_score > 0.3 {
        explanation.push_str("This code uses object-oriented patterns:\n");
        for ind in profile.indicators.iter().filter(|i| i.paradigm() == Paradigm::ObjectOriented) {
            explanation.push_str(&format!("  - {} at line {}\n", ind.category, ind.line));
        }
    }

    if profile.fp_score > 0.3 {
        explanation.push_str("This code uses functional patterns:\n");
        for ind in profile.indicators.iter().filter(|i| i.paradigm() == Paradigm::Functional) {
            explanation.push_str(&format!("  - {} at line {}\n", ind.category, ind.line));
        }
    }

    explanation
}
```

## Configuration

The detector can be configured for different analysis needs:

```rust
let config = ParadigmConfig {
    // Minimum score difference to declare a dominant paradigm
    dominance_threshold: 0.2,

    // Weight adjustments for different indicator categories
    category_weights: CategoryWeights::default(),

    // Language-specific pattern sets
    language_patterns: Some(LanguagePatterns::rust()),
};

let detector = ParadigmDetector::new(config);
```

## Related Components

- [Indicator Types](indicators.md) - Detailed indicator categories
- [Detection Algorithm](detection.md) - How detection works
- [API Patterns](api-patterns.md) - Mining API usage patterns
- [Domain Patterns](domain-patterns.md) - Rholang and MeTTa patterns
