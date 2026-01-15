# Paradigm Detection

The `ParadigmDetector` analyzes source code to identify programming paradigm patterns through lexical pattern matching.

## How Detection Works

The detector uses a pattern-based approach that scans source code for paradigm-indicative tokens and constructs:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Detection Algorithm                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Input: Source Code                                                     │
│       │                                                                  │
│       ▼                                                                  │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  1. Pattern Matching                                             │   │
│   │     • Scan for regex patterns per paradigm                       │   │
│   │     • Record match positions and categories                      │   │
│   └───────────────────────────────┬─────────────────────────────────┘   │
│                                   │                                      │
│                                   ▼                                      │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  2. Indicator Collection                                         │   │
│   │     • Create ParadigmIndicator for each match                    │   │
│   │     • Assign category and paradigm                               │   │
│   └───────────────────────────────┬─────────────────────────────────┘   │
│                                   │                                      │
│                                   ▼                                      │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  3. Score Calculation                                            │   │
│   │     • Weight indicators by category                              │   │
│   │     • Normalize scores to 0.0 - 1.0                              │   │
│   └───────────────────────────────┬─────────────────────────────────┘   │
│                                   │                                      │
│                                   ▼                                      │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  4. Profile Construction                                         │   │
│   │     • Combine scores into ParadigmProfile                        │   │
│   │     • Determine dominant paradigm                                │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│   Output: ParadigmProfile                                                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## ParadigmDetector

### Construction

```rust
use libgrammstein::topic::paradigm::{ParadigmDetector, ParadigmConfig};

// Default configuration
let detector = ParadigmDetector::new(ParadigmConfig::default());

// Custom configuration
let config = ParadigmConfig {
    dominance_threshold: 0.25,
    ..Default::default()
};
let detector = ParadigmDetector::new(config);
```

### Analyzing Code

The primary method is `analyze()`:

```rust
pub fn analyze(&self, code: &str) -> ParadigmProfile
```

This method:
1. Scans the code for all registered patterns
2. Creates indicators for each match
3. Calculates weighted scores
4. Returns a complete profile

### Example Analysis

```rust
let detector = ParadigmDetector::new(ParadigmConfig::default());

let typescript_code = r#"
interface Repository<T> {
    findById(id: string): Promise<T>;
    save(entity: T): Promise<T>;
}

class UserRepository implements Repository<User> {
    private cache: Map<string, User>;

    constructor(private db: Database) {
        this.cache = new Map();
    }

    async findById(id: string): Promise<User> {
        return this.cache.get(id) ?? await this.db.query<User>('users', id);
    }

    async save(user: User): Promise<User> {
        const saved = await this.db.insert('users', user);
        this.cache.set(saved.id, saved);
        return saved;
    }
}
"#;

let profile = detector.analyze(typescript_code);

println!("Paradigm Analysis:");
println!("  OOP: {:.1}%", profile.oop_score * 100.0);
println!("  FP: {:.1}%", profile.fp_score * 100.0);
println!("  Reactive: {:.1}%", profile.reactive_score * 100.0);
println!("  Procedural: {:.1}%", profile.procedural_score * 100.0);
println!("  Dominant: {:?}", profile.dominant_paradigm());
```

Output:
```
Paradigm Analysis:
  OOP: 72.3%
  FP: 15.2%
  Reactive: 8.1%
  Procedural: 4.4%
  Dominant: ObjectOriented
```

## Pattern Categories

The detector searches for patterns across multiple categories:

### OOP Patterns

| Pattern | Regex | Weight |
|---------|-------|--------|
| Class definition | `\bclass\s+\w+` | 1.0 |
| Interface | `\binterface\s+\w+` | 0.9 |
| Extends | `\bextends\s+\w+` | 0.8 |
| Implements | `\bimplements\s+\w+` | 0.8 |
| Constructor | `\bconstructor\s*\(` | 0.7 |
| Private/Public | `\b(private|public|protected)\b` | 0.5 |
| This reference | `\bthis\.` | 0.4 |
| New operator | `\bnew\s+\w+` | 0.6 |

### FP Patterns

| Pattern | Regex | Weight |
|---------|-------|--------|
| Map | `\.map\s*\(` | 0.8 |
| Filter | `\.filter\s*\(` | 0.8 |
| Reduce | `\.reduce\s*\(` | 0.9 |
| Lambda | `\|\w*\|\s*\{` (Rust) or `=>\s*\{?` (JS) | 0.7 |
| Const binding | `\bconst\s+\w+\s*=` | 0.4 |
| Pure function | `\bfn\s+\w+.*->\s*\w+` | 0.6 |
| Compose | `\bcompose\b` | 0.9 |
| Pipe | `\bpipe\b` or `\|>` | 0.9 |

### Reactive Patterns

| Pattern | Regex | Weight |
|---------|-------|--------|
| Observable | `\bObservable\b` | 1.0 |
| Subscribe | `\.subscribe\s*\(` | 0.9 |
| Subject | `\bSubject\b` | 0.9 |
| Event handler | `\.on\w+\s*=` or `addEventListener` | 0.7 |
| Stream | `\bStream\b` | 0.8 |
| Emit | `\.emit\s*\(` | 0.7 |

### Procedural Patterns

| Pattern | Regex | Weight |
|---------|-------|--------|
| For loop | `\bfor\s*\(` | 0.6 |
| While loop | `\bwhile\s*\(` | 0.6 |
| Mutable var | `\blet\s+mut\b` (Rust) or `\bvar\s+\w+` (JS) | 0.5 |
| If statement | `\bif\s*\(` | 0.3 |
| Function def | `\bfunction\s+\w+` | 0.4 |

## Scoring Algorithm

Scores are calculated using weighted indicator counts:

```rust
impl ParadigmDetector {
    fn calculate_scores(&self, indicators: &[ParadigmIndicator]) -> (f64, f64, f64, f64) {
        let mut oop_weight = 0.0;
        let mut fp_weight = 0.0;
        let mut reactive_weight = 0.0;
        let mut procedural_weight = 0.0;

        for indicator in indicators {
            let weight = self.category_weight(indicator.category);

            match indicator.paradigm() {
                Paradigm::ObjectOriented => oop_weight += weight,
                Paradigm::Functional => fp_weight += weight,
                Paradigm::Reactive => reactive_weight += weight,
                Paradigm::Procedural => procedural_weight += weight,
                Paradigm::Mixed => {} // Mixed is a result, not an input
            }
        }

        // Normalize to 0.0 - 1.0
        let total = oop_weight + fp_weight + reactive_weight + procedural_weight;
        if total == 0.0 {
            return (0.25, 0.25, 0.25, 0.25); // No indicators found
        }

        (
            oop_weight / total,
            fp_weight / total,
            reactive_weight / total,
            procedural_weight / total,
        )
    }
}
```

## Determining Dominant Paradigm

The dominant paradigm is selected based on score differences:

```rust
impl ParadigmProfile {
    pub fn dominant_paradigm(&self) -> Paradigm {
        let scores = [
            (self.oop_score, Paradigm::ObjectOriented),
            (self.fp_score, Paradigm::Functional),
            (self.reactive_score, Paradigm::Reactive),
            (self.procedural_score, Paradigm::Procedural),
        ];

        // Find highest score
        let (max_score, max_paradigm) = scores.iter()
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .unwrap();

        // Find second highest
        let second_max = scores.iter()
            .filter(|(_, p)| p != max_paradigm)
            .map(|(s, _)| s)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        // If difference is small, code is mixed
        if max_score - second_max < self.config.dominance_threshold {
            Paradigm::Mixed
        } else {
            *max_paradigm
        }
    }
}
```

## Language-Specific Detection

Different languages have different syntax for the same paradigm concepts:

### Rust

```rust
// OOP-like patterns in Rust
impl UserService {
    pub fn new(repo: UserRepository) -> Self {
        Self { repo }
    }

    pub fn get_user(&self, id: UserId) -> Option<User> {
        self.repo.find(id)
    }
}

// FP patterns in Rust
let doubled: Vec<i32> = numbers.iter()
    .map(|x| x * 2)
    .filter(|x| x > &10)
    .collect();
```

### JavaScript/TypeScript

```javascript
// OOP patterns
class UserService extends BaseService {
    constructor(private repo: UserRepository) {
        super();
    }
}

// FP patterns
const result = items
    .map(item => transform(item))
    .filter(item => item.valid)
    .reduce((acc, item) => acc + item.value, 0);

// Reactive patterns
observable$
    .pipe(
        map(x => x * 2),
        filter(x => x > 10)
    )
    .subscribe(console.log);
```

### Python

```python
# OOP patterns
class UserService:
    def __init__(self, repository):
        self._repository = repository

    def get_user(self, user_id):
        return self._repository.find(user_id)

# FP patterns
doubled = list(map(lambda x: x * 2, numbers))
evens = list(filter(lambda x: x % 2 == 0, numbers))
total = functools.reduce(lambda a, b: a + b, numbers)
```

## Advanced Usage

### Batch Analysis

Analyze multiple files efficiently:

```rust
use rayon::prelude::*;

fn analyze_codebase(files: &[PathBuf]) -> Vec<(PathBuf, ParadigmProfile)> {
    let detector = ParadigmDetector::new(ParadigmConfig::default());

    files.par_iter()
        .map(|path| {
            let code = std::fs::read_to_string(path).unwrap();
            let profile = detector.analyze(&code);
            (path.clone(), profile)
        })
        .collect()
}
```

### Custom Pattern Sets

Add custom patterns for specific frameworks:

```rust
let mut config = ParadigmConfig::default();

// Add React-specific patterns
config.add_pattern(
    IndicatorCategory::OopComponent,
    r"\bclass\s+\w+\s+extends\s+(React\.)?Component",
    0.9,
);

// Add Redux patterns
config.add_pattern(
    IndicatorCategory::FpReducer,
    r"\bfunction\s+\w+Reducer\s*\(",
    0.8,
);

let detector = ParadigmDetector::new(config);
```

### Trend Analysis

Track paradigm evolution over time:

```rust
struct ParadigmTrend {
    commit: String,
    timestamp: DateTime<Utc>,
    profile: ParadigmProfile,
}

fn analyze_git_history(repo: &Repository) -> Vec<ParadigmTrend> {
    let detector = ParadigmDetector::new(ParadigmConfig::default());

    repo.commits()
        .map(|commit| {
            let code = repo.get_file_at_commit(&commit, "src/main.rs");
            ParadigmTrend {
                commit: commit.id().to_string(),
                timestamp: commit.time(),
                profile: detector.analyze(&code),
            }
        })
        .collect()
}
```

## Performance Considerations

The detector is designed for efficiency:

- **Compiled regexes**: Patterns are compiled once at construction
- **Thread-safe**: Detector can be shared across threads (`Send + Sync`)
- **Streaming**: Processes code without building full AST
- **Memory efficient**: Only stores indicator metadata

For large codebases, use parallel processing:

```rust
use rayon::prelude::*;

let profiles: Vec<ParadigmProfile> = code_files
    .par_iter()
    .map(|code| detector.analyze(code))
    .collect();
```

## See Also

- [Overview](overview.md) - Paradigm detection introduction
- [Indicators](indicators.md) - Indicator types and categories
- [API Patterns](api-patterns.md) - Mining API usage patterns
- [Domain Patterns](domain-patterns.md) - Rholang and MeTTa patterns
