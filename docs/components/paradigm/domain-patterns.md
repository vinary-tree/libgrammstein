# Domain-Specific Patterns

libgrammstein includes specialized pattern catalogs for domain-specific languages, particularly Rholang and MeTTa which are core to the F1R3FLY.io ecosystem.

## Overview

Domain patterns capture language-specific idioms and constructs that don't map directly to general programming paradigms. These patterns help with:

- Understanding code semantics in specialized languages
- Detecting best practices and anti-patterns
- Enabling intelligent code completion
- Supporting language-specific refactoring

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Domain Pattern Detection                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    Rholang Patterns                              │    │
│  │  • Process Composition (Par, Sequential, Choice)                │    │
│  │  • Channel Operations (Send, Receive, Peek)                     │    │
│  │  • Contract Patterns (Factory, Registry, Token)                 │    │
│  │  • Concurrency Idioms (Barriers, Locks, Actors)                 │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    MeTTa Patterns                                │    │
│  │  • Atom Definitions (Symbols, Expressions, Grounded)            │    │
│  │  • Type Patterns (Type annotations, Inference)                  │    │
│  │  • Reasoning Patterns (Unification, Chaining)                   │    │
│  │  • Knowledge Representation (Ontologies, Relations)             │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Rholang Pattern Catalog

Rholang is a concurrent process calculus language designed for blockchain and distributed systems.

### RholangPatternCatalog

The catalog of Rholang-specific patterns:

```rust
pub struct RholangPatternCatalog {
    patterns: HashMap<RholangPatternCategory, Vec<RholangPattern>>,
}

pub enum RholangPatternCategory {
    ProcessComposition,
    ChannelOperation,
    Contract,
    Concurrency,
    DataStructure,
    NameBinding,
    Match,
    Bundle,
    Unforgeable,
    SystemCall,
    StateManagement,
    AccessControl,
}
```

### Process Composition Patterns

Rholang processes combine using parallel, sequential, and choice operators:

#### Par Composition (`|`)

Parallel composition runs processes concurrently:

```rholang
// Two processes running in parallel
for (@x <- chan1) { process1!(x) } |
for (@y <- chan2) { process2!(y) }
```

**Detection Pattern:**
```rust
RholangPattern {
    category: ProcessComposition,
    name: "ParComposition",
    regex: r"\|\s*(for|new|match|Nil|@)",
    description: "Parallel process composition",
}
```

#### Sequential Composition

Sequential execution using channel synchronization:

```rholang
// Process1 completes, then process2 runs
new ack in {
  process1!(ack) |
  for (_ <- ack) { process2!() }
}
```

**Detection Pattern:**
```rust
RholangPattern {
    category: ProcessComposition,
    name: "SequentialSync",
    regex: r"for\s*\([^)]+<-\s*(\w+)\)\s*\{[^}]*\}\s*\|\s*[^|]+!\s*\(\1\)",
    description: "Sequential synchronization via acknowledgment channel",
}
```

#### Choice (Select)

Non-deterministic choice between processes:

```rholang
// Select first available message
select {
  for (@x <- chan1) { handleChan1!(x) }
  for (@y <- chan2) { handleChan2!(y) }
}
```

### Channel Operation Patterns

Rholang uses channels for all communication:

#### Send (`!`)

Send data on a channel:

```rholang
channel!(data)           // Persistent send
channel!!(data)          // Peek-able send
```

**Detection Patterns:**
```rust
// Standard send
RholangPattern {
    category: ChannelOperation,
    name: "Send",
    regex: r"(\w+)\s*!\s*\(",
    description: "Channel send operation",
}

// Persistent send (peek)
RholangPattern {
    category: ChannelOperation,
    name: "PersistentSend",
    regex: r"(\w+)\s*!!\s*\(",
    description: "Persistent/peekable send",
}
```

#### Receive (`for ... <- ...`)

Receive data from a channel:

```rholang
for (@data <- channel) { process!(data) }    // Linear receive
for (@data <= channel) { process!(data) }    // Persistent receive (replicated)
```

**Detection Patterns:**
```rust
// Linear receive
RholangPattern {
    category: ChannelOperation,
    name: "LinearReceive",
    regex: r"for\s*\([^)]+<-\s*\w+\)",
    description: "Linear (consuming) receive",
}

// Replicated receive
RholangPattern {
    category: ChannelOperation,
    name: "ReplicatedReceive",
    regex: r"for\s*\([^)]+<=\s*\w+\)",
    description: "Replicated (persistent) receive",
}
```

### Contract Patterns

Common smart contract patterns in Rholang:

#### Factory Contract

Creates new contract instances:

```rholang
new factory in {
  contract factory(@config, return) = {
    new instance in {
      // Initialize instance with config
      contract instance(@method, args, ret) = {
        // Method implementation
      } |
      return!(instance)
    }
  }
}
```

**Detection Pattern:**
```rust
RholangPattern {
    category: Contract,
    name: "Factory",
    regex: r"contract\s+\w+\s*\([^)]*return\s*\)\s*=\s*\{\s*new\s+\w+\s+in",
    description: "Factory contract pattern",
}
```

#### Registry Pattern

Stores and retrieves named resources:

```rholang
new registry, lookup, register in {
  contract register(@name, @value, ack) = {
    registry!(name, value) |
    ack!(true)
  } |
  contract lookup(@name, return) = {
    for (@n, @v <- registry) {
      if (n == name) { return!(v) | registry!(n, v) }
      else { registry!(n, v) | lookup!(name, return) }
    }
  }
}
```

#### Token Contract

ERC20-like token implementation:

```rholang
new token in {
  contract token(@"transfer", @from, @to, @amount, return) = {
    // Transfer logic with balance checks
  } |
  contract token(@"balance", @account, return) = {
    // Balance lookup
  }
}
```

### Concurrency Patterns

Patterns for managing concurrent access:

#### Mutex/Lock

Mutual exclusion:

```rholang
new mutex in {
  mutex!(Nil) |  // Initialize with token

  // Acquire lock
  for (_ <- mutex) {
    // Critical section
    mutex!(Nil)  // Release lock
  }
}
```

#### Barrier

Synchronization point for multiple processes:

```rholang
new barrier in {
  contract barrier(@n, ready) = {
    new count in {
      count!(0) |
      for (@c <- count) {
        if (c + 1 == n) { ready!(true) }
        else { count!(c + 1) }
      }
    }
  }
}
```

#### Actor Pattern

Message-passing concurrency:

```rholang
new actor in {
  contract actor(@msg, reply) = {
    match msg {
      ("get", key) => { /* handle get */ }
      ("set", key, value) => { /* handle set */ }
    }
  }
}
```

### Using the Rholang Catalog

```rust
use libgrammstein::topic::paradigm::{
    RholangPatternCatalog,
    DomainPatternDetector,
};

let catalog = RholangPatternCatalog::default();
let detector = DomainPatternDetector::new(catalog);

let rholang_code = r#"
new channel in {
  contract channel(@msg, ack) = {
    @stdout!(msg) | ack!(true)
  } |
  channel!("Hello", *ack) |
  for (_ <- ack) { Nil }
}
"#;

let matches = detector.detect(rholang_code);

for m in matches {
    println!("Found {} pattern: {}",
             m.category, m.pattern_name);
    println!("  At: {}..{}", m.start, m.end);
}
```

---

## MeTTa Pattern Catalog

MeTTa is a meta-language for hypergraph-based AI and symbolic reasoning.

### MettaPatternCatalog

The catalog of MeTTa-specific patterns:

```rust
pub struct MettaPatternCatalog {
    patterns: HashMap<MettaPatternCategory, Vec<MettaPattern>>,
}

pub enum MettaPatternCategory {
    AtomDefinition,
    TypePattern,
    FunctionDefinition,
    QueryPattern,
    Unification,
    Import,
    Assertion,
    Inference,
    GroundedAtom,
    SpaceOperation,
    MetaProgramming,
    PatternMatching,
}
```

### Atom Definition Patterns

MeTTa operates on atoms in a hypergraph:

#### Symbol Atoms

Simple named entities:

```metta
; Symbol definition
(= (my-symbol) value)

; Typed symbol
(: my-symbol Type)
```

**Detection Pattern:**
```rust
MettaPattern {
    category: AtomDefinition,
    name: "SymbolDefinition",
    regex: r"\(=\s+\(\w+\)\s+",
    description: "Symbol atom definition",
}
```

#### Expression Atoms

Compound structures:

```metta
; Expression with multiple elements
(my-function arg1 arg2 arg3)

; Nested expressions
(outer (inner x y) z)
```

#### Grounded Atoms

Atoms with external implementation:

```metta
; Grounded function (implemented in host language)
(: + (-> Number Number Number))

; Calling grounded function
(+ 1 2)  ; Returns 3
```

### Type Patterns

MeTTa has a rich type system:

#### Type Declarations

```metta
; Declare atom type
(: my-func (-> InputType OutputType))

; Polymorphic type
(: map (-> (-> $a $b) (List $a) (List $b)))

; Type alias
(= (MyType) (List Int))
```

**Detection Patterns:**
```rust
// Type annotation
MettaPattern {
    category: TypePattern,
    name: "TypeAnnotation",
    regex: r"\(:\s+\w+\s+\(",
    description: "Type annotation",
}

// Arrow type (function type)
MettaPattern {
    category: TypePattern,
    name: "ArrowType",
    regex: r"\(->\s+",
    description: "Function type signature",
}

// Type variable
MettaPattern {
    category: TypePattern,
    name: "TypeVariable",
    regex: r"\$\w+",
    description: "Polymorphic type variable",
}
```

### Function Definition Patterns

#### Named Functions

```metta
; Function definition with pattern matching
(= (factorial 0) 1)
(= (factorial $n) (* $n (factorial (- $n 1))))

; Multi-argument function
(= (add $x $y) (+ $x $y))
```

**Detection Pattern:**
```rust
MettaPattern {
    category: FunctionDefinition,
    name: "FunctionDef",
    regex: r"\(=\s+\(\w+\s+",
    description: "Named function definition",
}
```

#### Lambda Expressions

```metta
; Anonymous function
(lambda ($x) (+ $x 1))

; Multi-parameter lambda
(lambda ($x $y) (+ $x $y))
```

### Query and Unification Patterns

#### Match Queries

```metta
; Query for matching patterns
(match &self (parent $x $y) $x)

; Query with condition
(match &self (person $name $age) (if (> $age 18) $name))
```

**Detection Pattern:**
```rust
MettaPattern {
    category: QueryPattern,
    name: "MatchQuery",
    regex: r"\(match\s+&\w+\s+",
    description: "Pattern matching query",
}
```

#### Unification

```metta
; Unify expressions
(unify (foo $x) (foo bar) $x)

; Unification in patterns
(= (find-pair $x $y)
   (match &self (pair $x $y) (pair $x $y)))
```

### Inference Patterns

#### Forward Chaining

```metta
; Assert knowledge
(add-atom &kb (parent alice bob))
(add-atom &kb (parent bob charlie))

; Rule for transitivity
(= (grandparent $x $z)
   (match &kb (parent $x $y)
     (match &kb (parent $y $z) (grandparent $x $z))))
```

#### Backward Chaining

```metta
; Query with backward reasoning
(: derive (-> Query Result))
(= (derive (grandparent $x $z))
   (let* (($y (derive (parent $x $y)))
          ($z (derive (parent $y $z))))
     (grandparent $x $z)))
```

### Knowledge Representation Patterns

#### Ontology Structures

```metta
; Class hierarchy
(: Animal Type)
(: Mammal Type)
(: Dog Type)

(isa Mammal Animal)
(isa Dog Mammal)

; Instance
(isa fido Dog)
```

#### Relations

```metta
; Binary relation
(: owns (-> Person Thing Prop))
(owns alice car1)

; N-ary relation
(: transaction (-> Account Account Amount Time Prop))
(transaction acc1 acc2 100 now)
```

### Space Operations

MeTTa organizes atoms into spaces:

```metta
; Create new space
(new-space)

; Add atoms to space
(add-atom &my-space (fact 1 2 3))

; Query space
(get-atoms &my-space)

; Bind space
(bind! &kb (new-space))
```

**Detection Patterns:**
```rust
MettaPattern {
    category: SpaceOperation,
    name: "SpaceReference",
    regex: r"&\w+",
    description: "Space reference",
}

MettaPattern {
    category: SpaceOperation,
    name: "AddAtom",
    regex: r"\(add-atom\s+&\w+",
    description: "Add atom to space",
}
```

### Using the MeTTa Catalog

```rust
use libgrammstein::topic::paradigm::{
    MettaPatternCatalog,
    DomainPatternDetector,
};

let catalog = MettaPatternCatalog::default();
let detector = DomainPatternDetector::new(catalog);

let metta_code = r#"
; Define a type
(: factorial (-> Int Int))

; Define the function
(= (factorial 0) 1)
(= (factorial $n)
   (* $n (factorial (- $n 1))))

; Use it
!(factorial 5)
"#;

let matches = detector.detect(metta_code);

for m in matches {
    println!("Found {} pattern: {}",
             m.category, m.pattern_name);
}
```

---

## DomainPatternDetector

The unified interface for domain pattern detection:

```rust
pub struct DomainPatternDetector<C: PatternCatalog> {
    catalog: C,
}

impl<C: PatternCatalog> DomainPatternDetector<C> {
    pub fn new(catalog: C) -> Self {
        Self { catalog }
    }

    pub fn detect(&self, code: &str) -> Vec<PatternMatch> {
        let mut matches = Vec::new();

        for (category, patterns) in self.catalog.patterns() {
            for pattern in patterns {
                for m in pattern.regex.find_iter(code) {
                    matches.push(PatternMatch {
                        category: category.clone(),
                        pattern_name: pattern.name.clone(),
                        start: m.start(),
                        end: m.end(),
                        matched_text: m.as_str().to_string(),
                    });
                }
            }
        }

        matches
    }
}
```

### Pattern Match Result

```rust
pub struct PatternMatch {
    /// Pattern category
    pub category: String,
    /// Specific pattern name
    pub pattern_name: String,
    /// Start position in source
    pub start: usize,
    /// End position in source
    pub end: usize,
    /// Matched text
    pub matched_text: String,
}
```

## Extending with Custom Patterns

Add patterns for your own domain languages:

```rust
use libgrammstein::topic::paradigm::{
    PatternCatalog,
    DomainPattern,
};

pub struct MyLanguagePatternCatalog {
    patterns: HashMap<String, Vec<DomainPattern>>,
}

impl MyLanguagePatternCatalog {
    pub fn new() -> Self {
        let mut patterns = HashMap::new();

        patterns.insert("Control".to_string(), vec![
            DomainPattern {
                name: "IfThenElse".to_string(),
                regex: Regex::new(r"\bif\s+.+\s+then\s+.+\s+else\b").unwrap(),
                description: "Conditional expression".to_string(),
            },
        ]);

        patterns.insert("DataDef".to_string(), vec![
            DomainPattern {
                name: "Record".to_string(),
                regex: Regex::new(r"\brecord\s+\w+\s*\{").unwrap(),
                description: "Record type definition".to_string(),
            },
        ]);

        Self { patterns }
    }
}

impl PatternCatalog for MyLanguagePatternCatalog {
    fn patterns(&self) -> &HashMap<String, Vec<DomainPattern>> {
        &self.patterns
    }
}
```

## Integration with Other Components

### With Paradigm Detection

Combine domain patterns with paradigm analysis:

```rust
let paradigm_detector = ParadigmDetector::new(ParadigmConfig::default());
let domain_detector = DomainPatternDetector::new(RholangPatternCatalog::default());

let profile = paradigm_detector.analyze(rholang_code);
let domain_matches = domain_detector.detect(rholang_code);

println!("Paradigm: {:?}", profile.dominant_paradigm());
println!("Domain patterns found: {}", domain_matches.len());
```

### With Code Embeddings

Use domain patterns to enhance embeddings:

```rust
// Weight embedding dimensions based on detected patterns
let domain_matches = domain_detector.detect(code);
let has_contract = domain_matches.iter()
    .any(|m| m.category == "Contract");

let embedding = if has_contract {
    contract_aware_embedder.embed(code)
} else {
    general_embedder.embed(code)
};
```

## See Also

- [Overview](overview.md) - Paradigm detection introduction
- [Detection](detection.md) - Paradigm detector usage
- [Indicators](indicators.md) - Indicator types and categories
- [API Patterns](api-patterns.md) - Mining API usage patterns
