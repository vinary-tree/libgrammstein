# Paradigm Indicators

Indicators are the atomic evidence of paradigm usage in source code. Each indicator represents a matched pattern that suggests a particular programming style.

## Core Types

### ParadigmIndicator

Represents a single paradigm indicator found in code:

```rust
pub struct ParadigmIndicator {
    /// The category of this indicator
    pub category: IndicatorCategory,
    /// Start position in source code (byte offset)
    pub start: usize,
    /// End position in source code (byte offset)
    pub end: usize,
    /// The matched text
    pub matched_text: String,
}
```

### IndicatorCategory

The classification of each indicator. Categories are organized by paradigm:

```rust
pub enum IndicatorCategory {
    // Object-Oriented Categories
    OopClass,
    OopInheritance,
    OopInterface,
    OopEncapsulation,
    OopConstructor,
    OopMethod,

    // Functional Categories
    FpHigherOrder,
    FpLambda,
    FpImmutability,
    FpComposition,
    FpPattern,

    // Reactive Categories
    ReactiveObservable,
    ReactiveSubscription,
    ReactiveOperator,
    ReactiveEvent,

    // Procedural Categories
    ProceduralLoop,
    ProceduralMutation,
    ProceduralControl,
    ProceduralFunction,
}
```

## Category Details

### Object-Oriented Categories

#### OopClass

Identifies class definitions:

```rust
// Matches
class UserService { }           // JavaScript/TypeScript
class UserService: BaseService  // Python
struct User { }                 // Rust (with impl blocks)
```

| Language | Pattern |
|----------|---------|
| JavaScript | `\bclass\s+\w+` |
| Python | `\bclass\s+\w+` |
| Rust | `\bstruct\s+\w+` + `\bimpl\s+` |

#### OopInheritance

Identifies inheritance relationships:

```rust
// Matches
class Admin extends User { }         // JavaScript
class Admin(User): pass              // Python
impl Deref for SmartPointer { }      // Rust trait impl
```

| Language | Pattern |
|----------|---------|
| JavaScript | `\bextends\s+\w+` |
| Python | `\bclass\s+\w+\(\w+\)` |
| Rust | `\bimpl\s+\w+\s+for\s+` |

#### OopInterface

Identifies interface definitions and implementations:

```rust
// Matches
interface Repository<T> { }          // TypeScript
class UserRepo implements Repository // TypeScript
trait Serialize { }                  // Rust
```

| Language | Pattern |
|----------|---------|
| TypeScript | `\binterface\s+\w+` |
| TypeScript | `\bimplements\s+\w+` |
| Rust | `\btrait\s+\w+` |

#### OopEncapsulation

Identifies access modifiers and encapsulation:

```rust
// Matches
private repository: Repository;      // TypeScript
protected _name: string;             // TypeScript
pub fn new() -> Self { }             // Rust
```

| Language | Pattern |
|----------|---------|
| JavaScript/TS | `\b(private|public|protected)\b` |
| Rust | `\bpub(\s*\(crate\))?\s+(fn|struct|mod)` |
| Python | `\b_\w+` (convention) |

#### OopConstructor

Identifies constructor patterns:

```rust
// Matches
constructor(private db: Database) { }  // TypeScript
def __init__(self, db):                // Python
fn new(db: Database) -> Self { }       // Rust
```

| Language | Pattern |
|----------|---------|
| JavaScript/TS | `\bconstructor\s*\(` |
| Python | `\bdef\s+__init__\s*\(` |
| Rust | `\bfn\s+new\s*\(` |

#### OopMethod

Identifies method definitions on classes/objects:

```rust
// Matches
getUser(id: string): User { }        // TypeScript
def get_user(self, id):              // Python
fn get_user(&self, id: UserId) { }   // Rust
```

| Language | Pattern |
|----------|---------|
| JavaScript/TS | Method in class body |
| Python | `\bdef\s+\w+\(self` |
| Rust | `\bfn\s+\w+\(&self` |

---

### Functional Categories

#### FpHigherOrder

Identifies higher-order function usage:

```rust
// Matches
items.map(transform)                 // All languages
items.filter(predicate)              // All languages
items.reduce(accumulator, initial)   // All languages
items.fold(initial, folder)          // Rust
```

| Function | Weight | Description |
|----------|--------|-------------|
| `map` | 0.8 | Transform each element |
| `filter` | 0.8 | Select elements by predicate |
| `reduce` | 0.9 | Accumulate to single value |
| `fold` | 0.9 | Accumulate with initial value |
| `flatMap` | 0.9 | Map and flatten |
| `forEach` | 0.5 | Side-effectful iteration |

#### FpLambda

Identifies anonymous functions and closures:

```rust
// Matches
x => x * 2                           // JavaScript arrow
(x) => { return x * 2; }             // JavaScript arrow
lambda x: x * 2                      // Python
|x| x * 2                            // Rust closure
|x: i32| -> i32 { x * 2 }            // Rust typed closure
```

| Language | Pattern |
|----------|---------|
| JavaScript | `\w+\s*=>\s*` |
| Python | `\blambda\s+\w+:` |
| Rust | `\|\w*\|\s*` |

#### FpImmutability

Identifies immutable binding patterns:

```rust
// Matches
const value = 42;                    // JavaScript
let value = 42;                      // Rust (immutable by default)
readonly property: string;           // TypeScript
```

| Language | Pattern |
|----------|---------|
| JavaScript | `\bconst\s+\w+\s*=` |
| TypeScript | `\breadonly\s+\w+` |
| Rust | `\blet\s+\w+\s*=` (without `mut`) |

#### FpComposition

Identifies function composition patterns:

```rust
// Matches
compose(f, g)(x)                     // JavaScript
pipe(f, g, h)(x)                     // JavaScript
f >> g                               // F#/Haskell-style
x |> f |> g                          // Elixir/F# pipe
```

| Pattern | Weight | Description |
|---------|--------|-------------|
| `compose` | 0.9 | Right-to-left composition |
| `pipe` | 0.9 | Left-to-right composition |
| `>>` | 0.9 | Composition operator |
| `\|>` | 0.9 | Pipe operator |

#### FpPattern

Identifies pattern matching (strongly FP):

```rust
// Matches
match value {                        // Rust
    Some(x) => x,
    None => default,
}

case value of                        // Haskell
    Just x -> x
    Nothing -> default
```

| Language | Pattern |
|----------|---------|
| Rust | `\bmatch\s+\w+\s*\{` |
| Haskell | `\bcase\s+\w+\s+of` |
| Scala | `\bmatch\s*\{` |

---

### Reactive Categories

#### ReactiveObservable

Identifies observable stream creation:

```rust
// Matches
Observable.create(observer => { })    // RxJS
new Observable(subscriber => { })     // RxJS
from([1, 2, 3])                       // RxJS
of(1, 2, 3)                           // RxJS
interval(1000)                        // RxJS
```

| Pattern | Weight | Description |
|---------|--------|-------------|
| `Observable` | 1.0 | Observable type |
| `Subject` | 0.9 | Multicast observable |
| `BehaviorSubject` | 0.9 | Stateful subject |
| `from` | 0.7 | Convert to observable |
| `of` | 0.7 | Create from values |

#### ReactiveSubscription

Identifies subscriptions to streams:

```rust
// Matches
observable.subscribe(value => { })    // RxJS
observable.subscribe({                // RxJS
    next: value => { },
    error: err => { },
    complete: () => { }
})
```

| Pattern | Weight | Description |
|---------|--------|-------------|
| `.subscribe(` | 0.9 | Subscription |
| `.unsubscribe(` | 0.7 | Cleanup |
| `Subscription` | 0.7 | Subscription type |

#### ReactiveOperator

Identifies reactive operators:

```rust
// Matches
observable.pipe(
    map(x => x * 2),
    filter(x => x > 10),
    debounceTime(300),
    switchMap(x => fetch(x))
)
```

| Operator | Weight | Category |
|----------|--------|----------|
| `debounceTime` | 0.9 | Timing |
| `throttleTime` | 0.9 | Timing |
| `switchMap` | 0.9 | Transformation |
| `mergeMap` | 0.9 | Transformation |
| `combineLatest` | 0.9 | Combination |
| `withLatestFrom` | 0.9 | Combination |

#### ReactiveEvent

Identifies event handling patterns:

```rust
// Matches
button.addEventListener('click', handler)  // DOM
element.onclick = handler                  // DOM
$(element).on('click', handler)            // jQuery
fromEvent(button, 'click')                 // RxJS
```

| Pattern | Weight | Description |
|---------|--------|-------------|
| `addEventListener` | 0.7 | DOM events |
| `.on(` | 0.6 | Event binding |
| `fromEvent` | 0.8 | RxJS event source |
| `.emit(` | 0.7 | Event emission |

---

### Procedural Categories

#### ProceduralLoop

Identifies imperative loop constructs:

```rust
// Matches
for (let i = 0; i < n; i++) { }      // JavaScript
while (condition) { }                 // All languages
for item in items { }                 // Rust
loop { break; }                       // Rust
```

| Pattern | Weight | Description |
|---------|--------|-------------|
| `for (` | 0.6 | C-style for loop |
| `while` | 0.6 | While loop |
| `do...while` | 0.6 | Do-while loop |
| `loop` | 0.5 | Infinite loop (Rust) |

#### ProceduralMutation

Identifies mutable state patterns:

```rust
// Matches
let mut counter = 0;                  // Rust
var total = 0;                        // JavaScript
counter++;                            // Increment
value = newValue;                     // Reassignment
```

| Pattern | Weight | Description |
|---------|--------|-------------|
| `let mut` | 0.5 | Rust mutable binding |
| `var` | 0.5 | JavaScript var |
| `++` / `--` | 0.6 | Increment/decrement |
| `+=` / `-=` | 0.5 | Compound assignment |

#### ProceduralControl

Identifies control flow statements:

```rust
// Matches
if (condition) { } else { }           // All languages
switch (value) { case x: }            // JavaScript
break;                                // Loop exit
continue;                             // Loop continue
return value;                         // Early return
```

| Pattern | Weight | Description |
|---------|--------|-------------|
| `if` | 0.3 | Conditional (low weight - universal) |
| `switch` | 0.4 | Switch statement |
| `break` | 0.4 | Loop exit |
| `continue` | 0.4 | Loop continue |
| `goto` | 0.8 | Goto (strongly procedural) |

#### ProceduralFunction

Identifies standalone function definitions:

```rust
// Matches
function calculateTotal(items) { }    // JavaScript
def calculate_total(items):           // Python (without self)
fn calculate_total(items: &[Item])    // Rust (free function)
```

| Language | Pattern |
|----------|---------|
| JavaScript | `\bfunction\s+\w+\s*\(` |
| Python | `\bdef\s+\w+\([^s]` (no self) |
| Rust | `\bfn\s+\w+\s*\(` (outside impl) |

---

## Working with Indicators

### Extracting Indicators

```rust
let profile = detector.analyze(code);

// Get all indicators
for indicator in &profile.indicators {
    println!("{:?} at {}..{}: '{}'",
             indicator.category,
             indicator.start,
             indicator.end,
             indicator.matched_text);
}
```

### Filtering by Paradigm

```rust
// Get only OOP indicators
let oop_indicators: Vec<_> = profile.indicators.iter()
    .filter(|i| matches!(i.category,
        IndicatorCategory::OopClass |
        IndicatorCategory::OopInheritance |
        IndicatorCategory::OopInterface |
        IndicatorCategory::OopEncapsulation |
        IndicatorCategory::OopConstructor |
        IndicatorCategory::OopMethod
    ))
    .collect();
```

### Paradigm from Category

Each indicator can report its paradigm:

```rust
impl ParadigmIndicator {
    pub fn paradigm(&self) -> Paradigm {
        match self.category {
            IndicatorCategory::OopClass |
            IndicatorCategory::OopInheritance |
            IndicatorCategory::OopInterface |
            IndicatorCategory::OopEncapsulation |
            IndicatorCategory::OopConstructor |
            IndicatorCategory::OopMethod => Paradigm::ObjectOriented,

            IndicatorCategory::FpHigherOrder |
            IndicatorCategory::FpLambda |
            IndicatorCategory::FpImmutability |
            IndicatorCategory::FpComposition |
            IndicatorCategory::FpPattern => Paradigm::Functional,

            IndicatorCategory::ReactiveObservable |
            IndicatorCategory::ReactiveSubscription |
            IndicatorCategory::ReactiveOperator |
            IndicatorCategory::ReactiveEvent => Paradigm::Reactive,

            IndicatorCategory::ProceduralLoop |
            IndicatorCategory::ProceduralMutation |
            IndicatorCategory::ProceduralControl |
            IndicatorCategory::ProceduralFunction => Paradigm::Procedural,
        }
    }
}
```

### Grouping by Category

```rust
use std::collections::HashMap;

fn group_by_category(indicators: &[ParadigmIndicator])
    -> HashMap<IndicatorCategory, Vec<&ParadigmIndicator>>
{
    let mut groups: HashMap<IndicatorCategory, Vec<_>> = HashMap::new();

    for indicator in indicators {
        groups.entry(indicator.category)
            .or_default()
            .push(indicator);
    }

    groups
}

// Usage
let groups = group_by_category(&profile.indicators);

for (category, indicators) in groups {
    println!("{:?}: {} occurrences", category, indicators.len());
}
```

## Indicator Statistics

### Category Distribution

```rust
fn category_distribution(profile: &ParadigmProfile)
    -> HashMap<IndicatorCategory, usize>
{
    let mut dist = HashMap::new();

    for indicator in &profile.indicators {
        *dist.entry(indicator.category).or_insert(0) += 1;
    }

    dist
}
```

### Paradigm Distribution

```rust
fn paradigm_distribution(profile: &ParadigmProfile)
    -> HashMap<Paradigm, usize>
{
    let mut dist = HashMap::new();

    for indicator in &profile.indicators {
        *dist.entry(indicator.paradigm()).or_insert(0) += 1;
    }

    dist
}
```

### Source Location Analysis

```rust
fn indicators_by_line(code: &str, indicators: &[ParadigmIndicator])
    -> HashMap<usize, Vec<&ParadigmIndicator>>
{
    let line_starts: Vec<_> = std::iter::once(0)
        .chain(code.match_indices('\n').map(|(i, _)| i + 1))
        .collect();

    let mut by_line: HashMap<usize, Vec<_>> = HashMap::new();

    for indicator in indicators {
        let line = line_starts.partition_point(|&start| start <= indicator.start);
        by_line.entry(line).or_default().push(indicator);
    }

    by_line
}
```

## See Also

- [Overview](overview.md) - Paradigm detection introduction
- [Detection](detection.md) - How detection works
- [API Patterns](api-patterns.md) - Mining API usage patterns
- [Domain Patterns](domain-patterns.md) - Rholang and MeTTa patterns
