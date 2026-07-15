# Semantic Corrector: CPG Graph Rules and Name Similarity

The **semantic corrector** is the layer that is supposed to catch what spelling and syntax cannot:
a variable that is *valid* but *wrong*, a binding written and never read, a fallible call whose
error is never handled. Its inputs are the [Code Property Graph](../cpg.md) — the fused
AST + control-flow + data-flow graph of Yamaguchi et al. [[1]](#references) — and a registry of the
project's known variables. Its output is a ranked list of `Correction`s carrying data-flow and
control-flow provenance.

> **Honest status, up front.** The type is called `SemanticCorrector`, it owns a
> `GnnSemanticScorer`, and it tags its output `CorrectionSource::Neural`. **No neural network
> runs.** `GnnSemanticScorer::node_embeddings` is initialized empty and is never written by any
> code path in the crate, so the graph convolution never executes and `score_node` returns $`0.0`$
> unconditionally. What actually runs is (a) two **deterministic CPG graph rules** and (b) a
> **Levenshtein name-similarity** search over registered variables. Both are useful; neither is
> learned. The details are in [What actually runs](#honest-status-graph-rules-and-name-similarity-no-gnn),
> and this page documents the real behavior throughout.

> **Scope.** Source of truth: [`src/code/correctors/semantic.rs`](../../../../src/code/correctors/semantic.rs).
> The graph it reads: [CPG](../cpg.md). The scorer it delegates detection to, and the target GNN
> architecture: [GNN](../gnn.md). The corrections it emits are merged by the
> [Ensemble](ensemble.md).

## Notation

| Symbol | Meaning |
|---|---|
| $`G = (\mathcal{N}, \mathcal{E})`$ | the CPG: nodes and typed edges |
| $`\nu \in \mathcal{N}`$ | a CPG node: `id`, `kind`, `name`, byte `location` |
| $`q`$ | the **query** name: the identifier under suspicion |
| $`n`$ | a **candidate** name drawn from the variable registry |
| $`d_L(a,b)`$ | Levenshtein distance (see [Lexical](lexical.md#levenshtein-distance)) |
| $`\mathrm{sim}(a,b)`$ | normalized name similarity — eq. $`(\mathrm{S1})`$ |
| $`k_n`$ | `use_count`: how many times $`n`$ was **registered** |
| $`u(n)`$ | the usage boost — eq. $`(\mathrm{S2})`$ |
| $`\iota`$ | a `SemanticIssue`: node, `IssueType`, confidence, optional suggestion |
| $`\theta_{\text{sem}}`$ | `min_confidence`, default $`0.5`$ — gates **issues**, not corrections |

## What & why

Token- and grammar-level correction are blind to meaning. `total` and `subtotal` are both spelled
correctly and both parse; only the program's *data flow* reveals that one of them was never
assigned. This is the class of fault that Allamanis et al. [[2]](#references) attack by learning
over program graphs, and the CPG is precisely such a graph. The design intent of this module is
therefore: run a graph neural network over the CPG, and let it localize the fault.

The architecture that intent points at is the spectral graph convolution of Kipf and Welling
[[3]](#references) — one layer propagates each node's features to its neighbours through a
symmetrically normalized adjacency:

```math
\begin{array}{lr}
\displaystyle H^{(l+1)} \;=\; \sigma\!\left( \hat{A}\, H^{(l)} W^{(l)} \right),
\qquad
\hat{A} \;=\; \tilde{D}^{-\frac{1}{2}}\,\tilde{A}\,\tilde{D}^{-\frac{1}{2}} & \text{(S0)}
\end{array}
```

Everything needed to *feed* $`(\mathrm{S0})`$ exists and is tested: `GnnSemanticScorer::extract_features`
builds the node and edge tensors, and `GnnFeatures::to_adjacency_list` / `to_node_matrix` project
them into the dense forms a tensor library expects. See [GNN](../gnn.md) for the feature algebra.
**What does not exist is the forward pass, the weights $`W^{(l)}`$, and any training procedure.**
$`(\mathrm{S0})`$ is the target, not the implementation, and is shown here only so the gap is
unambiguous.

## Honest status: graph rules and name similarity, no GNN

The corrector exposes **two independent surfaces**, and it is essential to know which one you are
calling, because they share no machinery at all.

| Surface | Methods | Reads the CPG? | Uses the GNN? | What it actually does |
|---|---|---|---|---|
| **Trait surface** | `correct_token`, `correct_range` | **No** | **No** | Levenshtein similarity against the registered-variable names |
| **Graph surface** | `analyze_parsed`, `analyze_cpg` | Yes | Only its rule engine | two deterministic graph rules over data flow and call names |

### The trait surface does no semantic analysis whatsoever

`CodeCorrector::correct_token` fires only for `TokenType::Identifier`, only when the identifier is
**absent** from `known_variables`, and then simply ranks the registered names by string similarity:

```math
\begin{array}{lr}
\displaystyle c \;=\; 0.7 \cdot \mathrm{sim}(q, n), \qquad \text{kept only if } \mathrm{sim}(q,n) > 0.5 & \text{(S3)}
\end{array}
```

No graph is consulted; no node is scored. It is a second lexical corrector with a different
dictionary (the variable registry) and a different metric (normalized Levenshtein instead of raw
distance) — yet its output is stamped `CorrectionSource::Neural` and
`CorrectionKind::VariableMisuse`. That provenance is misleading, and consumers who filter on
`source == Neural` expecting learned evidence will be disappointed. Note also that
`correct_range` forces `TokenType::Identifier`, so *any* byte range handed to it is treated as an
identifier.

Because $`q \notin`$ `known_variables` implies $`q \neq n`$, hence $`d_L \geq 1`$, hence
$`\mathrm{sim} < 1`$, the confidence from $`(\mathrm{S3})`$ is bounded:
$`c \in (0.35,\ 0.7)`$. This bound is what the [ensemble's calibration
analysis](ensemble.md#calibration-what-actually-survives-the-default-thresholds) turns on.

### The graph surface runs exactly two rules

`analyze_cpg` delegates detection to `GnnSemanticScorer::detect_issues`, which — despite the name —
executes no model. It walks the CPG once and applies two hand-written rules:

| Rule | Condition | `IssueType` | $`\iota`$ confidence | `suggestion` |
|---|---|---|---|---|
| **Unused binding** | a `Variable` node with an incoming `DfgWrite`/`DfgFlow` edge but **no** outgoing `DfgRead`/`DfgFlow` | `UnusedBinding` | $`0.6`$ | `Some("Variable may be unused")` |
| **Unhandled error** | a `Call` node whose name is `unwrap` or `expect` | `MissingErrorHandling` | $`0.75`$ | (set by the scorer) |

`analyze_cpg` then applies two filters: it drops issues below $`\theta_{\text{sem}} = 0.5`$ (both
rules clear it, at $`0.6`$ and $`0.75`$), and it drops issue types whose `check_*` flag is off.

**A consequence worth stating plainly: `detect_issues` never emits `VariableMisuse`, `TypeError`,
`NullDereference`, `ApiMisuse`, `ResourceLeak`, or `Anomaly`.** The `issue_to_corrections` match
arms for those types are therefore **unreachable from `analyze_parsed`** in the shipped crate —
including the `VariableMisuse` arm, which is the only caller of `find_variable_misuse`, the only
consumer of the GNN's `variable_misuse_candidates`, and the only place the variable registry meets
the graph. The most interesting code in the module is dead code today, reachable only by
constructing a `SemanticIssue` yourself and calling the (public) helpers directly.

So `analyze_parsed` produces exactly two shapes of correction:

| From | `CorrectionKind` | `CorrectionSource` | Confidence | Note |
|---|---|---|---|---|
| `UnusedBinding` | `Deletion` | `DataFlow` | $`0.6 \times 0.6 = 0.36`$ | deletes the binding's span; requires `suggestion.is_some()` |
| `MissingErrorHandling` | `Other` | `ControlFlow` | $`0.75 \times 0.5 = 0.375`$ | **replacement = original**, so `is_noop()` is `true` |

The second is deliberately a **no-op advisory**: it carries the message *"Consider adding error
handling"* and an edit that changes nothing, because a real fix would need language-specific
knowledge the corrector does not have. It is a diagnostic wearing a `Correction`'s clothes; do not
apply it blindly.

### Dead state: `known_functions`, `type_name`, `scope_level`

Three pieces of carefully-threaded state are **written and never read**:

- **`known_functions`** and the whole `FunctionInfo` struct (`param_types`, `return_type`,
  `arity`). `register_function` inserts into the map; nothing in the crate ever queries it, and
  `register_function` has no callers — not even the ensemble, which exposes only
  `register_variables`.
- **`VariableInfo::type_name`.** The ensemble's `register_variables` takes
  `&[(String, Option<String>)]` and faithfully stores the declared type, and
  `find_variable_misuse`'s comment promises to score on "*type compatibility (if known)*". It does
  not: only `use_count` is read. Type information entered here goes nowhere.
- **`VariableInfo::scope_level`.** Stored on first insert (the ensemble always passes $`0`$), never
  consulted, and silently ignored on re-registration.

Note also that `use_count` counts **registrations**, not uses: it is incremented once per
`register_variable` call, so a variable registered once has $`k_n = 1`$ and hence zero usage boost.

## Theory of what does run

### Normalized name similarity

The corrector converts Levenshtein distance into a bounded similarity by dividing by the longer
string:

```math
\begin{array}{lr}
\displaystyle \mathrm{sim}(a,b) \;=\; 1 \;-\; \frac{d_L(a,b)}{\max\bigl(\lvert a \rvert,\ \lvert b \rvert\bigr)}
\;\in\; [0, 1] & \text{(S1)}
\end{array}
```

with $`\mathrm{sim}(a,a) = 1`$ and $`\mathrm{sim} = 0`$ if either operand is empty. The range
follows from the Lemma in [Lexical](lexical.md#the-length-prefilter-is-sound):
$`d_L(a,b) \leq \max(\lvert a \rvert, \lvert b \rvert)`$ always, so the quotient never exceeds $`1`$.
This is the standard normalization that turns a distance into a similarity while keeping long-name
comparisons on the same scale as short ones — `count` versus `counter` scores
$`1 - 2/7 \approx 0.71`$, while `foo` versus `bar` scores $`0`$.

### The variable-misuse score

`find_variable_misuse` blends orthography with popularity. For a registered name $`n`$ with
`use_count` $`k_n`$:

```math
\begin{array}{lr}
\displaystyle u(n) \;=\; \frac{\max\bigl(\ln k_n,\ 0\bigr)}{10},
\qquad
s(q, n) \;=\; 0.7 \cdot \mathrm{sim}(q, n) \;+\; 0.3 \cdot u(n) & \text{(S2)}
\end{array}
```

Candidates with $`s > 0.3`$ are kept, deduplicated against the graph-derived candidates, sorted
descending, and truncated to `max_candidates`. The logarithm makes the popularity term deliberately
weak — a name registered once contributes $`u = 0`$, and it takes $`k_n = e^{10} \approx 22{,}026`$
registrations to reach $`u = 1`$ — so orthography dominates, and a variable must satisfy roughly
$`\mathrm{sim} > 0.43`$ to survive on name alone.

> **A commensurability caveat.** `find_variable_misuse` merges two candidate lists that are scored
> on **different scales**: the graph-derived list from `GnnSemanticScorer::variable_misuse_candidates`
> uses a **character-bigram Jaccard** similarity, while the registry-derived list uses
> $`(\mathrm{S2})`$, built on **normalized Levenshtein**. The two are then sorted against one
> another as though they were the same quantity. They are not, and the resulting ranking mixes
> units.

## The algorithm, literately

```
function correct_token(token):                        ▸ TRAIT SURFACE — no CPG, no GNN
    if token.token_type ≠ Identifier: return []
    if token.text ∈ known_variables:  return []       ▸ a known name is assumed correct
    cands <- []
    for n in known_variables.keys():                  ▸ O(V) full scan
        s <- name_similarity(token.text, n)           ▸ eq. (S1); own Wagner-Fischer DP
        if s > 0.5: cands.append((n, s))
    sort cands by s descending
    for (n, s) in cands[.. max_candidates]:
        emit Correction {
            kind: VariableMisuse, span: [off, off+len), replacement: n,
            confidence: s * 0.7,                      ▸ eq. (S3)
            source: Neural,                           ▸ ⚠ misleading: nothing neural ran
            context: "Unknown identifier, did you mean '{n}'?",
        }

function analyze_parsed(parsed, cpg):                  ▸ GRAPH SURFACE
    issues <- analyze_cpg(cpg)
    corrections <- concat( issue_to_corrections(ι, cpg, parsed.source) for ι in issues )
    sort corrections by confidence descending
    return corrections

function analyze_cpg(cpg):
    issues <- gnn_scorer.detect_issues(cpg)            ▸ TWO deterministic rules; no model
    retain ι where ι.confidence ≥ min_confidence       ▸ 0.5; both rules clear it
    retain ι where the matching check_* flag is set    ▸ MissingErrorHandling: always kept
    return issues

function issue_to_corrections(ι, cpg, source):
    ν <- cpg.all_nodes().find(n -> n.id = ι.node_idx)  ▸ ⚠ O(|N|) linear scan, per issue
    if ν is None: return []
    (b0, b1) <- ν.location;  original <- source[b0 .. b1)
    match ι.issue_type:
        UnusedBinding when ι.suggestion is Some ->
            emit Deletion  [b0,b1) -> ""      conf = ι.confidence * 0.6   source = DataFlow
        MissingErrorHandling ->
            emit Other     [b0,b1) -> original conf = ι.confidence * 0.5  source = ControlFlow
                                               ▸ a NO-OP: replacement = original
        VariableMisuse ->                      ▸ UNREACHABLE: detect_issues never emits it
            for (n, s) in find_variable_misuse(cpg, original, ι.node_idx):
                emit VariableMisuse [b0,b1) -> n   conf = ι.confidence * s   source = Neural
        TypeError when ι.suggestion is Some -> ▸ UNREACHABLE
            emit TypeError  conf = ι.confidence    source = TypeInference
        _ when ι.suggestion is Some ->         ▸ UNREACHABLE
            emit Other      conf = ι.confidence    source = Neural
```

![The GNN semantic scorer over the CPG, with its deterministic graph-rule fallback](../../../diagrams/codecorr-gnn.svg)

*Figure 1. The feature-extraction path (left) is real and produces GNN-ready tensors; the scoring
path it would feed is not wired. What the semantic corrector consumes today is the deterministic
rule engine and the lexical fallback on the right. This figure is shared with [GNN](../gnn.md),
which develops the feature algebra in full.*

## Engineering

### Types

```rust
pub struct SemanticCorrectorConfig {
    pub min_confidence: f64,        // default 0.5   -- gates ISSUES, not corrections
    pub max_candidates: usize,      // default 5
    pub check_variable_misuse: bool,// default true  -- moot: never emitted
    pub check_unused_bindings: bool,// default true  -- the live switch
    pub check_type_errors: bool,    // default true  -- moot: never emitted
    pub gnn_config: GnnConfig,      // forwarded to GnnSemanticScorer; unused by the rules
}

pub struct VariableInfo {
    pub name: String,
    pub type_name: Option<String>,  // DEAD: written, never read
    pub scope_level: usize,         // DEAD: written, never read
    pub use_count: usize,           // live: the k_n of (S2); counts REGISTRATIONS
}
```

`max_edit_distance` returns $`3`$ (hard-coded, wider than the lexical corrector's $`2`$ on the
grounds that "semantic corrections can involve larger changes"); `name()` returns
`"SemanticCorrector"`.

### A hand-rolled DP that should not exist

`SemanticCorrector::levenshtein_distance` is a **private, full-matrix Wagner–Fischer**
implementation:

```rust
let mut dp = vec![vec![0usize; n + 1]; m + 1];   // O(m*n) words, m+1 heap allocations
```

This is $`O(mn)`$ time **and** $`O(mn)`$ space, in a `Vec<Vec<usize>>` whose rows are scattered
across the heap. The sibling [Lexical corrector](lexical.md) solved the identical problem by
delegating to `liblevenshtein::distance::standard_distance`, which is bit-parallel
($`O(mn/w)`$ time, $`O(1)`$ allocation) for the ASCII identifiers that dominate here. Since
`correct_token` runs this DP against **every** registered variable, a single token costs $`V`$
matrix allocations. Replacing the body with a call to `standard_distance` is a strictly-better,
behavior-preserving change — the semantics of the two are identical (insert / delete / substitute).

Two smaller defects ride along:

- **Bytes versus characters.** `levenshtein_distance` counts **characters** (it collects
  `Vec<char>`), but `name_similarity` divides by `a.len().max(b.len())` — **bytes**. For non-ASCII
  identifiers the denominator is too large, so $`(\mathrm{S1})`$ **over-estimates** similarity. The
  same class of bug appears in the lexical corrector's prefilter, in the opposite direction.
- **The `.max(0.0)` in $`(\mathrm{S2})`$ is unreachable.** `use_count` starts at $`1`$ and only
  grows, so $`\ln k_n \geq 0`$ always. It is harmless defensive code.

### Complexity

Let $`V`$ be the number of registered variables, $`m`$ the mean identifier length, and
$`\lvert \mathcal{N} \rvert`$, $`\lvert \mathcal{E} \rvert`$ the CPG's node and edge counts.

| Operation | Time | Allocations |
|---|---|---|
| `name_similarity` | $`O(m^2)`$ | $`O(m)`$ `Vec`s, $`O(m^2)`$ words |
| `correct_token` | $`O(V \cdot m^2)`$ | $`O(V \cdot m)`$ |
| `detect_issues` | $`O(\lvert \mathcal{N} \rvert + \lvert \mathcal{E} \rvert)`$ | incident-edge queries, no edge buffer |
| `issue_to_corrections` | $`O(\lvert \mathcal{N} \rvert)`$ **per issue** | linear `find` over all nodes |
| `analyze_parsed` | $`O(\lvert \mathcal{N} \rvert + \lvert \mathcal{E} \rvert + \lvert I \rvert \cdot \lvert \mathcal{N} \rvert)`$ | $`\lvert I \rvert`$ = surviving issues |
| `find_variable_misuse` | $`O(\lvert \mathcal{N} \rvert \cdot m + V \cdot m^2)`$ | graph candidates plus registry scan |

The $`\lvert I \rvert \cdot \lvert \mathcal{N} \rvert`$ term is avoidable: the CPG node lookup is a
linear `all_nodes().find(...)` per issue, where a `HashMap<usize, &CpgNode>` built once would make
it $`O(1)`$.

### Concurrency

`SemanticCorrector<L>` is `Send + Sync` when `L` is. `analyze_cpg`, `analyze_parsed`, and the trait
methods all take `&self`; only `register_variable` / `register_function` need `&mut self`. The
`GnnSemanticScorer` it owns holds nothing but its config and an empty embedding cache, so it is
cheap to clone or construct.

## Usage

The two surfaces, used as they are meant to be:

```rust
use libgrammstein::code::ast::CodeParser; // note: NOT re-exported at `code::`
use libgrammstein::code::correction::CodeCorrector;
use libgrammstein::code::correctors::semantic::{SemanticCorrector, SemanticCorrectorConfig};
use libgrammstein::code::cpg::CodePropertyGraph;
use libgrammstein::code::language::{TokenContext, TokenType};
use libgrammstein::code::tokenizer::CodeToken;
use libgrammstein::code::Python;
use std::sync::Arc;

let python = Arc::new(Python::new());
let mut corrector = SemanticCorrector::with_defaults(Arc::clone(&python));

// Populate the registry. The Option<String> type is stored but never read (see Dead state).
corrector.register_variable("calculate_total".to_string(), Some("int".to_string()), 0);
corrector.register_variable("calculate_average".to_string(), None, 0);

// --- Trait surface: pure name similarity, eq. (S1)/(S3). ---
let token = CodeToken::new("calulate_total", 0, 0, 0, TokenType::Identifier, "identifier");
let context = TokenContext::new(TokenType::Identifier);
for c in corrector.correct_token(&token, &context) {
    // -> "calculate_total";  sim = 1 - 1/15 = 0.933;  confidence = 0.653
    println!("{} -> {} ({:.3})", c.original, c.replacement, c.confidence);
}

// --- Graph surface: the two deterministic CPG rules. ---
let mut parser = CodeParser::new(Arc::clone(&python))?;
let parsed = parser.parse("def f(items):\n    unused = 1\n    return sum(items)\n")?;
let cpg = CodePropertyGraph::from_parsed_code(&parsed);

for c in corrector.analyze_parsed(&parsed, &cpg) {
    // UnusedBinding -> a Deletion at confidence 0.36, source = DataFlow.
    println!("{:?} {:?} {:.2}: {}", c.kind, c.source, c.confidence,
             c.context.as_deref().unwrap_or(""));
}
# Ok::<(), libgrammstein::code::AstError>(())
```

To disable the one rule that actually fires, turn its check off:

```rust
let config = SemanticCorrectorConfig {
    check_unused_bindings: false, // the only live check_* switch
    ..Default::default()
};
```

Bear in mind that both graph-surface corrections land at confidence $`0.36`$ and $`0.375`$, and the
ensemble multiplies them by `semantic_weight = 0.25` before applying a $`0.3`$ floor — so by default
**neither survives**. See
[Ensemble](ensemble.md#calibration-what-actually-survives-the-default-thresholds).

## References

1. F. Yamaguchi, N. Golde, D. Arp & K. Rieck (2014). *Modeling and Discovering Vulnerabilities
   with Code Property Graphs.* IEEE Symposium on Security and Privacy, 590–604.
   [doi:10.1109/SP.2014.44](https://doi.org/10.1109/SP.2014.44)
2. M. Allamanis, M. Brockschmidt & M. Khademi (2018). *Learning to Represent Programs with Graphs.*
   ICLR 2018. [arXiv:1711.00740](https://arxiv.org/abs/1711.00740)
3. T. N. Kipf & M. Welling (2017). *Semi-Supervised Classification with Graph Convolutional
   Networks.* ICLR 2017. [arXiv:1609.02907](https://arxiv.org/abs/1609.02907)
4. V. I. Levenshtein (1966). *Binary codes capable of correcting deletions, insertions, and
   reversals.* Soviet Physics Doklady 10(8), 707–710.

## See also

- [Correctors Overview](overview.md) — the shared `CodeCorrector` contract
- [GNN](../gnn.md) — the feature pipeline, the issue taxonomy, and the target architecture
- [CPG](../cpg.md) — the graph the two rules walk, and its `DfgRead` / `DfgWrite` edges
- [Lexical Corrector](lexical.md) — the same Levenshtein machinery, done with liblevenshtein
- [Ensemble Corrector](ensemble.md) — why semantic corrections need a corroborating source by default
- [Code Embeddings](../embeddings.md) — the transformer features a trained model would consume
