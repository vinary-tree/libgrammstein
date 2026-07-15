# Dictionary Backend — feeding the Levenshtein automaton

[← hierarchical correction](./hierarchical-correction.md)

> **Notation.** Mathematical prose uses MathJax delimited for GitHub-flavored
> Markdown: inline math is a backtick span wrapped in dollar signs, and display
> math is a fenced block whose info-string is `math`. Bare dollar delimiters are
> never used. PlantUML diagram labels are the one exception: they use Unicode
> glyphs, since SVG cannot typeset MathJax.

Dimension 1 (orthographic correction) builds a **Levenshtein automaton** over the
in-vocabulary words and intersects it with a query to enumerate every dictionary
word within edit distance $`k`$. A natural question at Google-Books scale: to feed
that automaton, must the $`\sim\!13`$-million-term vocabulary be exported to an
in-memory trie, or can the automaton run directly over the memory-mapped
persistent vocabulary?

The code-verified answer:

1. The `DoubleArrayTrie` (DAT) export is **not** necessary. It remains an
   optional optimization for *small, static* dictionaries (≈30 ns reads), but it
   is the wrong default at Google-Books scale, where materializing the whole
   vocabulary in RAM defeats the purpose of persistent storage.
2. The automaton runs **directly over the persistent trie**. `SharedVocabARTrie`
   implements `libdictenstein::Dictionary`, and its node handle now descends the
   full depth of the trie, so a `Transducer<SharedVocabARTrie>` enumerates
   multi-character corrections against the memory-mapped overlay with **no
   in-RAM materialization** — no full-vocabulary copy, no $`O(N)`$ rebuild,
   eviction and persistence intact.

![Feeding the Levenshtein automaton directly from the persistent vocabulary trie. Dictionary::root and DictionaryNode::transition walk the lock-free overlay; each transitioned node keeps an Arc to its overlay node and re-resolves children on demand, so the automaton descends the full depth of the trie and enumerates every in-vocabulary word within edit distance k with no in-RAM materialization. The optional DoubleArrayTrie export is shown as a side branch reserved for small, static dictionaries.](../../diagrams/correction-dictionary-backend.svg)

**Figure.** Feeding the Levenshtein automaton directly from the persistent
vocabulary trie; the green path is the default in-place traversal, the dashed
blue branch is the optional materialized trie for small static dictionaries.
*(Rendered from `docs/diagrams/correction-dictionary-backend.puml`.)*

---

## The traversal contract

A Levenshtein automaton needs exactly three things from its dictionary: a **root
node**, a **child transition** by character, and an **is-final** test. That is
the `libdictenstein::Dictionary` / `DictionaryNode` trait pair:

```rust
// libdictenstein/src/lib.rs
pub trait Dictionary {
    type Node: DictionaryNode;
    fn root(&self) -> Self::Node;
    fn contains(&self, term: &str) -> bool { /* default: walk from root */ }
    fn len(&self) -> Option<usize>;
}
pub trait DictionaryNode: Clone + Send + Sync {
    type Unit: CharUnit;                 // `char` for the UTF-8 vocab trie
    fn is_final(&self) -> bool;
    fn transition(&self, unit: Self::Unit) -> Option<Self>;
}
```

The automaton simulates $`A(w, k)`$ — the set of strings within edit distance $`k`$
of the query $`w`$ — and walks it in lockstep with the dictionary graph, pruning a
subtree the instant no automaton state survives. The recognition cost is thus
$`O(|w|)`$-bounded, independent of the dictionary size (Schulz & Mihov 2002).
**Crucially, this requires `transition` to descend the full depth of the trie.**

## How the persistent trie satisfies the contract

`SharedVocabARTrie` (a bare `Arc<PersistentVocabARTrie>` after libdictenstein's
F4 lock-collapse) implements `Dictionary`, and `contains()` walks the real
lock-free overlay. Its associated node type is

```rust
// libdictenstein/src/persistent_artrie/vocab/mod.rs
impl Dictionary for SharedVocabARTrie {
    type Node = VocabTrieNodeRef;
    fn root(&self) -> Self::Node { /* overlay-backed root */ }
    /* … */
}

/// The shared overlay handle — the SAME node type the byte and char tries expose.
pub type VocabTrieNodeRef = OverlayDictionaryNode<CharKey, u64>;
```

`VocabTrieNodeRef` is an alias for the canonical `OverlayDictionaryNode<CharKey,
u64>`. Each node **owns an `Arc` to its overlay node**, so `transition` and
`edges` re-resolve a child's children on demand and hand back a node that itself
holds the child `Arc`. A caller — a liblevenshtein `Transducer`, or any generic
`DictionaryNode` walk — can therefore descend the **full depth** of the trie.
`root()` returns a node that owns the overlay-root `Arc`, so it outlives the
transient no-lock read shim; and because the vocabulary never evicts, every
overlay child is resident (`Child::InMem`), so no disk fault can interrupt the
walk.

> **History.** An earlier `VocabTrieNodeRef` was a shallow depth-1 *snapshot*
> struct whose `transition` stamped every returned child with an empty
> `children` vector — "we can't get the child's children without trie access" —
> so a transducer over it dead-ended after a single character and could only
> match depth-1 terms. That was the "returns childless nodes" bug reported by
> libgrammstein. It was fixed in libdictenstein by superseding the snapshot with
> the shared `OverlayDictionaryNode`, which holds the child `Arc` and re-resolves
> children on demand; the truncation is gone. libgrammstein required **no**
> change to the generic layer to benefit — it already accepts any working
> `Dictionary`.

## What the corrector does

`HierarchicalCorrector` builds the spelling automaton **directly over the
persistent vocabulary** — the spelling dictionary handle is the vocabulary trie
itself, moved in as a cheap `Arc`:

```rust
use liblevenshtein::transducer::{Transducer, Algorithm};

// `vocabulary: SharedVocabARTrie` — the live, memory-mapped persistent trie.
// No `iter_terms()`, no DAWG/DAT materialization: the automaton walks the trie.
let transducer = Transducer::new(vocabulary, Algorithm::Transposition);
for candidate in transducer.query_with_distance("teh", 2) {
    // candidate.term ∈ {"the", "ten", "tea", …}; candidate.distance is the edit cost
}
```

At full Google-Books scale this holds only the memory-mapped overlay resident,
not a second in-RAM copy of the vocabulary — exactly the memory cost a
materialized `DynamicDawgChar` / `DoubleArrayTrie` would incur. The regression
test `levenshtein_layer_over_persistent_vocab_descends_full_depth`
([`src/integration/corrector.rs`](../../../src/integration/corrector.rs)) guards
the full-depth descent: it recovers `"quikc"` → `"quick"` (a distance-1
transposition reachable only at depth 5) over a `SharedVocabARTrie`.

## When a materialized trie is still the right tool

The generic `LevenshteinCorrectionLayer` is parameterized over any
`D: libdictenstein::Dictionary`, so a caller with a **small, static** dictionary
can still build the layer over an in-memory automaton:

```rust
use libdictenstein::dynamic_dawg::char::DynamicDawgChar;

// Small, fixed word list → a frozen, suffix-shared DAWG is a fine backend.
let dawg = DynamicDawgChar::<()>::from_terms(vocabulary.iter_terms());
let layer = LevenshteinCorrectionLayer::new(dawg, edit_config);
```

`DynamicDawgChar` shares suffixes and is markedly more compact than a
double-array for a natural vocabulary; `DoubleArrayTrie` trades that compactness
for ≈30 ns reads once frozen. Both are **in-memory**, so they are appropriate
only when the dictionary is small enough to hold residently — which is why the
façade defaults to the persistent trie.

## Design consequence

`LevenshteinCorrectionLayer` is generic over
`D: libdictenstein::Dictionary + Clone + Send + Sync` (the exact bound set that
liblevenshtein's `Transducer` requires — copied from
`lling-llang/src/integration/liblevenshtein_bridge.rs::fuzzy_lookup`) and holds
exactly **one** `Transducer<D>`, built at construction. It therefore works over
*any* conforming dictionary — the persistent `SharedVocabARTrie` the façade uses
by default, or a materialized `DynamicDawgChar` for a small static word list —
with no change to the layer. See [pipeline-assembly.md](./pipeline-assembly.md).

## References

- Schulz, K. & Mihov, S. (2002). *Fast String Correction with Levenshtein
  Automata.* IJDAR. DOI: [10.1007/s10032-002-0082-8](https://doi.org/10.1007/s10032-002-0082-8)
