# libgrammstein diagrams

Every diagram in the libgrammstein documentation is authored in **PlantUML**
(`*.puml`) and committed alongside its rendered **`*.svg`**. The Markdown docs
embed the `.svg`; the `.puml` is the source of truth. There is no Mermaid in this
repository.

## Rendering

SVGs are committed, so a clone needs no tooling to *read* the docs. To *regenerate*
them after editing a `.puml` (requires [PlantUML](https://plantuml.com);
`plantuml` on `PATH`):

```sh
# from the repository root — regenerate every diagram
for f in docs/diagrams/*.puml; do plantuml -tsvg "$f"; done

# or a single diagram (each .puml's header carries its own Render: line)
plantuml -tsvg docs/diagrams/correction-component-map.puml
```

> **Render `<latex>` diagrams under an LTS JVM (Java ≤ 22).** JLaTeXMath draws some
> symbols at their Latin-1 font codepoints, and **Java 26 regressed** the rendering
> of the soft-hyphen codepoint `U+00AD` — it silently drops whatever glyph is mapped
> there, which includes `\otimes` (verified: `\otimes` renders under Java 11/17/21/22
> and is blank under Java 26). If your default `java` is 26+, run PlantUML on an LTS
> JVM, e.g. `` `/usr/lib/jvm/java-21-openjdk/bin/java -jar` `` `/usr/share/java/plantuml/plantuml.jar -tsvg <file>.puml`
> (or `archlinux-java set java-21-openjdk`). Diagrams without `<latex>` math render
> identically on any JVM.

## Conventions

- **House style, not `!theme`.** Each file sets explicit `skinparam`s (background
  `#FAFAFA`, `linetype ortho`, rounded white rectangles, `ArrowColor #37474F`) and
  colours each node *per concept*. The repo-wide semantic colour legend is
  documented in the header of [`architecture.puml`](./architecture.puml).
- **Self-documenting header.** Every `.puml` opens with a `'` comment block that
  says what the diagram draws, a `' Render:` line with its exact command, and a
  `' Source of truth for Figure N in <doc> §M` provenance back-reference. That
  header — not this index — is the authority for a diagram's exact figure number.
- **Math in labels uses PlantUML `<latex>`.** Mathematical formulae in diagram
  labels are typeset with PlantUML `<latex>…</latex>` — the bundled JLaTeXMath
  renders vector math straight into the SVG — honouring the docs' "LaTeX for math
  prose" rule (e.g. the `arch-*` and `correction-*` diagrams). Non-mathematical
  Unicode (box-drawing, flow arrows, bullet separators) stays as glyphs (see the
  Notation note in
  [`../integration/lling-llang/hierarchical-correction.md`](../integration/lling-llang/hierarchical-correction.md)).
- **Colour activities with the suffix form.** In activity diagrams write
  `:label; <<#546E7A>>`, never the prefix form `#546E7A:label;`. The prefix form is
  deprecated and **PlantUML silently discards the colour**, so the node renders
  uncoloured *and* the image gains a "This syntax is deprecated" banner. Enforced by
  `deprecated-activity-colour` below.

## Linting

```sh
raku scripts/lint-docs.raku              # text rules: fast, no JVM
raku scripts/lint-docs.raku --render     # + rendered-SVG diagnostics (~2 min, renders all 147)
raku scripts/lint-docs.raku --fix        # repair the auto-fixable kinds in place
raku scripts/lint-docs.raku --list-rules # what each rule is and why it exists
```

Wired into the formal gate: `make -C formal lint-docs` (part of `source-hygiene-fast`)
and `make -C formal lint-diagrams` (part of `check`). Exit codes: `0` clean, `1`
violations, `2` usage/environment error.

**Why a linter at all.** Every rule encodes a defect that shipped and that nothing
caught, because both renderers **fail silently**:

| Failure mode | Why it is invisible |
|---|---|
| Deprecated activity colour | PlantUML exits `0`, prints nothing to stderr, and writes the warning *into the image*. `-checkonly` reports nothing. |
| Error graphics, leaked `<latex>` | Same — rendered into the SVG, exit `0`. |
| Dropped glyph | A missing glyph is still a well-formed `<image>`, so counting images proves nothing; the rule decodes each embedded LaTeX image and asserts it drew ≥1 path. This is the `\otimes`/Java-26 class described above. |
| `\text{a\_b}` in Markdown | MathJax does not expand backslash macros inside `\text{}`, so it renders a **literal backslash**. Note this is *renderer-specific*: JLaTeXMath handles `\_` correctly, so the rule is Markdown-scoped and diagram sources are left alone. |
| `\operatorname`, `` `$x$` `` | GitHub refuses to render the span, or renders it as inert code. |

`--render` **pins Java 21** for the same reason the render command above does: under
Java 26 the `U+00AD` regression would make the dropped-glyph rule fire on every
`\otimes`.

## Index

**146 diagrams**, grouped below by subsystem (the filename prefix *is* the group).
Each row gives the diagram's stem — the files on disk are `<stem>.puml` and its
committed `<stem>.svg` — and a one-line statement of what it draws, taken from that
diagram's own header comment. Each group names the documents that consume it.

| Group | Prefixes | Count |
|---|---|---:|
| [Architecture](#architecture) | `architecture`, `arch-*` | 9 |
| [Statistical core — n-gram](#statistical-core--n-gram) | `ngram-*`, `mkn-*` | 4 |
| [Hybrid](#hybrid) | `hybrid-*` | 2 |
| [Embeddings](#embeddings) | `embedding-*` | 6 |
| [Scoring](#scoring) | `scoring-*` | 2 |
| [Generation](#generation) | `generation-*` | 2 |
| [Corpus & dictionary](#corpus--dictionary) | `corpus-*`, `dict-*` | 14 |
| [Language](#language) | `lang-*` | 3 |
| [Acoustic](#acoustic) | `acoustic-*` | 5 |
| [Neural](#neural) | `neural-*` | 9 |
| [Retrieval (RAG)](#retrieval-rag) | `rag-*` | 7 |
| [Topic & paradigm](#topic--paradigm) | `topic-*`, `paradigm-*` | 10 |
| [Code](#code) | `code-*`, `codecorr-*`, `codeemb-*`, `subtree-*` | 29 |
| [LaTeX](#latex) | `latex-*` | 7 |
| [PDF](#pdf) | `pdf-*` | 3 |
| [Correction (lling-llang)](#correction-lling-llang) | `correction-*`, `lling-*`, `grammar-corrector-*` | 13 |
| [Storage & Levenshtein](#storage--levenshtein) | `levenshtein-*` | 4 |
| [Cron](#cron) | `cron-*` | 2 |
| [CLI & training](#cli--training) | `cli-*`, `training-*` | 8 |
| [Examples](#examples) | `example-*` | 4 |
| [Cross-cutting](#cross-cutting) | `formal-verification`, `importer-eviction`, `cpg-triad` | 3 |

### Architecture

Consumed by [`../architecture/`](../architecture/).

| Diagram | Draws |
|---|---|
| `architecture` | The layered architecture of the crate — and the repo-wide semantic colour legend every other diagram inherits |
| `arch-layer-contracts` | The trait seams between layers: what each layer promises the layer above and requires from the layer below |
| `arch-dataflow` | Corpus bytes → tokens → counts → a smoothed model, and the mirror-image query path back out; colour encodes stage *kind*, not crate |
| `arch-serialization` | The two model-persistence paths: the fast native format welded to one dictionary backend, and the backend-agnostic portable one |
| `arch-threading` | The thread inventory: which threads exist, what each owns, and which synchronization primitive mediates every shared edge |
| `arch-memory-budget` | The heap decomposition of a full Google Books import — every term, its bound, and the one term that was unbounded |
| `arch-importer-pipeline` | One prefix file's journey from a remote `.gz` to a durable MKN-annotated shard, in three concurrent swimlanes |
| `arch-importer-sharding` | Shard routing, and the anatomy of a single shard: the overlay / mmap image / WAL three-tier stack |
| `arch-checkpoint-state` | The two state machines that make a crash survivable — a prefix's lifecycle and a shard's sync lifecycle — each a live TLA+ model |

### Statistical core — n-gram

Consumed by [`../components/ngram/`](../components/ngram/).

| Diagram | Draws |
|---|---|
| `ngram-training` | Two-pass training: lock-free parallel counting, then Kneser-Ney continuation statistics and discounts from the count-of-counts |
| `ngram-query` | The read-only query surface: one `log_prob` call fans out into at most $`n`$ trie lookups, one per backoff level |
| `ngram-trie-storage` | Key encoding: words interned to vocabulary indices, LEB128-varint encoded, each byte carried as a Latin-1 char |
| `mkn-backoff` | The Modified Kneser-Ney backoff recursion: one node per order's discounted estimate, one edge per backoff weight $`\gamma(h)`$ |

### Hybrid

Consumed by [`../components/hybrid/`](../components/hybrid/).

| Diagram | Draws |
|---|---|
| `hybrid-scoring` | n-gram $`\oplus`$ embedding fusion in `HybridLanguageModel`: parallel branches, a strategy switch, a lock-free memo cache |
| `hybrid-oov` | The two *independent* OOV surfaces: the always-on floor inside `score`, and the standalone `OovHandler` the model never calls |

### Embeddings

Consumed by [`../components/embedding/`](../components/embedding/).

| Diagram | Draws |
|---|---|
| `embedding-subword` | FastText-style composition: a word becomes the mean of its character n-gram bucket vectors, averaged with its word row |
| `embedding-skipgram` | One skip-gram update with negative sampling: the enriched input vector scored against the true context and $`k`$ noise words |
| `embedding-bpe` | Byte-Pair Encoding: the trainer greedily merges the most frequent adjacent pair; the tokenizer replays those merges by rank |
| `embedding-similarity` | `most_similar` / `analogy`: an enriched query scored by cosine against the *raw* embedding rows — the asymmetry — then ranked |
| `embedding-phonetic` | Phonetic-enhanced similarity: orthographic cosine blended with the cosine of Zompist-normalized forms |
| `embedding-acoustic` | Acoustic word embedding: variable-length frames → encoder → pooling → optional text projection → an indexable fixed vector |

### Scoring

Consumed by [`../components/scoring/`](../components/scoring/) and [`../api/scoring.md`](../api/scoring.md).

| Diagram | Draws |
|---|---|
| `scoring-perplexity` | How `Perplexity::corpus_perplexity` turns a stream of corpus sentences into PP, an OOV rate, and the token/sentence tallies |
| `scoring-sentence` | The three views of one sentence's probability — and the length bias that makes ranking by raw `log_prob` a trap |

### Generation

Consumed by [`../components/generation/`](../components/generation/).

| Diagram | Draws |
|---|---|
| `generation-autoregressive-loop` | `TextGenerator`'s loop: window the context, score the vocabulary, decode, stop-test, append |
| `generation-sampling` | `sample_token`: score ⇒ temperature ⇒ softmax ⇒ sort ⇒ top-k ⇒ nucleus ⇒ renormalize ⇒ draw |

### Corpus & dictionary

Consumed by [`../components/corpus/`](../components/corpus/) and [`../components/dictionary/`](../components/dictionary/).

| Diagram | Draws |
|---|---|
| `corpus-pipeline` | The corpus module end-to-end: a source of bytes becomes a stream of sentences (muted nodes = opt-in, not wired by default) |
| `corpus-reader-trait` | The `CorpusReader` contract: two required methods, three concrete readers, and the blanket impl that lets the CLI dispatch at run time |
| `corpus-formats` | How a path plus a `--format` flag becomes a `Box<dyn CorpusReader>` — the format is *declared*, never sniffed |
| `corpus-streaming` | Why a 20 GB dump fits on a laptop: what is resident at each rung of the lazy pull chain, and the unit of residency per reader |
| `corpus-prefetch-sequence` | `PrefetchingReader`: one producer thread, a bounded channel, and the three shutdown paths (Done, Error, early Drop) |
| `corpus-quality-dedup` | The two gates a sentence must clear: eight ordered quality predicates (first failure wins), then the deduplication mode |
| `corpus-wikipedia-state` | `WikipediaIterator` as a four-flag state machine over quick-xml pull events; a document is emitted only at `</page>` |
| `corpus-gutenberg-strip` | `strip_gutenberg_boilerplate`: choosing the half-open window between the first matching START and END markers |
| `dict-pipeline` | The two ways to obtain a dictionary: the standalone extractor ⇒ builder path, and the zero-copy view over an existing n-gram model |
| `dict-backends` | `SpellingDictionary` vs `VocabularyDictionary`: what backs each, what each costs, and what each can answer |
| `dict-build` | `DictionaryBuilder`: two entry points, one four-step construction, and the descending-frequency sort that makes `rank()` free |
| `dict-extract-concurrency` | Why `WordExtractor::add_sentence` takes `&self`: DashMap shards the keyspace, and the two counters are atomics |
| `dict-normalize` | `normalize_word`, the five-step gauntlet — and how the default config silently discards "don't" and "well-known" |
| `dict-file-format` | The two on-disk forms of a `SpellingDictionary`: the versioned binary container and the always-available tab-separated text |

### Language

Consumed by [`../components/language/`](../components/language/).

| Diagram | Draws |
|---|---|
| `lang-detection` | `detect_language`: whatlang info → confidence gate → ISO 639-1 remap → `LanguageTag`, and its three failure modes |
| `lang-tokenizer` | `create_tokenizer` dispatch: the primary language subtag selects a `Tokenizer` impl (and which impls are never selected) |
| `lang-registry` | `ModelRegistry::scan` builds a per-language index; `find_best_match` resolves a tag through a three-step fallback chain |

### Acoustic

Consumed by [`../components/acoustic/`](../components/acoustic/).

| Diagram | Draws |
|---|---|
| `acoustic-mfcc-pipeline` | The signal-processing trunk shared by every `FeatureExtractor` method, with its four output taps |
| `acoustic-mel-filterbank` | How the triangular mel filterbank bins a linear-frequency power spectrum into perceptual channels |
| `acoustic-streaming` | `StreamingFeatureExtractor`: chunks accumulate, complete frames are emitted, overlap is retained, flush pads |
| `acoustic-model-architecture` | Neural acoustic models: the shared encode → project → log-softmax skeleton, Linear vs Transformer |
| `acoustic-asr-cascade` | Where the acoustic stage sits in the lling-llang ASR cascade $`H \circ C \circ L \circ G`$ |

### Neural

Consumed by [`../components/neural/`](../components/neural/).

| Diagram | Draws |
|---|---|
| `neural-overview` | Map of the `neural` module: one `ModernBertModel` loaded once and shared by `Arc` across embedder, rescorer, and summarizer |
| `neural-modernbert-architecture` | ModernBERT-base as the crate drives it: RoPE embeddings, 22 pre-norm GeGLU layers, and the two exits the wrapper exposes |
| `neural-embedder` | `embed` and `embed_batch` — both `&self`, so one embedder serves many threads; the batch path partitions on the cache first |
| `neural-cache` | `EmbeddingCache`: a lock-free DashMap payload plus a mutex-guarded VecDeque for LRU recency — where a lock *is* and *is not* taken |
| `neural-kv-cache` | The KV-cache family, its growth law, and its two traps — drawn muted because no call site in the crate constructs it |
| `neural-rescorer` | `rescore_paths`: the n-gram beam's top-k paths scored by MLM pseudo-perplexity, mixed with the n-gram score, re-sorted |
| `neural-pseudo-perplexity` | MLM pseudo-perplexity of one sentence: $`T`$ masked copies, $`T`$ forward passes, one gathered log-probability each |
| `neural-summarizer-pipeline` | `Summarizer::extractive` end-to-end, with its two short-circuits and the index-provenance the MMR step depends on |
| `neural-mmr` | Maximal Marginal Relevance as instantiated here: the query is replaced by the document centroid, so relevance means centrality |

### Retrieval (RAG)

Consumed by [`../components/rag/`](../components/rag/).

| Diagram | Draws |
|---|---|
| `rag-pipeline` | The end-to-end RAG data path: index-time ingestion meeting query-time retrieval at the shared `RagIndex` |
| `rag-document` | The document model: `DocumentBuilder` → `Document` → `DocumentMeta`, plus the synopsis provenance fork |
| `rag-index` | `RagIndex<B>` as the join of a geometric backend, a metadata map, and an optional topic model — and its on-disk layout |
| `rag-backends` | The `RetrievalBackend` contract, its two implementations, and the size-driven selection rule |
| `rag-hnsw` | HNSW search: a hierarchy of proximity graphs, greedy descent from the entry point, then a best-first beam at layer 0 |
| `rag-builder` | Sequential vs rayon-parallel index building — both sharing one embedder and one summarizer, whose methods are thread-safe |
| `rag-retriever` | The query pipeline: text → embedding → backend top-k → metadata join → threshold/synopsis filter → ranked results |

### Topic & paradigm

Consumed by [`../components/topic/`](../components/topic/) and [`../components/paradigm/`](../components/paradigm/).

| Diagram | Draws |
|---|---|
| `topic-pipeline` | The BERTopic-style pipeline end-to-end: embed → cluster → describe → model |
| `topic-clustering` | The agglomerative merge loop: distance matrix → find-min → Lance-Williams update |
| `topic-dendrogram` | A binary merge tree with a horizontal cut yielding two clusters |
| `topic-ctfidf` | Class-based TF-IDF: tokenize → atomic vocabulary and per-topic term counts → score → top-k |
| `paradigm-overview` | The three engines of the paradigm subsystem, and the artifacts that flow between them |
| `paradigm-indicators` | The indicator taxonomy: 4 primary paradigms, 19 categories, 170 pattern definitions, and the three weight tiers |
| `paradigm-detection` | The `analyze_tokens` dataflow: indexed pattern probe → confidence → weighted accumulation → density normalisation → profile |
| `paradigm-dominance` | `dominant_paradigm`: the two-test decision rule that turns four scores into one verdict — or None, or Mixed |
| `paradigm-prefixspan` | PrefixSpan by prefix projection, traced on the five file-I/O sequences of the api_patterns suite |
| `paradigm-domain-catalogs` | The two F1R3FLY.io DSL catalogs and the matcher that runs them |

### Code

Consumed by [`../components/code/`](../components/code/), [`../components/code-embeddings/`](../components/code-embeddings/), and [`../components/subtree/`](../components/subtree/).

**Core** (`code-*`)

| Diagram | Draws |
|---|---|
| `code-overview` | The module end-to-end: source → tree-sitter → analysis lanes → ensemble → ranked corrections |
| `code-pipeline` | `CorrectionPipeline::analyze` — six phases feeding a bounded streaming top-k collector |
| `code-ast-flow` | The AST parsing pipeline: cache probe → tree-sitter → `ParsedCode` → owned `AstNode`, plus the incremental-edit path |
| `code-tokenizer` | Tokenization: descend to AST leaves, classify by (text, node kind), filter, enrich `TokenContext` |
| `code-language` | The `CodeLanguage` trait and its supporting token-classification vocabulary |
| `code-languages` | The five shipped `CodeLanguage` implementations, their feature gates and their tree-sitter grammars |
| `code-correction` | The correction data model: `Correction`, its taxonomy, the `CodeCorrector` trait, and ranked candidates |
| `code-cpg-construction` | The three-pass CPG build over one AST: `build_from_ast` → `build_cfg` → `build_dfg` into a petgraph `DiGraph` |

**Correctors** (`codecorr-*`)

| Diagram | Draws |
|---|---|
| `codecorr-correctors` | The three-layer stack: parse once, fan out to the lexical, grammar, and semantic correctors, aggregate in the ensemble |
| `codecorr-lexical` | `LexicalCorrector`: route a token to the dictionary bank for its type, fuzzy-match via liblevenshtein, score the candidates |
| `codecorr-grammar` | `GrammarCorrector`: seed an Earley chart from the PCFG, read off the admissible terminals, derive replace / delete / insert repairs |
| `codecorr-ensemble` | `EnsembleCorrector`: gather weighted candidates, group by (replacement, span), boost on agreement, filter and rank |
| `codecorr-earley` | The Earley chart cycle behind grammar-constrained decoding: predict, scan, complete, then project into a decoding token mask |
| `codecorr-pcfg` | PCFG training: harvest productions from named AST nodes, count them, normalize per left-hand side into a distribution |
| `codecorr-wfst` | PCFG ⇒ WFST: depth-bounded unrolling of productions into weighted arcs over a semiring, for lling-llang composition |
| `codecorr-gnn` | The GNN semantic scorer: featurize the CPG, propagate with graph convolutions, emit issues — with a deterministic rule fallback |
| `codecorr-treeminer` | TreeminerD frequent-subtree mining: vertical representation, level-wise extension, anti-monotone support pruning |
| `codecorr-embeddings` | `CodeEmbedder`: format/architecture auto-detection, backend dispatch, cache, and cosine similarity over snippets |

**Embeddings** (`codeemb-*`)

| Diagram | Draws |
|---|---|
| `codeemb-pipeline` | The single-snippet pipeline shared by all three ONNX embedders: cache probe → tokenize → session → pool → L2 → cache insert |
| `codeemb-module-map` | The two distinct types both named `CodeEmbedder` — the *trait* in `neural::code` and the *struct* in `code::embeddings` |
| `codeemb-pooling` | How a rank-3 hidden-state tensor collapses to one vector, and where the three shipped embedders disagree (mean-pool vs CLS) |
| `codeemb-cache` | `CodeEmbeddingCache`: key derivation splits on code length, the DashMap probe is lock-free, and eviction is *arbitrary*, not LRU |
| `codeemb-codet5` | CodeT5+ `embed_code`: optional language prefix, tokenize and truncate, ONNX run, rank-dispatch, L2 normalize, cache |
| `codeemb-unixcoder` | UniXcoder's three prefix-selected modes in the paper, versus what the shipped embedder actually feeds the graph |
| `codeemb-graphcodebert` | GraphCodeBERT's two input channels: the token channel is wired; the data-flow channel the paper adds is *not* |
| `codeemb-ensemble` | `EnsembleCodeEmbedder`: members queried in sequence, vectors fused by one of four strategies, the result L2-normalized |

**Subtree mining** (`subtree-*`)

| Diagram | Draws |
|---|---|
| `subtree-pipeline` | Source code to frequent subtree patterns, via TreeMinerD over a `FlatTree` forest |
| `subtree-encoding` | A small AST and its depth-first `FlatTree` encoding, with (label, depth, scope) |
| `subtree-treeminer` | The TreeMinerD level-wise loop: vertical representation → 1-subtrees → extend → prune |

### LaTeX

Consumed by [`../components/latex/`](../components/latex/).

| Diagram | Draws |
|---|---|
| `latex-pipeline` | End-to-end LaTeX scoring: tokenize → detect modes → per-mode n-gram and heuristic scorer, with optional neural rescoring |
| `latex-modeaware-tokenizer` | The mode-aware lexer state machine: a math stack toggles Text vs Math, so one character lexes differently by mode |
| `latex-ngram-modes` | Mode-separated n-gram scoring: carve the token stream into homogeneous regions, route each to its model, fuse the log-probs |
| `latex-scorer-components` | The self-contained heuristic scorer: structural, fluency, and coherence proxies fused by weight, then a variance confidence |
| `latex-rescorer-fallback` | Neural rescoring with graceful degradation: each component yields `Some`/`None`, and the combiner renormalises over what is present |
| `latex-embedding-topk` | `LaTeXEmbedder` internals, and the $`O(n \log k)`$ bounded min-heap behind top-k nearest-neighbour selection |
| `latex-rag-retrieval` | Equation retrieval: id/domain indices, then an exact linear cosine scan with a similarity floor, fronted by a quantised cache |

### PDF

Consumed by [`../components/sources/pdf.md`](../components/sources/pdf.md).

| Diagram | Draws |
|---|---|
| `pdf-pipeline` | `PdfExtractor::extract` — route, shell out to a Python backend, post-process; dashed edges mark config that never reaches the backend |
| `pdf-router` | `PdfRouter::route` — the full backend-selection decision procedure, in evaluation order |
| `pdf-postprocess` | `PostProcessor::process_content` — six order-dependent passes, each gated by a flag |

### Correction (lling-llang)

The hierarchical spelling/grammar correction family. Consumed by
[`../integration/lling-llang/`](../integration/lling-llang/) — chiefly
[hierarchical-correction.md](../integration/lling-llang/hierarchical-correction.md),
[pipeline-assembly.md](../integration/lling-llang/pipeline-assembly.md), and
[dictionary-backend.md](../integration/lling-llang/dictionary-backend.md), plus the
[multi-shard grammar corrector](../integration/multi-shard-grammar-corrector.md) design
record (the `grammar-corrector-*` and `correction-sharded-routing` diagrams).

| Diagram | Draws |
|---|---|
| `correction-component-map` | The five crates and the artifacts flowing into a correction; duallity is detached because libgrammstein does not depend on it |
| `correction-dimensions` | The $`N`$ heterogeneous matching dimensions, each with its own metric and engine, fused into one joint score by a semiring |
| `correction-wfst-composition` | Why the joint search is *literally* N-dimensional: composition ⇒ a product automaton over state tuples ⇒ an $`n`$-D DP grid |
| `correction-pipeline` | Correction as a lattice that is expanded, pruned, reweighted, then collapsed |
| `correction-cascade` | The $`T_{\text{lex}} \circ T_{\text{gram}}`$ cascade — two Levenshtein automata over a term-id alphabet, decoded by a history-indexed beam |
| `correction-sharded-routing` | One n-gram lookup under sharding: route `(first-token, order)` by the importer's own pure function, open that one shard read-only, view it as `u64` n-grams — anchored vs. fanout neighbors |
| `grammar-corrector-seam` | `GrammarCore<P: NgramViewSource>` holds the whole beam decoder; `SingleView` / `ShardedView` are the two sources, and both public correctors are thin newtypes over the identical path |
| `grammar-corrector-colocation` | Co-location: the importer's store router and the query's view router are the *same* pure function of `(first_token, order)`, so the shard that stores a key answers it |
| `grammar-corrector-readonly-query` | One `count()` lookup in read-only query mode: `view_for` routes and opens the shard read-only (never create / evict / checkpoint); an absent shard yields `None` — read as no n-gram evidence |
| `correction-dependency-contract` | The one-directional, acyclic, feature-gated crate dependency DAG |
| `correction-dictionary-backend` | Feeding the Levenshtein automaton straight from the mmap'd persistent trie — no in-RAM materialization |
| `lling-architecture` | libgrammstein as the language model behind an lling-llang `LayerPipeline`; the `LanguageModel` trait is the seam between the crates |
| `lling-pipeline-usage` | The four layers transforming one lattice (tokenize → expand → prune → reweight), traced on "teh quikc brwon fox" |

### Storage & Levenshtein

Consumed by [`../integration/liblevenshtein/`](../integration/liblevenshtein/).

| Diagram | Draws |
|---|---|
| `levenshtein-crate-map` | The ownership map: liblevenshtein supplies the automata, libdictenstein the dictionary backends, and who imports what from whom |
| `levenshtein-automaton` | Why the automaton is *simulated*, never materialized: one trie edge consumed = one automaton transition, in lock-step |
| `levenshtein-backends` | The backend-selection decision tree: durability first, then mutability, then the read/space trade-off |
| `levenshtein-pathmap-pipeline` | The two distinct ways PathMap enters the crate — as an in-memory backend, and as the mmap'd production artifact |

### Cron

Consumed by [`../util/cron/cron-manager.md`](../util/cron/cron-manager.md).

| Diagram | Draws |
|---|---|
| `cron-architecture` | Threading: N submitter threads share cloned `CronHandle`s, one dedicated thread owns the heap, and every cross-thread edge is lock-free |
| `cron-state` | `CronStateMachine`: the complete (state, event) → state relation, including the state that is declared but never entered |

### CLI & training

Consumed by [`../cli/`](../cli/) and [`../training/`](../training/).

| Diagram | Draws |
|---|---|
| `cli-commands` | The complete `grammstein` command surface as clap declares it: seven groups, their subcommands, and their positional arguments |
| `cli-workflow` | The canonical operator workflow — and the fact that `train hybrid` consumes two already-trained models, never a corpus |
| `cli-google-books-import` | Where each memory and reliability flag of `train import-google-books` acts on the download and write paths |
| `training-ngram-pipeline` | N-gram training, both paths: the in-memory library pipeline, and the CLI's checkpointed WAL-backed accumulator |
| `training-embedding-pipeline` | Subword-embedding training as executed: vocabulary pass, matrix init, negative-sampling table, one decayed pass per epoch |
| `training-hybrid-assembly` | Why `train hybrid` is an *assembly* step, not a training step, and how a single `--alpha` maps onto each strategy |
| `training-tuning-loop` | The hyperparameter search loop expressed in the commands that implement each step, and its two cost classes |
| `training-large-corpora-memory` | Every place a large corpus can blow up the heap, the mechanism that bounds it, and the operator's three-way choice |

### Examples

Consumed by [`../examples/`](../examples/).

| Diagram | Draws |
|---|---|
| `example-train-eval` | The train → fuse → evaluate → persist workflow, coloured by the component that owns each stage |
| `example-perplexity-eval` | How a test corpus becomes PP and OOV statistics: one streaming pass, log-space accumulation, exactly one `exp()` at the end |
| `example-spell-correction` | The spell-correction example as a code flow: what is constructed once, and what happens on every `correct()` call |
| `example-domain-adaptation` | Domain adaptation as a two-expert mixture whose weight is fitted on a held-out in-domain dev set, reported on a disjoint test set |

### Cross-cutting

Diagrams that belong to no single subsystem.

| Diagram | Draws | Consumed by |
|---|---|---|
| `formal-verification` | The formal-methods coverage map: which TLA+ specs and Rocq proofs cover which importer components | [`../architecture/overview.md`](../architecture/overview.md) |
| `importer-eviction` | The Google Books importer dataflow and the overlay-heap eviction that fixed the OOM | [`../architecture/google-books-importer.md`](../architecture/google-books-importer.md) |
| `cpg-triad` | The Code Property Graph: one tree-sitter parse fusing AST + CFG + DFG into the joint graph three correctors consume | [`../components/code/cpg.md`](../components/code/cpg.md) |
