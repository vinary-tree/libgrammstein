# Architecture Overview

This document provides a high-level view of libgrammstein's architecture, explaining how components fit together and the design principles behind them.

## Design Goals

libgrammstein is designed with four primary goals:

1. **Hybrid Scoring**: Combine the strengths of N-gram models (precise local context) with embeddings (semantic similarity, OOV handling)

2. **WFST Integration**: Seamlessly integrate with lling-llang's lattice-based text correction pipeline

3. **Efficiency**: Handle large corpora (10GB+) and serve millions of queries per second

4. **Modularity**: Each component is independently usable and testable

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           libgrammstein                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        Hybrid Layer                                  │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐│   │
│  │  │               HybridLanguageModel                               ││   │
│  │  │   - Interpolates N-gram and embedding scores                    ││   │
│  │  │   - Implements LanguageModel trait                              ││   │
│  │  │   - LRU cache for hot queries                                   ││   │
│  │  └─────────────────────────────────────────────────────────────────┘│   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│           │                                     │                           │
│           ▼                                     ▼                           │
│  ┌─────────────────────────┐     ┌─────────────────────────────────────┐   │
│  │     N-gram Layer        │     │       Embedding Layer               │   │
│  │                         │     │                                     │   │
│  │  ┌───────────────────┐  │     │  ┌─────────────────────────────┐   │   │
│  │  │    NgramModel     │  │     │  │    SubwordEmbedding         │   │   │
│  │  │ - Modified KN     │  │     │  │ - Word embeddings           │   │   │
│  │  │ - Order 3-5       │  │     │  │ - Subword (char n-gram)     │   │   │
│  │  │ - Backoff chain   │  │     │  │ - BPE tokenizer             │   │   │
│  │  └────────┬──────────┘  │     │  └────────────┬────────────────┘   │   │
│  │           │             │     │               │                     │   │
│  │  ┌────────▼──────────┐  │     │  ┌────────────▼────────────────┐   │   │
│  │  │ KneserNeySmooth   │  │     │  │   Skip-gram Trainer         │   │   │
│  │  │ - D1, D2, D3+     │  │     │  │ - Negative sampling         │   │   │
│  │  │ - Continuation    │  │     │  │ - Window size               │   │   │
│  │  └───────────────────┘  │     │  └─────────────────────────────┘   │   │
│  └─────────────────────────┘     └─────────────────────────────────────┘   │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Storage Layer                                   │   │
│  │  ┌───────────────────┐  ┌───────────────────┐  ┌─────────────────┐  │   │
│  │  │  DynamicDawgChar  │  │ PathMapDictionary │  │ DoubleArrayTrie │  │   │
│  │  │  (mutable, serde) │  │ (distributed)     │  │ (static, fast)  │  │   │
│  │  └───────────────────┘  └───────────────────┘  └─────────────────┘  │   │
│  │                           liblevenshtein-rust                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
           │
           │ implements LanguageModel trait
           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              lling-llang                                     │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      LanguageModelLayer                               │   │
│  │    Wraps LanguageModel trait for use in CorrectionLayer pipelines    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Neural, RAG, and Topic Layers

The following layers provide advanced neural capabilities, document retrieval, and topic modeling:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Neural Layer                                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │ ModernBertModel │──│ ModernBert      │──│   ModernBertRescorer        │  │
│  │ (149M params)   │  │ Embedder        │  │   (beam search rescoring)   │  │
│  │ - 8K context    │  │ - 768-dim       │  │   - pseudo-perplexity       │  │
│  │ - MLM head      │  │ - CLS/Mean pool │  │   - embedding coherence     │  │
│  └────────┬────────┘  └────────┬────────┘  └─────────────────────────────┘  │
│           │                    │                                             │
│           │           ┌────────┴────────┐                                   │
│           │           │   Summarizer    │                                   │
│           │           │ - MMR selection │                                   │
│           │           │ - Extractive    │                                   │
│           │           └─────────────────┘                                   │
└───────────┼─────────────────────┼───────────────────────────────────────────┘
            │                     │
            ▼                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              RAG Layer                                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │  IndexBuilder   │──│   RagIndex<B>   │──│       Retriever<B>          │  │
│  │ - auto-embed    │  │ - storage+query │  │ - text→embedding→results   │  │
│  │ - auto-synopsis │  │ - topic model   │  │ - batch retrieval          │  │
│  └─────────────────┘  └────────┬────────┘  └─────────────────────────────┘  │
│                                │                                             │
│  ┌─────────────────────────────┼────────────────────────────────────────┐   │
│  │            RetrievalBackend │                                         │   │
│  │  ┌──────────────────────┐  └────┐  ┌─────────────────────────────┐   │   │
│  │  │ ExactCosineBackend   │       │  │       HnswBackend           │   │   │
│  │  │ - BLAS-accelerated   │       │  │ - approx NN, O(log n)       │   │   │
│  │  │ - O(n) query, <1M    │       │  │ - scalable, >1M docs        │   │   │
│  │  └──────────────────────┘       │  └─────────────────────────────┘   │   │
│  └─────────────────────────────────┴────────────────────────────────────┘   │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             Topic Layer                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │ TopicExtractor  │──│   TopicModel    │──│         Topic               │  │
│  │ - HAC cluster   │  │ - topics map    │  │ - id, keywords, desc        │  │
│  │ - c-TF-IDF      │  │ - assignments   │  │ - document_count            │  │
│  └────────┬────────┘  └─────────────────┘  └─────────────────────────────┘  │
│           │                                                                  │
│  ┌────────┴────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │ Hierarchical    │  │     CtfIdf      │  │       Dendrogram            │  │
│  │ Clustering      │  │ - keywords/topic│  │ - scipy-compatible          │  │
│  │ - Ward/Complete │  │ - min_df/max_df │  │ - cut_tree/cut_by_distance  │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Code Correction Layer

The Code Correction layer provides multi-language code error detection and correction:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Code Correction Layer                              │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        CorrectionPipeline<L>                            │ │
│  │    Parse → Tokenize → Analyze → Correct → Rank → AnalysisResult        │ │
│  └───────────────────────────────────┬────────────────────────────────────┘ │
│                                      │                                       │
│  ┌───────────────────────────────────┴───────────────────────────────────┐  │
│  │                        EnsembleCorrector<L>                            │  │
│  │                                                                        │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │  │
│  │  │   Lexical    │  │   Grammar    │  │   Semantic   │                 │  │
│  │  │  Corrector   │  │  Corrector   │  │  Corrector   │                 │  │
│  │  │              │  │              │  │              │                 │  │
│  │  │ Levenshtein  │  │ PCFG/Earley  │  │  GNN + CPG   │                 │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                 │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                       │
│  ┌───────────────────────────────────┴───────────────────────────────────┐  │
│  │                       Code Analysis Layer                              │  │
│  │                                                                        │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────────┐   │  │
│  │  │   CodeParser    │  │  CodeTokenizer  │  │ CodePropertyGraph    │   │  │
│  │  │   (tree-sitter) │  │  (token types)  │  │ (AST + CFG + DFG)    │   │  │
│  │  └─────────────────┘  └─────────────────┘  └──────────────────────┘   │  │
│  │                                                                        │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │                    CodeLanguage Trait                            │  │  │
│  │  │  Python │ Rust │ JavaScript │ Rholang │ MeTTa │ Custom...       │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                       │
│  ┌───────────────────────────────────┴───────────────────────────────────┐  │
│  │                      Advanced Components                               │  │
│  │                                                                        │  │
│  │  ┌───────────────┐  ┌───────────────┐  ┌────────────────────────┐     │  │
│  │  │   WeightedCFG │  │ GnnSemantic   │  │     CodeEmbedder       │     │  │
│  │  │   + Trainer   │  │    Scorer     │  │ (UniXcoder/GraphCode)  │     │  │
│  │  └───────────────┘  └───────────────┘  └────────────────────────┘     │  │
│  │                                                                        │  │
│  │  ┌───────────────┐  ┌───────────────────────────────────────────┐     │  │
│  │  │ GrammarConstr │  │              WFST Export                   │     │  │
│  │  │ (Earley)      │  │  (lling-llang integration)                │     │  │
│  │  └───────────────┘  └───────────────────────────────────────────┘     │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Relationships

### Training Pipeline

```
Corpus Files                                 Trained Models
    │                                              ▲
    ▼                                              │
┌─────────────────┐                       ┌────────┴────────┐
│  CorpusReader   │                       │   save()/load() │
│  - Wikipedia    │                       │                 │
│  - Gutenberg    │                       │  ┌───────────┐  │
│  - Plaintext    │                       │  │ NgramModel│  │
└────────┬────────┘                       │  └───────────┘  │
         │                                │                 │
         │ sentences()                    │  ┌───────────┐  │
         ▼                                │  │ Embedding │  │
┌─────────────────┐                       │  └───────────┘  │
│   Tokenizer     │                       │                 │
│  - Sentence     │                       │  ┌───────────┐  │
│  - Word         │                       │  │  Hybrid   │  │
└────────┬────────┘                       │  └───────────┘  │
         │                                └─────────────────┘
         │ tokens                                 ▲
         ▼                                        │
┌─────────────────────────────────────────────────┤
│                                                 │
│  ┌─────────────────┐     ┌─────────────────┐   │
│  │  NgramTrainer   │     │ EmbeddingTrainer│   │
│  │                 │     │                 │   │
│  │  - Count ngrams │     │  - Skip-gram    │   │
│  │  - Compute MKN  │     │  - Neg sampling │   │
│  │  - Store in trie│     │  - Subwords     │   │
│  └────────┬────────┘     └────────┬────────┘   │
│           │                       │            │
│           └───────────┬───────────┘            │
│                       │                        │
│              ┌────────▼────────┐               │
│              │ HybridLanguage  │               │
│              │     Model       │───────────────┘
│              └─────────────────┘
│
│                    Training
└─────────────────────────────────────────────────
```

### Inference Pipeline

```
Input: ["the", "quick", "brown", "fox"]
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     HybridLanguageModel.score_sequence()                 │
│                                                                         │
│  ┌─────────────────────────┐     ┌─────────────────────────────────┐   │
│  │  Check LRU Cache        │────►│  Cache Hit: Return cached score │   │
│  └───────────┬─────────────┘     └─────────────────────────────────┘   │
│              │                                                          │
│              │ Cache Miss                                               │
│              ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  For each token in sequence:                                     │   │
│  │                                                                  │   │
│  │  ┌───────────────────────────────────────────────────────────┐  │   │
│  │  │  log P(token | context) =                                  │  │   │
│  │  │    λ₁ * ngram.log_prob(token, context) +                   │  │   │
│  │  │    λ₂ * embedding.similarity_score(token, context)         │  │   │
│  │  └───────────────────────────────────────────────────────────┘  │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│              │                                                          │
│              ▼                                                          │
│  ┌─────────────────────────┐                                           │
│  │  Sum log probabilities  │                                           │
│  │  Cache result           │                                           │
│  │  Return total           │                                           │
│  └─────────────────────────┘                                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
           │
           ▼
Output: log P(sequence) = -23.5
```

## Key Abstractions

### MutableMappedDictionary (from liblevenshtein)

libgrammstein stores N-grams in trie dictionaries provided by liblevenshtein:

```rust
pub trait MutableMappedDictionary<Value>: Dictionary {
    /// Insert a key-value pair
    fn insert_with_value(&mut self, key: &str, value: Value);

    /// Update existing or insert new
    fn update_or_insert<F>(&mut self, key: &str, default: Value, f: F)
    where F: FnOnce(&mut Value);

    /// Get value by key
    fn get_value(&self, key: &str) -> Option<&Value>;
}
```

N-grams are stored as pipe-separated strings: `"the|quick|brown"` → `NgramEntry { count: 42, ... }`

### LanguageModel Trait (from lling-llang)

libgrammstein implements lling-llang's `LanguageModel` trait for integration:

```rust
pub trait LanguageModel: Send + Sync {
    /// Score a complete sequence: log P(w₁, w₂, ..., wₙ)
    fn score_sequence(&self, tokens: &[&str]) -> f64;

    /// Score a continuation: log P(next | prefix)
    fn score_continuation(&self, prefix: &[&str], next: &str) -> f64;
}
```

This trait uses `&[&str]` (strings), not vocabulary IDs, ensuring libgrammstein doesn't need to know about lling-llang's lattice internals.

## Design Decisions

### Why Hybrid Models?

| Model Type | Strengths | Weaknesses |
|------------|-----------|------------|
| N-gram | Precise local context, fast lookup, well-understood | OOV problem, sparse data for long contexts |
| Embeddings | Semantic similarity, handles OOV via subwords | Ignores word order, slower |
| **Hybrid** | Best of both: local precision + semantic coverage | Slightly more complex |

### Why liblevenshtein for Storage?

1. **Trie Structure**: Natural fit for N-gram prefix matching
2. **Multiple Backends**: DynamicDawg (mutable), PathMap (distributed), DoubleArray (fast)
3. **DictZipper**: Efficient traversal for backoff computation
4. **Existing Integration**: Already used by lling-llang for spelling correction

### Why Rayon for Parallelism?

Language model training is CPU-bound and embarrassingly parallel:
- Sentence processing is independent
- Skip-gram windows are independent
- Batch inference is independent

Rayon's work-stealing scheduler optimally utilizes all cores without the overhead of async runtimes.

### Why `&[&str]` in the Trait Interface?

The `LanguageModel` trait uses strings rather than vocabulary IDs because:
1. **Decoupling**: LM doesn't need to know about lattice vocabulary
2. **Simplicity**: No ID translation layer needed
3. **Flexibility**: Works with any tokenization scheme
4. **Testing**: Easier to test with literal strings

## Thread Safety

libgrammstein is designed for concurrent access:

| Component | Thread Safety Mechanism |
|-----------|------------------------|
| `NgramModel<D>` | `Arc<D>` where `D: Send + Sync` |
| `SubwordEmbedding` | Immutable + `Arc<DashMap>` cache |
| `HybridLanguageModel` | `Mutex<LruCache>` for hot cache |
| `HybridConfig` | Plain data (Copy) |

All models implement `Send + Sync`, satisfying lling-llang's requirements.

## Performance Characteristics

| Operation | Time Complexity | Notes |
|-----------|-----------------|-------|
| N-gram lookup | O(n) | n = N-gram order |
| Embedding lookup | O(d + s) | d = dimension, s = subwords |
| Hybrid score | O(n + d + s) | Cache amortizes repeated queries |
| Training (N-gram) | O(C × n) | C = corpus tokens |
| Training (Embedding) | O(C × w × d × e) | w = window, e = epochs |

## Memory Layout

```
NgramModel
├── dictionary: Arc<D>          # Trie with NgramEntry values
│   ├── "the" → NgramEntry { count: 1M, ... }
│   ├── "the|quick" → NgramEntry { count: 50K, ... }
│   └── "the|quick|brown" → NgramEntry { count: 5K, ... }
├── smoothing: KneserNeySmoothing
│   ├── d1: f64                 # Discount for count=1
│   ├── d2: f64                 # Discount for count=2
│   └── d3_plus: f64            # Discount for count>=3
└── vocab_size: usize

SubwordEmbedding
├── word_embeddings: Array2<f32>      # [vocab_size × dim]
├── subword_embeddings: Array2<f32>   # [bucket_count × dim]
├── word_to_idx: HashMap<String, usize>
├── idx_to_word: Vec<String>
└── cache: Arc<DashMap<String, Array1<f32>>>

HybridLanguageModel
├── ngram: NgramModel<D>
├── embedding: SubwordEmbedding
├── config: HybridConfig
│   ├── ngram_weight: f64       # λ₁
│   └── embedding_weight: f64   # λ₂
└── cache: Mutex<LruCache<CacheKey, f64>>
```

## Next Steps

- [Data Flow](data-flow.md): Detailed data flow through the system
- [Threading Model](threading-model.md): Concurrency patterns
- [N-gram Overview](../components/ngram/overview.md): N-gram model details
- [Neural Overview](../components/neural/overview.md): ModernBERT-based neural components
- [RAG Overview](../components/rag/overview.md): Document indexing and retrieval
- [Topic Overview](../components/topic/overview.md): BERTopic-style topic modeling
- [lling-llang Integration](../integration/lling-llang/overview.md): Integration architecture
