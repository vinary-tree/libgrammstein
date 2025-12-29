# Training on Large Corpora

This guide covers strategies for training models on large-scale text corpora.

## Memory Challenges

Large corpora present several challenges:

| Challenge | Cause | Solution |
|-----------|-------|----------|
| Corpus doesn't fit in RAM | Loading all text | Streaming readers |
| Model doesn't fit | Large vocabulary | Filtering, compression |
| Slow training | Sequential processing | Parallelization |
| Disk space | Storing corpus | Streaming from HTTP |

## Streaming Corpus Readers

### Wikipedia Streaming

Process Wikipedia dumps without downloading:

```rust
use libgrammstein::corpus::{WikipediaReader, WikipediaConfig, LoadStrategy};

// HTTP streaming - processes on-the-fly
let config = WikipediaConfig {
    load_strategy: LoadStrategy::HttpStreaming,
    max_articles: Some(1_000_000),  // Limit for testing
    ..Default::default()
};

let reader = WikipediaReader::from_url(
    "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"
)?;

// Train directly from stream
let model = TrainerBuilder::new(dictionary)
    .order(5)
    .train(&reader)?;
```

### Chunked Processing

Process large files in chunks:

```rust
use std::io::{BufRead, BufReader};
use std::fs::File;

fn process_large_file<D, F>(path: &str, chunk_size: usize, mut processor: F)
where
    D: MutableMappedDictionary<Value = NgramEntry>,
    F: FnMut(&[String]) -> Result<()>,
{
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut chunk = Vec::with_capacity(chunk_size);

    for line in reader.lines() {
        chunk.push(line?);

        if chunk.len() >= chunk_size {
            processor(&chunk)?;
            chunk.clear();
        }
    }

    // Process remaining
    if !chunk.is_empty() {
        processor(&chunk)?;
    }
}
```

## Vocabulary Reduction

### Frequency Filtering

Remove rare words:

```rust
let model = TrainerBuilder::new(dictionary)
    .order(5)
    .min_word_freq(10)  // Words appearing < 10 times ignored
    .train(&reader)?;
```

Memory impact:
| min_word_freq | Vocabulary Reduction | Memory Savings |
|---------------|---------------------|----------------|
| 1 | 0% | 0% |
| 5 | ~50% | ~30% |
| 10 | ~70% | ~50% |
| 100 | ~90% | ~70% |

### Vocabulary Capping

Limit vocabulary size:

```rust
fn cap_vocabulary(word_counts: &mut HashMap<String, u64>, max_vocab: usize) {
    if word_counts.len() <= max_vocab {
        return;
    }

    // Sort by frequency
    let mut sorted: Vec<_> = word_counts.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));

    // Keep only top words
    let to_remove: Vec<_> = sorted[max_vocab..].iter()
        .map(|(w, _)| w.to_string())
        .collect();

    for word in to_remove {
        word_counts.remove(&word);
    }
}
```

## N-gram Pruning

### Count-based Pruning

Remove low-count n-grams:

```rust
fn prune_ngrams<D>(model: &mut NgramModel<D>, min_count: u64)
where
    D: IterableDictionary + MutableMappedDictionary<Value = NgramEntry>,
{
    let to_remove: Vec<String> = model.trie()
        .iter_entries()
        .filter(|(_, entry)| entry.count() < min_count)
        .map(|(key, _)| key)
        .collect();

    for key in to_remove {
        model.trie_mut().remove(&key);
    }
}
```

### Entropy-based Pruning

Keep only informative n-grams:

```rust
fn entropy_prune<D>(model: &mut NgramModel<D>, threshold: f64)
where
    D: IterableDictionary + MutableMappedDictionary<Value = NgramEntry>,
{
    // Remove n-grams that add little information beyond backoff
    // (Advanced technique - requires careful implementation)
}
```

## Parallel Processing

### Multi-threaded Training

libgrammstein uses Rayon for parallel processing:

```rust
// Configure thread pool
rayon::ThreadPoolBuilder::new()
    .num_threads(16)  // Use 16 threads
    .build_global()
    .unwrap();

// Training automatically parallelizes
let model = TrainerBuilder::new(dictionary)
    .order(5)
    .batch_size(100_000)  // Larger batches = better parallelism
    .train(&reader)?;
```

### Distributed Training

For very large corpora, split across machines:

```rust
// Machine 1: Train on first half
let model1 = train_on_shard("shard1.txt")?;
model1.save("model_shard1.bin")?;

// Machine 2: Train on second half
let model2 = train_on_shard("shard2.txt")?;
model2.save("model_shard2.bin")?;

// Merge (on coordinator)
let merged = merge_models(&[model1, model2])?;
```

## Checkpointing

### Periodic Saves

Save progress during long training:

```rust
use std::time::{Duration, Instant};

struct CheckpointManager {
    interval: Duration,
    last_checkpoint: Instant,
    path: PathBuf,
}

impl CheckpointManager {
    fn maybe_checkpoint<D>(&mut self, model: &NgramModel<D>) -> Result<()>
    where
        D: MutableMappedDictionary<Value = NgramEntry> + Serialize,
    {
        if self.last_checkpoint.elapsed() >= self.interval {
            let checkpoint_path = self.path.join(format!(
                "checkpoint_{}.bin",
                chrono::Utc::now().format("%Y%m%d_%H%M%S")
            ));
            model.save(&checkpoint_path)?;
            self.last_checkpoint = Instant::now();
            log::info!("Checkpoint saved: {:?}", checkpoint_path);
        }
        Ok(())
    }
}
```

### Resume from Checkpoint

```rust
fn resume_training(checkpoint_path: &str, corpus: &impl CorpusReader) -> Result<NgramModel<D>> {
    // Load checkpoint
    let mut model: NgramModel<D> = NgramModel::load(checkpoint_path)?;

    // Continue training (implementation depends on your needs)
    // For n-grams, you might track which sentences were processed
    // and resume from there

    Ok(model)
}
```

## Memory-Mapped Files

For very large models:

```rust
use memmap2::MmapMut;

fn mmap_model_storage(path: &str, size: usize) -> Result<MmapMut> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(path)?;

    file.set_len(size as u64)?;

    unsafe { MmapMut::map_mut(&file) }
}
```

## Dictionary Backend Selection

| Backend | Memory | Build Time | Query Time |
|---------|--------|------------|------------|
| `PathMapDictionary` | High | Fast | Fast |
| `DynamicDawgChar` | Low | Medium | Medium |
| `DoubleArrayTrieChar` | Lowest | Slow | Fastest |

For large corpora, use compressed backends:

```rust
// Start with DynamicDawgChar for training (allows updates)
let training_dict = DynamicDawgChar::new();
let model = TrainerBuilder::new(training_dict)
    .order(5)
    .train(&reader)?;

// Save to portable format
model.save_portable("model.portable.bin")?;

// Load into DoubleArrayTrieChar for production (smallest, fastest)
let production_model = NgramModel::load_portable(
    "model.portable.bin",
    || DoubleArrayTrieChar::new()
)?;
```

## Memory Estimation

Estimate memory requirements before training:

```rust
fn estimate_memory(
    corpus_words: u64,
    vocab_size: u64,
    order: usize,
    dim: usize,  // For embeddings
) -> (u64, u64) {
    // N-gram estimate (rough)
    // Assumes ~10 bytes per n-gram entry average
    let ngram_entries = corpus_words * order as u64;
    let ngram_memory = ngram_entries * 10;

    // Embedding estimate
    // vocab_size * dim * 4 bytes (f32)
    // + bucket_count * dim * 4 bytes
    let bucket_count = 2_000_000u64;
    let embedding_memory = (vocab_size + bucket_count) * dim as u64 * 4;

    (ngram_memory, embedding_memory)
}

let (ngram_mb, embed_mb) = estimate_memory(1_000_000_000, 1_000_000, 5, 100);
println!("Estimated N-gram memory: {} MB", ngram_mb / 1_000_000);
println!("Estimated Embedding memory: {} MB", embed_mb / 1_000_000);
```

## CLI for Large Corpora

```bash
# Train with checkpoints
grammstein train ngram enwiki.xml.bz2 model.bin \
    --order 5 \
    --min-count 10 \
    --checkpoint ./checkpoints \
    --checkpoint-interval 100000

# Resume from checkpoint
grammstein train ngram enwiki.xml.bz2 model.bin \
    --resume ./checkpoints/latest.ckpt

# HTTP streaming (no download required)
grammstein train ngram \
    "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2" \
    model.bin \
    --order 5
```

## Best Practices

1. **Profile first** - Understand where memory is going
2. **Stream when possible** - Don't load entire corpus into RAM
3. **Filter aggressively** - Remove rare words early
4. **Checkpoint frequently** - Long jobs can fail
5. **Use appropriate backends** - Compressed for storage, fast for serving
6. **Monitor progress** - Know when something is wrong

## Scaling Guidelines

| Corpus Size | Order | min_count | Approx. Memory |
|-------------|-------|-----------|----------------|
| 10M words | 5 | 5 | 2-4 GB |
| 100M words | 5 | 10 | 10-20 GB |
| 1B words | 5 | 20 | 50-100 GB |
| 10B+ words | 3-4 | 50+ | 200+ GB |

## See Also

- [N-gram Training](ngram.md) - Training workflow
- [Hyperparameters](hyperparameters.md) - Tuning for size/quality
- [CLI Reference](../cli/README.md) - Command-line options
