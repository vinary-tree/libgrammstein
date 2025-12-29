# grammstein CLI Reference

The `grammstein` CLI provides command-line tools for training and querying language models.

## Installation

Build the CLI with the `cli` feature:

```bash
cargo build --release --features cli
```

The binary is located at `target/release/grammstein`.

## Global Options

```
-v, --verbose    Enable verbose output
-q, --quiet      Suppress progress bars and status messages
-h, --help       Print help
-V, --version    Print version
```

## Commands

### corpus - Corpus Processing

Process and analyze text corpora.

#### corpus stats

Display corpus statistics including word counts, vocabulary, and token distribution.

```bash
grammstein corpus stats <PATH> [OPTIONS]

Options:
  --top <N>       Show top N most frequent words (default: 10)
  --format <FMT>  Output format: text, json (default: text)
```

Example:
```bash
grammstein corpus stats corpus.txt
grammstein corpus stats wikipedia.xml.bz2 --top 20
```

#### corpus sample

Sample random sentences from a corpus.

```bash
grammstein corpus sample <PATH> [OPTIONS]

Options:
  -n, --count <N>    Number of sentences to sample (default: 5)
  --seed <SEED>      Random seed for reproducibility
```

Example:
```bash
grammstein corpus sample corpus.txt -n 10
```

#### corpus detect

Detect the language of a corpus.

```bash
grammstein corpus detect <PATH>
```

Example:
```bash
grammstein corpus detect corpus.txt
# Output:
# Detected language: en (English)
# Confidence: 99.8%
# Reliable: yes
```

---

### train - Model Training

Train n-gram and embedding models.

#### train ngram

Train an n-gram language model.

```bash
grammstein train ngram <CORPUS> <OUTPUT> [OPTIONS]

Options:
  --order <N>           N-gram order (default: 5)
  --min-count <N>       Minimum n-gram count (default: 1)
  --checkpoint <PATH>   Save checkpoints to path
  --checkpoint-interval <N>
                        Checkpoint every N sentences (default: 100000)
  --resume <PATH>       Resume from checkpoint
```

Example:
```bash
grammstein train ngram corpus.txt model.bin --order 5
grammstein train ngram large-corpus.txt model.bin --checkpoint ./checkpoints
```

#### train embedding

Train subword embeddings.

```bash
grammstein train embedding <CORPUS> <OUTPUT> [OPTIONS]

Options:
  --dim <N>            Embedding dimension (default: 100)
  --window <N>         Context window size (default: 5)
  --min-count <N>      Minimum word count (default: 5)
  --epochs <N>         Training epochs (default: 5)
  --neg-samples <N>    Negative samples (default: 5)
  --learning-rate <F>  Initial learning rate (default: 0.05)
  --checkpoint <PATH>  Save checkpoints to path
```

Example:
```bash
grammstein train embedding corpus.txt embed.bin --dim 300 --epochs 10
```

#### train hybrid

Train a hybrid model (n-gram + embeddings).

```bash
grammstein train hybrid <CORPUS> <OUTPUT> [OPTIONS]

Options:
  --ngram-order <N>    N-gram order (default: 5)
  --embed-dim <N>      Embedding dimension (default: 100)
  --lambda <F>         Interpolation weight for n-gram (default: 0.5)
```

Example:
```bash
grammstein train hybrid corpus.txt hybrid.bin --lambda 0.7
```

---

### models - Model Information

Inspect and manage trained models.

#### models info

Display model information.

```bash
grammstein models info <MODEL>
```

Example:
```bash
grammstein models info model.bin
# Output:
# Model Information
#
# Path: model.bin
# Type: NgramModel<DynamicDawgChar>
# Size: 10.42 KiB
#
# N-gram component:
#   Order:       3
#   Vocab size:  291
#   Smoothing:   Modified Kneser-Ney
```

#### models list

List n-grams in a model.

```bash
grammstein models list <MODEL> [OPTIONS]

Options:
  -n, --limit <N>     Maximum entries to show (default: 100)
  --prefix <PREFIX>   Filter by prefix
```

---

### query - Query Models

Score text and find similar words.

#### query score

Score a sentence or continuation.

```bash
grammstein query score <MODEL> <TEXT> [OPTIONS]

Options:
  --mode <MODE>     Scoring mode: sentence, continuation (default: sentence)
```

Example:
```bash
grammstein query score model.bin "the quick brown fox"
# Output:
# Tokens: the quick brown fox
# Mode:   sentence
#
# Log probability: -5.6733
# Perplexity:      291.00
```

#### query completions

Get top completions for a context.

```bash
grammstein query completions <MODEL> <CONTEXT> [OPTIONS]

Options:
  -n, --count <N>    Number of completions (default: 10)
```

Example:
```bash
grammstein query completions model.bin "the quick"
# Output:
# Top completions for "the quick":
#   1. brown         -2.345  (P=0.0956)
#   2. fox           -3.012  (P=0.0492)
```

#### query similar

Find similar words (embedding models only).

```bash
grammstein query similar <MODEL> <WORD> [OPTIONS]

Options:
  -n, --count <N>    Number of similar words (default: 10)
```

---

### convert - Format Conversion

Convert between model formats.

```bash
grammstein convert <INPUT> <OUTPUT> [OPTIONS]

Options:
  --format <FMT>    Output format: binary, zstd, json
  --compress        Enable compression (zstd)
```

---

### eval - Model Evaluation

Evaluate model performance.

```bash
grammstein eval <MODEL> <TEST_CORPUS> [OPTIONS]

Options:
  --metric <METRIC>    Metric: perplexity, accuracy (default: perplexity)
  --batch-size <N>     Batch size (default: 1000)
```

Example:
```bash
grammstein eval model.bin test.txt
# Output:
# Evaluation Results
#
# Sentences:   1000
# Tokens:      25431
# Perplexity:  142.56
# Time:        1.23s
```

---

### repl - Interactive Mode

Start an interactive REPL for model exploration.

```bash
grammstein repl <MODEL>
```

REPL Commands:
- `score <text>` - Score a sentence
- `complete <context>` - Get completions
- `similar <word>` - Find similar words (embedding)
- `info` - Show model info
- `help` - Show help
- `quit` / `exit` - Exit REPL

Example session:
```
grammstein> score the quick brown fox
Log probability: -5.6733
Perplexity:      291.00

grammstein> complete the quick
  1. brown         -2.345  (P=0.0956)
  2. fox           -3.012  (P=0.0492)

grammstein> quit
```

---

## Environment Variables

- `GRAMMSTEIN_CACHE_DIR` - Cache directory for downloaded corpora
- `GRAMMSTEIN_LOG_LEVEL` - Log level: debug, info, warn, error

## Exit Codes

- `0` - Success
- `1` - General error
- `2` - Invalid arguments

## Examples

### Complete Workflow

```bash
# 1. Analyze corpus
grammstein corpus stats wikipedia-en.txt

# 2. Train n-gram model
grammstein train ngram wikipedia-en.txt ngram.bin \
  --order 5 \
  --checkpoint ./checkpoints

# 3. Check model
grammstein models info ngram.bin

# 4. Query model
grammstein query score ngram.bin "artificial intelligence"

# 5. Interactive exploration
grammstein repl ngram.bin
```

### Training with HTTP Streaming

```bash
# Train directly from Wikipedia dump URL
grammstein train ngram \
  "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2" \
  model.bin \
  --checkpoint ./checkpoints
```
