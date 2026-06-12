//! Compare two artrie files to verify n-gram content equivalence.
//!
//! This tool compares vocabularies and n-gram tries using true on-disk streaming
//! iteration. It does NOT load entire tries into memory, instead processing one
//! entry at a time to enable comparison of arbitrarily large tries.
//!
//! # Comparison Strategy
//!
//! 1. **Vocabulary comparison**: Finds terms that exist in only one vocabulary
//! 2. **N-gram comparison**: Bidirectional streaming comparison
//!    - Forward pass: For each entry in trie1, decode → re-encode → lookup in trie2
//!    - Reverse pass: For each entry in trie2, check existence in trie1
//!
//! # Memory Efficiency
//!
//! - O(1) memory per entry during iteration
//! - O(k) memory for single n-gram decoding (k = ngram order, typically ≤5)
//! - Vocabulary reverse Vec in memory (~100MB for 10M words)
//! - Buffer pool cache: 64MB (configurable by libdictenstein)
//!
//! # Usage
//!
//! ```bash
//! cargo run --bin compare_artries --features cli,google-books -- \
//!     --trie1 bak-sharded-e2e/english.artrie \
//!     --vocab1 bak-sharded-e2e/english.vocab.artrie \
//!     --trie2 bak-sharded-e2e-2/english.artrie \
//!     --vocab2 bak-sharded-e2e-2/english.vocab.artrie \
//!     --max-mismatches 100
//! ```

#[cfg(feature = "mimalloc-alloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use clap::Parser;
use libdictenstein::persistent_artrie::char::PersistentARTrieChar;
use libgrammstein::ngram::vocabulary::{
    decode_ngram_key, encode_ngram_key_existing, open_vocabulary, SharedVocabARTrie,
};
use std::collections::HashSet;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "compare_artries")]
#[command(about = "Compare two artrie files to verify n-gram content equivalence")]
struct Args {
    /// Path to the first trie file
    #[arg(long)]
    trie1: PathBuf,

    /// Path to the first vocabulary file
    #[arg(long)]
    vocab1: PathBuf,

    /// Path to the second trie file
    #[arg(long)]
    trie2: PathBuf,

    /// Path to the second vocabulary file
    #[arg(long)]
    vocab2: PathBuf,

    /// Only show first N mismatches (0 for all)
    #[arg(long, default_value = "100")]
    max_mismatches: usize,

    /// Show progress during comparison
    #[arg(long, short = 'v')]
    verbose: bool,
}

/// Result of vocabulary comparison.
#[derive(Default)]
struct VocabComparisonResult {
    /// Terms that exist only in vocabulary 1.
    only_in_1: Vec<String>,
    /// Terms that exist only in vocabulary 2.
    only_in_2: Vec<String>,
    /// Total terms in vocabulary 1.
    vocab1_count: u64,
    /// Total terms in vocabulary 2.
    vocab2_count: u64,
}

/// A count mismatch between tries.
struct CountMismatch {
    /// The n-gram as a vector of words.
    ngram: Vec<String>,
    /// Count in trie 1.
    count1: u64,
    /// Count in trie 2.
    count2: u64,
}

/// An entry missing from one trie.
struct MissingEntry {
    /// The n-gram as a vector of words.
    ngram: Vec<String>,
    /// Count in the trie that has it.
    count: u64,
}

/// Result of n-gram comparison.
#[derive(Default)]
struct NgramComparisonResult {
    /// Number of entries in trie 1.
    trie1_count: u64,
    /// Number of entries in trie 2.
    trie2_count: u64,
    /// Number of entries that failed to decode in trie 1.
    decode_failures_1: u64,
    /// Number of entries that failed to decode in trie 2.
    decode_failures_2: u64,
    /// Entries with different counts between tries.
    count_mismatches: Vec<CountMismatch>,
    /// Entries in trie 2 but missing in trie 1.
    missing_in_1: Vec<MissingEntry>,
    /// Entries in trie 1 but missing in trie 2.
    missing_in_2: Vec<MissingEntry>,
    /// Whether the comparison was truncated due to max_mismatches.
    truncated: bool,
}

impl NgramComparisonResult {
    fn total_errors(&self) -> usize {
        self.count_mismatches.len() + self.missing_in_1.len() + self.missing_in_2.len()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging to see libdictenstein warnings
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .format_timestamp(None)
        .init();

    let args = Args::parse();

    // Load vocabularies using SharedVocabARTrie for O(1) reverse lookups
    println!("Loading vocabulary 1: {:?}", args.vocab1);
    let vocab1 = open_vocabulary(&args.vocab1)?;
    println!("  {} terms in vocabulary 1", vocab1.read().len());

    println!("Loading vocabulary 2: {:?}", args.vocab2);
    let vocab2 = open_vocabulary(&args.vocab2)?;
    println!("  {} terms in vocabulary 2", vocab2.read().len());

    // Compare vocabularies
    println!("\nComparing vocabularies...");
    let vocab_result = compare_vocabularies(&vocab1, &vocab2, args.verbose)?;
    print_vocab_results(&vocab_result);

    // Load n-gram tries (lazy loading - only root loaded initially)
    println!("\nLoading trie 1: {:?}", args.trie1);
    let trie1 = PersistentARTrieChar::<u64>::open(&args.trie1)?;
    println!("  Trie 1 opened successfully");

    println!("Loading trie 2: {:?}", args.trie2);
    let trie2 = PersistentARTrieChar::<u64>::open(&args.trie2)?;
    println!("  Trie 2 opened successfully");

    // Compare n-grams using streaming iteration
    println!("\nComparing n-grams (streaming)...");
    let ngram_result = compare_ngrams_streaming(
        &trie1,
        &vocab1,
        &trie2,
        &vocab2,
        args.max_mismatches,
        args.verbose,
    )?;

    // Print results
    print_ngram_results(&ngram_result, args.max_mismatches);

    // Determine exit code
    let has_vocab_errors = !vocab_result.only_in_1.is_empty() || !vocab_result.only_in_2.is_empty();
    let has_ngram_errors = !ngram_result.count_mismatches.is_empty()
        || !ngram_result.missing_in_1.is_empty()
        || !ngram_result.missing_in_2.is_empty();

    if has_vocab_errors || has_ngram_errors {
        let total_errors = vocab_result.only_in_1.len()
            + vocab_result.only_in_2.len()
            + ngram_result.total_errors();
        println!("\nFAILED: {} total differences found", total_errors);
        std::process::exit(1);
    } else {
        println!("\nPASS: All vocabularies and n-grams match!");
        Ok(())
    }
}

/// Compare vocabularies to find terms that exist in only one.
fn compare_vocabularies(
    vocab1: &SharedVocabARTrie,
    vocab2: &SharedVocabARTrie,
    verbose: bool,
) -> Result<VocabComparisonResult, Box<dyn std::error::Error>> {
    let mut result = VocabComparisonResult::default();

    // Get all terms from vocab1 using iter_terms()
    let mut terms1 = HashSet::new();
    {
        let guard = vocab1.read();
        for term in guard.iter_terms() {
            result.vocab1_count += 1;
            terms1.insert(term);
        }
    }

    // Get all terms from vocab2 using iter_terms()
    let mut terms2 = HashSet::new();
    {
        let guard = vocab2.read();
        for term in guard.iter_terms() {
            result.vocab2_count += 1;
            terms2.insert(term);
        }
    }

    // Find symmetric difference
    for term in &terms1 {
        if !terms2.contains(term) {
            result.only_in_1.push(term.clone());
        }
    }

    for term in &terms2 {
        if !terms1.contains(term) {
            result.only_in_2.push(term.clone());
        }
    }

    // Sort for consistent output
    result.only_in_1.sort();
    result.only_in_2.sort();

    if verbose {
        println!(
            "  Vocab 1: {} terms, Vocab 2: {} terms",
            result.vocab1_count, result.vocab2_count
        );
        println!(
            "  Only in vocab 1: {}, Only in vocab 2: {}",
            result.only_in_1.len(),
            result.only_in_2.len()
        );
    }

    Ok(result)
}

/// Compare n-grams using streaming iteration (one entry at a time).
fn compare_ngrams_streaming(
    trie1: &PersistentARTrieChar<u64>,
    vocab1: &SharedVocabARTrie,
    trie2: &PersistentARTrieChar<u64>,
    vocab2: &SharedVocabARTrie,
    max_mismatches: usize,
    verbose: bool,
) -> Result<NgramComparisonResult, Box<dyn std::error::Error>> {
    let mut result = NgramComparisonResult::default();
    let max_to_track = if max_mismatches == 0 {
        usize::MAX
    } else {
        max_mismatches
    };

    // Forward pass: trie1 → trie2
    if verbose {
        println!("  Forward pass: checking trie1 entries against trie2...");
    }

    if let Some(entries) = trie1.iter_prefix_with_values("")? {
        for (key1, value1) in entries {
            // Skip metadata
            if key1.starts_with('\x00') {
                continue;
            }
            result.trie1_count += 1;

            // Decode to word indices
            let indices = decode_ngram_key(&key1);

            // Reverse lookup to get words (O(1) per index)
            let guard = vocab1.read();
            let words: Vec<String> = indices
                .iter()
                .filter_map(|&idx| {
                    if idx == 0 {
                        return None; // index 0 is reserved
                    }
                    guard.get_term(idx)
                })
                .collect();
            drop(guard);

            if words.len() != indices.len() {
                result.decode_failures_1 += 1;
                if verbose && result.decode_failures_1 <= 5 {
                    eprintln!(
                        "  Warning: Could not decode all indices for key (indices: {:?})",
                        indices
                    );
                }
                continue;
            }

            // Re-encode with vocab2's indices
            let word_refs: Vec<&str> = words.iter().map(|s| s.as_str()).collect();
            match encode_ngram_key_existing(&word_refs, vocab2) {
                Some(key2) => {
                    // Lookup in trie2 (on-disk, lazy loaded)
                    match trie2.get(&key2) {
                        Some(value2) => {
                            if value2 != value1 {
                                if result.count_mismatches.len() < max_to_track {
                                    result.count_mismatches.push(CountMismatch {
                                        ngram: words,
                                        count1: value1,
                                        count2: value2,
                                    });
                                }
                            }
                        }
                        None => {
                            if result.missing_in_2.len() < max_to_track {
                                result.missing_in_2.push(MissingEntry {
                                    ngram: words,
                                    count: value1,
                                });
                            }
                        }
                    }
                }
                None => {
                    // Word not in vocab2 - this means a vocab mismatch
                    if result.missing_in_2.len() < max_to_track {
                        result.missing_in_2.push(MissingEntry {
                            ngram: words,
                            count: value1,
                        });
                    }
                }
            }

            // Early exit if max mismatches reached
            if max_mismatches > 0 && result.total_errors() >= max_mismatches {
                result.truncated = true;
                break;
            }
        }
    }

    // Reverse pass: trie2 → trie1 (find entries only in trie2)
    if !result.truncated {
        if verbose {
            println!("  Reverse pass: checking trie2 entries against trie1...");
        }

        if let Some(entries) = trie2.iter_prefix_with_values("")? {
            for (key2, value2) in entries {
                if key2.starts_with('\x00') {
                    continue;
                }
                result.trie2_count += 1;

                let indices = decode_ngram_key(&key2);
                let guard = vocab2.read();
                let words: Vec<String> = indices
                    .iter()
                    .filter_map(|&idx| {
                        if idx == 0 {
                            return None;
                        }
                        guard.get_term(idx)
                    })
                    .collect();
                drop(guard);

                if words.len() != indices.len() {
                    result.decode_failures_2 += 1;
                    continue;
                }

                let word_refs: Vec<&str> = words.iter().map(|s| s.as_str()).collect();
                match encode_ngram_key_existing(&word_refs, vocab1) {
                    Some(key1) => {
                        // Only check existence (values already compared in forward pass)
                        if trie1.get(&key1).is_none() {
                            if result.missing_in_1.len() < max_to_track {
                                result.missing_in_1.push(MissingEntry {
                                    ngram: words,
                                    count: value2,
                                });
                            }
                        }
                    }
                    None => {
                        if result.missing_in_1.len() < max_to_track {
                            result.missing_in_1.push(MissingEntry {
                                ngram: words,
                                count: value2,
                            });
                        }
                    }
                }

                if max_mismatches > 0 && result.total_errors() >= max_mismatches {
                    result.truncated = true;
                    break;
                }
            }
        }
    }

    Ok(result)
}

/// Print vocabulary comparison results.
fn print_vocab_results(result: &VocabComparisonResult) {
    println!("\n=== Vocabulary Comparison ===");
    println!("Vocabulary 1: {} terms", result.vocab1_count);
    println!("Vocabulary 2: {} terms", result.vocab2_count);

    if !result.only_in_1.is_empty() {
        println!("\nTerms only in vocabulary 1: {}", result.only_in_1.len());
        for (i, term) in result.only_in_1.iter().take(10).enumerate() {
            println!("  {}. {}", i + 1, term);
        }
        if result.only_in_1.len() > 10 {
            println!("  ... and {} more", result.only_in_1.len() - 10);
        }
    }

    if !result.only_in_2.is_empty() {
        println!("\nTerms only in vocabulary 2: {}", result.only_in_2.len());
        for (i, term) in result.only_in_2.iter().take(10).enumerate() {
            println!("  {}. {}", i + 1, term);
        }
        if result.only_in_2.len() > 10 {
            println!("  ... and {} more", result.only_in_2.len() - 10);
        }
    }

    if result.only_in_1.is_empty() && result.only_in_2.is_empty() {
        println!("\nVocabularies match exactly.");
    }
}

/// Print n-gram comparison results.
fn print_ngram_results(result: &NgramComparisonResult, max_mismatches: usize) {
    println!("\n=== N-gram Comparison ===");
    println!("Trie 1: {} entries", result.trie1_count);
    println!("Trie 2: {} entries", result.trie2_count);

    if result.decode_failures_1 > 0 {
        println!(
            "Decode failures in trie 1: {} (indices not in vocabulary)",
            result.decode_failures_1
        );
    }
    if result.decode_failures_2 > 0 {
        println!(
            "Decode failures in trie 2: {} (indices not in vocabulary)",
            result.decode_failures_2
        );
    }

    if !result.count_mismatches.is_empty() {
        println!("\nCount mismatches: {}", result.count_mismatches.len());
        for mismatch in result.count_mismatches.iter().take(10) {
            println!(
                "  {}: {} vs {}",
                mismatch.ngram.join("|"),
                mismatch.count1,
                mismatch.count2
            );
        }
        if result.count_mismatches.len() > 10 {
            println!("  ... and {} more", result.count_mismatches.len() - 10);
        }
    }

    if !result.missing_in_2.is_empty() {
        println!("\nMissing in trie 2: {}", result.missing_in_2.len());
        for entry in result.missing_in_2.iter().take(10) {
            println!("  {}: {}", entry.ngram.join("|"), entry.count);
        }
        if result.missing_in_2.len() > 10 {
            println!("  ... and {} more", result.missing_in_2.len() - 10);
        }
    }

    if !result.missing_in_1.is_empty() {
        println!("\nMissing in trie 1: {}", result.missing_in_1.len());
        for entry in result.missing_in_1.iter().take(10) {
            println!("  {}: {}", entry.ngram.join("|"), entry.count);
        }
        if result.missing_in_1.len() > 10 {
            println!("  ... and {} more", result.missing_in_1.len() - 10);
        }
    }

    if result.truncated {
        println!(
            "\n(Comparison truncated after {} mismatches)",
            max_mismatches
        );
    }

    if result.count_mismatches.is_empty()
        && result.missing_in_1.is_empty()
        && result.missing_in_2.is_empty()
    {
        println!("\nN-grams match exactly.");
    }
}
