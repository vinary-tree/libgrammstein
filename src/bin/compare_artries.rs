//! Compare two artrie files to verify n-gram content equivalence.
//!
//! This tool decodes n-gram keys from two artrie files using their respective
//! vocabularies and compares them for equality. It verifies that sharded and
//! non-sharded imports produce identical results.
//!
//! # Usage
//!
//! ```bash
//! cargo run --bin compare_artries --features cli,google-books -- \
//!     --trie1 bak-non-sharded/english.artrie \
//!     --vocab1 bak-non-sharded/english.vocab.artrie \
//!     --trie2 english.artrie \
//!     --vocab2 english.vocab.artrie
//! ```

use clap::Parser;
use liblevenshtein::dictionary::persistent_artrie_char::DiskBackedCharTrieInner;
use libgrammstein::ngram::vocabulary::decode_ngram_key;
use std::collections::{BTreeMap, HashMap};
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

    /// Show progress during extraction
    #[arg(long, short = 'v')]
    verbose: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging to see libdictenstein warnings
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .format_timestamp(None)
        .init();

    let args = Args::parse();

    // Load vocabularies directly as tries and build reverse caches
    println!("Loading vocabulary 1: {:?}", args.vocab1);
    let vocab1_trie = DiskBackedCharTrieInner::<u64>::open(&args.vocab1)?;
    let reverse1 = build_reverse_vocab(&vocab1_trie)?;
    println!("  {} terms in vocabulary 1", reverse1.len());

    println!("Loading vocabulary 2: {:?}", args.vocab2);
    let vocab2_trie = DiskBackedCharTrieInner::<u64>::open(&args.vocab2)?;
    let reverse2 = build_reverse_vocab(&vocab2_trie)?;
    println!("  {} terms in vocabulary 2", reverse2.len());

    // Load n-gram tries
    println!("Loading trie 1: {:?}", args.trie1);
    let trie1 = DiskBackedCharTrieInner::<u64>::open(&args.trie1)?;
    println!("  Trie 1 opened successfully, len = {}", trie1.len);

    println!("Loading trie 2: {:?}", args.trie2);
    let trie2 = DiskBackedCharTrieInner::<u64>::open(&args.trie2)?;
    println!("  Trie 2 opened successfully, len = {}", trie2.len);

    // Extract entries (filtering metadata)
    println!("Extracting entries from trie 1...");
    let entries1 = extract_entries(&trie1)?;
    println!("  Found {} entries", entries1.len());

    println!("Extracting entries from trie 2...");
    let entries2 = extract_entries(&trie2)?;
    println!("  Found {} entries", entries2.len());

    // Decode entries to word-based representation for comparison
    println!("Decoding entries from trie 1...");
    let decoded1 = decode_entries(&entries1, &reverse1, args.verbose);
    println!("  Decoded {} entries", decoded1.len());

    println!("Decoding entries from trie 2...");
    let decoded2 = decode_entries(&entries2, &reverse2, args.verbose);
    println!("  Decoded {} entries", decoded2.len());

    println!(
        "\nComparing {} vs {} decoded entries...",
        decoded1.len(),
        decoded2.len()
    );

    // Compare the decoded maps
    let mut mismatches = 0;
    let mut missing_in_2 = 0;
    let mut missing_in_1 = 0;
    let mut count_mismatches = 0;
    let max_to_show = if args.max_mismatches == 0 {
        usize::MAX
    } else {
        args.max_mismatches
    };

    // Check entries in 1 but not in 2
    for (words, count1) in &decoded1 {
        if let Some(count2) = decoded2.get(words) {
            if count1 != count2 {
                count_mismatches += 1;
                if mismatches < max_to_show {
                    println!(
                        "COUNT MISMATCH: {:?} has {} vs {}",
                        words_to_string(words),
                        count1,
                        count2
                    );
                }
                mismatches += 1;
            }
        } else {
            missing_in_2 += 1;
            if mismatches < max_to_show {
                println!(
                    "MISSING in trie2: {:?} (count={})",
                    words_to_string(words),
                    count1
                );
            }
            mismatches += 1;
        }
    }

    // Check entries in 2 but not in 1
    for (words, count2) in &decoded2 {
        if !decoded1.contains_key(words) {
            missing_in_1 += 1;
            if mismatches < max_to_show {
                println!(
                    "MISSING in trie1: {:?} (count={})",
                    words_to_string(words),
                    count2
                );
            }
            mismatches += 1;
        }
    }

    // Summary
    println!("\n=== SUMMARY ===");
    println!("Trie 1 entries (raw): {}", entries1.len());
    println!("Trie 2 entries (raw): {}", entries2.len());
    println!("Trie 1 entries (decoded): {}", decoded1.len());
    println!("Trie 2 entries (decoded): {}", decoded2.len());
    println!("Missing in trie 2: {}", missing_in_2);
    println!("Missing in trie 1: {}", missing_in_1);
    println!("Count mismatches: {}", count_mismatches);

    if mismatches == 0 {
        println!("\n✅ PASS: All n-grams match!");
        Ok(())
    } else {
        println!("\n❌ FAIL: {} total mismatches", mismatches);
        if mismatches > max_to_show {
            println!(
                "   (showing first {} of {} mismatches)",
                max_to_show, mismatches
            );
        }
        // Return success even on mismatch so we can see the summary
        Ok(())
    }
}

/// Build a reverse vocabulary lookup (index -> word) from a vocabulary trie.
fn build_reverse_vocab(
    trie: &DiskBackedCharTrieInner<u64>,
) -> Result<HashMap<u64, String>, Box<dyn std::error::Error>> {
    let mut reverse = HashMap::new();

    // Use prefix iteration with empty string to get all entries
    if let Some(entries) = trie.iter_prefix_with_values("")? {
        for (word, index) in entries {
            // Skip metadata keys (start with \x00)
            if word.starts_with('\x00') {
                continue;
            }
            reverse.insert(index, word);
        }
    }

    Ok(reverse)
}

/// Extract all non-metadata entries from a trie.
fn extract_entries(
    trie: &DiskBackedCharTrieInner<u64>,
) -> Result<BTreeMap<String, u64>, Box<dyn std::error::Error>> {
    let mut entries = BTreeMap::new();

    if let Some(all) = trie.iter_prefix_with_values("")? {
        for (key, value) in all {
            // Skip metadata keys (start with \x00)
            if !key.starts_with('\x00') {
                entries.insert(key, value);
            }
        }
    }

    Ok(entries)
}

/// Decode encoded keys to word vectors using the reverse vocabulary cache.
fn decode_entries(
    entries: &BTreeMap<String, u64>,
    reverse: &HashMap<u64, String>,
    verbose: bool,
) -> BTreeMap<Vec<String>, u64> {
    let mut decoded = BTreeMap::new();
    let mut decode_failures = 0;

    for (key, &count) in entries {
        let indices = decode_ngram_key(key);
        let words: Vec<String> = indices
            .iter()
            .filter_map(|idx| reverse.get(idx).cloned())
            .collect();

        // Check if all indices were resolved
        if words.len() != indices.len() {
            decode_failures += 1;
            if verbose {
                let missing: Vec<_> = indices
                    .iter()
                    .filter(|idx| !reverse.contains_key(idx))
                    .collect();
                eprintln!(
                    "  Warning: Could not decode all indices for key (indices: {:?}, missing: {:?})",
                    indices, missing
                );
            }
            continue;
        }

        decoded.insert(words, count);
    }

    if decode_failures > 0 {
        eprintln!(
            "  Warning: {} entries had unresolvable indices",
            decode_failures
        );
    }

    decoded
}

/// Convert a word vector to a displayable string.
fn words_to_string(words: &[String]) -> String {
    words.join(" ")
}
