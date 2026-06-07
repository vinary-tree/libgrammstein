//! Diagnostic tool to inspect checkpoint state from artrie and JSON files.
//!
//! This tool helps debug resume issues by showing the exact checkpoint state,
//! including prefix completion status, n-gram counts, and WAL file sizes.
//!
//! # Usage
//!
//! ```bash
//! # Inspect a specific directory
//! cargo run --release --bin dump_checkpoint --features cli,google-books -- \
//!     --dir bak-sharded-interrupted/
//!
//! # Compare multiple directories
//! cargo run --release --bin dump_checkpoint --features cli,google-books -- \
//!     --dir bak-sharded-interrupted/ --dir bak-sharded-completed/ --dir .
//! ```

#[cfg(feature = "mimalloc-alloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use clap::Parser;
use libdictenstein::persistent_artrie::PersistentARTrie;
use libdictenstein::persistent_vocab_artrie::PersistentVocabARTrie;
use liblevenshtein::dictionary::Dictionary;
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Checkpoint state inspector for debugging resume issues.
#[derive(Parser)]
#[command(name = "dump_checkpoint")]
#[command(about = "Inspect checkpoint state from artrie files for debugging")]
struct Args {
    /// Directory or directories to inspect (can specify multiple with --dir)
    #[arg(long = "dir", value_name = "PATH")]
    directories: Vec<PathBuf>,

    /// Prefix for the model files (e.g., "english" for english.checkpoint.artrie)
    #[arg(long, default_value = "english")]
    prefix: String,

    /// Show all prefixes, not just completed ones
    #[arg(long, short = 'a')]
    all_prefixes: bool,

    /// Show raw checkpoint keys from trie
    #[arg(long)]
    raw_keys: bool,

    /// Verbose output
    #[arg(long, short = 'v')]
    verbose: bool,
}

/// On-disk u64 LE magic for `PersistentARTrie<V>` (mirrors
/// `libdictenstein::persistent_artrie_core::disk_manager::MAGIC_NUMBER`).
///
/// Both byte-keyed (`PersistentARTrie`) and the legacy char-keyed
/// (`PersistentARTrieChar`) tries share this file-header magic because they
/// use the same `DiskManager` — they differ only in arena layout. So we
/// cannot distinguish them by file header alone; this binary supports only
/// the current byte-keyed format (post-Phase-10e). To inspect legacy
/// pre-migration files, use an earlier binary build.
const PART_MAGIC_U64: u64 = 0x5041_5254_0001_0000;

/// 4-byte ASCII tag at the start of a `PersistentVocabARTrie` file.
const VOCB_MAGIC: &[u8; 4] = b"VOCB";

/// Quick sanity check: does the file start with the byte-trie `PART` u64 magic?
/// Used to fail fast with a clear error if a non-artrie file is given.
fn looks_like_byte_artrie(path: &Path) -> Result<bool, Box<dyn std::error::Error>> {
    let mut f = std::fs::File::open(path)?;
    let mut buf = [0u8; 8];
    if f.read(&mut buf)? < 8 {
        return Ok(false);
    }
    Ok(u64::from_le_bytes(buf) == PART_MAGIC_U64)
}

/// Quick sanity check: does the file start with the vocab `VOCB` ASCII magic?
fn looks_like_vocab(path: &Path) -> Result<bool, Box<dyn std::error::Error>> {
    let mut f = std::fs::File::open(path)?;
    let mut buf = [0u8; 4];
    if f.read(&mut buf)? < 4 {
        return Ok(false);
    }
    Ok(&buf == VOCB_MAGIC)
}

/// Checkpoint key constants (duplicated from checkpoint.rs for standalone binary)
const CHECKPOINT_KEY_PREFIX: &str = "\x00__ckpt__";
const CHECKPOINT_VERSION_KEY: &str = "\x00__ckpt__:version";
const CHECKPOINT_MKN_PHASE_KEY: &str = "\x00__ckpt__:mkn_phase";
const CHECKPOINT_TIMESTAMP_KEY: &str = "\x00__ckpt__:timestamp";
const CHECKPOINT_NGRAMS_PROCESSED_KEY: &str = "\x00__ckpt__:ngrams_processed";
const CHECKPOINT_UNIQUE_NGRAMS_KEY: &str = "\x00__ckpt__:unique_ngrams";
const CHECKPOINT_FILES_PROCESSED_KEY: &str = "\x00__ckpt__:files_processed";
const CHECKPOINT_NGRAMS_BY_ORDER_PREFIX: &str = "\x00__ckpt__:ngrams_by_order:";
const CHECKPOINT_PREFIX_KEY_PREFIX: &str = "\x00__ckpt__:prefix:";
const CHECKPOINT_ORDER_COMPLETE_PREFIX: &str = "\x00__ckpt__:order_complete:";
const CHECKPOINT_BITMAP_PREFIX: &str = "\x00__ckpt__:bitmap:";

/// Prefix status codes
const STATUS_COMPLETED: u64 = 1;
const STATUS_IN_PROGRESS: u64 = 2;
const STATUS_FAILED: u64 = 3;

/// Bitmap state encoding
const BITMAP_STATE_IN_PROGRESS: u8 = 0b01;
const BITMAP_STATE_COMPLETED: u8 = 0b10;
const BITMAP_STATE_FAILED: u8 = 0b11;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .format_timestamp(None)
        .init();

    let args = Args::parse();

    if args.directories.is_empty() {
        eprintln!("Error: No directories specified. Use --dir <path> to specify directories.");
        std::process::exit(1);
    }

    for dir in &args.directories {
        println!("\n{}", "=".repeat(80));
        println!("Directory: {}", dir.display());
        println!("{}", "=".repeat(80));

        if let Err(e) = inspect_directory(dir, &args) {
            eprintln!("Error inspecting {}: {}", dir.display(), e);
        }
    }

    Ok(())
}

fn inspect_directory(dir: &PathBuf, args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    // Check for WAL files first - this is crucial
    println!("\n--- WAL Files ---");
    check_wal_files(dir, &args.prefix)?;

    // Check for JSON checkpoint
    let json_path = dir.join(format!("{}.checkpoint.json", args.prefix));
    if json_path.exists() {
        println!("\n--- JSON Checkpoint ---");
        inspect_json_checkpoint(&json_path)?;
    } else {
        println!("\n--- JSON Checkpoint ---");
        println!("  Not found: {}", json_path.display());
    }

    // Check for trie-based checkpoint
    let trie_path = dir.join(format!("{}.checkpoint.artrie", args.prefix));
    if trie_path.exists() {
        println!("\n--- Trie Checkpoint ---");
        inspect_trie_checkpoint(&trie_path, args)?;
    } else {
        println!("\n--- Trie Checkpoint ---");
        println!("  Not found: {}", trie_path.display());
    }

    // Check vocabulary file
    let vocab_path = dir.join(format!("{}.vocab.artrie", args.prefix));
    if vocab_path.exists() {
        println!("\n--- Vocabulary ---");
        inspect_vocabulary(&vocab_path)?;
    }

    // Check sharding checkpoint if it exists
    let shard_checkpoint = dir
        .join(format!("{}_shards", args.prefix))
        .join("checkpoint.json");
    if shard_checkpoint.exists() {
        println!("\n--- Sharding Checkpoint ---");
        inspect_sharding_checkpoint(&shard_checkpoint)?;
    }

    Ok(())
}

fn check_wal_files(dir: &PathBuf, prefix: &str) -> Result<(), Box<dyn std::error::Error>> {
    let wal_patterns = [
        format!("{}.wal", prefix),
        format!("{}.vocab.wal", prefix),
        format!("{}.checkpoint.wal", prefix),
        format!("{}.mkn.wal", prefix),
    ];

    for pattern in &wal_patterns {
        let wal_path = dir.join(pattern);
        if wal_path.exists() {
            let metadata = std::fs::metadata(&wal_path)?;
            let size = metadata.len();
            let size_str = format_size(size);

            // WAL files with just headers are 64 bytes
            let status = if size <= 64 {
                "empty (checkpointed)"
            } else if size > 1_000_000 {
                "LARGE - NOT CHECKPOINTED!"
            } else {
                "has pending data"
            };

            println!("  {} - {} ({})", pattern, size_str, status);
        }
    }

    // Check for wal_archive
    let archive_dir = dir.join("wal_archive");
    if archive_dir.exists() && archive_dir.is_dir() {
        let count = std::fs::read_dir(&archive_dir)?.count();
        println!("  wal_archive/ - {} archived WAL files", count);
    }

    Ok(())
}

fn inspect_json_checkpoint(path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let file = std::fs::File::open(path)?;
    let value: serde_json::Value = serde_json::from_reader(file)?;

    if let Some(version) = value.get("version").and_then(|v| v.as_u64()) {
        println!("  Version: {}", version);
    }

    if let Some(timestamp) = value.get("timestamp").and_then(|v| v.as_str()) {
        println!("  Timestamp: {}", timestamp);
    }

    if let Some(stats) = value.get("stats").and_then(|v| v.as_object()) {
        println!("  Stats:");
        if let Some(ngrams) = stats.get("ngrams_processed").and_then(|v| v.as_u64()) {
            println!("    N-grams processed: {}", ngrams);
        }
        if let Some(unique) = stats.get("unique_ngrams").and_then(|v| v.as_u64()) {
            println!("    Unique n-grams: {}", unique);
        }
        if let Some(files) = stats.get("files_processed").and_then(|v| v.as_u64()) {
            println!("    Files processed: {}", files);
        }
    }

    if let Some(order_progress) = value.get("order_progress").and_then(|v| v.as_object()) {
        println!("  Order Progress:");
        for (order, progress) in order_progress {
            if let Some(progress_obj) = progress.as_object() {
                let is_complete = progress_obj
                    .get("is_complete")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);

                let completed_count = progress_obj
                    .get("completed_prefixes")
                    .and_then(|v| v.as_array())
                    .map(|a| a.len())
                    .unwrap_or(0);

                let in_progress_count = progress_obj
                    .get("in_progress_prefixes")
                    .and_then(|v| v.as_array())
                    .map(|a| a.len())
                    .unwrap_or(0);

                let failed_count = progress_obj
                    .get("failed_prefixes")
                    .and_then(|v| v.as_array())
                    .map(|a| a.len())
                    .unwrap_or(0);

                let ngrams = progress_obj
                    .get("ngrams_processed")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);

                println!(
                    "    Order {}: completed={}, in_progress={}, failed={}, ngrams={}, is_complete={}",
                    order, completed_count, in_progress_count, failed_count, ngrams, is_complete
                );

                // Show completed prefixes
                if let Some(completed) = progress_obj
                    .get("completed_prefixes")
                    .and_then(|v| v.as_array())
                {
                    let prefixes: Vec<&str> = completed.iter().filter_map(|v| v.as_str()).collect();
                    if !prefixes.is_empty() {
                        let prefix_str = if prefixes.len() > 10 {
                            format!(
                                "{} (and {} more)",
                                prefixes[..10].join(", "),
                                prefixes.len() - 10
                            )
                        } else {
                            prefixes.join(", ")
                        };
                        println!("      Completed: {}", prefix_str);
                    }
                }

                // Show in-progress prefixes (important for debugging)
                if let Some(in_progress) = progress_obj
                    .get("in_progress_prefixes")
                    .and_then(|v| v.as_array())
                {
                    let prefixes: Vec<&str> =
                        in_progress.iter().filter_map(|v| v.as_str()).collect();
                    if !prefixes.is_empty() {
                        println!("      In Progress: {}", prefixes.join(", "));
                    }
                }
            }
        }
    }

    Ok(())
}

/// Inspect a trie-format checkpoint file. Expects the current byte-keyed
/// `PersistentARTrie<u64>` format (post-Phase-10e). Rejects vocab files (which
/// have `"VOCB"` magic) with a clear message pointing at `inspect_vocabulary`.
fn inspect_trie_checkpoint(path: &PathBuf, args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    if looks_like_vocab(path)? {
        return Err(format!(
            "{} has VOCB (vocab) magic — inspect with the vocabulary path, not the trie path",
            path.display()
        )
        .into());
    }
    if !looks_like_byte_artrie(path)? {
        return Err(format!(
            "{} does not have the PART byte-trie magic in its first 8 bytes — \
             not a current-format checkpoint trie",
            path.display()
        )
        .into());
    }

    let trie = PersistentARTrie::<u64>::open(path)?;
    inspect_trie_checkpoint_inner(&trie, args)
}

/// Read all checkpoint fields from a byte-keyed `PersistentARTrie<u64>` and
/// print a human-readable summary.
fn inspect_trie_checkpoint_inner(
    store: &PersistentARTrie<u64>,
    args: &Args,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("  Trie size: {} entries", store_len(store));

    // Load basic metadata
    if let Some(version) = get_checkpoint_value(store, CHECKPOINT_VERSION_KEY) {
        println!("  Version: {}", version);
    }

    if let Some(timestamp) = get_checkpoint_value(store, CHECKPOINT_TIMESTAMP_KEY) {
        let dt = chrono::DateTime::from_timestamp(timestamp as i64, 0);
        if let Some(dt) = dt {
            println!("  Timestamp: {}", dt.format("%Y-%m-%d %H:%M:%S UTC"));
        }
    }

    if let Some(ngrams) = get_checkpoint_value(store, CHECKPOINT_NGRAMS_PROCESSED_KEY) {
        println!("  N-grams processed: {}", ngrams);
    }

    if let Some(unique) = get_checkpoint_value(store, CHECKPOINT_UNIQUE_NGRAMS_KEY) {
        println!("  Unique n-grams: {}", unique);
    }

    if let Some(files) = get_checkpoint_value(store, CHECKPOINT_FILES_PROCESSED_KEY) {
        println!("  Files processed: {}", files);
    }

    if let Some(mkn_phase) = get_checkpoint_value(store, CHECKPOINT_MKN_PHASE_KEY) {
        let phase_str = match mkn_phase {
            0 => "NotStarted",
            100 => "Pass1Complete",
            200 => "Complete",
            n if (1..100).contains(&n) => "Pass1InProgress",
            n if (101..200).contains(&n) => "Pass2InProgress",
            _ => "Unknown",
        };
        println!("  MKN Phase: {} ({})", mkn_phase, phase_str);
    }

    // Show ngrams by order
    println!("  N-grams by order:");
    for order in 1..=5u8 {
        let key = format!("{}{}", CHECKPOINT_NGRAMS_BY_ORDER_PREFIX, order);
        if let Some(count) = get_checkpoint_value(store, &key) {
            if count > 0 {
                println!("    Order {}: {}", order, count);
            }
        }
    }

    // Load prefix states using v3 bitmap format or v2 key-per-prefix format
    println!("  Prefix states by order:");

    for order in 1..=5u8 {
        // Check if order is complete
        let complete_key = format!("{}{}", CHECKPOINT_ORDER_COMPLETE_PREFIX, order);
        let is_complete = get_checkpoint_value(store, &complete_key)
            .map(|v| v == 1)
            .unwrap_or(false);

        // Try v3 bitmap format first
        let prefix_len = if order == 1 { 1u8 } else { 2u8 };
        let mut states = load_bitmap_states(store, order, prefix_len)?;

        // If no bitmap states, try v2 key-per-prefix format
        if states.is_empty() {
            states = load_v2_prefix_states(store, order)?;
        }

        if is_complete || !states.is_empty() {
            let completed: Vec<_> = states
                .iter()
                .filter(|(_, s)| *s == "Completed")
                .map(|(p, _)| p.as_str())
                .collect();
            let in_progress: Vec<_> = states
                .iter()
                .filter(|(_, s)| *s == "InProgress")
                .map(|(p, _)| p.as_str())
                .collect();
            let failed: Vec<_> = states
                .iter()
                .filter(|(_, s)| *s == "Failed")
                .map(|(p, _)| p.as_str())
                .collect();

            println!(
                "    Order {}: completed={}, in_progress={}, failed={}, is_complete={}",
                order,
                completed.len(),
                in_progress.len(),
                failed.len(),
                is_complete
            );

            if args.all_prefixes || args.verbose {
                if !completed.is_empty() {
                    let prefix_str = if completed.len() > 20 {
                        format!(
                            "{} (and {} more)",
                            completed[..20].join(", "),
                            completed.len() - 20
                        )
                    } else {
                        completed.join(", ")
                    };
                    println!("      Completed: {}", prefix_str);
                }
            }

            if !in_progress.is_empty() {
                println!("      In Progress: {}", in_progress.join(", "));
            }

            if !failed.is_empty() {
                println!("      Failed: {}", failed.join(", "));
            }
        }
    }

    // Show raw keys if requested
    if args.raw_keys {
        println!("\n  Raw checkpoint keys:");
        let entries = iter_prefix_pairs(store, CHECKPOINT_KEY_PREFIX);
        for (key, value) in entries {
            // Clean up key for display
            let display_key = key.replace('\x00', "\\0");
            println!("    {} = {}", display_key, value);
        }
    }

    Ok(())
}

/// Total entry count of a byte-keyed checkpoint trie.
fn store_len(store: &PersistentARTrie<u64>) -> usize {
    <PersistentARTrie<u64> as Dictionary>::len(store).unwrap_or(0)
}

/// Look up a `u64` checkpoint value by string key.
fn get_checkpoint_value(store: &PersistentARTrie<u64>, key: &str) -> Option<u64> {
    store.get_value_bytes(key.as_bytes())
}

/// Collect all `(term, value)` pairs whose term begins with `prefix`.
/// Returns an empty Vec if no such prefix exists in the store.
fn iter_prefix_pairs(store: &PersistentARTrie<u64>, prefix: &str) -> Vec<(String, u64)> {
    let iter = match store.iter_prefix_with_values(prefix.as_bytes()) {
        Some(it) => it,
        None => return Vec::new(),
    };
    iter.map(|(k, v)| (String::from_utf8_lossy(&k).into_owned(), v))
        .collect()
}

fn load_bitmap_states(
    store: &PersistentARTrie<u64>,
    order: u8,
    prefix_len: u8,
) -> Result<HashMap<String, String>, Box<dyn std::error::Error>> {
    let max_index: u16 = if prefix_len == 1 { 26 } else { 676 };
    let prefixes_per_chunk = 32usize;
    let num_chunks = (max_index as usize + prefixes_per_chunk - 1) / prefixes_per_chunk;

    // Load chunks
    let mut chunks = vec![0u64; num_chunks];
    let mut has_any = false;

    for chunk_idx in 0..num_chunks {
        let key = format!("{}{}:{}", CHECKPOINT_BITMAP_PREFIX, order, chunk_idx);
        if let Some(value) = get_checkpoint_value(store, &key) {
            chunks[chunk_idx] = value;
            if value != 0 {
                has_any = true;
            }
        }
    }

    if !has_any {
        return Ok(HashMap::new());
    }

    // Unpack states
    let mut states = HashMap::new();
    for index in 0..max_index {
        let chunk_idx = (index as usize) / prefixes_per_chunk;
        let bit_pos = ((index as usize) % prefixes_per_chunk) * 2;

        let state_bits = ((chunks[chunk_idx] >> bit_pos) & 0b11) as u8;
        let state = match state_bits {
            0b00 => continue, // NotStarted
            BITMAP_STATE_IN_PROGRESS => "InProgress",
            BITMAP_STATE_COMPLETED => "Completed",
            BITMAP_STATE_FAILED => "Failed",
            _ => continue,
        };

        let prefix = index_to_prefix(index, prefix_len);
        states.insert(prefix, state.to_string());
    }

    Ok(states)
}

fn load_v2_prefix_states(
    store: &PersistentARTrie<u64>,
    order: u8,
) -> Result<HashMap<String, String>, Box<dyn std::error::Error>> {
    let prefix_key_prefix = format!("{}{}:", CHECKPOINT_PREFIX_KEY_PREFIX, order);
    let mut states = HashMap::new();

    let entries = iter_prefix_pairs(store, &prefix_key_prefix);
    for (key, status_code) in entries {
        if let Some(prefix) = key.strip_prefix(&prefix_key_prefix) {
            let state = match status_code {
                STATUS_COMPLETED => "Completed",
                STATUS_IN_PROGRESS => "InProgress",
                STATUS_FAILED => "Failed",
                _ => continue,
            };
            states.insert(prefix.to_string(), state.to_string());
        }
    }

    Ok(states)
}

fn index_to_prefix(index: u16, prefix_len: u8) -> String {
    match prefix_len {
        1 => {
            let c = (b'a' + index as u8) as char;
            c.to_string()
        }
        2 => {
            let c1 = (b'a' + (index / 26) as u8) as char;
            let c2 = (b'a' + (index % 26) as u8) as char;
            format!("{}{}", c1, c2)
        }
        _ => format!("?{}", index),
    }
}

/// Inspect a vocabulary file. Expects the current `PersistentVocabARTrie`
/// format (`"VOCB"` magic) — produced by all imports since Phase 10e.
fn inspect_vocabulary(path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    if !looks_like_vocab(path)? {
        return Err(format!(
            "{} does not have the VOCB vocab magic in its first 4 bytes — \
             not a current-format vocabulary file",
            path.display()
        )
        .into());
    }

    let vocab = PersistentVocabARTrie::open(path)?;
    let count = <PersistentVocabARTrie as Dictionary>::len(&vocab).unwrap_or(0);
    println!("  Vocabulary entries: {}", count);

    // Show first 5 sample terms via iter_terms (uses the reverse index for
    // O(1) NodeRef → term backtrack).
    for (idx, term) in vocab.iter_terms().take(5).enumerate() {
        println!("    [{}] {}", idx, term);
    }
    Ok(())
}

fn inspect_sharding_checkpoint(path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let file = std::fs::File::open(path)?;
    let value: serde_json::Value = serde_json::from_reader(file)?;

    if let Some(state) = value.get("import_state") {
        println!("  Import state: {:?}", state);
    }

    if let Some(shards) = value.get("shards").and_then(|v| v.as_object()) {
        println!("  Shards: {}", shards.len());

        // Count in-progress shards
        let in_progress: Vec<_> = shards
            .iter()
            .filter(|(_, v)| v.get("current_prefix").and_then(|p| p.as_str()).is_some())
            .collect();

        if !in_progress.is_empty() {
            println!("  In-progress shards: {}", in_progress.len());
            for (key, _) in in_progress {
                println!("    {}", key);
            }
        }
    }

    Ok(())
}

fn format_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}
