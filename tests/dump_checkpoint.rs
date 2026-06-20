//! Integration tests for the `dump_checkpoint` binary.
//!
//! Fixtures are synthesized at test time (real on-disk artrie files written
//! into a `tempfile::TempDir`) rather than committing binary blobs to the repo.
//! Tests exercise the magic-number auto-detection paths added to handle the
//! Phase 10e migration from `PersistentARTrieChar` (legacy) to byte-keyed
//! `PersistentARTrie` for n-gram shards and `PersistentVocabARTrie` for
//! vocabularies.

// The `dump_checkpoint` binary these tests spawn has
// `required-features = ["cli", "google-books"]`, so gate the tests on the same set —
// otherwise `cargo test --features google-books` (without `cli`) compiles the tests but
// not the binary, and they fail to spawn it.
#![cfg(all(feature = "cli", feature = "google-books"))]

use libdictenstein::persistent_artrie::vocab::PersistentVocabARTrie;
use libdictenstein::persistent_artrie::PersistentARTrie;
use std::path::Path;
use std::process::Command;
use tempfile::TempDir;

// Checkpoint keys mirrored from src/sources/google_books/checkpoint.rs
const VERSION_KEY: &str = "\x00__ckpt__:version";
const NGRAMS_PROCESSED_KEY: &str = "\x00__ckpt__:ngrams_processed";
const MKN_PHASE_KEY: &str = "\x00__ckpt__:mkn_phase";

fn dump_checkpoint_bin() -> &'static str {
    env!("CARGO_BIN_EXE_dump_checkpoint")
}

/// Create a minimal byte-keyed (`PersistentARTrie<u64>`) checkpoint with
/// version + ngram count + MKN phase. Returns the trie path.
///
/// Uses `sync()` (WAL flush) — matches the libdictenstein test pattern.
/// `open()` replays the WAL on next load, so synced data is durable.
fn make_byte_checkpoint(dir: &Path) {
    let path = dir.join("english.checkpoint.artrie");
    let trie = PersistentARTrie::<u64>::create(&path).expect("PersistentARTrie::create failed");
    trie.insert_with_value(VERSION_KEY, 3);
    trie.insert_with_value(NGRAMS_PROCESSED_KEY, 1_234_567);
    trie.insert_with_value(MKN_PHASE_KEY, 200);
    trie.sync().expect("byte trie sync failed");
}

/// Create a minimal `PersistentVocabARTrie` vocabulary file with 3 sample terms.
fn make_vocab(dir: &Path) {
    let path = dir.join("english.vocab.artrie");
    let vocab = PersistentVocabARTrie::create_with_start_index(&path, 1)
        .expect("PersistentVocabARTrie::create_with_start_index failed");
    vocab.insert("hello").expect("insert hello");
    vocab.insert("world").expect("insert world");
    vocab.insert("ngram").expect("insert ngram");
    vocab.checkpoint().expect("vocab checkpoint failed");
}

fn run_dump(args: &[&str]) -> std::process::Output {
    Command::new(dump_checkpoint_bin())
        .args(args)
        .output()
        .expect("failed to spawn dump_checkpoint")
}

#[test]
fn inspects_current_byte_checkpoint() {
    let tmp = TempDir::new().expect("tempdir");
    make_byte_checkpoint(tmp.path());

    let output = run_dump(&["--dir", tmp.path().to_str().unwrap()]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "binary exited non-zero: stderr={}",
        stderr
    );
    assert!(
        stdout.contains("Trie Checkpoint"),
        "missing 'Trie Checkpoint' header in stdout: {}",
        stdout
    );
    assert!(
        stdout.contains("Version: 3"),
        "missing 'Version: 3' in stdout: {}\nstderr: {}",
        stdout,
        stderr
    );
    assert!(
        stdout.contains("1234567"),
        "missing '1234567' (ngrams_processed) in stdout: {}",
        stdout
    );
    // MKN phase 200 = Complete
    assert!(
        stdout.contains("MKN Phase: 200") && stdout.contains("Complete"),
        "expected MKN Phase 200/Complete in stdout: {}",
        stdout
    );
}

#[test]
fn inspects_current_vocabulary() {
    let tmp = TempDir::new().expect("tempdir");
    make_vocab(tmp.path());

    let output = run_dump(&["--dir", tmp.path().to_str().unwrap()]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "binary exited non-zero: stderr={}",
        stderr
    );
    assert!(
        stdout.contains("Vocabulary"),
        "missing 'Vocabulary' header in stdout: {}",
        stdout
    );
    assert!(
        stdout.contains("Vocabulary entries: 3"),
        "expected 'Vocabulary entries: 3' in stdout: {}",
        stdout
    );
    // At least one of the inserted terms appears in the sample output
    let any_term = stdout.contains("hello") || stdout.contains("world") || stdout.contains("ngram");
    assert!(any_term, "expected sample terms in stdout: {}", stdout);
}

#[test]
fn rejects_unknown_magic_with_clear_error() {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("english.checkpoint.artrie");
    // Write a file with non-PART/ARTC/VOCB magic
    std::fs::write(&path, b"BADMAGIC\x00\x00\x00\x00").expect("write garbage file");

    let output = run_dump(&["--dir", tmp.path().to_str().unwrap()]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Unrecognized trie format") || stderr.contains("Error inspecting"),
        "expected a clear error in stderr, got: {}",
        stderr
    );
}

#[test]
fn roundtrip_save_then_dump_byte_checkpoint() {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("english.checkpoint.artrie");

    {
        let trie = PersistentARTrie::<u64>::create(&path).expect("PersistentARTrie::create");
        trie.insert_with_value(VERSION_KEY, 4);
        trie.insert_with_value(NGRAMS_PROCESSED_KEY, 42_000_000);
        trie.insert_with_value(MKN_PHASE_KEY, 200);
        trie.sync().expect("sync");
    }

    let output = run_dump(&["--dir", tmp.path().to_str().unwrap()]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "binary exited non-zero: stderr={}",
        stderr
    );
    assert!(
        stdout.contains("Version: 4"),
        "missing Version: 4 in stdout: {}",
        stdout
    );
    assert!(
        stdout.contains("42000000"),
        "missing ngrams count in stdout: {}",
        stdout
    );
    assert!(
        stdout.contains("MKN Phase: 200") && stdout.contains("Complete"),
        "expected MKN Phase 200/Complete: {}",
        stdout
    );
}
