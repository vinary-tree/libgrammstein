//! Self-contained demo of the `T_lex ∘ T_gram` grammar corrector.
//!
//! Builds a tiny in-memory n-gram **count** store (LEB128-varint term-id keys) and
//! a spelling vocabulary, then corrects sentences exhibiting the full error
//! taxonomy: a substitution, a transposition, a *missing* word (word-level
//! insertion), and an *extraneous* word (word-level deletion) — plus a phonetic
//! sound-alike when the articulatory path is enabled.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example grammar_correct --features lling-llang-integration
//! # or, to enable the articulatory (sound-alike) T_lex path:
//! cargo run --example grammar_correct --features phonetic-correction
//! ```

use std::collections::HashMap;

use libdictenstein::dynamic_dawg::DynamicDawg;
use libgrammstein::integration::{GrammarCorrector, GrammarCorrectorConfig};
use libgrammstein::ngram::vocabulary::{create_vocabulary, encode_varint};

fn main() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let vocabulary = create_vocabulary(&dir.path().join("vocab")).expect("vocabulary");
    let order = 3usize;

    // A small corpus; repetition strengthens the trigram counts relative to the
    // (unseen) corrupted inputs.
    let sentences = [
        "the quick brown fox jumps over the lazy dog",
        "the quick brown fox jumps over the lazy dog",
        "the quick brown fox",
        "the quick brown fox",
        "the quick brown dog",
        "the lazy brown fox",
        "please answer the phone",
    ];

    // Count all 1..=order n-grams with sentence boundaries, keyed by concatenated
    // varints of each word's term-id — the importer's on-disk key format.
    let mut counts: HashMap<Vec<u64>, u64> = HashMap::new();
    for sentence in sentences {
        let mut words: Vec<&str> = vec!["<s>"; order - 1];
        words.extend(sentence.split_whitespace());
        words.push("</s>");
        let ids: Vec<u64> = words
            .iter()
            .map(|w| vocabulary.insert(w).expect("vocabulary insert"))
            .collect();
        for width in 1..=order {
            for window in ids.windows(width) {
                *counts.entry(window.to_vec()).or_insert(0) += 1;
            }
        }
    }
    let store = DynamicDawg::<u64>::new();
    for (sequence, count) in &counts {
        let mut key = Vec::new();
        for &id in sequence {
            encode_varint(id, &mut key);
        }
        store.insert_bytes_with_value(&key, *count);
    }

    let corrector =
        GrammarCorrector::from_store(vocabulary, store, order, GrammarCorrectorConfig::default());

    let cases = [
        ("substitution", "teh quick brown fox"),
        ("transposition", "the quikc brown fox"),
        ("missing word", "the quick fox"),
        ("extraneous word", "the the quick brown fox"),
        ("phonetic sound-alike", "please answer the fone"),
    ];

    println!("T_lex ∘ T_gram grammar corrector");
    println!(
        "(phonetics: {})\n",
        if corrector.config().use_phonetics {
            "on"
        } else {
            "off"
        }
    );
    for (label, input) in cases {
        match corrector.correct(input).first() {
            Some(best) => println!(
                "  [{label}]\n    in:  {input:?}\n    out: {:?}  (cost {:.3})\n",
                best.text, best.cost
            ),
            None => println!("  [{label}]\n    in:  {input:?}\n    out: <no hypothesis>\n"),
        }
    }
}
