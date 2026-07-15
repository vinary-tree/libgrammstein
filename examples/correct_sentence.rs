//! Self-contained demo of the hierarchical spelling/grammar corrector.
//!
//! Builds a tiny in-memory n-gram language model and spelling vocabulary, then
//! corrects the misspelled sentence `"teh quikc brwon fox"` into
//! `"the quick brown fox"`.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example correct_sentence --features lling-llang-integration
//! ```

use std::io::Write;

use libdictenstein::dynamic_dawg::DynamicDawg;
use libgrammstein::corpus::PlaintextReader;
use libgrammstein::integration::corrector::{CorrectorConfig, HierarchicalCorrector};
use libgrammstein::ngram::vocabulary::create_vocabulary;
use libgrammstein::ngram::{NgramEntry, TrainerBuilder};

fn main() {
    // A scratch directory for the corpus file and the persistent vocabulary.
    let dir = tempfile::TempDir::new().expect("create temp dir");

    // 1. Write a tiny training corpus. Repetition strengthens the in-vocabulary
    //    n-gram counts relative to the (unseen) misspellings.
    let sentence = "the quick brown fox jumps over the lazy dog";
    let corpus = format!("{sentence}\n").repeat(20);
    let corpus_path = dir.path().join("corpus.txt");
    {
        let mut file = std::fs::File::create(&corpus_path).expect("create corpus file");
        file.write_all(corpus.as_bytes())
            .expect("write corpus file");
    }

    // 2. Train a trigram model over an in-memory byte-native (term-id) backend.
    let reader = PlaintextReader::from_file(&corpus_path).expect("open corpus");
    let dictionary = DynamicDawg::<NgramEntry>::new();
    let model = TrainerBuilder::new(dictionary)
        .order(3)
        .train(reader)
        .expect("train n-gram model");

    // 3. Seed the spelling vocabulary with the correct words.
    let vocabulary = create_vocabulary(&dir.path().join("vocab")).expect("create vocabulary");
    for word in [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
    ] {
        vocabulary.insert(word).expect("insert vocabulary word");
    }

    // 4. Assemble the corrector and correct a misspelled sentence.
    let corrector =
        HierarchicalCorrector::from_ngram_model(vocabulary, model, CorrectorConfig::default());

    let input = "teh quikc brwon fox";
    let results = corrector.correct(input);

    println!("input:  {input:?}");
    match results.first() {
        Some(best) => println!("output: {:?}  (score {:.4})", best.text, best.score),
        None => println!("output: <no hypothesis>"),
    }
}
