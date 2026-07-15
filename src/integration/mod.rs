//! Integration with lling-llang WFST framework.
//!
//! This module provides:
//! - `LanguageModel` trait implementation for lattice rescoring
//! - WFST export for n-gram models (eager and lazy)
//! - Vocabulary mapping between words and WFST labels
//!
//! # WFST Export
//!
//! The WFST export creates a transducer representing the n-gram language model:
//! - States represent n-gram histories (previous n-1 words)
//! - Transitions represent word emissions with log probability weights
//! - Backoff ε-transitions implement smoothing fallback
//!
//! ```ignore
//! use libgrammstein::ngram::NgramModel;
//! use libgrammstein::integration::{NgramWfstExport, FromLogProb};
//! use lling_llang::semiring::LogWeight;
//!
//! let model: NgramModel<D> = /* ... */;
//! let (wfst, vocab) = model.to_wfst::<LogWeight>();
//!
//! // Use in ASR cascade
//! let cascade = CascadeBuilder::new()
//!     .grammar(wfst)
//!     .build();
//! ```
//!
//! # Language Model Scoring
//!
//! ```ignore
//! use libgrammstein::integration::GrammsteinLanguageModel;
//! use libgrammstein::ngram::NgramModel;
//! use lling_llang::layers::LanguageModel;
//!
//! let ngram_model = NgramModel::load("model.bin")?;
//! let lm = GrammsteinLanguageModel::from_ngram(ngram_model);
//!
//! // Use with lling-llang's LanguageModelLayer
//! let score = lm.score_sequence(&["the", "quick", "brown", "fox"]);
//! ```

#[cfg(feature = "lling-llang-integration")]
pub mod corrector;
/// The `T_lex ∘ T_gram` term-id noisy-channel cascade (the grammar corrector).
///
/// It composes two liblevenshtein automata directly over the term-id alphabet,
/// without the lling-llang lattice machinery the sibling [`corrector`] uses — but,
/// like the rest of this module, it is compiled only under the
/// `lling-llang-integration` feature (`lib.rs` gates the whole `integration` tree).
/// The articulatory `T_lex` path additionally requires `phonetic-correction`; the
/// sharded backend ([`sharded_grammar_corrector`]) additionally requires `google-books`.
pub mod grammar_corrector;
pub mod lazy_ngram;
mod lling_llang;
/// The sharded `T_lex ∘ T_gram` cascade over a Google-Books shard store — the same
/// beam decoder as [`grammar_corrector`], reading views from a `ShardCoordinator`.
/// Additionally requires the `google-books` feature (for the shard coordinator).
#[cfg(feature = "google-books")]
pub mod sharded_grammar_corrector;
pub mod vocabulary;
pub mod wfst_export;

#[cfg(feature = "lling-llang-integration")]
pub use self::corrector::{
    CorrectionResult, CorrectorConfig, CorrectorError, EditConfig, HierarchicalCorrector,
    LevenshteinCorrectionLayer,
};
pub use self::grammar_corrector::{
    GrammarCore, GrammarCorrection, GrammarCorrector, GrammarCorrectorConfig, GrammarNeighbor,
    LexCandidate, NgramViewSource, SingleView, DEFAULT_BACKOFF_ALPHA,
};
pub use self::lazy_ngram::{NgramHistoryKey, NgramLazyWfst, NgramStateRegistry, NgramStateSource};
pub use self::lling_llang::GrammsteinLanguageModel;
#[cfg(feature = "google-books")]
pub use self::sharded_grammar_corrector::{ShardedGrammarCorrector, ShardedView};
pub use self::vocabulary::{WordId, WordVocabulary, EOS_WORD_ID, UNK_WORD_ID};
pub use self::wfst_export::{
    FromLogProb, NgramTransducerBuilder, NgramWfstBuilder, NgramWfstExport,
};
