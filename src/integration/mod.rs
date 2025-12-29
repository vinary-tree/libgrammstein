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

mod lling_llang;
pub mod lazy_ngram;
pub mod vocabulary;
pub mod wfst_export;

pub use self::lling_llang::GrammsteinLanguageModel;
pub use self::vocabulary::{WordId, WordVocabulary, EOS_WORD_ID, UNK_WORD_ID};
pub use self::wfst_export::{FromLogProb, NgramWfstExport, NgramWfstBuilder, NgramTransducerBuilder};
pub use self::lazy_ngram::{NgramStateSource, NgramStateRegistry, NgramHistoryKey, NgramLazyWfst};
