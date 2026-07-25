//! End-to-end hierarchical spelling / grammar corrector.
//!
//! This module wires three F1R3FLY.io components into a single, decode-ready
//! text-correction pipeline:
//!
//! 1. **liblevenshtein** — a Levenshtein automaton ([`Transducer`]) built once
//!    over a `libdictenstein::Dictionary` handle. It proposes in-vocabulary
//!    correction candidates for out-of-vocabulary (OOV) tokens.
//! 2. **libgrammstein** — the Google-Books n-gram (or hybrid) language model,
//!    exposed through [`GrammsteinLanguageModel`] as an lling-llang
//!    `LanguageModel`.
//! 3. **lling-llang** — the WFST lattice framework. Correction alternatives are
//!    materialized as a weighted DAG; the language model rescores the edges; a
//!    Viterbi / n-best decode extracts the most fluent path.
//!
//! # Pipeline
//!
//! ```text
//!   "teh quikc brwon fox"
//!            │  whitespace tokenize + one original edge per token
//!            ▼
//!   ┌──────────────────────┐   ┌────────────────────────────────────────────┐
//!   │  input Lattice       │   │  positions 0 ─ 1 ─ 2 ─ 3 ─ 4                │
//!   │  (originals only)    │   │  teh · quikc · brwon · fox                  │
//!   └──────────┬───────────┘   └────────────────────────────────────────────┘
//!              ▼
//!   ┌──────────────────────┐   LevenshteinCorrectionLayer (spelling)
//!   │  + correction edges  │   OOV token → Transducer.query_with_distance
//!   │  the · quick · brown │   cost(edge) = edit distance
//!   └──────────┬───────────┘
//!              ▼
//!   ┌──────────────────────┐   CfgFilterLayer (optional, only if a grammar
//!   │  grammatical paths   │   is supplied — there is no general English CFG)
//!   └──────────┬───────────┘
//!              ▼
//!   ┌──────────────────────┐   LanguageModelLayer (rescoring)
//!   │  LM-reweighted edges │   w' = (1−λ)·w + λ·(−log P(word | history))
//!   └──────────┬───────────┘
//!              ▼   viterbi / nbest  (⊕ = min over paths, ⊗ = + along a path)
//!   "the quick brown fox"
//! ```
//!
//! The layer type [`LevenshteinCorrectionLayer`] is fully generic over the
//! semiring `W`, the lattice backend `B`, and the dictionary handle `D`. The
//! [`HierarchicalCorrector`] façade fixes the concrete instantiation to
//! `TropicalWeight` + `HashMapBackend` and a `SharedVocabARTrie` spelling
//! dictionary, which is the common deployment configuration.
//!
//! # Example
//!
//! ```ignore
//! use libgrammstein::integration::corrector::{CorrectorConfig, HierarchicalCorrector};
//!
//! let corrector = HierarchicalCorrector::from_checkpoint(
//!     std::path::Path::new("model_dir"),
//!     CorrectorConfig::default(),
//! )?;
//!
//! let best = corrector.correct("teh quikc brwon fox");
//! assert_eq!(best[0].text, "the quick brown fox");
//! ```

use std::collections::HashSet;
use std::marker::PhantomData;
#[cfg(feature = "serde-extras")]
use std::path::Path;

// `DynamicDawgChar` backs the small static spelling fixtures in the test module;
// it is not on the default correction path, which traverses the persistent
// vocabulary trie.
#[cfg(test)]
use libdictenstein::dynamic_dawg::char::DynamicDawgChar;
use libdictenstein::{Dictionary, DictionaryNode};
use liblevenshtein::transducer::{Algorithm, Candidate, Transducer};

use lling_llang::backend::{HashMapBackend, LatticeBackend};
use lling_llang::cfg::Grammar;
use lling_llang::lattice::{EdgeMetadata, Lattice, LatticeBuilder, NodeId};
use lling_llang::layers::rescoring::LanguageModelLayer;
use lling_llang::layers::{CfgFilterLayer, CorrectionLayer, LayerResult};
use lling_llang::path::{nbest, viterbi};
use lling_llang::semiring::{Semiring, TropicalWeight};

use crate::integration::GrammsteinLanguageModel;
use crate::ngram::store::NgramLookup;
use crate::ngram::vocabulary::{SharedVocabARTrie, VocabularyError};
use crate::ngram::{NgramEntry, NgramModel};

/// Configuration for the [`LevenshteinCorrectionLayer`] spelling stage.
///
/// These knobs control how the Levenshtein automaton proposes candidates and
/// how many of them are injected into the lattice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EditConfig {
    /// Maximum edit distance searched by the automaton for each OOV token.
    pub max_distance: usize,
    /// Levenshtein algorithm variant (standard, transposition, merge-and-split).
    pub algorithm: Algorithm,
    /// Upper bound on the number of correction edges added per OOV token.
    ///
    /// Candidates are ranked by ascending edit distance (ties broken
    /// lexicographically), so the closest matches are retained.
    pub max_corrections_per_word: usize,
    /// Tokens shorter than this (counted in Unicode scalar values) are never
    /// corrected. Prevents thrashing on 1–2 character tokens whose neighborhood
    /// is dense.
    pub min_word_length: usize,
    /// Whether to retain the original (possibly OOV) token as an alternative.
    ///
    /// When `true` (the default) every input edge is preserved and correction
    /// edges are added alongside it. When `false`, an OOV token that produced at
    /// least one correction is *replaced* by its corrections; if a token
    /// produced no candidates its original edge is always kept so the lattice
    /// stays connected.
    pub keep_original: bool,
}

impl Default for EditConfig {
    fn default() -> Self {
        Self {
            max_distance: 2,
            // Damerau-Levenshtein: adjacent transpositions (e.g. "teh" → "the")
            // are among the most common typo classes (Damerau, 1964) and cost a
            // single edit under this algorithm rather than two substitutions, so
            // corrections stay close to the input and cooperate with the LM.
            algorithm: Algorithm::Transposition,
            max_corrections_per_word: 8,
            min_word_length: 2,
            keep_original: true,
        }
    }
}

/// Resolve the input-sequence position of a lattice node.
///
/// Mirrors the fallback used by lling-llang's own layers: prefer the node's
/// recorded `position`, otherwise fall back to the raw node index.
#[inline]
fn node_position<W: Semiring, B: LatticeBackend>(lattice: &Lattice<W, B>, node: NodeId) -> usize {
    lattice
        .node(node)
        .and_then(|n| n.position)
        .unwrap_or(node.value() as usize)
}

/// A first-class lling-llang [`CorrectionLayer`] that expands a lattice with
/// Levenshtein spelling corrections drawn from a `libdictenstein::Dictionary`.
///
/// A single [`Transducer`] is constructed at layer-construction time and reused
/// for every [`CorrectionLayer::apply`] call — the automaton is **never** rebuilt
/// per word. For each input edge whose label is not present in the dictionary
/// (and is long enough), the transducer enumerates every dictionary term within
/// [`EditConfig::max_distance`] edits and adds one correction edge per candidate,
/// weighted by the edit distance in the target semiring.
///
/// The layer never mutates its input: [`CorrectionLayer::apply`] returns a brand
/// new lattice that clones the input backend and copies the original edges.
///
/// # Type Parameters
///
/// - `W`: the semiring the lattice is weighted over (must be constructible from
///   an `f64` cost, i.e. `W: From<f64>`; the edit distance becomes the cost).
/// - `B`: the lattice vocabulary backend.
/// - `D`: the dictionary handle (`libdictenstein::Dictionary`). The associated
///   character unit must convert to/from `char`, matching the bound set that
///   liblevenshtein's [`Transducer`] query iterators require.
pub struct LevenshteinCorrectionLayer<W, B, D>
where
    D: Dictionary,
{
    /// The Levenshtein automaton, built once over `D`.
    transducer: Transducer<D>,
    /// Candidate-generation configuration.
    config: EditConfig,
    /// `W` and `B` only participate in the `CorrectionLayer` impl; the
    /// `fn() -> (W, B)` shape keeps the layer `Send + Sync` regardless of `W`/`B`.
    _marker: PhantomData<fn() -> (W, B)>,
}

impl<W, B, D> LevenshteinCorrectionLayer<W, B, D>
where
    D: Dictionary + Clone + Send + Sync,
    D::Node: Send + Sync,
    <D::Node as DictionaryNode>::Unit: Into<char> + TryFrom<char> + Copy + Send + Sync,
{
    /// Build a layer, constructing the Levenshtein automaton over `dictionary`.
    pub fn new(dictionary: D, config: EditConfig) -> Self {
        Self {
            transducer: Transducer::new(dictionary, config.algorithm),
            config,
            _marker: PhantomData,
        }
    }

    /// Build a layer from an already-constructed transducer.
    ///
    /// Useful when the automaton is shared or pre-warmed elsewhere. The
    /// `config.algorithm` should match the algorithm the transducer was built
    /// with; the transducer's own algorithm governs querying.
    pub fn from_transducer(transducer: Transducer<D>, config: EditConfig) -> Self {
        Self {
            transducer,
            config,
            _marker: PhantomData,
        }
    }

    /// Borrow the underlying dictionary handle.
    #[inline]
    pub fn dictionary(&self) -> &D {
        self.transducer.dictionary()
    }

    /// Borrow the candidate-generation configuration.
    #[inline]
    pub fn config(&self) -> &EditConfig {
        &self.config
    }

    /// Collect deduplicated, distance-ranked correction candidates for `word`.
    ///
    /// The original token is excluded (it is either preserved separately or
    /// intentionally dropped by the caller). At most
    /// [`EditConfig::max_corrections_per_word`] candidates are returned.
    fn candidates_for(&self, word: &str) -> Vec<Candidate> {
        let mut candidates: Vec<Candidate> = self
            .transducer
            .query_with_distance(word, self.config.max_distance)
            .collect();

        // Deterministic ordering: closest first, then lexicographic.
        candidates.sort_by(|a, b| {
            a.distance
                .cmp(&b.distance)
                .then_with(|| a.term.cmp(&b.term))
        });

        let mut seen: HashSet<&str> = HashSet::with_capacity(candidates.len());
        let mut selected: Vec<Candidate> =
            Vec::with_capacity(candidates.len().min(self.config.max_corrections_per_word));
        for candidate in &candidates {
            if selected.len() >= self.config.max_corrections_per_word {
                break;
            }
            // Never re-propose the original spelling as a "correction".
            if candidate.term == word {
                continue;
            }
            if seen.insert(candidate.term.as_str()) {
                selected.push(candidate.clone());
            }
        }
        selected
    }
}

impl<W, B, D> CorrectionLayer<W, B> for LevenshteinCorrectionLayer<W, B, D>
where
    W: Semiring + From<f64>,
    B: LatticeBackend,
    D: Dictionary + Clone + Send + Sync,
    D::Node: Send + Sync,
    <D::Node as DictionaryNode>::Unit: Into<char> + TryFrom<char> + Copy + Send + Sync,
{
    fn name(&self) -> &str {
        "levenshtein-correction"
    }

    fn apply(&self, lattice: &Lattice<W, B>) -> LayerResult<Lattice<W, B>> {
        let num_nodes = lattice.num_nodes();
        let avg_edges = lattice.num_edges() / num_nodes.max(1);
        // Reserve room for the originals plus up to `max_corrections_per_word`
        // additional edges at each position.
        let mut builder = LatticeBuilder::<W, B>::with_capacity(
            lattice.backend().clone(),
            num_nodes,
            avg_edges + self.config.max_corrections_per_word + 1,
        );

        for edge in lattice.edges() {
            let source_pos = node_position(lattice, edge.source);
            let target_pos = node_position(lattice, edge.target);

            // An edge is a correction target iff it carries a word that is long
            // enough and is not already present in the dictionary.
            let correction_target = match lattice.edge_word(edge) {
                Some(word) => {
                    word.chars().count() >= self.config.min_word_length
                        && !self.transducer.dictionary().contains(word)
                }
                None => false,
            };

            if !correction_target {
                // Non-target edges (epsilon, in-vocabulary, or too short) are
                // copied verbatim, preserving their weight and metadata.
                builder.add_correction_by_id(
                    source_pos,
                    target_pos,
                    edge.label,
                    edge.weight,
                    edge.metadata.clone(),
                );
                continue;
            }

            let word = lattice
                .edge_word(edge)
                .expect("correction_target is only true when the edge has a word");
            let candidates = self.candidates_for(word);

            for candidate in &candidates {
                let cost = W::from(candidate.distance as f64);
                let distance = u8::try_from(candidate.distance).unwrap_or(u8::MAX);
                builder.add_correction(
                    source_pos,
                    target_pos,
                    &candidate.term,
                    cost,
                    EdgeMetadata::correction(distance),
                );
            }

            // Preserve the original token when requested, or unconditionally
            // when no correction was produced (otherwise the position would
            // have no outgoing edge and the lattice would be disconnected).
            if self.config.keep_original || candidates.is_empty() {
                builder.add_correction_by_id(
                    source_pos,
                    target_pos,
                    edge.label,
                    edge.weight,
                    edge.metadata.clone(),
                );
            }
        }

        let end_pos = node_position(lattice, lattice.end());
        Ok(builder.build(end_pos))
    }

    fn estimated_reduction(&self) -> f64 {
        // The layer expands (never prunes) the lattice.
        1.0
    }
}

/// A single ranked correction hypothesis returned by
/// [`HierarchicalCorrector::correct`].
#[derive(Clone, Debug, PartialEq)]
pub struct CorrectionResult {
    /// The corrected text (whitespace-joined tokens of the decoded path).
    pub text: String,
    /// Total path weight (lower is better under the tropical semiring).
    pub score: f64,
}

/// Configuration for the [`HierarchicalCorrector`] façade.
#[derive(Clone, Debug, PartialEq)]
pub struct CorrectorConfig {
    /// Maximum Levenshtein edit distance for spelling candidates.
    pub max_edit_distance: usize,
    /// Levenshtein algorithm variant used by the spelling layer.
    pub algorithm: Algorithm,
    /// Maximum correction edges injected per out-of-vocabulary token.
    pub max_corrections_per_word: usize,
    /// Minimum token length (Unicode scalar values) eligible for correction.
    pub min_word_length: usize,
    /// Whether original tokens are retained alongside corrections.
    pub keep_original: bool,
    /// Number of hypotheses to return. `1` (the default) runs Viterbi; values
    /// greater than one run the lazy n-best enumeration.
    pub k_best: usize,
    /// Language-model interpolation weight `λ ∈ [0, 1]` passed to the
    /// `LanguageModelLayer` (`0` ignores the LM, `1` uses only the LM).
    pub lm_weight: f64,
    /// File name (within the checkpoint directory) of the persistent
    /// vocabulary, used by [`HierarchicalCorrector::from_checkpoint`].
    pub vocabulary_filename: String,
    /// File name (within the checkpoint directory) of the serialized n-gram
    /// model, used by [`HierarchicalCorrector::from_checkpoint`].
    pub model_filename: String,
}

impl Default for CorrectorConfig {
    fn default() -> Self {
        Self {
            max_edit_distance: 2,
            algorithm: Algorithm::Transposition,
            max_corrections_per_word: 8,
            min_word_length: 2,
            keep_original: true,
            k_best: 1,
            // The LM interpolation must overcome the edit-distance cost that a
            // correction edge carries relative to the (zero-cost) original OOV
            // token. A weight around 0.75 lets fluency dominate — which is what
            // spelling correction wants — while still letting edit distance act
            // as a tie-breaker among comparably-fluent candidates.
            lm_weight: 0.75,
            vocabulary_filename: "vocabulary".to_string(),
            model_filename: "model.bin".to_string(),
        }
    }
}

impl CorrectorConfig {
    /// Project the spelling-relevant knobs into an [`EditConfig`].
    fn edit_config(&self) -> EditConfig {
        EditConfig {
            max_distance: self.max_edit_distance,
            algorithm: self.algorithm,
            max_corrections_per_word: self.max_corrections_per_word,
            min_word_length: self.min_word_length,
            keep_original: self.keep_original,
        }
    }
}

/// Errors that can arise while assembling a [`HierarchicalCorrector`] from disk.
#[derive(Debug, thiserror::Error)]
pub enum CorrectorError {
    /// The persistent vocabulary could not be opened.
    #[error("vocabulary error: {0}")]
    Vocabulary(#[from] VocabularyError),

    /// The n-gram model could not be loaded / deserialized.
    #[error("model error: {0}")]
    Model(#[from] crate::Error),

    /// A required checkpoint artifact was missing.
    #[error("checkpoint artifact not found: {0}")]
    MissingArtifact(String),
}

/// Concrete spelling dictionary used by the façade: the live, memory-mapped
/// persistent vocabulary trie.
///
/// The Levenshtein [`Transducer`] drives correction by traversing its dictionary
/// node-by-node (`root` → `transition` → …) to arbitrary depth. `SharedVocabARTrie`
/// implements `libdictenstein::Dictionary`, and its node handle
/// (`VocabTrieNodeRef` = `OverlayDictionaryNode<CharKey, u64>`) keeps an `Arc` to
/// each overlay node it hands back, so `transition`/`edges` re-resolve a child's
/// children on demand and descend the **full depth** of the trie (libdictenstein
/// `src/persistent_artrie/vocab/mod.rs`). The automaton therefore runs *directly
/// over the persistent vocabulary* — no full-vocabulary RAM copy, no `O(N)`
/// rebuild, eviction and persistence intact — which is what makes the corrector
/// viable at Google-Books scale.
///
/// Because [`LevenshteinCorrectionLayer`] is generic over any fully-traversable
/// `Dictionary`, a caller who wants a frozen, static automaton for a *small*
/// dictionary can still build the layer over a `DynamicDawgChar` /
/// `DoubleArrayTrie` instead; the façade defaults to the persistent trie, the
/// correct choice at scale.
type SpellingDictionary = SharedVocabARTrie;

/// The concrete spelling layer used by the [`HierarchicalCorrector`] façade.
type VocabSpellingLayer =
    LevenshteinCorrectionLayer<TropicalWeight, HashMapBackend, SpellingDictionary>;

/// Hierarchical spelling + grammar + language-model corrector.
///
/// This is the batteries-included façade over the generic
/// [`LevenshteinCorrectionLayer`]. It fixes the semiring to `TropicalWeight` and
/// the backend to `HashMapBackend`, uses a `SharedVocabARTrie` as the spelling
/// dictionary, and rescores with a libgrammstein n-gram / hybrid model.
///
/// Construct one with [`HierarchicalCorrector::from_checkpoint`] (load from
/// disk), [`HierarchicalCorrector::from_ngram_model`] (in-memory n-gram), or
/// [`HierarchicalCorrector::from_parts`] (in-memory, n-gram or hybrid). An
/// optional CFG grammar can be attached with
/// [`HierarchicalCorrector::with_grammar`].
pub struct HierarchicalCorrector {
    spelling: VocabSpellingLayer,
    language_model: LanguageModelLayer,
    grammar: Option<Grammar>,
    config: CorrectorConfig,
}

impl HierarchicalCorrector {
    /// Assemble a corrector from an in-memory vocabulary and language model.
    ///
    /// `vocabulary` supplies the spelling dictionary (both the membership test
    /// and the fuzzy candidate source). `language_model` may wrap either a pure
    /// n-gram model or a hybrid n-gram + embedding model.
    pub fn from_parts<S>(
        vocabulary: SharedVocabARTrie,
        language_model: GrammsteinLanguageModel<S>,
        config: CorrectorConfig,
    ) -> Self
    where
        S: NgramLookup + Send + Sync + 'static,
    {
        // The Levenshtein automaton runs directly over the persistent vocabulary
        // trie — moving the cheap `Arc` handle in, never materializing the terms.
        let spelling = LevenshteinCorrectionLayer::new(vocabulary, config.edit_config());
        let language_model =
            LanguageModelLayer::new(Box::new(language_model)).with_weight(config.lm_weight);
        Self {
            spelling,
            language_model,
            grammar: None,
            config,
        }
    }

    /// Convenience constructor for a pure n-gram language model.
    pub fn from_ngram_model<S>(
        vocabulary: SharedVocabARTrie,
        model: NgramModel<S>,
        config: CorrectorConfig,
    ) -> Self
    where
        S: NgramLookup + Send + Sync + 'static,
    {
        Self::from_parts(
            vocabulary,
            GrammsteinLanguageModel::from_ngram(model),
            config,
        )
    }

    /// Attach a CFG grammar. When present, a `CfgFilterLayer` runs between the
    /// spelling and language-model stages, pruning ungrammatical paths.
    ///
    /// This is a genuinely optional slot: no grammar means the CFG stage is
    /// simply skipped (there is no general-purpose English CFG to default to).
    pub fn with_grammar(mut self, grammar: Grammar) -> Self {
        self.grammar = Some(grammar);
        self
    }

    /// Borrow the effective configuration.
    #[inline]
    pub fn config(&self) -> &CorrectorConfig {
        &self.config
    }

    /// Whether a CFG grammar is attached.
    #[inline]
    pub fn has_grammar(&self) -> bool {
        self.grammar.is_some()
    }

    /// Correct `text`, returning ranked hypotheses best-first.
    ///
    /// The input is tokenized on whitespace and turned into a linear lattice of
    /// original edges. The spelling layer, the optional CFG filter, and the
    /// language-model rescoring layer are applied in sequence, and the resulting
    /// lattice is decoded with Viterbi (when `k_best == 1`) or n-best.
    ///
    /// An empty input yields an empty result vector.
    pub fn correct(&self, text: &str) -> Vec<CorrectionResult> {
        let tokens: Vec<&str> = text.split_whitespace().collect();
        if tokens.is_empty() {
            return Vec::new();
        }

        // Build the linear input lattice: one zero-cost original edge per token.
        let mut builder = LatticeBuilder::<TropicalWeight, _>::with_capacity(
            HashMapBackend::new(),
            tokens.len() + 1,
            1,
        );
        for (index, token) in tokens.iter().enumerate() {
            builder.add_correction(
                index,
                index + 1,
                token,
                TropicalWeight::one(),
                EdgeMetadata::original(),
            );
        }
        let input = builder.build(tokens.len());

        // Stage 1: Levenshtein spelling corrections. This layer only ever fails
        // to `apply` on a cyclic lattice; the linear lattice above is acyclic.
        let spelled = self
            .spelling
            .apply(&input)
            .expect("spelling layer cannot fail on an acyclic lattice");

        // Stage 2 (optional): CFG filtering. Constructed per call because
        // `CfgFilterLayer` borrows the grammar owned by `self`.
        let filtered = match &self.grammar {
            Some(grammar) => CfgFilterLayer::new(grammar)
                .apply(&spelled)
                .expect("cfg filter layer applies to an acyclic lattice"),
            None => spelled,
        };

        // Stage 3: language-model rescoring.
        let mut rescored = self
            .language_model
            .apply(&filtered)
            .expect("language-model layer cannot fail on an acyclic lattice");

        // Decode.
        if self.config.k_best <= 1 {
            let result = viterbi(&mut rescored);
            if !result.success {
                return Vec::new();
            }
            vec![CorrectionResult {
                text: result.path.to_words(&rescored).join(" "),
                score: result.path.weight.value(),
            }]
        } else {
            let paths = nbest(&mut rescored, self.config.k_best);
            let mut results = Vec::with_capacity(paths.len());
            for path in &paths {
                results.push(CorrectionResult {
                    text: path.to_words(&rescored).join(" "),
                    score: path.weight.value(),
                });
            }
            results
        }
    }
}

/// Disk-loading constructor. Requires `serde-extras` (transitively enabled by
/// the `google-books` feature) for the serialized n-gram model format.
#[cfg(feature = "serde-extras")]
impl HierarchicalCorrector {
    /// Assemble a corrector from a checkpoint directory.
    ///
    /// The directory is expected to contain:
    /// - a persistent vocabulary at `config.vocabulary_filename`
    ///   (default `"vocabulary"`), opened via [`open_vocabulary`]; and
    /// - a portable [`NgramModel`] at `config.model_filename` (default
    ///   `"model.bin"`), written by [`NgramModel::save_portable`].
    ///
    /// The language model is reconstructed as a byte-native
    /// `TermIdStore<DynamicDawg<NgramEntry>>` **over the same persistent
    /// vocabulary** used for spelling, so both share one term-id space. The
    /// model's stamped vocabulary fingerprint is verified against that
    /// vocabulary, rejecting a mismatched `(model, vocabulary)` pair rather than
    /// scoring against the wrong vocabulary (see
    /// [`NgramModel::from_portable_with_vocabulary`]).
    pub fn from_checkpoint(
        checkpoint_dir: &Path,
        config: CorrectorConfig,
    ) -> Result<Self, CorrectorError> {
        use crate::ngram::vocabulary::open_vocabulary;
        use libdictenstein::dynamic_dawg::DynamicDawg;

        let vocabulary_path = checkpoint_dir.join(&config.vocabulary_filename);
        let model_path = checkpoint_dir.join(&config.model_filename);

        if !model_path.exists() {
            return Err(CorrectorError::MissingArtifact(
                model_path.display().to_string(),
            ));
        }

        let vocabulary = open_vocabulary(&vocabulary_path)?;

        // Read the portable model and rebuild it over the *shared* persistent
        // vocabulary (fingerprint-checked), rather than a private copy.
        let portable = {
            let file = std::fs::File::open(&model_path).map_err(crate::Error::from)?;
            let reader = std::io::BufReader::new(file);
            crate::bincode_compat::deserialize_from::<_, crate::ngram::PortableNgramModel>(reader)
                .map_err(crate::Error::from)?
        };
        let model = NgramModel::from_portable_with_vocabulary(
            portable,
            DynamicDawg::<NgramEntry>::new(),
            vocabulary.clone(),
        )?;

        Ok(Self::from_ngram_model(vocabulary, model, config))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::PlaintextReader;
    use crate::ngram::vocabulary::create_vocabulary;
    use crate::ngram::TrainerBuilder;
    use libdictenstein::dynamic_dawg::DynamicDawg;
    use std::io::Write;
    use std::path::Path;
    use tempfile::TempDir;

    /// The `LevenshteinCorrectionLayer` must inject a correction edge for the
    /// real word "the" when fed the OOV token "teh".
    #[test]
    fn levenshtein_layer_adds_correction_edge() {
        let dictionary =
            DynamicDawgChar::<()>::from_terms(vec!["the", "ten", "tea", "quick", "brown", "fox"]);
        let layer: LevenshteinCorrectionLayer<TropicalWeight, HashMapBackend, _> =
            LevenshteinCorrectionLayer::new(dictionary, EditConfig::default());

        // Single-token lattice for the misspelling "teh".
        let mut builder = LatticeBuilder::<TropicalWeight, _>::new(HashMapBackend::new());
        builder.add_correction(0, 1, "teh", TropicalWeight::one(), EdgeMetadata::original());
        let input = builder.build(1);

        let output = layer.apply(&input).expect("spelling layer applies");

        let words: Vec<&str> = output
            .edges()
            .iter()
            .filter_map(|edge| output.edge_word(edge))
            .collect();

        assert!(
            words.contains(&"the"),
            "expected a correction edge for 'the', got {words:?}"
        );
        // The original token is retained by default.
        assert!(
            words.contains(&"teh"),
            "expected the original token 'teh' to be preserved, got {words:?}"
        );
        // Under the default Damerau-Levenshtein algorithm, "teh" → "the" is a
        // single adjacent transposition, so the correction edge carries distance 1.
        let the_edge = output
            .edges()
            .iter()
            .find(|edge| output.edge_word(edge) == Some("the"))
            .expect("the correction edge exists");
        assert_eq!(the_edge.metadata.edit_distance, Some(1));
        assert!(!the_edge.metadata.is_original);
    }

    /// In-vocabulary tokens must not be corrected (only their original edge is
    /// kept), and short tokens below `min_word_length` are left untouched.
    #[test]
    fn levenshtein_layer_skips_in_vocabulary_and_short_tokens() {
        let dictionary = DynamicDawgChar::<()>::from_terms(vec!["fox", "the"]);
        let layer: LevenshteinCorrectionLayer<TropicalWeight, HashMapBackend, _> =
            LevenshteinCorrectionLayer::new(dictionary, EditConfig::default());

        let mut builder = LatticeBuilder::<TropicalWeight, _>::new(HashMapBackend::new());
        builder.add_correction(0, 1, "fox", TropicalWeight::one(), EdgeMetadata::original());
        let input = builder.build(1);

        let output = layer.apply(&input).expect("spelling layer applies");
        // Exactly one edge: the in-vocabulary original, no corrections.
        assert_eq!(output.num_edges(), 1);
        assert_eq!(
            output.edge_word(&output.edges()[0]),
            Some("fox"),
            "in-vocabulary token should be preserved without corrections"
        );
    }

    /// Build a small vocabulary trie seeded with the given words.
    fn build_vocabulary(dir: &Path, words: &[&str]) -> SharedVocabARTrie {
        let vocab = create_vocabulary(&dir.join("vocab")).expect("create vocabulary");
        for word in words {
            vocab.insert(word).expect("insert vocabulary word");
        }
        vocab
    }

    /// Regression test for the libdictenstein `VocabTrieNodeRef` full-depth
    /// descent fix. A [`LevenshteinCorrectionLayer`] built *directly* over the
    /// persistent [`SharedVocabARTrie`] (no materialized DAWG) must recover a
    /// correction whose edit lies past the first character.
    ///
    /// Before the fix, the vocab trie's node handle returned childless nodes, so a
    /// transducer over it dead-ended after one character and could only match
    /// single-character terms. `"quikc"` → `"quick"` is a distance-1 adjacent
    /// transposition reachable only by descending `q → u → i → k → c` (depth 5),
    /// so recovering it proves the automaton descends the full depth of the
    /// persistent trie — the whole point of dropping the DAWG materialization.
    #[test]
    fn levenshtein_layer_over_persistent_vocab_descends_full_depth() {
        let dir = TempDir::new().expect("temp dir");
        let vocabulary = build_vocabulary(dir.path(), &["quick", "quack", "brown", "the"]);

        let layer: LevenshteinCorrectionLayer<TropicalWeight, HashMapBackend, _> =
            LevenshteinCorrectionLayer::new(vocabulary, EditConfig::default());

        let mut builder = LatticeBuilder::<TropicalWeight, _>::new(HashMapBackend::new());
        builder.add_correction(
            0,
            1,
            "quikc",
            TropicalWeight::one(),
            EdgeMetadata::original(),
        );
        let input = builder.build(1);

        let output = layer.apply(&input).expect("spelling layer applies");
        let words: Vec<&str> = output
            .edges()
            .iter()
            .filter_map(|edge| output.edge_word(edge))
            .collect();

        assert!(
            words.contains(&"quick"),
            "expected the full-depth correction 'quick' for 'quikc' over the \
             persistent vocab trie, got {words:?}"
        );
    }

    /// End-to-end: correct "teh quikc brwon fox" to "the quick brown fox" using
    /// a small n-gram model trained on the target sentence.
    #[test]
    fn end_to_end_corrects_sentence() {
        let dir = TempDir::new().expect("temp dir");

        // Corpus: repeat the target sentence so the in-vocabulary n-grams have
        // strong counts relative to the (unseen) misspellings.
        let sentence = "the quick brown fox jumps over the lazy dog";
        let corpus = format!("{sentence}\n").repeat(20);
        let corpus_path = dir.path().join("corpus.txt");
        {
            let mut file = std::fs::File::create(&corpus_path).expect("create corpus");
            file.write_all(corpus.as_bytes()).expect("write corpus");
        }

        // Train a trigram model over a PathMap-backed dictionary.
        let reader = PlaintextReader::from_file(&corpus_path).expect("open corpus");
        let dictionary = DynamicDawg::<NgramEntry>::new();
        let model = TrainerBuilder::new(dictionary)
            .order(3)
            .train(reader)
            .expect("train n-gram model");

        // Spelling vocabulary: the correct words of the sentence.
        let vocabulary = build_vocabulary(
            dir.path(),
            &[
                "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
            ],
        );

        let corrector =
            HierarchicalCorrector::from_ngram_model(vocabulary, model, CorrectorConfig::default());

        let results = corrector.correct("teh quikc brwon fox");
        assert!(!results.is_empty(), "expected at least one hypothesis");
        assert_eq!(
            results[0].text, "the quick brown fox",
            "top hypothesis should be the corrected sentence"
        );
    }

    /// The n-best path must surface multiple ranked hypotheses in cost order.
    #[test]
    fn end_to_end_k_best_returns_multiple() {
        let dir = TempDir::new().expect("temp dir");
        // Use the richer 9-word corpus: it gives the LM enough discriminative
        // power to separate "the" from the OOV "teh".
        let sentence = "the quick brown fox jumps over the lazy dog";
        let corpus = format!("{sentence}\n").repeat(20);
        let corpus_path = dir.path().join("corpus.txt");
        {
            let mut file = std::fs::File::create(&corpus_path).expect("create corpus");
            file.write_all(corpus.as_bytes()).expect("write corpus");
        }

        let reader = PlaintextReader::from_file(&corpus_path).expect("open corpus");
        let dictionary = DynamicDawg::<NgramEntry>::new();
        let model = TrainerBuilder::new(dictionary)
            .order(3)
            .train(reader)
            .expect("train n-gram model");

        let vocabulary = build_vocabulary(
            dir.path(),
            &[
                "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
            ],
        );

        let config = CorrectorConfig {
            k_best: 3,
            ..CorrectorConfig::default()
        };
        let corrector = HierarchicalCorrector::from_ngram_model(vocabulary, model, config);

        let results = corrector.correct("teh quikc brwon fox");
        assert!(
            results.len() >= 2,
            "expected multiple hypotheses, got {}",
            results.len()
        );
        assert_eq!(results[0].text, "the quick brown fox");
        // Hypotheses must be ordered by non-decreasing cost.
        for pair in results.windows(2) {
            assert!(
                pair[0].score <= pair[1].score + 1e-9,
                "n-best results must be sorted by ascending score"
            );
        }
    }

    /// An empty input yields no hypotheses.
    #[test]
    fn correct_empty_input_is_empty() {
        let dir = TempDir::new().expect("temp dir");
        let corpus_path = dir.path().join("corpus.txt");
        {
            let mut file = std::fs::File::create(&corpus_path).expect("create corpus");
            file.write_all(b"the quick brown fox\n")
                .expect("write corpus");
        }
        let reader = PlaintextReader::from_file(&corpus_path).expect("open corpus");
        let dictionary = DynamicDawg::<NgramEntry>::new();
        let model = TrainerBuilder::new(dictionary)
            .order(2)
            .train(reader)
            .expect("train n-gram model");
        let vocabulary = build_vocabulary(dir.path(), &["the", "quick", "brown", "fox"]);
        let corrector =
            HierarchicalCorrector::from_ngram_model(vocabulary, model, CorrectorConfig::default());

        assert!(corrector.correct("").is_empty());
        assert!(corrector.correct("   ").is_empty());
    }
}
