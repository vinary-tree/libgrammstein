//! # Grammar corrector — the `T_lex ∘ T_gram` noisy-channel cascade
//!
//! This module reimplements text correction as a **two-level noisy-channel WFST
//! cascade over a shared term-id alphabet**, distinct from the string-level
//! lattice pipeline in [`super::corrector`]. Where the [`HierarchicalCorrector`]
//! proposes per-token string substitutions and rescores a linear lattice, the
//! [`GrammarCorrector`] composes two Levenshtein automata on **term-ids** and
//! decodes word-level insertions, deletions, and substitutions jointly with the
//! n-gram counts.
//!
//! [`HierarchicalCorrector`]: super::corrector::HierarchicalCorrector
//!
//! ## Noisy-channel model
//!
//! Given an observed (possibly corrupted) sentence `` $`x`$ `` we seek the
//! correction `` $`w`$ `` that maximizes the posterior. By Bayes' rule the
//! normalizing `` $`P(x)`$ `` is constant in `` $`w`$ ``, so
//!
//! ```math
//! \hat{w} \;=\; \operatorname*{arg\,max}_{w} P(w \mid x)
//!            \;=\; \operatorname*{arg\,max}_{w} \underbrace{P(x \mid w)}_{\text{channel}}\;\underbrace{P(w)}_{\text{source}} .
//! ```
//!
//! The **source model** `` $`P(w)`$ `` is estimated from the n-gram count store by
//! *stupid backoff* (Brants et al. 2007); the **channel model** `` $`P(x \mid w)`$ ``
//! is factored into two edit stages composed on the term-id alphabet
//! `` $`\Sigma = \{\,\text{term-id}\,\}`$ ``:
//!
//! - **`T_lex`** (character level) — for each observed token, a fuzzy lexicon
//!   maps its *characters* to candidate vocabulary **term-ids** with a combined
//!   edit + articulatory-phonetic cost. Spelling and phonetic similarity are one
//!   mechanism here, not a downstream rescore.
//! - **`T_gram`** (word level) — a `u64` Levenshtein automaton over the n-gram
//!   count store, viewed as a dictionary of term-id sequences via [`U64NgramView`],
//!   recovers word-level substitutions / insertions / deletions against the
//!   *known* n-grams and carries their frequency value.
//!
//! Working in the log semiring turns the product into a sum, so the MAP decode is
//! a **minimum-cost** path:
//!
//! ```math
//! \hat{w} \;=\; \operatorname*{arg\,min}_{w}\;
//!   \underbrace{c_{\text{lex}}(x \mid w)}_{\text{spelling / phonetic}}
//!   \;+\; \underbrace{c_{\text{gram}}(x \mid w)}_{\text{word insert / delete}}
//!   \;+\; \underbrace{\lambda \cdot \bigl(-\log S(w)\bigr)}_{\text{language model}} ,
//! ```
//!
//! where `` $`S(w)=\prod_i S(w_i \mid w_{i-k+1..i-1})`$ `` is the stupid-backoff score.
//!
//! ## Stupid-backoff source score
//!
//! From raw n-gram counts `` $`f(\cdot)`$ `` (all orders `` $`1..\text{order}`$ `` are
//! stored), the conditional score is defined recursively, longest history first:
//!
//! ```math
//! S(w \mid h) \;=\;
//!   \begin{cases}
//!     \dfrac{f(h\,w)}{f(h)} & f(h\,w) > 0,\\[1.2ex]
//!     \alpha \cdot S\!\bigl(w \mid h'\bigr) & \text{otherwise (}h' = h \text{ without its oldest word)},
//!   \end{cases}
//!   \qquad
//!   S(w) \;=\; \dfrac{f(w)}{N},
//! ```
//!
//! with backoff weight `` $`\alpha \in (0,1)`$ `` and corpus size `` $`N=\sum_v f(v)`$ ``.
//! Unseen words are floored at `` $`1/(N+1)`$ `` so `` $`-\log S`$ `` stays finite. Unlike
//! Kneser–Ney, stupid backoff needs only the plain counts the varint store holds,
//! and is the standard estimator for web-scale count models — apt for Google Books.
//!
//! ## Decode — a layered beam over the term-id lattice
//!
//! The n-gram store holds windows of length `` $`\le`$ `` `order`, so a whole
//! sentence is decoded by chaining windows with a left-to-right, history-indexed
//! beam search — the classic stack decoder for the equation above:
//!
//! ```text
//!   layer i (i tokens consumed)                     layer i+1
//!   ┌───────────────────────────┐                  ┌───────────────┐
//!   │ hypothesis                │  ── substitute ─▶│  + w  (T_lex)  │
//!   │  emitted:  w₀ … w_{m-1}   │      c_lex + λ·(−log S(w∣h))       │
//!   │  history:  h = w_{m-k+1…}  │  ── delete ─────▶│  (drop x_i)    │
//!   │  cost:     g               │      c_del        │               │
//!   └───────────────────────────┘  ── insert ──────▶│  + v then w    │
//!            │  successor oracle: v ∈ known n-gram continuations of h │
//!            ▼         c_ins + λ·(−log S(v∣h))       └───────────────┘
//!   beam-prune layer i+1 to `beam_width` by ascending cost g
//! ```
//!
//! - **Substitution / keep** — every [`T_lex`](Self::lex_candidates) candidate of
//!   token `` $`x_i`$ `` (the exact match is one, at cost 0).
//! - **Deletion** — drop `` $`x_i`$ `` for a fixed `deletion_penalty` (removes an
//!   extraneous word, e.g. `the the quick → the quick`).
//! - **Insertion** — emit a word `` $`v`$ `` *without* consuming `` $`x_i`$ ``,
//!   drawn from the **known continuations** of the current history `` $`h`$ `` via
//!   the [`U64NgramView`] successor oracle and required to *bridge* to a candidate
//!   of `` $`x_i`$ `` (i.e. `` $`h \, v \, x_i`$ `` is a stored n-gram). This adds
//!   a missing word (e.g. `quick fox → quick brown fox`) only where the store has
//!   evidence for it.
//!
//! ## Worked error taxonomy
//!
//! | Error class      | Example                         | Handled by |
//! |------------------|---------------------------------|------------|
//! | substitution     | `teh → the`                     | `T_lex` (edit) |
//! | transposition    | `quikc → quick`                 | `T_lex` (Damerau) |
//! | phonetic         | `fone → phone`                  | `T_lex` (articulatory discount) |
//! | missing word     | `quick fox → quick brown fox`   | `T_gram` insertion (successor oracle) |
//! | extraneous word  | `the the quick → the quick`     | `T_gram` deletion |
//! | real-word (context) | `form the list → from the list` | `T_lex` candidate + n-gram rescore |
//!
//! ## Relationship to the shipped corrector
//!
//! [`GrammarCorrector`] is additive and opt-in: constructing and calling it is
//! explicit, and the existing [`HierarchicalCorrector`] is unchanged, so default
//! behavior and performance are preserved.
//!
//! [`HierarchicalCorrector`]: super::corrector::HierarchicalCorrector

use libdictenstein::{Dictionary, DictionaryNode, MappedDictionary, MappedDictionaryNode};
use liblevenshtein::transducer::{Algorithm, Transducer};

use crate::ngram::u64_view::{U64NgramNode, U64NgramView, VarintByteUnit};
use crate::ngram::vocabulary::SharedVocabARTrie;

/// Default stupid-backoff weight `` $`\alpha`$ `` (Brants et al. 2007 report 0.4 as
/// near-optimal across scales).
pub const DEFAULT_BACKOFF_ALPHA: f64 = 0.4;

/// Configuration for the [`GrammarCorrector`].
///
/// Every knob has a defensible default (see [`Default`]); the aggressive
/// word-level operations (insertion, deletion) are enabled but penalized so they
/// only win when the source model clearly prefers them.
#[derive(Clone, Debug, PartialEq)]
pub struct GrammarCorrectorConfig {
    /// Maximum character edit distance searched by `T_lex` per token.
    pub max_char_edit_distance: usize,
    /// Maximum word edit distance searched by `T_gram` in
    /// [`grammar_neighbors`](GrammarCorrector::grammar_neighbors).
    pub max_word_edit_distance: usize,
    /// Upper bound on `T_lex` candidates retained per token (closest first).
    pub max_lex_candidates_per_token: usize,
    /// Maximum consecutive word insertions considered in a single gap.
    pub max_insertions_per_gap: usize,
    /// Levenshtein variant for both automata (Damerau/transposition by default).
    pub algorithm: Algorithm,
    /// Tokens shorter than this (in Unicode scalar values) are not spell-corrected.
    pub min_word_length: usize,
    /// Additive cost charged for deleting an observed token.
    pub deletion_penalty: f64,
    /// Additive cost charged for inserting a word not present in the input.
    pub insertion_penalty: f64,
    /// Cost charged for keeping an out-of-vocabulary token verbatim.
    pub oov_penalty: f64,
    /// Language-model mixing weight `` $`\lambda`$ `` applied to `` $`-\log S`$ ``.
    pub lm_weight: f64,
    /// Stupid-backoff weight `` $`\alpha`$ `` per backoff level.
    pub backoff_alpha: f64,
    /// Beam width retained at each decode layer.
    pub beam_width: usize,
    /// Number of ranked hypotheses returned by [`correct`](GrammarCorrector::correct).
    pub k_best: usize,
    /// Whether to fold articulatory-phonetic costs into `T_lex`
    /// (requires the `phonetic-correction` feature; ignored without it).
    pub use_phonetics: bool,
    /// Beginning-of-sentence marker used to seed the decode history. When this
    /// term is in the vocabulary (i.e. the store was trained with boundaries), the
    /// first word is scored `` $`P(w \mid \texttt{<s>} \cdots)`$ ``; otherwise
    /// boundary modeling is skipped.
    pub bos_token: String,
    /// End-of-sentence marker scored after the last word. When in the vocabulary,
    /// it penalizes truncated corrections (a word that rarely ends a sentence),
    /// counteracting the shorter-is-cheaper length bias.
    pub eos_token: String,
}

impl Default for GrammarCorrectorConfig {
    fn default() -> Self {
        Self {
            max_char_edit_distance: 2,
            max_word_edit_distance: 1,
            max_lex_candidates_per_token: 8,
            max_insertions_per_gap: 1,
            // Damerau–Levenshtein: adjacent transpositions cost a single edit,
            // matching the most common typo class (Damerau, 1964).
            algorithm: Algorithm::Transposition,
            min_word_length: 2,
            deletion_penalty: 3.0,
            insertion_penalty: 3.0,
            oov_penalty: 6.0,
            lm_weight: 1.0,
            backoff_alpha: DEFAULT_BACKOFF_ALPHA,
            beam_width: 16,
            k_best: 1,
            use_phonetics: cfg!(feature = "phonetic-correction"),
            bos_token: "<s>".to_string(),
            eos_token: "</s>".to_string(),
        }
    }
}

/// A `T_lex` candidate correction for a single observed token: an in-vocabulary
/// word, its term-id, and the combined edit + articulatory-phonetic cost.
#[derive(Clone, Debug, PartialEq)]
pub struct LexCandidate {
    /// The corrected word.
    pub term: String,
    /// The vocabulary term-id of `term` (`0` marks an out-of-vocabulary fallback).
    pub id: u64,
    /// Combined character edit + phonetic cost (`0.0` for an exact match).
    pub cost: f64,
}

/// A `T_gram` neighbor: a stored term-id sequence within word-edit distance of a
/// query, together with its edit distance and n-gram frequency.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GrammarNeighbor {
    /// The stored (corrected) term-id sequence.
    pub ids: Vec<u64>,
    /// Word-level edit distance from the queried sequence.
    pub distance: usize,
    /// The stored n-gram frequency of `ids`.
    pub frequency: u64,
}

/// A ranked correction hypothesis returned by [`GrammarCorrector::correct`].
#[derive(Clone, Debug, PartialEq)]
pub struct GrammarCorrection {
    /// The corrected sentence (space-joined words).
    pub text: String,
    /// Total minimum-cost decode score (lower is better).
    pub cost: f64,
}

/// One partial decode hypothesis in the beam.
#[derive(Clone, Debug, Default)]
struct Hypothesis {
    /// Words emitted so far (the corrected prefix), for output.
    emitted: Vec<String>,
    /// Term-ids parallel to `emitted`, for the successor-oracle / scoring history.
    ids: Vec<u64>,
    /// Accumulated `` $`-\log`$ ``-domain cost (lower is better).
    cost: f64,
}

/// A per-operation source of `u64` (term-id) n-gram views — the single seam that
/// lets the beam decoder ([`GrammarCore`]) run identically over a **single** count
/// store or a **sharded** one.
///
/// The decoder never holds "the store"; instead, for each lookup it asks for the
/// view whose ROOT can walk the specific n-gram it is about to score. For a single
/// store every request returns the same whole-store view; for a sharded store the
/// request routes `(first_id, key_len)` to the shard that physically holds that
/// n-gram (co-located with how the importer routed it), so `count`, the successor
/// oracle, and `grammar_neighbors` each read from the correct shard.
///
/// The associated-type bounds pin the projections the word-level Levenshtein engine
/// requires — the view's node yields term-ids (`Unit = u64`) and frequencies
/// (`Value = u64`) — so [`GrammarCore`] needs no repeated `where` clauses.
pub trait NgramViewSource: Clone + Send + Sync + 'static {
    /// The view's node: term-id edges (`Unit = u64`) carrying n-gram frequencies
    /// (`Value = u64`).
    type Node: MappedDictionaryNode<Value = u64> + DictionaryNode<Unit = u64>;
    /// The `u64`-unit dictionary view walked by the transducer / oracle.
    type View: Dictionary<Node = Self::Node>
        + MappedDictionary<Value = u64>
        + Clone
        + Send
        + Sync
        + 'static;

    /// The view whose ROOT can walk any stored `key_len`-gram beginning with
    /// `first_id`. `None` when no populated view holds that first-token/length
    /// (a sharded store may have no shard for an absent first-token, which the
    /// decoder treats as "no n-gram evidence").
    fn view_for(&self, first_id: u64, key_len: usize) -> Option<Self::View>;

    /// A view rooted at the WHOLE store, if this source can provide one.
    ///
    /// A single store can (it *is* the whole store); a sharded store cannot,
    /// because unigrams (and, generally, the roots of every length class) are
    /// spread across all shards. This is used only where the decoder genuinely
    /// needs the un-rooted store — the successor oracle at an EMPTY history
    /// (an insertion before the first token, reachable only without a boundary
    /// model) and an empty-sequence `grammar_neighbors` query. Returning `None`
    /// honestly disables just those two whole-store operations for a sharded
    /// source (see [`grammar_neighbors_fanout`] for the sharded completeness path).
    ///
    /// [`grammar_neighbors_fanout`]: #
    fn whole_view(&self) -> Option<Self::View>;
}

/// A [`NgramViewSource`] over a single, whole count store `D` — the non-sharded
/// case. Every `view_for` / `whole_view` returns the one [`U64NgramView`] over the
/// store, ignoring the routing arguments.
///
/// `D` presents its keys as LEB128-varint term-id sequences over a
/// [`VarintByteUnit`] carrier (`u8` for the importer `PersistentARTrie<u64>` /
/// `SharedARTrie<u64>`, `char` for an in-memory `DynamicDawgChar`) and its value as
/// the raw `u64` frequency.
#[derive(Clone, Debug)]
pub struct SingleView<D> {
    store: D,
}

impl<D> SingleView<D> {
    /// Wrap a whole count store as a single-view source.
    #[inline]
    pub fn new(store: D) -> Self {
        Self { store }
    }

    /// Borrow the underlying count store.
    #[inline]
    pub fn store(&self) -> &D {
        &self.store
    }
}

impl<D> NgramViewSource for SingleView<D>
where
    D: MappedDictionary<Value = u64> + Clone + Send + Sync + 'static,
    <D as Dictionary>::Node: MappedDictionaryNode<Value = u64>,
    <<D as Dictionary>::Node as DictionaryNode>::Unit: VarintByteUnit,
{
    type Node = U64NgramNode<<D as Dictionary>::Node>;
    type View = U64NgramView<D>;

    #[inline]
    fn view_for(&self, _first_id: u64, _key_len: usize) -> Option<Self::View> {
        // A single store is not sharded: routing is a no-op, the whole store answers.
        Some(U64NgramView::new(self.store.clone()))
    }

    #[inline]
    fn whole_view(&self) -> Option<Self::View> {
        Some(U64NgramView::new(self.store.clone()))
    }
}

/// The `T_lex ∘ T_gram` grammar-correction beam decoder, generic over the n-gram
/// view source `P`.
///
/// This holds the whole decode algorithm — `T_lex` character correction, the
/// stupid-backoff source score, and the layered noisy-channel beam — parameterized
/// only by *where the n-gram views come from* ([`NgramViewSource`]). The public
/// [`GrammarCorrector`] (single store) and `ShardedGrammarCorrector` (sharded store)
/// are thin delegating newtypes over `GrammarCore<SingleView<D>>` and
/// `GrammarCore<ShardedView>` respectively, so both share this exact code path.
///
/// Term-id-native throughout: `T_lex` yields term-ids, `T_gram` corrects term-id
/// sequences, and the source score is stupid backoff over the stored counts.
pub struct GrammarCore<P: NgramViewSource> {
    vocabulary: SharedVocabARTrie,
    source: P,
    order: usize,
    total_count: u64,
    bos_id: Option<u64>,
    eos_id: Option<u64>,
    config: GrammarCorrectorConfig,
}

impl<P: NgramViewSource> GrammarCore<P> {
    /// Build a decoder from a vocabulary and an n-gram view `source`, given the
    /// model `order` and the corpus token total `total_count`
    /// (`` $`N=\sum_v f(v)`$ ``, the stupid-backoff base normalizer).
    ///
    /// The boundary markers `config.bos_token` / `config.eos_token` are resolved
    /// against the vocabulary; when absent (a store trained without boundaries),
    /// boundary modeling is silently skipped.
    pub fn new(
        vocabulary: SharedVocabARTrie,
        source: P,
        order: usize,
        total_count: u64,
        config: GrammarCorrectorConfig,
    ) -> Self {
        let bos_id = vocabulary.get_index(config.bos_token.as_str());
        let eos_id = vocabulary.get_index(config.eos_token.as_str());
        Self {
            vocabulary,
            source,
            order: order.max(1),
            total_count,
            bos_id,
            eos_id,
            config,
        }
    }

    /// Borrow the effective configuration.
    #[inline]
    pub fn config(&self) -> &GrammarCorrectorConfig {
        &self.config
    }

    /// The corpus token total `` $`N`$ `` used as the stupid-backoff base normalizer.
    #[inline]
    pub fn total_count(&self) -> u64 {
        self.total_count
    }

    /// Borrow the underlying n-gram view source (the routing seam). Used by the
    /// sharded corrector's opt-in all-shards `grammar_neighbors_fanout`.
    #[inline]
    pub(crate) fn source(&self) -> &P {
        &self.source
    }

    // ── T_lex: characters → term-id candidates ──────────────────────────────

    /// `T_lex` — the fuzzy lexicon. Map an observed token's characters to ranked
    /// in-vocabulary `(term, term-id, cost)` candidates.
    ///
    /// With the `phonetic-correction` feature and `use_phonetics`, candidates are
    /// scored by articulatory-weighted edit distance (a sound-alike substitution
    /// costs a fraction of a full edit); otherwise by plain edit distance. A token
    /// with no in-vocabulary neighbor falls back to itself at `oov_penalty`, so the
    /// decode lattice always stays connected.
    pub fn lex_candidates(&self, token: &str) -> Vec<LexCandidate> {
        let mut out = Vec::with_capacity(self.config.max_lex_candidates_per_token + 1);

        // Tokens below the minimum length are kept verbatim rather than thrashing
        // over a dense short-string neighborhood.
        let too_short = token.chars().count() < self.config.min_word_length;

        if !too_short {
            #[cfg(feature = "phonetic-correction")]
            if self.config.use_phonetics {
                if let Some(cands) = self.phonetic_lex_candidates(token) {
                    out.extend(cands);
                }
            }

            if out.is_empty() {
                let transducer = Transducer::new(self.vocabulary.clone(), self.config.algorithm);
                for candidate in
                    transducer.query_with_distance(token, self.config.max_char_edit_distance)
                {
                    // Skip vocabulary metadata keys (\x00-prefixed control strings).
                    if candidate.term.chars().any(|c| c.is_control()) {
                        continue;
                    }
                    if let Some(id) = self.vocabulary.get_index(&candidate.term) {
                        out.push(LexCandidate {
                            term: candidate.term,
                            id,
                            cost: candidate.distance as f64,
                        });
                    }
                }
            }
        }

        out.sort_by(|a, b| a.cost.total_cmp(&b.cost).then_with(|| a.term.cmp(&b.term)));
        out.dedup_by(|a, b| a.id == b.id);
        out.truncate(self.config.max_lex_candidates_per_token);

        if out.is_empty() {
            // Out-of-vocabulary (or intentionally uncorrected short token): keep it.
            out.push(LexCandidate {
                term: token.to_string(),
                id: 0,
                cost: if too_short {
                    0.0
                } else {
                    self.config.oov_penalty
                },
            });
        }
        out
    }

    /// Articulatory-phonetic `T_lex` path (feature-gated). Returns `None` when the
    /// token cannot be compiled into a phonetic pattern, so the caller falls back
    /// to the plain edit-distance path.
    #[cfg(feature = "phonetic-correction")]
    fn phonetic_lex_candidates(&self, token: &str) -> Option<Vec<LexCandidate>> {
        use liblevenshtein::phonetic::nfa::compile;
        use liblevenshtein::phonetic::regex::parse;
        use liblevenshtein::transducer::{ArticulatoryCosts, PhoneticTransducerChar};

        let pattern = escape_phonetic_pattern(token);
        let ast = parse(&pattern).ok()?;
        let nfa = compile(&ast).ok()?;
        let transducer = PhoneticTransducerChar::with_articulatory_costs(
            self.vocabulary.clone(),
            nfa,
            self.config.max_char_edit_distance as u8,
            ArticulatoryCosts::default(),
        );

        let mut out = Vec::new();
        for candidate in transducer.query_values_sorted(token) {
            if candidate.term.chars().any(|c| c.is_control()) {
                continue;
            }
            out.push(LexCandidate {
                term: candidate.term,
                id: candidate.value,
                cost: candidate.total_cost,
            });
        }
        Some(out)
    }

    // ── T_gram: term-id sequences → known-n-gram neighbors ───────────────────

    /// `T_gram` — the fuzzy grammar. Return the stored n-gram term-id sequences
    /// within `max_word_edit_distance` word edits of `term_ids`, each with its edit
    /// distance and frequency, via one `query_units_values` pass over the
    /// [`U64NgramView`].
    ///
    /// This is the word-level analogue of `T_lex`: the *same* Levenshtein engine
    /// that corrects a word's characters here corrects a sentence's words, once
    /// each word is a term-id. It recovers word substitution / insertion / deletion
    /// against the known n-grams (see the tests for the worked cases).
    pub fn grammar_neighbors(
        &self,
        term_ids: &[u64],
        max_word_edit_distance: usize,
    ) -> Vec<GrammarNeighbor> {
        // Anchored to the view holding `(first_id, len)`-rooted n-grams. For a single
        // store this is the whole store (identical to the pre-sharding behavior); for
        // a sharded store it returns exactly the neighbors co-located with the query's
        // first-token/length — see `ShardedGrammarCorrector::grammar_neighbors_fanout`
        // for cross-shard (first-token-edit) completeness. An empty query has no
        // first-token to route on, so it uses the whole store (single sources only).
        let view = match term_ids.first() {
            Some(&first) => self.source.view_for(first, term_ids.len()),
            None => self.source.whole_view(),
        };
        let Some(view) = view else {
            return Vec::new();
        };
        let transducer = Transducer::new(view, self.config.algorithm);
        transducer
            .query_units_values(term_ids, max_word_edit_distance)
            .map(|(ids, distance, frequency)| GrammarNeighbor {
                ids,
                distance,
                frequency,
            })
            .collect()
    }

    /// Stored count `` $`f(\text{ids})`$ `` of a term-id sequence (`0` if absent).
    ///
    /// Self-routing: the count of a `len`-gram beginning with `ids[0]` is read from
    /// the view for `(ids[0], len)`. An empty sequence — or a first-token that routes
    /// to no populated view (an absent shard) — counts as `0`.
    fn count(&self, ids: &[u64]) -> u64 {
        let Some(&first) = ids.first() else {
            return 0;
        };
        match self.source.view_for(first, ids.len()) {
            Some(view) => node_after(&view, ids)
                .and_then(|node| node.value())
                .unwrap_or(0),
            None => 0,
        }
    }

    // ── Source score: stupid backoff over the raw counts ─────────────────────

    /// Stupid-backoff score `` $`S(w \mid h)`$ `` for the term-id `word` given the
    /// term-id `history` (oldest first), per the module-doc recursion. The returned
    /// value lies in `` $`(0, 1]`$ `` and is floored so `` $`-\log S`$ `` is finite.
    ///
    /// Each level's numerator `` $`f(h\,w)`$ `` and denominator `` $`f(h)`$ `` route
    /// independently by their own `(first_id, len)` via [`count`](Self::count) — they
    /// may live in different shards under an order-sensitive granularity.
    fn stupid_backoff(&self, history: &[u64], word: u64) -> f64 {
        // Cap the history at `order - 1`, longest first; back off by dropping the
        // oldest token until an observed continuation is found.
        let max_hist = self.order.saturating_sub(1).min(history.len());
        let base = &history[history.len() - max_hist..];

        let mut sequence: Vec<u64> = Vec::with_capacity(max_hist + 1);
        for dropped in 0..=max_hist {
            let context = &base[dropped..];
            sequence.clear();
            sequence.extend_from_slice(context);
            sequence.push(word);

            let count_hw = self.count(&sequence);
            if count_hw > 0 {
                let count_h = if context.is_empty() {
                    self.total_count
                } else {
                    self.count(context)
                };
                if count_h > 0 {
                    let alpha = self.config.backoff_alpha.powi(dropped as i32);
                    return alpha * (count_hw as f64 / count_h as f64);
                }
            }
        }

        // Unseen even as a unigram: add-one floor over the corpus size.
        1.0 / (self.total_count as f64 + 1.0)
    }

    /// `` $`-\log S(w \mid h)`$ `` — the source-model cost contribution.
    #[inline]
    fn neg_log_score(&self, history: &[u64], word: u64) -> f64 {
        -self.stupid_backoff(history, word).ln()
    }

    /// The effective `n`-token history for scoring / the successor oracle: the last
    /// `n` emitted term-ids, left-padded with the beginning-of-sentence marker when
    /// the store models boundaries. Early in the sentence this yields e.g.
    /// `` $`[\texttt{<s>}, \texttt{<s>}]`$ `` so the first word is scored in context.
    fn effective_history(&self, ids: &[u64], n: usize) -> Vec<u64> {
        match self.bos_id {
            Some(bos) if n > 0 => {
                let mut history = Vec::with_capacity(n);
                let pad = n.saturating_sub(ids.len());
                for _ in 0..pad {
                    history.push(bos);
                }
                history.extend_from_slice(&ids[ids.len().saturating_sub(n)..]);
                history
            }
            _ => history_ids(ids, n),
        }
    }

    // ── Decode: the layered beam noisy-channel search ────────────────────────

    /// Correct `text`, returning up to `k_best` hypotheses best (lowest-cost) first.
    ///
    /// Whitespace-tokenizes the input and runs the history-indexed beam described
    /// in the [module docs](self): each layer expands substitutions (`T_lex`),
    /// deletions, and bridged insertions (the `T_gram` successor oracle), scoring
    /// every emission with `` $`\lambda\cdot(-\log S(w \mid h))`$ `` and pruning to
    /// `beam_width`. An empty input yields an empty result.
    pub fn correct(&self, text: &str) -> Vec<GrammarCorrection> {
        let tokens: Vec<&str> = text.split_whitespace().collect();
        if tokens.is_empty() {
            return Vec::new();
        }

        let history_len = self.order.saturating_sub(1);

        // Per-token T_lex candidate sets, computed once.
        let lex: Vec<Vec<LexCandidate>> = tokens.iter().map(|t| self.lex_candidates(t)).collect();

        let beam_width = self.config.beam_width.max(1);
        let mut beam: Vec<Hypothesis> = vec![Hypothesis::default()];

        for candidates in &lex {
            let mut next: Vec<Hypothesis> = Vec::new();

            for hypothesis in &beam {
                // Optional bridged insertions BEFORE consuming the token, then the
                // token is consumed (kept/substituted or deleted) from each. Every
                // scoring lookup self-routes to the view that holds it.
                let staged = self.insertion_expansions(hypothesis, candidates, history_len);

                for base in &staged {
                    let history = self.effective_history(&base.ids, history_len);

                    // Deletion: drop the observed token.
                    let mut deleted = base.clone();
                    deleted.cost += self.config.deletion_penalty;
                    next.push(deleted);

                    // Substitution / keep: emit each candidate correction.
                    for candidate in candidates {
                        let cost_lm =
                            self.config.lm_weight * self.neg_log_score(&history, candidate.id);
                        let mut hypothesis = base.clone();
                        hypothesis.emitted.push(candidate.term.clone());
                        hypothesis.ids.push(candidate.id);
                        hypothesis.cost += candidate.cost + cost_lm;
                        next.push(hypothesis);
                    }
                }
            }

            next.sort_by(|a, b| a.cost.total_cmp(&b.cost));
            next.truncate(beam_width);
            beam = next;
        }

        // End-of-sentence: score the transition to `</s>` from each final history,
        // penalizing truncated hypotheses (words that rarely end a sentence). Skipped
        // when the store has no boundary marker.
        if let Some(eos) = self.eos_id {
            for hypothesis in &mut beam {
                let history = self.effective_history(&hypothesis.ids, history_len);
                hypothesis.cost += self.config.lm_weight * self.neg_log_score(&history, eos);
            }
        }

        beam.sort_by(|a, b| a.cost.total_cmp(&b.cost));
        beam.into_iter()
            .take(self.config.k_best.max(1))
            .map(|hypothesis| GrammarCorrection {
                text: hypothesis.emitted.join(" "),
                cost: hypothesis.cost,
            })
            .collect()
    }

    /// Expand a hypothesis by 0..=`max_insertions_per_gap` bridged word insertions.
    ///
    /// An inserted word `v` must be a known continuation of the current history
    /// `` $`h`$ `` (an out-edge of the history node in the [`U64NgramView`]) *and*
    /// bridge to some candidate `c` of the token about to be consumed (`` $`h\,v\,c`$ ``
    /// is a stored n-gram). This keeps insertions data-driven and finite: only words
    /// the store has evidence for, only where they reconnect to the observation.
    fn insertion_expansions(
        &self,
        hypothesis: &Hypothesis,
        next_candidates: &[LexCandidate],
        history_len: usize,
    ) -> Vec<Hypothesis> {
        let mut result = vec![hypothesis.clone()];
        if self.config.max_insertions_per_gap == 0 || history_len == 0 {
            return result;
        }

        let beam_width = self.config.beam_width.max(1);
        let mut frontier = vec![hypothesis.clone()];

        for _ in 0..self.config.max_insertions_per_gap {
            let mut grown: Vec<Hypothesis> = Vec::new();

            for current in &frontier {
                let history = self.effective_history(&current.ids, history_len);

                // Successor view: the produced n-gram `history · v` has length
                // `history.len() + 1`, so fetch the view for THAT length (the
                // Adaptive-granularity caveat), then walk to the `history` node and
                // read its edges. An empty history routes to no shard, so a
                // boundary-less insertion before the first token uses the whole store
                // (single sources only; `None` disables it for a sharded source).
                let successor_view = match history.first() {
                    Some(&first) => self.source.view_for(first, history.len() + 1),
                    None => self.source.whole_view(),
                };
                let Some(successor_view) = successor_view else {
                    continue;
                };
                let node = match node_after(&successor_view, &history) {
                    Some(node) => node,
                    None => continue,
                };

                // `node.edges()` are exactly the known continuations `v` of the
                // history (`history · v` is a stored path — the successor oracle).
                for (candidate_id, _child) in node.edges() {
                    // Bridge check: after inserting `v`, the (order−1)-suffix of
                    // `history · v` followed by some candidate `c` must be a stored
                    // n-gram — i.e. `v` connects forward to what we are about to
                    // consume. (Walking from the deeper `history · v` node would ask
                    // for an over-length (order+1)-gram, which is never stored.)
                    let mut extended = history.clone();
                    extended.push(candidate_id);
                    let bridge_context = history_ids(&extended, history_len);
                    // The bridge n-gram `bridge_context · c` has length
                    // `bridge_context.len() + 1`; route by that produced length.
                    let bridge_view = match bridge_context.first() {
                        Some(&first) => self.source.view_for(first, bridge_context.len() + 1),
                        None => self.source.whole_view(),
                    };
                    let Some(bridge_view) = bridge_view else {
                        continue;
                    };
                    let bridge_node = match node_after(&bridge_view, &bridge_context) {
                        Some(bridge_node) => bridge_node,
                        None => continue,
                    };
                    let bridges = next_candidates
                        .iter()
                        .any(|c| bridge_node.transition(c.id).is_some());
                    if !bridges {
                        continue;
                    }
                    let word = match self.vocabulary.get_term(candidate_id) {
                        Some(word) => word,
                        None => continue,
                    };
                    let cost_lm =
                        self.config.lm_weight * self.neg_log_score(&history, candidate_id);
                    let mut inserted = current.clone();
                    inserted.emitted.push(word);
                    inserted.ids.push(candidate_id);
                    inserted.cost += self.config.insertion_penalty + cost_lm;
                    grown.push(inserted);
                }
            }

            if grown.is_empty() {
                break;
            }
            grown.sort_by(|a, b| a.cost.total_cmp(&b.cost));
            grown.truncate(beam_width);
            result.extend(grown.iter().cloned());
            frontier = grown;
        }

        result
    }
}

/// Walk the term-id sequence `ids` from a view's root, returning the node reached or
/// `None` if that sequence is not a path. A free function (not a method) so it stays
/// generic over the view type produced by any [`NgramViewSource`].
fn node_after<Vw>(view: &Vw, ids: &[u64]) -> Option<<Vw as Dictionary>::Node>
where
    Vw: Dictionary,
    <Vw as Dictionary>::Node: DictionaryNode<Unit = u64>,
{
    let mut node = view.root();
    for &id in ids {
        node = node.transition(id)?;
    }
    Some(node)
}

/// The `T_lex ∘ T_gram` grammar corrector over a shared term-id alphabet — a single,
/// whole count store `D`.
///
/// A thin delegating newtype over [`GrammarCore`]`<`[`SingleView`]`<D>>`: the public
/// API and behavior are identical to the pre-sharding corrector, so existing callers
/// and tests are unaffected. `D` presents its keys as LEB128-varint term-id sequences
/// over a [`VarintByteUnit`] carrier (`u8` for the importer `PersistentARTrie<u64>` /
/// `SharedARTrie<u64>`, `char` for an in-memory `DynamicDawgChar`) and its value as
/// the raw `u64` frequency.
pub struct GrammarCorrector<D>
where
    D: MappedDictionary<Value = u64> + Clone + Send + Sync + 'static,
    <D as Dictionary>::Node: MappedDictionaryNode<Value = u64>,
    <<D as Dictionary>::Node as DictionaryNode>::Unit: VarintByteUnit,
{
    core: GrammarCore<SingleView<D>>,
}

impl<D> GrammarCorrector<D>
where
    D: MappedDictionary<Value = u64> + Clone + Send + Sync + 'static,
    <D as Dictionary>::Node: MappedDictionaryNode<Value = u64>,
    <<D as Dictionary>::Node as DictionaryNode>::Unit: VarintByteUnit,
{
    /// Build a corrector from a vocabulary and a varint term-id count `store`,
    /// given the model `order` and the corpus token total `total_count`
    /// (`` $`N=\sum_v f(v)`$ ``, the stupid-backoff base normalizer).
    ///
    /// The boundary markers `config.bos_token` / `config.eos_token` are resolved
    /// against the vocabulary; when absent (a store trained without boundaries),
    /// boundary modeling is silently skipped.
    pub fn new(
        vocabulary: SharedVocabARTrie,
        store: D,
        order: usize,
        total_count: u64,
        config: GrammarCorrectorConfig,
    ) -> Self {
        Self {
            core: GrammarCore::new(
                vocabulary,
                SingleView::new(store),
                order,
                total_count,
                config,
            ),
        }
    }

    /// Build a corrector, deriving `total_count` by summing the stored unigram
    /// counts (the store's depth-1 nodes).
    ///
    /// Convenient for tests and small stores; for a web-scale store prefer [`new`]
    /// with a precomputed total, since this scans every unigram edge once.
    ///
    /// [`new`]: Self::new
    pub fn from_store(
        vocabulary: SharedVocabARTrie,
        store: D,
        order: usize,
        config: GrammarCorrectorConfig,
    ) -> Self {
        let total_count = sum_unigram_counts(&store);
        Self::new(vocabulary, store, order, total_count, config)
    }

    /// Borrow the effective configuration.
    #[inline]
    pub fn config(&self) -> &GrammarCorrectorConfig {
        self.core.config()
    }

    /// The corpus token total `` $`N`$ `` used as the stupid-backoff base normalizer.
    #[inline]
    pub fn total_count(&self) -> u64 {
        self.core.total_count()
    }

    /// `T_lex` — map an observed token's characters to ranked in-vocabulary
    /// `(term, term-id, cost)` candidates. See [`GrammarCore::lex_candidates`].
    pub fn lex_candidates(&self, token: &str) -> Vec<LexCandidate> {
        self.core.lex_candidates(token)
    }

    /// `T_gram` — the stored n-gram term-id sequences within `max_word_edit_distance`
    /// word edits of `term_ids`. See [`GrammarCore::grammar_neighbors`].
    pub fn grammar_neighbors(
        &self,
        term_ids: &[u64],
        max_word_edit_distance: usize,
    ) -> Vec<GrammarNeighbor> {
        self.core
            .grammar_neighbors(term_ids, max_word_edit_distance)
    }

    /// Correct `text`, returning up to `k_best` hypotheses best (lowest-cost) first.
    /// See [`GrammarCore::correct`].
    pub fn correct(&self, text: &str) -> Vec<GrammarCorrection> {
        self.core.correct(text)
    }
}

/// Sum the counts of every stored unigram (depth-1 node) of a whole count store, i.e.
/// the corpus size `` $`N=\sum_v f(v)`$ ``. Used by [`GrammarCorrector::from_store`];
/// a sharded corrector injects `N` at construction instead (unigrams span all shards,
/// so no single view can sum them).
fn sum_unigram_counts<D>(store: &D) -> u64
where
    D: MappedDictionary<Value = u64> + Clone + Send + Sync + 'static,
    <D as Dictionary>::Node: MappedDictionaryNode<Value = u64>,
    <<D as Dictionary>::Node as DictionaryNode>::Unit: VarintByteUnit,
{
    let view = U64NgramView::new(store.clone());
    let mut total = 0u64;
    for (_id, node) in view.root().edges() {
        total = total.saturating_add(node.value().unwrap_or(0));
    }
    total
}

/// Return the last `n` emitted term-ids (the successor-oracle / scoring history).
#[inline]
fn history_ids(ids: &[u64], n: usize) -> Vec<u64> {
    let start = ids.len().saturating_sub(n);
    ids[start..].to_vec()
}

/// Escape regular-expression metacharacters so a literal token compiles as a
/// phonetic pattern rather than an accidental regex.
#[cfg(feature = "phonetic-correction")]
fn escape_phonetic_pattern(token: &str) -> String {
    const META: &[char] = &[
        '\\', '.', '+', '*', '?', '(', ')', '[', ']', '{', '}', '|', '^', '$',
    ];
    let mut out = String::with_capacity(token.len() + 2);
    for c in token.chars() {
        if META.contains(&c) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ngram::vocabulary::{create_vocabulary, encode_varint};
    use libdictenstein::dynamic_dawg::DynamicDawg;
    use std::collections::HashMap;
    use tempfile::TempDir;

    /// A tiny end-to-end fixture: a persistent vocabulary plus a byte-carrier count
    /// store keyed by concatenated LEB128 varints (the importer key format), holding
    /// every 1..=`order` n-gram count of the supplied sentences.
    struct Fixture {
        _dir: TempDir,
        vocab: SharedVocabARTrie,
        store: DynamicDawg<u64>,
        order: usize,
    }

    impl Fixture {
        fn corrector(&self, config: GrammarCorrectorConfig) -> GrammarCorrector<DynamicDawg<u64>> {
            GrammarCorrector::from_store(self.vocab.clone(), self.store.clone(), self.order, config)
        }

        fn id(&self, word: &str) -> u64 {
            self.vocab
                .get_index(word)
                .expect("word is in the vocabulary")
        }

        fn ids(&self, words: &[&str]) -> Vec<u64> {
            words.iter().map(|w| self.id(w)).collect()
        }
    }

    /// Build a fixture by counting all 1..=`order` n-grams of `sentences`.
    fn build(sentences: &[&str], order: usize) -> Fixture {
        let dir = TempDir::new().expect("tempdir");
        let vocab = create_vocabulary(&dir.path().join("vocab")).expect("vocabulary");
        let mut counts: HashMap<Vec<u64>, u64> = HashMap::new();
        let insert = |w: &str| vocab.as_ref().insert(w).expect("vocabulary insert");
        for sentence in sentences {
            // Pad with (order-1) beginning-of-sentence markers and one end marker,
            // so the store models sentence boundaries (the standard n-gram convention).
            let mut words: Vec<&str> = vec!["<s>"; order.saturating_sub(1)];
            words.extend(sentence.split_whitespace());
            words.push("</s>");
            let ids: Vec<u64> = words.iter().map(|w| insert(w)).collect();
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
        Fixture {
            _dir: dir,
            vocab,
            store,
            order,
        }
    }

    /// A trigram corpus with discriminating counts for the worked error taxonomy.
    fn corpus() -> Fixture {
        build(
            &[
                "the quick brown fox",
                "the quick brown fox",
                "the quick brown fox",
                "the quick brown dog",
                "the lazy brown fox",
                "the quick brown fox jumps over",
            ],
            3,
        )
    }

    // ── T_lex ────────────────────────────────────────────────────────────────

    #[test]
    fn lex_candidates_recover_spelling() {
        let fx = corpus();
        let corrector = fx.corrector(GrammarCorrectorConfig::default());
        let candidates = corrector.lex_candidates("teh");
        assert!(
            candidates.iter().any(|c| c.term == "the"),
            "teh → the; got {candidates:?}"
        );
        let brown = corrector.lex_candidates("brown");
        assert!(
            brown.iter().any(|c| c.term == "brown" && c.cost == 0.0),
            "an in-vocabulary word is an exact (cost 0) candidate; got {brown:?}"
        );
    }

    #[test]
    fn lex_candidate_ids_match_vocabulary() {
        let fx = corpus();
        let corrector = fx.corrector(GrammarCorrectorConfig::default());
        for candidate in corrector.lex_candidates("quick") {
            assert_eq!(
                Some(candidate.id),
                fx.vocab.get_index(&candidate.term),
                "T_lex value must equal the vocabulary index of the term"
            );
        }
    }

    #[test]
    fn lex_keeps_out_of_vocabulary_token() {
        let fx = corpus();
        let corrector = fx.corrector(GrammarCorrectorConfig::default());
        let candidates = corrector.lex_candidates("zzzqqq");
        assert!(
            candidates.iter().any(|c| c.term == "zzzqqq" && c.id == 0),
            "an OOV token with no neighbor is kept verbatim; got {candidates:?}"
        );
    }

    // ── T_gram ────────────────────────────────────────────────────────────────

    #[test]
    fn grammar_neighbors_exact_is_lossless() {
        let fx = corpus();
        let corrector = fx.corrector(GrammarCorrectorConfig::default());
        let ids = fx.ids(&["the", "quick", "brown"]);
        let neighbors = corrector.grammar_neighbors(&ids, 0);
        assert_eq!(
            neighbors.len(),
            1,
            "exact query is lossless; got {neighbors:?}"
        );
        assert_eq!(neighbors[0].ids, ids);
        assert_eq!(neighbors[0].distance, 0);
        assert!(neighbors[0].frequency >= 1);
    }

    #[test]
    fn grammar_neighbors_recover_word_substitution() {
        let fx = corpus();
        let corrector = fx.corrector(GrammarCorrectorConfig::default());
        // [the, quick, fox] is NOT a stored trigram; its distance-1 neighbor
        // [the, quick, brown] (fox → brown) is.
        let observed = fx.ids(&["the", "quick", "fox"]);
        let target = fx.ids(&["the", "quick", "brown"]);
        let neighbors = corrector.grammar_neighbors(&observed, 1);
        assert!(
            neighbors.iter().any(|n| n.ids == target && n.distance == 1),
            "word substitution fox→brown recovered; got {neighbors:?}"
        );
    }

    // ── The cascade: worked error taxonomy ─────────────────────────────────────

    #[test]
    fn correct_fixes_substitution() {
        let fx = corpus();
        let corrector = fx.corrector(GrammarCorrectorConfig::default());
        let out = corrector.correct("teh quick brown fox");
        assert_eq!(out[0].text, "the quick brown fox", "got {out:?}");
    }

    #[test]
    fn correct_fixes_transposition() {
        let fx = corpus();
        let corrector = fx.corrector(GrammarCorrectorConfig::default());
        let out = corrector.correct("the quikc brown fox");
        assert_eq!(out[0].text, "the quick brown fox", "got {out:?}");
    }

    #[test]
    fn correct_deletes_extraneous_word() {
        let fx = corpus();
        let corrector = fx.corrector(GrammarCorrectorConfig::default());
        let out = corrector.correct("the the quick brown fox");
        assert_eq!(out[0].text, "the quick brown fox", "got {out:?}");
    }

    #[test]
    fn correct_inserts_missing_word() {
        let fx = corpus();
        let corrector = fx.corrector(GrammarCorrectorConfig::default());
        let out = corrector.correct("the quick fox");
        assert_eq!(out[0].text, "the quick brown fox", "got {out:?}");
    }

    #[test]
    fn correct_empty_input_is_empty() {
        let fx = corpus();
        let corrector = fx.corrector(GrammarCorrectorConfig::default());
        assert!(corrector.correct("   ").is_empty());
    }

    #[test]
    fn correct_leaves_clean_input_unchanged() {
        let fx = corpus();
        let corrector = fx.corrector(GrammarCorrectorConfig::default());
        let out = corrector.correct("the quick brown fox");
        assert_eq!(out[0].text, "the quick brown fox", "got {out:?}");
    }

    #[test]
    fn stupid_backoff_prefers_seen_continuation() {
        let fx = corpus();
        let corrector = fx.corrector(GrammarCorrectorConfig::default());
        let history = fx.ids(&["the", "quick"]);
        // "brown" follows "the quick" in the corpus; "over" does not. White-box:
        // `neg_log_score` now self-routes per lookup (no view argument), so reach
        // through the delegating newtype's `core` to the `GrammarCore` decoder.
        let seen = corrector.core.neg_log_score(&history, fx.id("brown"));
        let unseen = corrector.core.neg_log_score(&history, fx.id("over"));
        assert!(
            seen < unseen,
            "a seen continuation must cost less: brown={seen}, over={unseen}"
        );
    }

    /// The articulatory `T_lex` path recovers a sound-alike misspelling
    /// (`fone → phone`, the `f`↔`ph` substitution) that plain edit distance would
    /// rank no better than any other distance-2 neighbor.
    #[cfg(feature = "phonetic-correction")]
    #[test]
    fn lex_phonetic_soundalike_is_found() {
        let fx = build(&["please answer the phone", "please answer the phone"], 3);
        let corrector = fx.corrector(GrammarCorrectorConfig::default());
        assert!(
            corrector.config().use_phonetics,
            "phonetics on by default here"
        );
        let candidates = corrector.lex_candidates("fone");
        assert!(
            candidates.iter().any(|c| c.term == "phone"),
            "fone → phone via the articulatory path; got {candidates:?}"
        );
    }
}
