//! # Sharded grammar corrector — the `T_lex ∘ T_gram` cascade over a sharded corpus
//!
//! [`ShardedGrammarCorrector`] runs the exact same beam decoder as the single-store
//! [`GrammarCorrector`](super::grammar_corrector::GrammarCorrector) — they are both
//! thin newtypes over [`GrammarCore`] — but scores against the Google-Books n-gram
//! corpus, which is physically split across many byte-keyed persistent-trie shards.
//!
//! ## The one seam: [`ShardedView`]
//!
//! The decoder never holds "the store"; for each lookup it asks a
//! [`NgramViewSource`] for the `u64`-unit view whose ROOT can walk the specific
//! n-gram it is about to score. [`ShardedView`] answers `view_for(first_id, key_len)`
//! by:
//!
//! 1. reverse-mapping `first_id` to its vocabulary string via
//!    [`get_term`](libdictenstein::persistent_artrie::vocab::PersistentVocabARTrie::get_term);
//! 2. routing `(first_token, key_len)` with the very same pure function the importer
//!    used to *store* the n-gram — [`compute_shard_key_from_token`] — so the query
//!    shard equals the storage shard (co-location); and
//! 3. opening that shard **read-only** ([`ShardCoordinator::get_shard_readonly`]) and
//!    wrapping its lock-free byte trie in a [`U64NgramView`].
//!
//! Because routing is a pure function of `(first_token, produced_length)`, every
//! exact walk rooted at a fixed first term-id is confined to the shard `view_for`
//! returns — the property the decoder relies on.
//!
//! ## Read-only query mode (safety)
//!
//! [`get_shard_readonly`](ShardCoordinator::get_shard_readonly) never creates a shard,
//! never evicts (so it never triggers the eviction-path `checkpoint()` write), and
//! never arms overlay eviction. The decoder therefore performs **no writes** while
//! readers are active, so the lock-free trie reads never race a concurrent writer.
//! The coordinator should be opened with `max_open_shards = 0` (all shards resident)
//! for query mode: correction scores touch up to `order` distinct first-token shards,
//! and the read path never evicts, so a bounded residency cap would only accumulate
//! shards without reclaiming them. An absent first-token routes to a shard that was
//! never populated; `view_for` returns `None`, which the decoder reads as
//! "no n-gram evidence" (count `0` / no neighbors) — exactly correct.
//!
//! ## Corpus total `N`
//!
//! The stupid-backoff normalizer `` $`N=\sum_v f(v)`$ `` is **injected at
//! construction** ([`ShardedGrammarCorrector::new`]). No persisted corpus-token total
//! exists, and unigrams span every shard, so — unlike the single-store
//! [`from_store`](super::grammar_corrector::GrammarCorrector::from_store) — a sharded
//! corrector cannot derive `N` from a single view; the caller (the importer, which
//! accumulates it) supplies it.
//!
//! ## Neighbor completeness — anchored vs. fanout
//!
//! Under sharding a first-token EDIT lands in a *different* shard (and hash routing
//! destroys prefix locality), so the anchored [`grammar_neighbors`] — which walks the
//! single shard of the query's own first-token — cannot see first-token-edit
//! neighbors. This matches exactly what the decoder needs (its successor oracle only
//! ever consults co-located continuations), so it is the honest default. For full
//! soundness+completeness across first-token edits, use the opt-in
//! [`grammar_neighbors_fanout`], which walks **every** shard and merges — batch /
//! offline only.
//!
//! [`grammar_neighbors`]: ShardedGrammarCorrector::grammar_neighbors
//! [`grammar_neighbors_fanout`]: ShardedGrammarCorrector::grammar_neighbors_fanout
//! [`NgramViewSource`]: super::grammar_corrector::NgramViewSource
//! [`GrammarCore`]: super::grammar_corrector::GrammarCore
//! [`GrammarCorrector`]: super::grammar_corrector::GrammarCorrector

use std::collections::HashMap;
use std::sync::Arc;

use libdictenstein::persistent_artrie::SharedARTrie;
use libdictenstein::Dictionary;
use liblevenshtein::transducer::Transducer;

use crate::ngram::u64_view::{U64NgramNode, U64NgramView};
use crate::ngram::vocabulary::SharedVocabARTrie;
use crate::sources::google_books::sharding::{
    compute_shard_key_from_token, ShardCoordinator, ShardKey,
};

use super::grammar_corrector::{
    GrammarCore, GrammarCorrection, GrammarCorrectorConfig, GrammarNeighbor, LexCandidate,
    NgramViewSource,
};

/// A [`NgramViewSource`] over a sharded byte-trie corpus: it routes each requested
/// `(first_id, key_len)` to the shard that physically holds it and hands back a
/// read-only `u64`-unit view of that shard's trie.
///
/// See the [module docs](self) for the routing / read-only-safety / completeness
/// design.
#[derive(Clone)]
pub struct ShardedView {
    coordinator: Arc<ShardCoordinator>,
    vocabulary: SharedVocabARTrie,
}

impl ShardedView {
    /// Build a sharded view source over `coordinator` (the shard store) and
    /// `vocabulary` (the term-id ↔ string bijection used for routing).
    #[inline]
    pub fn new(coordinator: Arc<ShardCoordinator>, vocabulary: SharedVocabARTrie) -> Self {
        Self {
            coordinator,
            vocabulary,
        }
    }

    /// Borrow the shard coordinator.
    #[inline]
    pub fn coordinator(&self) -> &Arc<ShardCoordinator> {
        &self.coordinator
    }

    /// Open the read-only `u64` view for an already-known shard key, or `None` if the
    /// shard file does not exist. Used by both `view_for` (after routing) and the
    /// all-shards fanout.
    fn open_view(&self, key: &ShardKey) -> Option<U64NgramView<SharedARTrie<u64>>> {
        let shard = self.coordinator.get_shard_readonly(key).ok().flatten()?;
        // Clone the lock-free `Arc` trie out of the brief read guard: the trie's reads
        // are `&self` and lock-free, so the returned view outlives the guard.
        let trie = shard.read().trie_arc();
        Some(U64NgramView::new(trie))
    }

    /// The all-shards fuzzy-neighbor query behind
    /// [`ShardedGrammarCorrector::grammar_neighbors_fanout`]: walk every existing
    /// shard, run the word-level Levenshtein query against each, and merge by
    /// term-id sequence (keeping the min distance and max frequency). This is what
    /// restores first-token-edit completeness that the anchored query cannot see.
    fn neighbors_fanout(
        &self,
        term_ids: &[u64],
        max_word_edit_distance: usize,
        algorithm: liblevenshtein::transducer::Algorithm,
    ) -> Vec<GrammarNeighbor> {
        let shards = match self.coordinator.discover_shard_files() {
            Ok(shards) => shards,
            Err(error) => {
                log::warn!("grammar_neighbors_fanout: shard discovery failed: {error}");
                return Vec::new();
            }
        };

        let mut merged: HashMap<Vec<u64>, GrammarNeighbor> = HashMap::new();
        for (key, _path) in shards {
            let Some(view) = self.open_view(&key) else {
                continue;
            };
            let transducer = Transducer::new(view, algorithm);
            for (ids, distance, frequency) in
                transducer.query_units_values(term_ids, max_word_edit_distance)
            {
                merged
                    .entry(ids.clone())
                    .and_modify(|neighbor| {
                        // A term-id sequence is co-located in exactly one shard, so a
                        // duplicate across shards would be an anomaly; keep the
                        // strongest evidence defensively (min distance, max frequency).
                        if distance < neighbor.distance {
                            neighbor.distance = distance;
                        }
                        if frequency > neighbor.frequency {
                            neighbor.frequency = frequency;
                        }
                    })
                    .or_insert(GrammarNeighbor {
                        ids,
                        distance,
                        frequency,
                    });
            }
        }
        merged.into_values().collect()
    }
}

impl NgramViewSource for ShardedView {
    type Node = U64NgramNode<<SharedARTrie<u64> as Dictionary>::Node>;
    type View = U64NgramView<SharedARTrie<u64>>;

    fn view_for(&self, first_id: u64, key_len: usize) -> Option<Self::View> {
        // Reverse-map the first term-id to the string the importer routed on, then
        // route by the PRODUCED length `key_len` (the Adaptive-granularity caveat),
        // exactly as the importer's `route_tokens` did at storage time.
        let first_token = self.vocabulary.get_term(first_id)?;
        let key = compute_shard_key_from_token(
            &first_token,
            key_len as u8,
            &self.coordinator.config().granularity,
        );
        self.open_view(&key)
    }

    fn whole_view(&self) -> Option<Self::View> {
        // A sharded store has no single whole-store view: unigrams (and the roots of
        // every length class) are spread across all shards. This disables just the
        // whole-store operations — the successor oracle at an empty history and an
        // empty-sequence neighbor query — for the sharded path.
        None
    }
}

/// The `T_lex ∘ T_gram` grammar corrector over a **sharded** term-id count corpus.
///
/// A thin delegating newtype over [`GrammarCore`]`<`[`ShardedView`]`>` — it shares the
/// exact beam decoder with the single-store
/// [`GrammarCorrector`](super::grammar_corrector::GrammarCorrector), differing only in
/// where the n-gram views come from. See the [module docs](self) for read-only-mode
/// safety, corpus-total injection, and the anchored-vs-fanout neighbor contract.
pub struct ShardedGrammarCorrector {
    core: GrammarCore<ShardedView>,
}

impl ShardedGrammarCorrector {
    /// Build a sharded corrector.
    ///
    /// - `coordinator` — the shard store; should be opened with `max_open_shards = 0`
    ///   (all shards resident) for query mode (see the [module docs](self)).
    /// - `vocabulary` — the term-id ↔ string bijection (routing + `T_lex`).
    /// - `order` — the model order (max n-gram length).
    /// - `total_count` — the corpus token total `` $`N=\sum_v f(v)`$ `` (injected; no
    ///   persisted total exists and unigrams span all shards).
    /// - `config` — decoder configuration.
    pub fn new(
        coordinator: Arc<ShardCoordinator>,
        vocabulary: SharedVocabARTrie,
        order: usize,
        total_count: u64,
        config: GrammarCorrectorConfig,
    ) -> Self {
        if coordinator.config().max_open_shards != 0 {
            log::warn!(
                "ShardedGrammarCorrector: coordinator has max_open_shards={} (bounded \
                 residency). The read-only query path never evicts, so touched shards \
                 accumulate resident regardless; use max_open_shards=0 to make all-resident \
                 query mode explicit and avoid interleaving with any writer that would evict.",
                coordinator.config().max_open_shards
            );
        }
        let source = ShardedView::new(coordinator, vocabulary.clone());
        Self {
            core: GrammarCore::new(vocabulary, source, order, total_count, config),
        }
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
    /// `(term, term-id, cost)` candidates. See
    /// [`GrammarCore::lex_candidates`](super::grammar_corrector::GrammarCore::lex_candidates).
    pub fn lex_candidates(&self, token: &str) -> Vec<LexCandidate> {
        self.core.lex_candidates(token)
    }

    /// `T_gram` (anchored) — the stored n-gram term-id sequences within
    /// `max_word_edit_distance` word edits of `term_ids` that reside in the **same
    /// shard** as the query's own first-token/length.
    ///
    /// This is the honest default: it returns exactly what the decoder's co-located
    /// scoring relies on. It does **not** return first-token-edit neighbors (which
    /// live in other shards) — use [`grammar_neighbors_fanout`] for that.
    ///
    /// [`grammar_neighbors_fanout`]: Self::grammar_neighbors_fanout
    pub fn grammar_neighbors(
        &self,
        term_ids: &[u64],
        max_word_edit_distance: usize,
    ) -> Vec<GrammarNeighbor> {
        self.core
            .grammar_neighbors(term_ids, max_word_edit_distance)
    }

    /// `T_gram` (fanout) — the stored n-gram term-id sequences within
    /// `max_word_edit_distance` word edits of `term_ids` across **all** shards,
    /// merged and de-duplicated.
    ///
    /// This restores the soundness+completeness contract of the single-store
    /// `grammar_neighbors` under sharding, including first-token-edit neighbors. It
    /// opens and walks every shard, so it is a batch / offline operation — prefer the
    /// anchored [`grammar_neighbors`](Self::grammar_neighbors) on the query hot path.
    pub fn grammar_neighbors_fanout(
        &self,
        term_ids: &[u64],
        max_word_edit_distance: usize,
    ) -> Vec<GrammarNeighbor> {
        self.core.source().neighbors_fanout(
            term_ids,
            max_word_edit_distance,
            self.core.config().algorithm,
        )
    }

    /// Correct `text`, returning up to `k_best` hypotheses best (lowest-cost) first.
    /// See [`GrammarCore::correct`](super::grammar_corrector::GrammarCore::correct).
    pub fn correct(&self, text: &str) -> Vec<GrammarCorrection> {
        self.core.correct(text)
    }
}
