//! Lazy WFST implementation for n-gram models.
//!
//! This module provides a lazy state source for n-gram language models,
//! enabling on-demand state expansion during WFST traversal. This is
//! especially useful for large language models where materializing the
//! entire WFST upfront would be prohibitive.
//!
//! # Architecture
//!
//! The lazy implementation:
//! - Computes states on-demand as they are accessed during composition
//! - Caches computed states according to the configured policy
//! - Uses thread-safe state registration with `RwLock`
//!
//! # Example
//!
//! ```ignore
//! use libgrammstein::ngram::NgramModel;
//! use libgrammstein::integration::lazy_ngram::NgramStateSource;
//! use lling_llang::semiring::LogWeight;
//! use lling_llang::wfst::LazyWfstWrapper;
//!
//! let model: NgramModel<S> = /* ... */;
//! let source = NgramStateSource::<_, LogWeight>::new(Arc::new(model));
//! let lazy_wfst = LazyWfstWrapper::new(source);
//!
//! // States are computed on-demand during traversal
//! lazy_wfst.expand(0); // Compute start state
//! ```

use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

use lling_llang::semiring::Semiring;
use lling_llang::wfst::{LazyState, StateId, StateSource, WeightedTransition};
use smallvec::SmallVec;

use crate::ngram::store::{IterableNgramStore, NgramLookup};
use crate::ngram::NgramModel;

use super::vocabulary::{WordId, WordVocabulary};
use super::wfst_export::FromLogProb;

/// Key representing an n-gram history for state lookup.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum NgramHistoryKey {
    /// The start state (empty history with special handling).
    Start,
    /// Backoff state for a given history depth.
    /// An empty history Vec represents the unigram backoff state.
    Backoff {
        /// The history words leading to this backoff.
        history: Vec<WordId>,
    },
    /// A regular n-gram history state.
    History {
        /// The words comprising this history.
        words: Vec<WordId>,
    },
}

impl NgramHistoryKey {
    /// Create a start state key.
    pub fn start() -> Self {
        Self::Start
    }

    /// Create a backoff state key.
    pub fn backoff(history: Vec<WordId>) -> Self {
        Self::Backoff { history }
    }

    /// Create a history state key.
    pub fn history(words: Vec<WordId>) -> Self {
        Self::History { words }
    }
}

/// Thread-safe registry mapping history keys to state IDs.
#[derive(Debug)]
pub struct NgramStateRegistry {
    /// Mapping from state ID to history key.
    id_to_history: Vec<NgramHistoryKey>,
    /// Mapping from history key to state ID.
    history_to_id: HashMap<NgramHistoryKey, StateId>,
    /// Next available state ID.
    next_id: StateId,
}

impl NgramStateRegistry {
    /// Create a new registry with start and backoff states pre-registered.
    pub fn new() -> Self {
        let mut registry = Self {
            id_to_history: Vec::with_capacity(64),
            history_to_id: HashMap::with_capacity(64),
            next_id: 0,
        };

        // Register start state (ID 0)
        registry.register(NgramHistoryKey::Start);

        // Register unigram backoff state (ID 1)
        registry.register(NgramHistoryKey::Backoff { history: vec![] });

        registry
    }

    /// Register a history key, returning its state ID.
    ///
    /// If already registered, returns the existing ID.
    pub fn register(&mut self, key: NgramHistoryKey) -> StateId {
        if let Some(&id) = self.history_to_id.get(&key) {
            return id;
        }

        let id = self.next_id;
        self.next_id += 1;
        self.id_to_history.push(key.clone());
        self.history_to_id.insert(key, id);
        id
    }

    /// Get the history key for a state ID.
    pub fn get_history(&self, state: StateId) -> Option<&NgramHistoryKey> {
        self.id_to_history.get(state as usize)
    }

    /// Get the state ID for a history key.
    pub fn get_state(&self, key: &NgramHistoryKey) -> Option<StateId> {
        self.history_to_id.get(key).copied()
    }

    /// Get the number of registered states.
    pub fn len(&self) -> usize {
        self.id_to_history.len()
    }

    /// Check if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.id_to_history.is_empty()
    }
}

impl Default for NgramStateRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Lazy state source for n-gram language models.
///
/// Implements [`StateSource`] to enable lazy WFST construction where
/// states are computed on-demand during traversal.
///
/// # Thread Safety
///
/// The state registry is protected by a `RwLock`, allowing concurrent
/// reads during WFST traversal while serializing state registration.
pub struct NgramStateSource<S, W>
where
    S: NgramLookup,
    W: Semiring + FromLogProb,
{
    /// The underlying n-gram model.
    model: Arc<NgramModel<S>>,
    /// Word vocabulary.
    vocabulary: Arc<WordVocabulary>,
    /// Thread-safe state registry.
    state_registry: RwLock<NgramStateRegistry>,
    /// Phantom data for weight type.
    _weight: PhantomData<W>,
}

impl<S, W> NgramStateSource<S, W>
where
    S: IterableNgramStore,
    W: Semiring + FromLogProb,
{
    /// Create a new lazy state source from an n-gram model.
    pub fn new(model: Arc<NgramModel<S>>) -> Self {
        let vocabulary = Arc::new(Self::build_vocabulary(&model));

        Self {
            model,
            vocabulary,
            state_registry: RwLock::new(NgramStateRegistry::new()),
            _weight: PhantomData,
        }
    }

    /// Create with a pre-built vocabulary.
    pub fn with_vocabulary(model: Arc<NgramModel<S>>, vocabulary: Arc<WordVocabulary>) -> Self {
        Self {
            model,
            vocabulary,
            state_registry: RwLock::new(NgramStateRegistry::new()),
            _weight: PhantomData,
        }
    }

    /// Get a reference to the vocabulary.
    pub fn vocabulary(&self) -> &WordVocabulary {
        &self.vocabulary
    }

    /// Get a reference to the model.
    pub fn model(&self) -> &NgramModel<S> {
        &self.model
    }

    /// Build vocabulary from the n-gram model.
    fn build_vocabulary(model: &NgramModel<S>) -> WordVocabulary {
        let mut vocab = WordVocabulary::with_capacity(model.vocab_size());

        for (words, _entry) in model.iter_ngrams() {
            // A unigram is a single-word n-gram.
            if words.len() == 1 {
                vocab.add_word(&words[0]);
            }
        }

        vocab
    }

    /// Get or register a state for a history key.
    fn get_or_register_state(&self, key: NgramHistoryKey) -> StateId {
        // Try read lock first
        {
            let registry = self.state_registry.read().expect("Lock poisoned");
            if let Some(id) = registry.get_state(&key) {
                return id;
            }
        }

        // Need write lock to register
        let mut registry = self.state_registry.write().expect("Lock poisoned");
        registry.register(key)
    }

    /// Compute the start state.
    fn compute_start_state(&self) -> LazyState<WordId, W> {
        // Start state has epsilon transition to backoff and
        // transitions for all words based on unigram probabilities
        let mut transitions: SmallVec<[WeightedTransition<WordId, W>; 4]> = SmallVec::new();

        // Epsilon transition to unigram backoff state (state 1)
        transitions.push(WeightedTransition::new(0, None, None, 1, W::one()));

        LazyState::final_state(W::one(), transitions)
    }

    /// Compute a backoff state.
    fn compute_backoff_state(&self, history: &[WordId]) -> LazyState<WordId, W> {
        let mut transitions: SmallVec<[WeightedTransition<WordId, W>; 4]> = SmallVec::new();
        let order = self.model.order();

        // Convert history to strings for querying
        let history_strs: Vec<String> = history
            .iter()
            .filter_map(|&id| self.vocabulary.get_word(id).map(|s| s.to_string()))
            .collect();

        let history_refs: Vec<&str> = history_strs.iter().map(|s| s.as_str()).collect();

        // Add transitions for all words in vocabulary
        for (word, word_id) in self.vocabulary.iter().skip(2) {
            // Skip special tokens
            let log_prob = self.model.log_prob(word, &history_refs);

            // Skip zero-probability transitions
            if log_prob.is_finite() {
                let weight = W::from_log_prob(log_prob);

                // Compute target history
                let mut target_history = history.to_vec();
                target_history.push(word_id);
                if target_history.len() >= order {
                    target_history = target_history[target_history.len() - (order - 1)..].to_vec();
                }

                let target_key = NgramHistoryKey::History {
                    words: target_history,
                };
                let target_state = self.get_or_register_state(target_key);

                let source_state = if history.is_empty() {
                    1 // Unigram backoff state
                } else {
                    self.get_or_register_state(NgramHistoryKey::Backoff {
                        history: history.to_vec(),
                    })
                };

                transitions.push(WeightedTransition::new(
                    source_state,
                    Some(word_id),
                    Some(word_id),
                    target_state,
                    weight,
                ));
            }
        }

        LazyState::final_state(W::one(), transitions)
    }

    /// Compute a history state.
    fn compute_history_state(&self, words: &[WordId]) -> LazyState<WordId, W> {
        let mut transitions: SmallVec<[WeightedTransition<WordId, W>; 4]> = SmallVec::new();
        let order = self.model.order();

        // Convert history to strings for querying
        let history_strs: Vec<String> = words
            .iter()
            .filter_map(|&id| self.vocabulary.get_word(id).map(|s| s.to_string()))
            .collect();

        let history_refs: Vec<&str> = history_strs.iter().map(|s| s.as_str()).collect();

        // Get source state ID
        let source_state = self.get_or_register_state(NgramHistoryKey::History {
            words: words.to_vec(),
        });

        // Add transitions for all words
        for (word, word_id) in self.vocabulary.iter().skip(2) {
            let log_prob = self.model.log_prob(word, &history_refs);

            if log_prob.is_finite() {
                let weight = W::from_log_prob(log_prob);

                // Compute target history
                let mut target_history = words.to_vec();
                target_history.push(word_id);
                if target_history.len() >= order {
                    target_history = target_history[target_history.len() - (order - 1)..].to_vec();
                }

                let target_key = NgramHistoryKey::History {
                    words: target_history,
                };
                let target_state = self.get_or_register_state(target_key);

                transitions.push(WeightedTransition::new(
                    source_state,
                    Some(word_id),
                    Some(word_id),
                    target_state,
                    weight,
                ));
            }
        }

        // Add backoff epsilon transition
        if !words.is_empty() {
            let backoff_history = words[1..].to_vec();
            let backoff_key = if backoff_history.is_empty() {
                NgramHistoryKey::Backoff { history: vec![] }
            } else {
                NgramHistoryKey::History {
                    words: backoff_history,
                }
            };
            let backoff_state = self.get_or_register_state(backoff_key);

            transitions.push(WeightedTransition::new(
                source_state,
                None, // epsilon
                None,
                backoff_state,
                W::one(),
            ));
        }

        LazyState::final_state(W::one(), transitions)
    }
}

impl<S, W> Clone for NgramStateSource<S, W>
where
    S: IterableNgramStore,
    W: Semiring + FromLogProb,
{
    fn clone(&self) -> Self {
        Self {
            model: Arc::clone(&self.model),
            vocabulary: Arc::clone(&self.vocabulary),
            state_registry: RwLock::new(NgramStateRegistry::new()),
            _weight: PhantomData,
        }
    }
}

impl<S, W> StateSource<WordId, W> for NgramStateSource<S, W>
where
    S: IterableNgramStore + Send + Sync,
    W: Semiring + FromLogProb,
{
    fn compute_state(&self, state: StateId) -> LazyState<WordId, W> {
        // Get the history key for this state
        let history_key = {
            let registry = self.state_registry.read().expect("Lock poisoned");
            registry.get_history(state).cloned()
        };

        match history_key {
            Some(NgramHistoryKey::Start) => self.compute_start_state(),
            Some(NgramHistoryKey::Backoff { history }) => self.compute_backoff_state(&history),
            Some(NgramHistoryKey::History { words }) => self.compute_history_state(&words),
            None => {
                // Unknown state - return empty non-final
                LazyState::non_final(SmallVec::new())
            }
        }
    }

    fn start(&self) -> StateId {
        0 // Start state is always ID 0
    }

    fn num_states_hint(&self) -> Option<usize> {
        // Can't know in advance how many states will be expanded
        None
    }
}

/// Extension trait for NgramModel to provide lazy WFST.
pub trait NgramLazyWfst<S>
where
    S: IterableNgramStore + Clone,
{
    /// Create a lazy WFST state source for this n-gram model.
    ///
    /// The resulting state source can be wrapped with `LazyWfstWrapper`
    /// for use in lling-llang's WFST operations.
    ///
    /// # Type Parameters
    ///
    /// * `W` - The semiring weight type (must implement `FromLogProb`)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use libgrammstein::ngram::NgramModel;
    /// use libgrammstein::integration::lazy_ngram::NgramLazyWfst;
    /// use lling_llang::semiring::LogWeight;
    /// use lling_llang::wfst::LazyWfstWrapper;
    ///
    /// let model: NgramModel<S> = /* ... */;
    /// let source = model.to_lazy_wfst_source::<LogWeight>();
    /// let lazy_wfst = LazyWfstWrapper::new(source);
    /// ```
    fn to_lazy_wfst_source<W>(&self) -> NgramStateSource<S, W>
    where
        W: Semiring + FromLogProb;
}

impl<S> NgramLazyWfst<S> for NgramModel<S>
where
    S: IterableNgramStore + Clone,
{
    fn to_lazy_wfst_source<W>(&self) -> NgramStateSource<S, W>
    where
        W: Semiring + FromLogProb,
    {
        NgramStateSource::new(Arc::new(self.clone()))
    }
}

impl<S> NgramLazyWfst<S> for Arc<NgramModel<S>>
where
    S: IterableNgramStore + Clone,
{
    fn to_lazy_wfst_source<W>(&self) -> NgramStateSource<S, W>
    where
        W: Semiring + FromLogProb,
    {
        NgramStateSource::new(Arc::clone(self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::PlaintextReader;
    use crate::ngram::store::TermIdStore;
    use crate::ngram::{NgramEntry, TrainerBuilder};
    use libdictenstein::dynamic_dawg::DynamicDawg;
    use lling_llang::semiring::LogWeight;
    use lling_llang::wfst::LazyWfst;
    use lling_llang::wfst::LazyWfstWrapper;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_test_model() -> NgramModel<TermIdStore<DynamicDawg<NgramEntry>>> {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let content = "the quick brown fox the quick brown dog";
        let path = dir.path().join("test.txt");
        let mut file = std::fs::File::create(&path).expect("Failed to create test file");
        write!(file, "{}", content).expect("Failed to write test file");

        let reader = PlaintextReader::from_file(&path).expect("Failed to create reader");
        let dictionary = DynamicDawg::<NgramEntry>::new();
        TrainerBuilder::new(dictionary)
            .order(3)
            .train(reader)
            .expect("Training failed")
    }

    #[test]
    fn test_state_registry() {
        let mut registry = NgramStateRegistry::new();

        // Start and backoff should be pre-registered
        assert_eq!(registry.get_state(&NgramHistoryKey::Start), Some(0));
        assert_eq!(
            registry.get_state(&NgramHistoryKey::Backoff { history: vec![] }),
            Some(1)
        );

        // Register new states
        let id1 = registry.register(NgramHistoryKey::History { words: vec![5] });
        let id2 = registry.register(NgramHistoryKey::History { words: vec![5, 6] });

        assert_eq!(id1, 2);
        assert_eq!(id2, 3);

        // Duplicate registration returns same ID
        let id1_dup = registry.register(NgramHistoryKey::History { words: vec![5] });
        assert_eq!(id1_dup, id1);
    }

    #[test]
    fn test_ngram_state_source_creation() {
        let model = create_test_model();
        let source: NgramStateSource<_, LogWeight> = model.to_lazy_wfst_source();

        assert_eq!(source.start(), 0);
        assert!(source.vocabulary().len() > 2); // More than just special tokens
    }

    #[test]
    fn test_lazy_wfst_start_state() {
        let model = create_test_model();
        let source: NgramStateSource<_, LogWeight> = model.to_lazy_wfst_source();
        let mut lazy = LazyWfstWrapper::new(source);

        // Expand start state
        lazy.expand(0);
        assert!(lazy.is_expanded(0));

        // Should have transitions
        let transitions = lazy.transitions_lazy(0);
        assert!(!transitions.is_empty());
    }

    #[test]
    fn test_lazy_wfst_expansion() {
        let model = create_test_model();
        let source: NgramStateSource<_, LogWeight> = model.to_lazy_wfst_source();
        let mut lazy = LazyWfstWrapper::new(source);

        // Expand states progressively
        lazy.expand(0);
        let initial_count = lazy.computed_states();

        // Expand backoff state
        lazy.expand(1);
        assert!(lazy.computed_states() > initial_count);
    }
}
