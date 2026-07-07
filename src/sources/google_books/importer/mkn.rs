//! Modified Kneser-Ney smoothing statistics computation.
//!
//! After all n-grams are imported, the importer runs a post-processing pass
//! to compute the MKN statistics used by [`crate::ngram::smoothing`]:
//!
//! - **N1+(suffix)** — count of unique preceding words for each suffix.
//! - **N1+prefix(context)** — count of unique following words for each context.
//! - **n1/n2/n3/n4 frequency-of-frequencies** plus `total_unique` and
//!   `total_count` — used to derive the discount parameters.
//!
//! Two storage modes:
//! - **Single-trie**: iterate the trie in-process and write the stats into
//!   the same trie under `\x00…` metadata keys.
//! - **Sharded**: use the [`super::super::sharding::MknAggregator`] to fold
//!   per-shard partial results into the global stats keys.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use libdictenstein::persistent_artrie::PersistentARTrie;
use parking_lot::RwLock;

use crate::ngram::vocabulary::{decode_ngram_key_bytes, encode_indices_to_key_bytes};

use super::super::checkpoint::MknPhase;
use super::super::events::ImportEvent;
use super::super::sharding::MknAggregator;
use super::{GoogleBooksImporter, ImportError};

impl GoogleBooksImporter {
    /// Compute Modified Kneser-Ney smoothing statistics as a post-processing
    /// step, with optional event emission for TUI progress.
    ///
    /// This function computes MKN statistics differently based on storage mode:
    /// - **Single-trie**: Iterates over the trie and stores stats inline with special keys
    /// - **Sharded**: Uses MknAggregator to compute stats across all shards
    ///
    /// The statistics include:
    /// - N1+(suffix): Count of unique preceding words for each suffix
    /// - N1+prefix(context): Count of unique following words for each context
    ///
    /// These statistics are used by MKN smoothing to estimate probabilities
    /// for unseen n-grams based on lower-order distributions.
    pub(super) fn compute_mkn_stats_with_events(
        &mut self,
        event_tx: Option<&tokio::sync::broadcast::Sender<ImportEvent>>,
    ) -> Result<(), ImportError> {
        if self.checkpoint.mkn_phase == MknPhase::Complete {
            log::info!("MKN statistics already computed");
            return Ok(());
        }

        log::info!("Computing MKN statistics (post-processing)...");
        let mkn_start = std::time::Instant::now();
        let estimated_ngrams = self.total_ngrams.load(Ordering::Relaxed);

        // Emit MknStarted event
        let source = if self.storage.is_sharded() {
            "shards"
        } else {
            "single_trie"
        };
        if let Some(tx) = event_tx {
            log::debug!(
                "[IMPORTER] Sending MknStarted: source={}, estimated_ngrams={}",
                source,
                estimated_ngrams
            );
            let _ = tx.send(ImportEvent::MknStarted {
                source: source.to_string(),
                estimated_ngrams,
            });
        }

        let result = if self.storage.is_sharded() {
            // Sharded mode: use MknAggregator which iterates over all shards
            self.compute_mkn_stats_sharded_with_events(event_tx)
        } else {
            // Single-trie mode: iterate over trie and store stats inline
            self.compute_mkn_stats_single_trie_with_events(event_tx)
        };

        match result {
            Ok((continuation_entries, frequency_entries)) => {
                self.checkpoint.mkn_phase = MknPhase::Complete;
                self.save_checkpoint()?;

                let duration = mkn_start.elapsed();
                log::info!(
                    "MKN statistics computed successfully in {:.1}s",
                    duration.as_secs_f64()
                );

                // Emit MknCompleted event
                if let Some(tx) = event_tx {
                    log::debug!(
                        "[IMPORTER] Sending MknCompleted: continuation={}, frequency={}",
                        continuation_entries,
                        frequency_entries
                    );
                    let _ = tx.send(ImportEvent::MknCompleted {
                        continuation_entries,
                        frequency_entries,
                        duration,
                    });
                }

                Ok(())
            }
            Err(e) => {
                // Emit MknFailed event
                if let Some(tx) = event_tx {
                    let _ = tx.send(ImportEvent::MknFailed {
                        error: e.to_string(),
                    });
                }
                Err(e)
            }
        }
    }

    /// Compute MKN stats for sharded storage with optional event emission.
    ///
    /// Returns (continuation_entries, frequency_entries) counts on success.
    fn compute_mkn_stats_sharded_with_events(
        &self,
        event_tx: Option<&tokio::sync::broadcast::Sender<ImportEvent>>,
    ) -> Result<(u64, u64), ImportError> {
        let coordinator = self
            .storage
            .as_sharded()
            .ok_or_else(|| ImportError::Trie("Expected sharded storage".to_string()))?;

        // Phase 1: Compute MKN statistics (parallel over shards via rayon)
        log::info!("MKN Phase 1: Computing statistics across shards...");
        if let Some(tx) = event_tx {
            let _ = tx.send(ImportEvent::MknProgress {
                phase: 1,
                total_phases: 2,
                items_processed: 0,
                total_items: self.total_ngrams.load(Ordering::Relaxed),
                percent_complete: 0.0,
            });
        }

        let aggregator = MknAggregator::new(coordinator).with_cancellation_flag(&self.interrupted);
        let mkn_stats = aggregator
            .compute_all()
            .map_err(|e| ImportError::Trie(format!("Failed to compute MKN statistics: {}", e)))?;

        if let Some(tx) = event_tx {
            let _ = tx.send(ImportEvent::MknProgress {
                phase: 1,
                total_phases: 2,
                items_processed: self.total_ngrams.load(Ordering::Relaxed),
                total_items: self.total_ngrams.load(Ordering::Relaxed),
                percent_complete: 50.0,
            });
        }

        // Phase 2: Write MKN statistics to trie
        log::info!("MKN Phase 2: Writing statistics to MKN trie...");
        let mkn_path = self.config.output_path.with_extension("mkn.artrie");
        log::info!("Saving MKN statistics to {:?}...", mkn_path);

        let mkn_trie = PersistentARTrie::create(&mkn_path)
            .map_err(|e| ImportError::Trie(format!("Failed to create MKN trie: {}", e)))?;
        let mkn_trie = Arc::new(RwLock::new(mkn_trie));

        let mut continuation_entries = 0u64;
        let mut frequency_entries = 0u64;

        {
            let trie = mkn_trie.write();

            // Store frequency counts for each order
            for (order, counts) in mkn_stats.frequency_counts.iter().enumerate() {
                let prefix = format!("\x00order{}\x00", order);
                trie.upsert_bytes(format!("{}n1", prefix).as_bytes(), counts.n1)
                    .map_err(|e| ImportError::Trie(format!("Failed to write n1: {}", e)))?;
                trie.upsert_bytes(format!("{}n2", prefix).as_bytes(), counts.n2)
                    .map_err(|e| ImportError::Trie(format!("Failed to write n2: {}", e)))?;
                trie.upsert_bytes(format!("{}n3", prefix).as_bytes(), counts.n3)
                    .map_err(|e| ImportError::Trie(format!("Failed to write n3: {}", e)))?;
                trie.upsert_bytes(format!("{}n4", prefix).as_bytes(), counts.n4)
                    .map_err(|e| ImportError::Trie(format!("Failed to write n4: {}", e)))?;
                trie.upsert_bytes(
                    format!("{}total_unique", prefix).as_bytes(),
                    counts.total_unique,
                )
                .map_err(|e| ImportError::Trie(format!("Failed to write total_unique: {}", e)))?;
                trie.upsert_bytes(
                    format!("{}total_count", prefix).as_bytes(),
                    counts.total_count,
                )
                .map_err(|e| ImportError::Trie(format!("Failed to write total_count: {}", e)))?;
                frequency_entries += 6;
            }

            // Store continuation counts for each order
            for (order, conts) in mkn_stats.continuation_counts.iter().enumerate() {
                // Store predecessor counts (N1+(•w) - unique predecessors for each context)
                for (context, count) in &conts.predecessor_counts {
                    let mut key = format!("\x00N1+predecessor\x00{}\x00", order).into_bytes();
                    key.extend_from_slice(context);
                    trie.upsert_bytes(&key, *count).map_err(|e| {
                        ImportError::Trie(format!("Failed to write predecessor count: {}", e))
                    })?;
                    continuation_entries += 1;
                }

                // Store successor counts (N1+(w•) - unique successors for each context)
                for (context, count) in &conts.successor_counts {
                    let mut key = format!("\x00N1+successor\x00{}\x00", order).into_bytes();
                    key.extend_from_slice(context);
                    trie.upsert_bytes(&key, *count).map_err(|e| {
                        ImportError::Trie(format!("Failed to write successor count: {}", e))
                    })?;
                    continuation_entries += 1;
                }
            }

            // Checkpoint to persist
            trie.checkpoint()
                .map_err(|e| ImportError::Trie(format!("Failed to checkpoint MKN trie: {}", e)))?;
        }

        if let Some(tx) = event_tx {
            let _ = tx.send(ImportEvent::MknProgress {
                phase: 2,
                total_phases: 2,
                items_processed: continuation_entries + frequency_entries,
                total_items: continuation_entries + frequency_entries,
                percent_complete: 100.0,
            });
        }

        log::info!(
            "MKN statistics saved: {} orders with frequency and continuation counts",
            mkn_stats.max_order
        );

        Ok((continuation_entries, frequency_entries))
    }

    /// Compute MKN stats for single-trie storage with optional event emission.
    ///
    /// Returns (continuation_entries, frequency_entries) counts on success.
    ///
    /// N-grams are stored as varint-encoded byte keys (LEB128 encoding).
    /// This function decodes them to extract word indices for
    /// computing predecessor and successor contexts.
    fn compute_mkn_stats_single_trie_with_events(
        &self,
        event_tx: Option<&tokio::sync::broadcast::Sender<ImportEvent>>,
    ) -> Result<(u64, u64), ImportError> {
        // Collect unique (suffix, prefix) and (context, following) pairs
        // using HashSets for deduplication.
        //
        // We store contexts as varint-encoded keys (same format as n-gram keys)
        // and use u64 indices for efficient comparison.
        use std::collections::HashSet;
        let mut continuation_pairs: HashSet<(Vec<u8>, u64)> = HashSet::new();
        let mut unique_cont_pairs: HashSet<(Vec<u8>, u64)> = HashSet::new();

        // Frequency count accumulators
        let mut n1 = 0u64;
        let mut n2 = 0u64;
        let mut n3 = 0u64;
        let mut n4 = 0u64;
        let mut total_unique = 0u64;
        let mut total_count = 0u64;

        let estimated_ngrams = self.total_ngrams.load(Ordering::Relaxed);

        // Phase 1: Iterate all n-grams, collect pairs and compute frequency counts
        log::info!("Phase 1: Collecting continuation pairs and frequency counts from n-grams...");
        if let Some(tx) = event_tx {
            let _ = tx.send(ImportEvent::MknProgress {
                phase: 1,
                total_phases: 3,
                items_processed: 0,
                total_items: estimated_ngrams,
                percent_complete: 0.0,
            });
        }

        {
            // Single-trie MKN: iterate the n-gram data living in the
            // checkpoint trie (which IS the data trie in single-trie mode).
            let trie_arc = self.storage.checkpoint_trie();
            let trie = trie_arc.as_ref();
            // Collect all entries first to avoid lifetime issues with borrowed iterator
            let entries: Vec<(Vec<u8>, u64)> = trie
                .iter_prefix_with_values(b"")
                .map(|iter| iter.collect())
                .unwrap_or_default();
            for (ngram, count) in entries {
                // Skip metadata keys (they start with \x00)
                if ngram.starts_with(&[0x00]) {
                    continue;
                }

                // Accumulate frequency counts
                total_unique += 1;
                total_count += count;
                match count {
                    1 => n1 += 1,
                    2 => n2 += 1,
                    3 => n3 += 1,
                    4 => n4 += 1,
                    _ => {}
                }

                // Decode varint-encoded key to word indices
                let indices = decode_ngram_key_bytes(&ngram);
                if indices.len() >= 2 {
                    // MKN Pass 1: continuation counts (suffix → unique prefixes)
                    // e.g., indices [0, 1, 2] → prefix=0, suffix=encode([1, 2])
                    let prefix = indices[0];
                    let suffix = encode_indices_to_key_bytes(&indices[1..]);
                    continuation_pairs.insert((suffix, prefix));

                    // MKN Pass 2: unique continuations (context → unique following)
                    // e.g., indices [0, 1, 2] → context=encode([0, 1]), following=2
                    let context = encode_indices_to_key_bytes(&indices[..indices.len() - 1]);
                    let following = indices[indices.len() - 1];
                    unique_cont_pairs.insert((context, following));
                }
            }
        }

        if let Some(tx) = event_tx {
            let _ = tx.send(ImportEvent::MknProgress {
                phase: 1,
                total_phases: 3,
                items_processed: estimated_ngrams,
                total_items: estimated_ngrams,
                percent_complete: 33.0,
            });
        }

        log::info!(
            "Collected {} continuation pairs and {} unique continuation pairs",
            continuation_pairs.len(),
            unique_cont_pairs.len()
        );
        log::info!(
            "Frequency counts: n1={}, n2={}, n3={}, n4={}, total_unique={}, total_count={}",
            n1,
            n2,
            n3,
            n4,
            total_unique,
            total_count
        );

        // Phase 2: Compute and write continuation counts
        log::info!("Phase 2: Writing continuation statistics to trie...");
        if let Some(tx) = event_tx {
            let _ = tx.send(ImportEvent::MknProgress {
                phase: 2,
                total_phases: 3,
                items_processed: 0,
                total_items: continuation_pairs.len() as u64 + unique_cont_pairs.len() as u64,
                percent_complete: 33.0,
            });
        }

        let mut continuation_entries = 0u64;
        {
            // Single-trie MKN writes to the data trie (same as checkpoint
            // trie in single-trie mode).
            let trie_arc = self.storage.checkpoint_trie();
            let trie = trie_arc.as_ref();

            // Count unique prefixes per suffix (N1+(suffix))
            let mut suffix_counts: std::collections::HashMap<Vec<u8>, u64> =
                std::collections::HashMap::new();
            for (suffix, _prefix) in &continuation_pairs {
                *suffix_counts.entry(suffix.clone()).or_insert(0) += 1;
            }

            // Write continuation counts
            for (suffix, count) in &suffix_counts {
                let mut count_key = b"\x00N1+\x00".to_vec();
                count_key.extend_from_slice(suffix);
                trie.upsert_bytes(&count_key, *count).map_err(|e| {
                    ImportError::Trie(format!("Failed to write MKN continuation count: {}", e))
                })?;
                continuation_entries += 1;
            }

            // Count unique following words per context (N1+prefix(context))
            let mut context_counts: std::collections::HashMap<Vec<u8>, u64> =
                std::collections::HashMap::new();
            for (context, _following) in &unique_cont_pairs {
                *context_counts.entry(context.clone()).or_insert(0) += 1;
            }

            // Write unique continuation counts
            for (context, count) in &context_counts {
                let mut count_key = b"\x00N1+prefix\x00".to_vec();
                count_key.extend_from_slice(context);
                trie.upsert_bytes(&count_key, *count).map_err(|e| {
                    ImportError::Trie(format!(
                        "Failed to write MKN unique continuation count: {}",
                        e
                    ))
                })?;
                continuation_entries += 1;
            }
        }

        if let Some(tx) = event_tx {
            let _ = tx.send(ImportEvent::MknProgress {
                phase: 2,
                total_phases: 3,
                items_processed: continuation_entries,
                total_items: continuation_entries,
                percent_complete: 66.0,
            });
        }

        // Phase 3: Write frequency counts to trie
        log::info!("Phase 3: Writing frequency statistics to trie...");
        if let Some(tx) = event_tx {
            let _ = tx.send(ImportEvent::MknProgress {
                phase: 3,
                total_phases: 3,
                items_processed: 0,
                total_items: 6,
                percent_complete: 66.0,
            });
        }

        let mut frequency_entries = 0u64;
        {
            let trie_arc = self.storage.checkpoint_trie();
            let trie = trie_arc.as_ref();

            trie.upsert_bytes(b"\x00mkn\x00n1", n1)
                .map_err(|e| ImportError::Trie(format!("Failed to write MKN n1: {}", e)))?;
            frequency_entries += 1;

            trie.upsert_bytes(b"\x00mkn\x00n2", n2)
                .map_err(|e| ImportError::Trie(format!("Failed to write MKN n2: {}", e)))?;
            frequency_entries += 1;

            trie.upsert_bytes(b"\x00mkn\x00n3", n3)
                .map_err(|e| ImportError::Trie(format!("Failed to write MKN n3: {}", e)))?;
            frequency_entries += 1;

            trie.upsert_bytes(b"\x00mkn\x00n4", n4)
                .map_err(|e| ImportError::Trie(format!("Failed to write MKN n4: {}", e)))?;
            frequency_entries += 1;

            trie.upsert_bytes(b"\x00mkn\x00total_unique", total_unique)
                .map_err(|e| {
                    ImportError::Trie(format!("Failed to write MKN total_unique: {}", e))
                })?;
            frequency_entries += 1;

            trie.upsert_bytes(b"\x00mkn\x00total_count", total_count)
                .map_err(|e| {
                    ImportError::Trie(format!("Failed to write MKN total_count: {}", e))
                })?;
            frequency_entries += 1;
        }

        if let Some(tx) = event_tx {
            let _ = tx.send(ImportEvent::MknProgress {
                phase: 3,
                total_phases: 3,
                items_processed: frequency_entries,
                total_items: frequency_entries,
                percent_complete: 100.0,
            });
        }

        Ok((continuation_entries, frequency_entries))
    }
}
