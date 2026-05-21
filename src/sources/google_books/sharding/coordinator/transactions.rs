//! Coordinator-level prefix transactions for atomic, idempotent imports.
//!
//! Wraps a shard-level `PrefixTransaction` with the shard reference needed
//! to commit or abort the transaction. Used by the Google Books importer's
//! chunked-tx path; see the module docs on `super::CoordinatorPrefixTx`.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use super::{CoordinatorPrefixTx, CoordinatorResult, ShardCoordinator, ShardKey};

impl ShardCoordinator {
    // ========================================================================
    // Document Transaction API (for idempotent prefix imports)
    // ========================================================================

    /// Begin a prefix transaction for atomic, idempotent n-gram import.
    ///
    /// This creates a document transaction on the appropriate shard that buffers
    /// all n-gram inserts until `commit_prefix_tx()` is called. If interrupted
    /// before commit, the transaction is automatically discarded on recovery.
    ///
    /// # Key Properties
    ///
    /// - **Atomicity**: Either all n-grams are committed or none are
    /// - **Idempotency**: Uses SET semantics, so re-imports produce the same result
    /// - **Crash Safety**: Uncommitted transactions are discarded on WAL recovery
    ///
    /// # Arguments
    ///
    /// * `shard_key` - The shard key for this prefix (from `route_tokens`)
    /// * `prefix` - The prefix file being imported (used as document ID)
    ///
    /// # Returns
    ///
    /// A `CoordinatorPrefixTx` that must be passed to `tx_insert()` and
    /// eventually to `commit_prefix_tx()`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let shard_key = coordinator.route_tokens(&tokens);
    /// let mut tx = coordinator.begin_prefix_tx(&shard_key, "th")?;
    /// for (ngram, count) in ngrams {
    ///     coordinator.tx_insert(&mut tx, &ngram, count);
    /// }
    /// coordinator.commit_prefix_tx(tx)?;
    /// ```
    pub fn begin_prefix_tx(
        &self,
        shard_key: &ShardKey,
        prefix: &str,
    ) -> CoordinatorResult<CoordinatorPrefixTx> {
        let shard = self.get_or_create_shard(shard_key)?;
        let guard = shard.read();
        let inner_tx = guard.begin_prefix(prefix)?;

        Ok(CoordinatorPrefixTx {
            shard_key: shard_key.clone(),
            shard: Arc::clone(&shard),
            inner: Some(inner_tx),
        })
    }

    /// Insert an n-gram into a pending prefix transaction.
    ///
    /// The n-gram is buffered in memory and will be written atomically when
    /// the transaction is committed. Uses SET semantics (not increment),
    /// making re-imports idempotent.
    ///
    /// # Arguments
    ///
    /// * `tx` - The active transaction from `begin_prefix_tx()`
    /// * `ngram` - The n-gram key to insert
    /// * `count` - The n-gram count
    pub fn tx_insert(&self, tx: &mut CoordinatorPrefixTx, ngram: &[u8], count: u64) {
        let guard = tx.shard.read();
        if let Some(ref mut inner) = tx.inner {
            guard.tx_insert(inner, ngram, count);
        }
    }

    /// Commit a prefix transaction atomically and mark the prefix as completed.
    ///
    /// This:
    /// 1. Writes all buffered n-grams to the WAL as a single batch
    /// 2. Applies them to the trie atomically
    /// 3. Marks the prefix as completed in the shard's checkpoint state
    /// 4. Persists the shard checkpoint state to WAL (done by commit_prefix)
    /// 5. Updates and saves the global checkpoint
    ///
    /// # Durability Guarantee
    ///
    /// After this method returns successfully, the prefix completion is durable:
    /// - Shard checkpoint state is persisted to the shard's WAL
    /// - Global checkpoint is saved to disk
    ///
    /// This ensures that if the process crashes after commit_prefix_tx() returns,
    /// the prefix will be correctly marked as complete during recovery.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to commit (consumed)
    ///
    /// # Returns
    ///
    /// The number of n-grams that were committed.
    pub fn commit_prefix_tx(&self, mut tx: CoordinatorPrefixTx) -> CoordinatorResult<usize> {
        let inner_tx = tx.inner.take().expect("Transaction already consumed");
        let prefix = inner_tx.prefix.clone();

        // Commit the transaction (this now also updates and persists shard checkpoint state)
        let ngram_count = {
            let mut guard = tx.shard.write();
            guard.commit_prefix(inner_tx)?
        };

        // Update stats
        self.stats.unique_ngrams.fetch_add(ngram_count as u64, Ordering::Relaxed);

        // Mark prefix as completed in global checkpoint and force save.
        // We use save() instead of maybe_save() to ensure durability - this is
        // critical because the shard's checkpoint state has already recorded
        // the prefix as complete. If we skip saving the global checkpoint and
        // crash, recovery would rebuild the global checkpoint from shard states
        // correctly, but forcing the save here provides an extra safety margin
        // and keeps the global checkpoint in sync.
        if let Some(ref manager) = self.checkpoint_manager {
            let mut mgr = manager.lock();
            mgr.checkpoint_mut().complete_prefix(&tx.shard_key, &prefix);
            mgr.save()?;
        }

        Ok(ngram_count)
    }

    /// Commit a prefix transaction chunk WITHOUT marking the prefix as complete.
    ///
    /// This commits the buffered n-grams to the WAL but does not update the
    /// global checkpoint or mark the prefix as completed. Used for chunked
    /// imports of large prefix files to bound per-transaction memory usage.
    ///
    /// After committing a chunk, the caller should begin a new transaction
    /// for the next chunk via `begin_prefix_tx()`.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to commit (consumed)
    ///
    /// # Returns
    ///
    /// The number of n-grams that were committed in this chunk.
    pub fn commit_chunk_tx(&self, tx: &mut CoordinatorPrefixTx) -> CoordinatorResult<usize> {
        let inner_tx = tx.inner.take().expect("Transaction already consumed");

        let ngram_count = {
            let mut guard = tx.shard.write();
            guard.commit_chunk(inner_tx)?
        };

        // Update stats (but don't mark prefix as completed)
        self.stats.unique_ngrams.fetch_add(ngram_count as u64, Ordering::Relaxed);

        Ok(ngram_count)
    }

    /// Abort a prefix transaction, discarding all buffered n-grams.
    ///
    /// Use this if an error occurs during processing and you want to
    /// discard the partial work without committing it.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to abort (consumed)
    pub fn abort_prefix_tx(&self, mut tx: CoordinatorPrefixTx) -> CoordinatorResult<()> {
        if let Some(inner_tx) = tx.inner.take() {
            let guard = tx.shard.read();
            guard.abort_prefix(inner_tx)?;
        }
        Ok(())
    }
}
