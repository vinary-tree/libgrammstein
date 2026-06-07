------------------------ MODULE PersistentStorageBridge ------------------------
(*
 * Bridge model between libgrammstein's Google Books importer/checkpoint
 * lifecycle and libdictenstein's persistent trie/WAL/checkpoint contracts.
 *
 * The dependency contracts are recorded in:
 *   formal/dependencies/libdictenstein-contracts.md
 *
 * This model verifies the libgrammstein-side composition rule: a new importer
 * checkpoint claim can be published only after the underlying data, checkpoint
 * metadata, optional vocabulary state, and worker-drain evidence are durable.
 *)

EXTENDS TLC

CONSTANTS
    \* @type: Bool;
    NeedVocab,
    \* @type: Bool;
    Sharded

VARIABLES
    \* @type: Bool;
    prefix_committed,
    \* @type: Bool;
    data_visible,
    \* @type: Bool;
    data_durable,
    \* @type: Bool;
    sync_failed,
    \* @type: Bool;
    metadata_durable,
    \* @type: Bool;
    metadata_write_failed,
    \* @type: Bool;
    vocab_visible,
    \* @type: Bool;
    vocab_durable,
    \* @type: Bool;
    workers_drained,
    \* @type: Bool;
    graceful_cancel,
    \* @type: Bool;
    force_quit,
    \* @type: Bool;
    drain_failed,
    \* @type: Bool;
    checkpoint_claimed,
    \* @type: Bool;
    crashed,
    \* @type: Bool;
    reopened,
    \* @type: Bool;
    recoverable_prefix

ModelConstantsOK ==
    /\ NeedVocab \in BOOLEAN
    /\ Sharded \in BOOLEAN

TypeOK ==
    /\ ModelConstantsOK
    /\ prefix_committed \in BOOLEAN
    /\ data_visible \in BOOLEAN
    /\ data_durable \in BOOLEAN
    /\ sync_failed \in BOOLEAN
    /\ metadata_durable \in BOOLEAN
    /\ metadata_write_failed \in BOOLEAN
    /\ vocab_visible \in BOOLEAN
    /\ vocab_durable \in BOOLEAN
    /\ workers_drained \in BOOLEAN
    /\ graceful_cancel \in BOOLEAN
    /\ force_quit \in BOOLEAN
    /\ drain_failed \in BOOLEAN
    /\ checkpoint_claimed \in BOOLEAN
    /\ crashed \in BOOLEAN
    /\ reopened \in BOOLEAN
    /\ recoverable_prefix \in BOOLEAN

Init ==
    /\ prefix_committed = FALSE
    /\ data_visible = FALSE
    /\ data_durable = FALSE
    /\ sync_failed = FALSE
    /\ metadata_durable = FALSE
    /\ metadata_write_failed = FALSE
    /\ vocab_visible = FALSE
    /\ vocab_durable = FALSE
    /\ workers_drained = FALSE
    /\ graceful_cancel = FALSE
    /\ force_quit = FALSE
    /\ drain_failed = FALSE
    /\ checkpoint_claimed = FALSE
    /\ crashed = FALSE
    /\ reopened = FALSE
    /\ recoverable_prefix = FALSE

Active ==
    /\ ~crashed
    /\ ~reopened
    /\ ~force_quit
    /\ ~drain_failed
    /\ ~checkpoint_claimed

DependencyEvidenceReady ==
    /\ prefix_committed
    /\ data_visible
    /\ data_durable
    /\ workers_drained
    /\ ~metadata_write_failed
    /\ (~NeedVocab \/ (vocab_visible /\ vocab_durable))

CommitPrefixOk ==
    /\ Active
    /\ ~prefix_committed
    /\ prefix_committed' = TRUE
    /\ data_visible' = TRUE
    /\ UNCHANGED <<data_durable, sync_failed, metadata_durable,
                   metadata_write_failed, vocab_visible, vocab_durable,
                   workers_drained, graceful_cancel, force_quit, drain_failed,
                   checkpoint_claimed, crashed, reopened, recoverable_prefix>>

SyncDataOk ==
    /\ Active
    /\ prefix_committed
    /\ data_visible
    /\ ~data_durable
    /\ data_durable' = TRUE
    /\ UNCHANGED <<prefix_committed, data_visible, sync_failed,
                   metadata_durable, metadata_write_failed, vocab_visible,
                   vocab_durable, workers_drained, graceful_cancel, force_quit,
                   drain_failed, checkpoint_claimed, crashed, reopened,
                   recoverable_prefix>>

SyncDataFail ==
    /\ Active
    /\ prefix_committed
    /\ data_visible
    /\ ~data_durable
    /\ sync_failed' = TRUE
    /\ UNCHANGED <<prefix_committed, data_visible, data_durable,
                   metadata_durable, metadata_write_failed, vocab_visible,
                   vocab_durable, workers_drained, graceful_cancel, force_quit,
                   drain_failed, checkpoint_claimed, crashed, reopened,
                   recoverable_prefix>>

CheckpointVocabOk ==
    /\ Active
    /\ NeedVocab
    /\ ~vocab_durable
    /\ vocab_visible' = TRUE
    /\ vocab_durable' = TRUE
    /\ UNCHANGED <<prefix_committed, data_visible, data_durable, sync_failed,
                   metadata_durable, metadata_write_failed, workers_drained,
                   graceful_cancel, force_quit, drain_failed,
                   checkpoint_claimed, crashed, reopened, recoverable_prefix>>

DrainWorkers ==
    /\ Active
    /\ prefix_committed
    /\ ~workers_drained
    /\ workers_drained' = TRUE
    /\ UNCHANGED <<prefix_committed, data_visible, data_durable, sync_failed,
                   metadata_durable, metadata_write_failed, vocab_visible,
                   vocab_durable, graceful_cancel, force_quit, drain_failed,
                   checkpoint_claimed, crashed, reopened, recoverable_prefix>>

SaveCheckpointOk ==
    /\ Active
    /\ DependencyEvidenceReady
    /\ checkpoint_claimed' = TRUE
    /\ metadata_durable' = TRUE
    /\ UNCHANGED <<prefix_committed, data_visible, data_durable, sync_failed,
                   metadata_write_failed, vocab_visible, vocab_durable,
                   workers_drained, graceful_cancel, force_quit, drain_failed,
                   crashed, reopened, recoverable_prefix>>

SaveCheckpointFail ==
    /\ Active
    /\ prefix_committed
    /\ workers_drained
    /\ ~metadata_durable
    /\ metadata_write_failed' = TRUE
    /\ UNCHANGED <<prefix_committed, data_visible, data_durable, sync_failed,
                   metadata_durable, vocab_visible, vocab_durable,
                   workers_drained, graceful_cancel, force_quit, drain_failed,
                   checkpoint_claimed, crashed, reopened, recoverable_prefix>>

GracefulCancelCheckpoint ==
    /\ Active
    /\ DependencyEvidenceReady
    /\ graceful_cancel' = TRUE
    /\ checkpoint_claimed' = TRUE
    /\ metadata_durable' = TRUE
    /\ UNCHANGED <<prefix_committed, data_visible, data_durable, sync_failed,
                   metadata_write_failed, vocab_visible, vocab_durable,
                   workers_drained, force_quit, drain_failed, crashed,
                   reopened, recoverable_prefix>>

ForceQuitAbort ==
    /\ ~crashed
    /\ ~reopened
    /\ ~checkpoint_claimed
    /\ ~force_quit
    /\ force_quit' = TRUE
    /\ UNCHANGED <<prefix_committed, data_visible, data_durable, sync_failed,
                   metadata_durable, metadata_write_failed, vocab_visible,
                   vocab_durable, workers_drained, graceful_cancel,
                   drain_failed, checkpoint_claimed, crashed, reopened,
                   recoverable_prefix>>

DrainFailureAbort ==
    /\ ~crashed
    /\ ~reopened
    /\ ~checkpoint_claimed
    /\ ~drain_failed
    /\ drain_failed' = TRUE
    /\ UNCHANGED <<prefix_committed, data_visible, data_durable, sync_failed,
                   metadata_durable, metadata_write_failed, vocab_visible,
                   vocab_durable, workers_drained, graceful_cancel, force_quit,
                   checkpoint_claimed, crashed, reopened, recoverable_prefix>>

CrashAndReopen ==
    /\ ~crashed
    /\ ~reopened
    /\ crashed' = TRUE
    /\ reopened' = TRUE
    /\ recoverable_prefix' = (
        /\ checkpoint_claimed
        /\ data_visible
        /\ data_durable
        /\ metadata_durable
        /\ (~NeedVocab \/ (vocab_visible /\ vocab_durable)))
    /\ UNCHANGED <<prefix_committed, data_visible, data_durable, sync_failed,
                   metadata_durable, metadata_write_failed, vocab_visible,
                   vocab_durable, workers_drained, graceful_cancel, force_quit,
                   drain_failed, checkpoint_claimed>>

Next ==
    \/ CommitPrefixOk
    \/ SyncDataOk
    \/ SyncDataFail
    \/ CheckpointVocabOk
    \/ DrainWorkers
    \/ SaveCheckpointOk
    \/ SaveCheckpointFail
    \/ GracefulCancelCheckpoint
    \/ ForceQuitAbort
    \/ DrainFailureAbort
    \/ CrashAndReopen

\* @type: <<Bool, Bool, Bool, Bool, Bool, Bool, Bool, Bool, Bool, Bool, Bool, Bool, Bool, Bool, Bool, Bool>>;
vars == <<prefix_committed, data_visible, data_durable, sync_failed,
          metadata_durable, metadata_write_failed, vocab_visible,
          vocab_durable, workers_drained, graceful_cancel, force_quit,
          drain_failed, checkpoint_claimed, crashed, reopened,
          recoverable_prefix>>

Spec == Init /\ [][Next]_vars

ClaimRequiresDurableEvidence ==
    checkpoint_claimed =>
        /\ prefix_committed
        /\ data_visible
        /\ data_durable
        /\ metadata_durable
        /\ workers_drained
        /\ ~metadata_write_failed
        /\ ~force_quit
        /\ ~drain_failed
        /\ (~NeedVocab \/ (vocab_visible /\ vocab_durable))

RecoveredPrefixHasCheckpointEvidence ==
    recoverable_prefix =>
        /\ reopened
        /\ checkpoint_claimed
        /\ data_visible
        /\ data_durable
        /\ metadata_durable
        /\ (~NeedVocab \/ (vocab_visible /\ vocab_durable))

FailedSyncCannotClaimWithoutDurableData ==
    (sync_failed /\ ~data_durable) => ~checkpoint_claimed

ForceQuitPublishesNoNewClaim ==
    force_quit => ~checkpoint_claimed

DrainFailurePublishesNoNewClaim ==
    drain_failed => ~checkpoint_claimed

CheckpointAfterWorkerDrain ==
    checkpoint_claimed => workers_drained

MetadataFailureDoesNotClaim ==
    (metadata_write_failed /\ ~metadata_durable) => ~checkpoint_claimed

DataDurableRequiresVisible ==
    data_durable => data_visible

VocabularyDurableRequiresVisible ==
    vocab_durable => vocab_visible

VocabularyStableWhenRecovered ==
    (NeedVocab /\ recoverable_prefix) => vocab_visible /\ vocab_durable

SingleTrieClaimSharesBoundary ==
    (~Sharded /\ checkpoint_claimed) => data_durable /\ metadata_durable

ShardedClaimHasAuxiliaryMetadata ==
    (Sharded /\ checkpoint_claimed) => metadata_durable

Safety ==
    /\ TypeOK
    /\ ClaimRequiresDurableEvidence
    /\ RecoveredPrefixHasCheckpointEvidence
    /\ FailedSyncCannotClaimWithoutDurableData
    /\ ForceQuitPublishesNoNewClaim
    /\ DrainFailurePublishesNoNewClaim
    /\ CheckpointAfterWorkerDrain
    /\ MetadataFailureDoesNotClaim
    /\ DataDurableRequiresVisible
    /\ VocabularyDurableRequiresVisible
    /\ VocabularyStableWhenRecovered
    /\ SingleTrieClaimSharesBoundary
    /\ ShardedClaimHasAuxiliaryMetadata

=============================================================================
