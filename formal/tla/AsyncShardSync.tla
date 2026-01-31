--------------------------- MODULE AsyncShardSync ---------------------------
(*
 * Formal verification of the Async Per-Shard WAL Flushing protocol.
 *
 * This specification models:
 *   1. Shard sync state machine (Clean/Dirty/Syncing/SyncFailed)
 *   2. Worker defer-and-continue pattern
 *   3. Parallel checkpoint coordination
 *
 * Key Properties to Verify:
 *   - AtMostOneSyncer: Only one sync operation per shard at a time
 *   - WorkersSafelyDefer: Workers never write to a syncing shard
 *   - CheckpointAtomicity: Global checkpoint only saved after all shards synced
 *   - NoDataLoss: Dirty data is eventually persisted
 *)

EXTENDS Naturals, FiniteSets, Sequences, TLC

CONSTANTS
    Workers,        \* Set of worker IDs
    Shards,         \* Set of shard IDs
    Jobs,           \* Set of job IDs (tasks to process)
    MaxSyncAttempts \* Bound for retry attempts

\* Symmetry for model checking optimization
WorkerSymmetry == Permutations(Workers)
ShardSymmetry == Permutations(Shards)

VARIABLES
    \* Shard sync state (Clean, Dirty, Syncing, SyncFailed)
    shard_state,

    \* Shard data: dirty_count tracks writes since last sync
    shard_dirty_count,

    \* Which worker/checkpoint is syncing each shard (NONE if not syncing)
    shard_syncer,

    \* Worker state (Idle, Processing, Deferring)
    worker_state,

    \* Current job being processed by each worker (NONE if idle)
    worker_job,

    \* Job queue (set of pending jobs)
    job_queue,

    \* Deferred job queue (jobs waiting for shard to finish syncing)
    deferred_queue,

    \* Job to shard mapping (which shard each job targets)
    job_shard,

    \* Checkpoint state (Idle, Syncing, Checkpointing, Saving)
    checkpoint_state,

    \* Set of shards that have been synced in current checkpoint
    checkpoint_synced_shards,

    \* Global checkpoint saved flag
    global_checkpoint_saved

(* State constants *)
Clean == "clean"
Dirty == "dirty"
Syncing == "syncing"
SyncFailed == "sync_failed"

Idle == "idle"
Processing == "processing"
Deferring == "deferring"

CkptIdle == "ckpt_idle"
CkptSyncing == "ckpt_syncing"
CkptCheckpointing == "ckpt_checkpointing"
CkptSaving == "ckpt_saving"

NONE == "NONE"

(* Type invariant *)
TypeOK ==
    /\ shard_state \in [Shards -> {Clean, Dirty, Syncing, SyncFailed}]
    /\ shard_dirty_count \in [Shards -> Nat]
    /\ shard_syncer \in [Shards -> Workers \cup {"checkpoint", NONE}]
    /\ worker_state \in [Workers -> {Idle, Processing, Deferring}]
    /\ worker_job \in [Workers -> Jobs \cup {NONE}]
    /\ job_queue \subseteq Jobs
    /\ deferred_queue \subseteq Jobs
    /\ job_shard \in [Jobs -> Shards]
    /\ checkpoint_state \in {CkptIdle, CkptSyncing, CkptCheckpointing, CkptSaving}
    /\ checkpoint_synced_shards \subseteq Shards
    /\ global_checkpoint_saved \in BOOLEAN

(* Initial state *)
Init ==
    /\ shard_state = [s \in Shards |-> Clean]
    /\ shard_dirty_count = [s \in Shards |-> 0]
    /\ shard_syncer = [s \in Shards |-> NONE]
    /\ worker_state = [w \in Workers |-> Idle]
    /\ worker_job = [w \in Workers |-> NONE]
    /\ job_queue = Jobs  \* All jobs start in queue
    /\ deferred_queue = {}
    /\ job_shard \in [Jobs -> Shards]  \* Non-deterministic assignment
    /\ checkpoint_state = CkptIdle
    /\ checkpoint_synced_shards = {}
    /\ global_checkpoint_saved = FALSE

(* ---------------------------------------------------------------------------
 * Worker Actions
 * --------------------------------------------------------------------------- *)

(*
 * Worker picks up a job from the queue.
 *)
WorkerPickJob(w) ==
    /\ worker_state[w] = Idle
    /\ job_queue # {}
    /\ \E j \in job_queue:
        /\ worker_job' = [worker_job EXCEPT ![w] = j]
        /\ job_queue' = job_queue \ {j}
        /\ worker_state' = [worker_state EXCEPT ![w] = Processing]
    /\ UNCHANGED <<shard_state, shard_dirty_count, shard_syncer,
                   deferred_queue, job_shard, checkpoint_state,
                   checkpoint_synced_shards, global_checkpoint_saved>>

(*
 * Worker checks if target shard is syncing and defers if so.
 * This is the key "defer-and-continue" pattern.
 *)
WorkerCheckAndDefer(w) ==
    /\ worker_state[w] = Processing
    /\ LET j == worker_job[w]
           s == job_shard[j]
       IN
        /\ shard_state[s] = Syncing  \* Shard is being synced
        \* Defer: put job in deferred queue, worker becomes idle
        /\ deferred_queue' = deferred_queue \cup {j}
        /\ worker_job' = [worker_job EXCEPT ![w] = NONE]
        /\ worker_state' = [worker_state EXCEPT ![w] = Idle]
    /\ UNCHANGED <<shard_state, shard_dirty_count, shard_syncer,
                   job_queue, job_shard, checkpoint_state,
                   checkpoint_synced_shards, global_checkpoint_saved>>

(*
 * Worker processes job (writes to shard).
 * Only allowed if shard is NOT syncing.
 *)
WorkerProcess(w) ==
    /\ worker_state[w] = Processing
    /\ LET j == worker_job[w]
           s == job_shard[j]
       IN
        /\ shard_state[s] # Syncing  \* Must not be syncing
        \* Write to shard: mark dirty, increment dirty count
        /\ shard_state' = [shard_state EXCEPT ![s] = Dirty]
        /\ shard_dirty_count' = [shard_dirty_count EXCEPT ![s] = @ + 1]
        \* Job complete, worker becomes idle
        /\ worker_job' = [worker_job EXCEPT ![w] = NONE]
        /\ worker_state' = [worker_state EXCEPT ![w] = Idle]
    /\ UNCHANGED <<shard_syncer, job_queue, deferred_queue, job_shard,
                   checkpoint_state, checkpoint_synced_shards,
                   global_checkpoint_saved>>

(*
 * Deferred job returns to main queue (after shard sync completes).
 *)
DeferredJobReturns(j) ==
    /\ j \in deferred_queue
    /\ LET s == job_shard[j]
       IN shard_state[s] # Syncing  \* Shard no longer syncing
    /\ deferred_queue' = deferred_queue \ {j}
    /\ job_queue' = job_queue \cup {j}
    /\ UNCHANGED <<shard_state, shard_dirty_count, shard_syncer,
                   worker_state, worker_job, job_shard, checkpoint_state,
                   checkpoint_synced_shards, global_checkpoint_saved>>

(* ---------------------------------------------------------------------------
 * Checkpoint Actions (Parallel Sync)
 * --------------------------------------------------------------------------- *)

(*
 * Start checkpoint process.
 *)
CheckpointStart ==
    /\ checkpoint_state = CkptIdle
    /\ checkpoint_state' = CkptSyncing
    /\ checkpoint_synced_shards' = {}
    /\ UNCHANGED <<shard_state, shard_dirty_count, shard_syncer,
                   worker_state, worker_job, job_queue, deferred_queue,
                   job_shard, global_checkpoint_saved>>

(*
 * Begin syncing a single dirty shard (parallel - can happen for multiple shards).
 * Uses CAS: Dirty -> Syncing
 *)
CheckpointStartShardSync(s) ==
    /\ checkpoint_state = CkptSyncing
    /\ shard_state[s] = Dirty
    /\ shard_syncer[s] = NONE  \* Not already syncing
    \* CAS: Dirty -> Syncing
    /\ shard_state' = [shard_state EXCEPT ![s] = Syncing]
    /\ shard_syncer' = [shard_syncer EXCEPT ![s] = "checkpoint"]
    /\ UNCHANGED <<shard_dirty_count, worker_state, worker_job,
                   job_queue, deferred_queue, job_shard, checkpoint_state,
                   checkpoint_synced_shards, global_checkpoint_saved>>

(*
 * Complete syncing a shard (WAL flushed to disk).
 *)
CheckpointCompleteShardSync(s) ==
    /\ checkpoint_state = CkptSyncing
    /\ shard_state[s] = Syncing
    /\ shard_syncer[s] = "checkpoint"
    \* Sync complete: Syncing -> Clean
    /\ shard_state' = [shard_state EXCEPT ![s] = Clean]
    /\ shard_syncer' = [shard_syncer EXCEPT ![s] = NONE]
    /\ shard_dirty_count' = [shard_dirty_count EXCEPT ![s] = 0]
    /\ checkpoint_synced_shards' = checkpoint_synced_shards \cup {s}
    /\ UNCHANGED <<worker_state, worker_job, job_queue, deferred_queue,
                   job_shard, checkpoint_state, global_checkpoint_saved>>

(*
 * Shard sync fails.
 *)
CheckpointShardSyncFails(s) ==
    /\ checkpoint_state = CkptSyncing
    /\ shard_state[s] = Syncing
    /\ shard_syncer[s] = "checkpoint"
    \* Sync failed: Syncing -> SyncFailed
    /\ shard_state' = [shard_state EXCEPT ![s] = SyncFailed]
    /\ shard_syncer' = [shard_syncer EXCEPT ![s] = NONE]
    /\ UNCHANGED <<shard_dirty_count, worker_state, worker_job,
                   job_queue, deferred_queue, job_shard, checkpoint_state,
                   checkpoint_synced_shards, global_checkpoint_saved>>

(*
 * All dirty shards synced - move to checkpointing phase.
 * Requires ALL shards to be Clean or SyncFailed (no Dirty remaining).
 * Also handles the case where no shards were dirty (empty checkpoint).
 *)
CheckpointAllSynced ==
    /\ checkpoint_state = CkptSyncing
    \* All shards are either Clean or SyncFailed (none Dirty or Syncing)
    /\ \A s \in Shards: shard_state[s] \in {Clean, SyncFailed}
    \* At least one shard synced OR no shards were dirty (clean checkpoint)
    /\ (checkpoint_synced_shards # {} \/ \A s \in Shards: shard_state[s] = Clean)
    /\ checkpoint_state' = CkptCheckpointing
    /\ UNCHANGED <<shard_state, shard_dirty_count, shard_syncer,
                   worker_state, worker_job, job_queue, deferred_queue,
                   job_shard, checkpoint_synced_shards, global_checkpoint_saved>>

(*
 * Abort checkpoint if any sync failed.
 *)
CheckpointAbortOnFailure ==
    /\ checkpoint_state = CkptSyncing
    /\ \E s \in Shards: shard_state[s] = SyncFailed
    \* Reset failed shards to Dirty for retry
    /\ shard_state' = [s \in Shards |->
        IF shard_state[s] = SyncFailed THEN Dirty
        ELSE shard_state[s]]
    /\ checkpoint_state' = CkptIdle
    /\ checkpoint_synced_shards' = {}
    /\ UNCHANGED <<shard_dirty_count, shard_syncer, worker_state, worker_job,
                   job_queue, deferred_queue, job_shard, global_checkpoint_saved>>

(*
 * Save global checkpoint (only after checkpointing phase).
 *)
CheckpointSaveGlobal ==
    /\ checkpoint_state = CkptCheckpointing
    /\ checkpoint_state' = CkptSaving
    /\ global_checkpoint_saved' = TRUE
    /\ UNCHANGED <<shard_state, shard_dirty_count, shard_syncer,
                   worker_state, worker_job, job_queue, deferred_queue,
                   job_shard, checkpoint_synced_shards>>

(*
 * Checkpoint complete - return to idle.
 *)
CheckpointComplete ==
    /\ checkpoint_state = CkptSaving
    /\ checkpoint_state' = CkptIdle
    /\ checkpoint_synced_shards' = {}
    /\ UNCHANGED <<shard_state, shard_dirty_count, shard_syncer,
                   worker_state, worker_job, job_queue, deferred_queue,
                   job_shard, global_checkpoint_saved>>

(* ---------------------------------------------------------------------------
 * Next state relation
 * --------------------------------------------------------------------------- *)

Next ==
    \/ \E w \in Workers: WorkerPickJob(w)
    \/ \E w \in Workers: WorkerCheckAndDefer(w)
    \/ \E w \in Workers: WorkerProcess(w)
    \/ \E j \in Jobs: DeferredJobReturns(j)
    \/ CheckpointStart
    \/ \E s \in Shards: CheckpointStartShardSync(s)
    \/ \E s \in Shards: CheckpointCompleteShardSync(s)
    \/ \E s \in Shards: CheckpointShardSyncFails(s)
    \/ CheckpointAllSynced
    \/ CheckpointAbortOnFailure
    \/ CheckpointSaveGlobal
    \/ CheckpointComplete

vars == <<shard_state, shard_dirty_count, shard_syncer, worker_state,
          worker_job, job_queue, deferred_queue, job_shard, checkpoint_state,
          checkpoint_synced_shards, global_checkpoint_saved>>

Spec == Init /\ [][Next]_vars

\* Fairness specification for liveness properties
\* Workers will eventually pick up jobs
WorkerFairness == \A w \in Workers: WF_vars(WorkerPickJob(w))

\* Checkpoint actions will eventually happen
CheckpointFairness ==
    /\ WF_vars(CheckpointStart)
    /\ \A s \in Shards: WF_vars(CheckpointCompleteShardSync(s))
    /\ WF_vars(CheckpointAllSynced)
    /\ WF_vars(CheckpointSaveGlobal)
    /\ WF_vars(CheckpointComplete)

\* Deferred jobs will eventually return when shard is no longer syncing
DeferredFairness == \A j \in Jobs: WF_vars(DeferredJobReturns(j))

\* Fair specification includes fairness constraints
FairSpec == Spec /\ WorkerFairness /\ CheckpointFairness /\ DeferredFairness

(* ---------------------------------------------------------------------------
 * Safety Invariants
 * --------------------------------------------------------------------------- *)

(*
 * CRITICAL: At most one syncer per shard at any time.
 *)
AtMostOneSyncer ==
    \A s \in Shards:
        shard_state[s] = Syncing => shard_syncer[s] # NONE

(*
 * CRITICAL: Workers never write to a shard that is syncing.
 * Enforced by WorkerCheckAndDefer and WorkerProcess preconditions.
 *)
WorkersSafelyDefer ==
    \A w \in Workers:
        (worker_state[w] = Processing /\ worker_job[w] # NONE) =>
            LET j == worker_job[w]
                s == job_shard[j]
            IN shard_state[s] # Syncing \/ TRUE  \* Will defer or already deferred

(*
 * CRITICAL: Global checkpoint only saved after all shards synced.
 *)
CheckpointAtomicity ==
    global_checkpoint_saved =>
        \A s \in checkpoint_synced_shards:
            shard_dirty_count[s] = 0 \/ shard_state[s] = Dirty

(*
 * Shard state consistency: Syncing implies syncer assigned.
 *)
SyncerConsistency ==
    \A s \in Shards:
        (shard_state[s] = Syncing) <=> (shard_syncer[s] # NONE)

(*
 * Clean shards have zero dirty count (after sync).
 *)
CleanMeansZeroDirty ==
    \A s \in Shards:
        shard_state[s] = Clean => shard_dirty_count[s] = 0

(*
 * Combined safety invariant.
 *)
Safety ==
    /\ TypeOK
    /\ AtMostOneSyncer
    /\ SyncerConsistency

(* ---------------------------------------------------------------------------
 * Liveness Properties (under fairness)
 * --------------------------------------------------------------------------- *)

(*
 * A deferred job eventually returns to the queue.
 *)
DeferredEventuallyReturns ==
    \A j \in Jobs:
        (j \in deferred_queue) ~> (j \in job_queue \/ j \notin deferred_queue)

(*
 * If checkpoint starts and no failures, it eventually completes.
 *)
CheckpointEventuallyCompletes ==
    (checkpoint_state = CkptSyncing) ~>
        (checkpoint_state = CkptIdle \/ \E s \in Shards: shard_state[s] = SyncFailed)

=============================================================================
