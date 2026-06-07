--------------------------- MODULE AsyncShardSync ---------------------------
(*
 * Formal verification of the lock-free per-shard sync coordinator.
 *
 * This specification models the ShardSyncCoordinator state machine in
 * src/sources/google_books/sharding/shard.rs under the lock-free overlay write
 * path, where worker `increment_cas` writes proceed CONCURRENTLY with an
 * in-flight checkpoint sync. There is no defer-and-continue: workers never
 * block or queue behind a syncing shard.
 *
 * State machine (mirrors the Rust coordinator one-for-one):
 *   - mark_dirty      Clean   -> Dirty       (CAS; a NO-OP in any other state)
 *   - try_start_sync  Dirty   -> Syncing     (CAS; establishes the single syncer)
 *   - complete_sync   Syncing -> Clean       (unconditional)
 *   - fail_sync       Syncing -> SyncFailed
 *
 * Because mark_dirty only fires Clean -> Dirty (shard.rs:212), a write that
 * lands while a shard is Syncing leaves shard_state unchanged -- the resident
 * overlay absorbs the write and the coordinator state machine does not register
 * it. Whether that during-sync overlay write is captured by the checkpoint's
 * RCU snapshot is NOT modeled here; it is delegated to libdictenstein's
 * LockFreeDurableCheckpoint contract (formal/dependencies/libdictenstein-contracts.md).
 *
 * Key Properties to Verify:
 *   - AtMostOneSyncer:     Only one sync operation per shard at a time
 *   - SyncerConsistency:   A shard is Syncing iff a syncer is assigned to it
 *   - CheckpointAtomicity: Global checkpoint only saved after all targets synced
 *   - CleanMeansZeroDirty: A Clean shard carries no pending dirty writes
 *   - JobPartition:        Every job is queued, in-flight, or completed (no loss)
 *)

EXTENDS Naturals, FiniteSets, Sequences, TLC

CONSTANTS
    \* @type: Set(Str);
    Workers,        \* Set of worker IDs
    \* @type: Set(Str);
    Shards,         \* Set of shard IDs
    \* @type: Set(Str);
    Jobs,           \* Set of job IDs (tasks to process)
    \* @type: Int;
    MaxSyncAttempts \* Bound for retry attempts

\* Symmetry for model checking optimization
WorkerSymmetry == Permutations(Workers)
ShardSymmetry == Permutations(Shards)

VARIABLES
    \* Shard sync state (Clean, Dirty, Syncing, SyncFailed)
    \* @type: Str -> Str;
    shard_state,

    \* Shard data: dirty_count tracks writes since last sync
    \* @type: Str -> Int;
    shard_dirty_count,

    \* Which worker/checkpoint is syncing each shard (NONE if not syncing)
    \* @type: Str -> Str;
    shard_syncer,

    \* Worker state (Idle, Processing)
    \* @type: Str -> Str;
    worker_state,

    \* Current job being processed by each worker (NONE if idle)
    \* @type: Str -> Str;
    worker_job,

    \* Job queue (set of pending jobs)
    \* @type: Set(Str);
    job_queue,

    \* Jobs that have completed processing
    \* @type: Set(Str);
    completed_jobs,

    \* Job to shard mapping (which shard each job targets)
    \* @type: Str -> Str;
    job_shard,

    \* Checkpoint state (Idle, Syncing, Checkpointing, Saving)
    \* @type: Str;
    checkpoint_state,

    \* Set of shards that have been synced in current checkpoint
    \* @type: Set(Str);
    checkpoint_synced_shards,

    \* Dirty shards captured when the current checkpoint began
    \* @type: Set(Str);
    checkpoint_target_shards,

    \* Snapshot of target/synced shards at the last global checkpoint save
    \* @type: Set(Str);
    last_saved_target_shards,
    \* @type: Set(Str);
    last_saved_synced_shards,

    \* Global checkpoint saved flag
    \* @type: Bool;
    global_checkpoint_saved

(* State constants *)
Clean == "clean"
Dirty == "dirty"
Syncing == "syncing"
SyncFailed == "sync_failed"

Idle == "idle"
Processing == "processing"

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
    /\ worker_state \in [Workers -> {Idle, Processing}]
    /\ worker_job \in [Workers -> Jobs \cup {NONE}]
    /\ job_queue \subseteq Jobs
    /\ completed_jobs \subseteq Jobs
    /\ job_shard \in [Jobs -> Shards]
    /\ checkpoint_state \in {CkptIdle, CkptSyncing, CkptCheckpointing, CkptSaving}
    /\ checkpoint_synced_shards \subseteq Shards
    /\ checkpoint_target_shards \subseteq Shards
    /\ last_saved_target_shards \subseteq Shards
    /\ last_saved_synced_shards \subseteq Shards
    /\ global_checkpoint_saved \in BOOLEAN

(* Initial state *)
Init ==
    /\ shard_state = [s \in Shards |-> Clean]
    /\ shard_dirty_count = [s \in Shards |-> 0]
    /\ shard_syncer = [s \in Shards |-> NONE]
    /\ worker_state = [w \in Workers |-> Idle]
    /\ worker_job = [w \in Workers |-> NONE]
    /\ job_queue = Jobs  \* All jobs start in queue
    /\ completed_jobs = {}
    /\ job_shard \in [Jobs -> Shards]  \* Non-deterministic assignment
    /\ checkpoint_state = CkptIdle
    /\ checkpoint_synced_shards = {}
    /\ checkpoint_target_shards = {}
    /\ last_saved_target_shards = {}
    /\ last_saved_synced_shards = {}
    /\ global_checkpoint_saved = FALSE

(* ---------------------------------------------------------------------------
 * Worker Actions
 * --------------------------------------------------------------------------- *)

(*
 * Worker picks up a job from the queue.
 *)
WorkerPickJobBy(w, j) ==
    /\ worker_state[w] = Idle
    /\ worker_job[w] = NONE
    /\ j \in job_queue
    /\ worker_job' = [worker_job EXCEPT ![w] = j]
    /\ job_queue' = job_queue \ {j}
    /\ worker_state' = [worker_state EXCEPT ![w] = Processing]
    /\ UNCHANGED <<shard_state, shard_dirty_count, shard_syncer,
                   completed_jobs, job_shard, checkpoint_state,
                   checkpoint_synced_shards, checkpoint_target_shards,
                   last_saved_target_shards, last_saved_synced_shards,
                   global_checkpoint_saved>>

WorkerPickJob(w) ==
    \E j \in job_queue: WorkerPickJobBy(w, j)

(*
 * Worker processes job (writes to its target shard via the lock-free overlay).
 *
 * Unlike the retired defer-and-continue model, this fires REGARDLESS of the
 * shard's sync state: the overlay write proceeds concurrently with any
 * in-flight sync. The shard-state effect mirrors mark_dirty exactly --
 * Clean -> Dirty, and a NO-OP (state preserved) in Dirty / Syncing / SyncFailed.
 *)
WorkerProcessJob(w, j) ==
    /\ worker_state[w] = Processing
    /\ worker_job[w] = j
    /\ j \in Jobs
    /\ LET s == job_shard[j]
       IN
        \* mark_dirty semantics: only Clean -> Dirty; otherwise leave as-is.
        /\ shard_state' = [shard_state EXCEPT
                            ![s] = IF @ = Clean THEN Dirty ELSE @]
        /\ shard_dirty_count' = [shard_dirty_count EXCEPT ![s] = @ + 1]
        \* Job complete, worker becomes idle
        /\ worker_job' = [worker_job EXCEPT ![w] = NONE]
        /\ worker_state' = [worker_state EXCEPT ![w] = Idle]
        /\ completed_jobs' = completed_jobs \cup {j}
        /\ global_checkpoint_saved' = FALSE
    /\ UNCHANGED <<shard_syncer, job_queue, job_shard,
                   checkpoint_state, checkpoint_synced_shards,
                   checkpoint_target_shards, last_saved_target_shards,
                   last_saved_synced_shards>>

WorkerProcess(w) ==
    \E j \in Jobs: WorkerProcessJob(w, j)

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
    /\ checkpoint_target_shards' = {s \in Shards: shard_state[s] = Dirty}
    /\ UNCHANGED <<shard_state, shard_dirty_count, shard_syncer,
                   worker_state, worker_job, job_queue,
                   completed_jobs, job_shard, last_saved_target_shards,
                   last_saved_synced_shards, global_checkpoint_saved>>

(*
 * Begin syncing a single dirty shard (parallel - can happen for multiple shards).
 * Uses CAS: Dirty -> Syncing
 *)
CheckpointStartShardSync(s) ==
    /\ checkpoint_state = CkptSyncing
    /\ shard_state[s] = Dirty
    /\ s \in checkpoint_target_shards
    /\ shard_syncer[s] = NONE  \* Not already syncing
    \* CAS: Dirty -> Syncing
    /\ shard_state' = [shard_state EXCEPT ![s] = Syncing]
    /\ shard_syncer' = [shard_syncer EXCEPT ![s] = "checkpoint"]
    /\ UNCHANGED <<shard_dirty_count, worker_state, worker_job,
                   job_queue, job_shard, checkpoint_state,
                   checkpoint_synced_shards, checkpoint_target_shards,
                   completed_jobs, last_saved_target_shards,
                   last_saved_synced_shards, global_checkpoint_saved>>

(*
 * Complete syncing a shard (WAL flushed to disk).
 *
 * Mirrors complete_sync: Syncing -> Clean UNCONDITIONALLY, dirty_count reset.
 * A write that arrived while this shard was Syncing did not change shard_state
 * (mark_dirty no-op), so it is subsumed here; its overlay durability is a
 * libdictenstein LockFreeDurableCheckpoint concern, not a coordinator concern.
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
    /\ UNCHANGED <<worker_state, worker_job, job_queue,
                   completed_jobs, job_shard, checkpoint_state,
                   checkpoint_target_shards, last_saved_target_shards,
                   last_saved_synced_shards, global_checkpoint_saved>>

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
                   job_queue, job_shard, checkpoint_state,
                   checkpoint_synced_shards, checkpoint_target_shards,
                   completed_jobs, last_saved_target_shards,
                   last_saved_synced_shards, global_checkpoint_saved>>

(*
 * All dirty shards synced - move to checkpointing phase.
 * Requires ALL shards to be Clean or SyncFailed (no Dirty remaining).
 * Also handles the case where no shards were dirty (empty checkpoint).
 *)
CheckpointAllSynced ==
    /\ checkpoint_state = CkptSyncing
    \* Every shard targeted at checkpoint start has finished syncing successfully.
    /\ checkpoint_target_shards \subseteq checkpoint_synced_shards
    /\ \A s \in Shards: shard_syncer[s] = NONE
    /\ \A s \in checkpoint_target_shards: shard_state[s] # SyncFailed
    /\ checkpoint_state' = CkptCheckpointing
    /\ UNCHANGED <<shard_state, shard_dirty_count, shard_syncer,
                   worker_state, worker_job, job_queue,
                   completed_jobs, job_shard, checkpoint_synced_shards,
                   checkpoint_target_shards, last_saved_target_shards,
                   last_saved_synced_shards, global_checkpoint_saved>>

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
    /\ checkpoint_target_shards' = {}
    /\ UNCHANGED <<shard_dirty_count, shard_syncer, worker_state, worker_job,
                   job_queue, completed_jobs, job_shard,
                   last_saved_target_shards, last_saved_synced_shards,
                   global_checkpoint_saved>>

(*
 * Save global checkpoint (only after checkpointing phase).
 *)
CheckpointSaveGlobal ==
    /\ checkpoint_state = CkptCheckpointing
    /\ checkpoint_state' = CkptSaving
    /\ global_checkpoint_saved' = TRUE
    /\ last_saved_target_shards' = checkpoint_target_shards
    /\ last_saved_synced_shards' = checkpoint_synced_shards
    /\ UNCHANGED <<shard_state, shard_dirty_count, shard_syncer,
                   worker_state, worker_job, job_queue,
                   completed_jobs, job_shard, checkpoint_synced_shards,
                   checkpoint_target_shards>>

(*
 * Checkpoint complete - return to idle.
 *)
CheckpointComplete ==
    /\ checkpoint_state = CkptSaving
    /\ checkpoint_state' = CkptIdle
    /\ checkpoint_synced_shards' = {}
    /\ checkpoint_target_shards' = {}
    /\ UNCHANGED <<shard_state, shard_dirty_count, shard_syncer,
                   worker_state, worker_job, job_queue,
                   completed_jobs, job_shard, last_saved_target_shards,
                   last_saved_synced_shards, global_checkpoint_saved>>

(* ---------------------------------------------------------------------------
 * Next state relation
 * --------------------------------------------------------------------------- *)

Next ==
    \/ \E w \in Workers: WorkerPickJob(w)
    \/ \E w \in Workers: WorkerProcess(w)
    \/ CheckpointStart
    \/ \E s \in Shards: CheckpointStartShardSync(s)
    \/ \E s \in Shards: CheckpointCompleteShardSync(s)
    \/ \E s \in Shards: CheckpointShardSyncFails(s)
    \/ CheckpointAllSynced
    \/ CheckpointAbortOnFailure
    \/ CheckpointSaveGlobal
    \/ CheckpointComplete

vars == <<shard_state, shard_dirty_count, shard_syncer, worker_state,
          worker_job, job_queue, completed_jobs, job_shard,
          checkpoint_state, checkpoint_synced_shards, checkpoint_target_shards,
          last_saved_target_shards, last_saved_synced_shards,
          global_checkpoint_saved>>

Spec == Init /\ [][Next]_vars

\* Fairness specification for liveness properties
\* Workers will eventually pick up and process jobs (no defer step anymore).
WorkerFairness ==
    /\ \A w \in Workers: WF_vars(WorkerPickJob(w))
    /\ \A w \in Workers: WF_vars(WorkerProcess(w))

\* Checkpoint actions will eventually happen
CheckpointFairness ==
    /\ WF_vars(CheckpointStart)
    /\ \A s \in Shards: WF_vars(CheckpointStartShardSync(s))
    \* Strong fairness rules out an environment that always chooses failure
    \* whenever a shard sync is retried.
    /\ \A s \in Shards: SF_vars(CheckpointCompleteShardSync(s))
    /\ WF_vars(CheckpointAllSynced)
    /\ WF_vars(CheckpointAbortOnFailure)
    /\ WF_vars(CheckpointSaveGlobal)
    /\ WF_vars(CheckpointComplete)

\* Fair specification includes fairness constraints
FairSpec == Spec /\ WorkerFairness /\ CheckpointFairness

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
 * CRITICAL: Global checkpoint only saved after all shards synced.
 *)
CheckpointAtomicity ==
    global_checkpoint_saved =>
        last_saved_target_shards \subseteq last_saved_synced_shards

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

JobRepresented ==
    \A j \in Jobs:
        \/ j \in job_queue
        \/ j \in completed_jobs
        \/ \E w \in Workers: worker_job[w] = j

JobSetsDisjoint ==
    job_queue \cap completed_jobs = {}

WorkerJobsDisjointFromSets ==
    \A w \in Workers:
        worker_job[w] # NONE =>
            /\ worker_job[w] \notin job_queue
            /\ worker_job[w] \notin completed_jobs

WorkerJobsUnique ==
    \A w1, w2 \in Workers:
        worker_job[w1] # NONE /\ worker_job[w1] = worker_job[w2] => w1 = w2

ProcessingHasJob ==
    \A w \in Workers:
        worker_state[w] = Processing => worker_job[w] # NONE

IdleHasNoJob ==
    \A w \in Workers:
        worker_state[w] = Idle => worker_job[w] = NONE

WorkerStateJobConsistency ==
    /\ ProcessingHasJob
    /\ IdleHasNoJob

JobPartition ==
    /\ JobRepresented
    /\ JobSetsDisjoint
    /\ WorkerJobsDisjointFromSets
    /\ WorkerJobsUnique

(*
 * Once the checkpoint reaches the metadata-save phases, every shard captured
 * at checkpoint start has successfully synced.
 *)
CheckpointReadyToSave ==
    checkpoint_state \in {CkptCheckpointing, CkptSaving} =>
        checkpoint_target_shards \subseteq checkpoint_synced_shards

(*
 * Combined safety invariant.
 *)
Safety ==
    /\ TypeOK
    /\ AtMostOneSyncer
    /\ CheckpointAtomicity
    /\ SyncerConsistency
    /\ CleanMeansZeroDirty
    /\ WorkerStateJobConsistency
    /\ JobPartition
    /\ CheckpointReadyToSave

(* ---------------------------------------------------------------------------
 * Liveness Properties (under fairness)
 * --------------------------------------------------------------------------- *)

(*
 * If checkpoint starts and no failures, it eventually completes.
 *)
CheckpointEventuallyCompletes ==
    (checkpoint_state = CkptSyncing) ~>
        (checkpoint_state = CkptIdle \/ \E s \in Shards: shard_state[s] = SyncFailed)

(*
 * Every queued job eventually completes under fair worker scheduling.
 *)
AllJobsEventuallyComplete ==
    <>(completed_jobs = Jobs)

=============================================================================
