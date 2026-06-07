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

    \* Deferred job queue (jobs waiting for shard to finish syncing)
    \* @type: Set(Str);
    deferred_queue,

    \* Jobs that have completed processing
    \* @type: Set(Str);
    completed_jobs,

    \* Sticky flag recording whether a worker ever wrote while its shard synced
    \* @type: Bool;
    unsafe_write_attempted,

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
    /\ deferred_queue \subseteq Jobs
    /\ completed_jobs \subseteq Jobs
    /\ unsafe_write_attempted \in BOOLEAN
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
    /\ deferred_queue = {}
    /\ completed_jobs = {}
    /\ unsafe_write_attempted = FALSE
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
                   deferred_queue, completed_jobs, unsafe_write_attempted,
                   job_shard, checkpoint_state,
                   checkpoint_synced_shards, checkpoint_target_shards,
                   last_saved_target_shards, last_saved_synced_shards,
                   global_checkpoint_saved>>

WorkerPickJob(w) ==
    \E j \in job_queue: WorkerPickJobBy(w, j)

(*
 * Worker checks if target shard is syncing and defers if so.
 * This is the key "defer-and-continue" pattern.
 *)
WorkerCheckAndDeferJob(w, j) ==
    /\ worker_state[w] = Processing
    /\ worker_job[w] = j
    /\ j \in Jobs
    /\ LET s == job_shard[j]
       IN
        /\ shard_state[s] = Syncing  \* Shard is being synced
        \* Defer: put job in deferred queue, worker becomes idle
        /\ deferred_queue' = deferred_queue \cup {j}
        /\ worker_job' = [worker_job EXCEPT ![w] = NONE]
        /\ worker_state' = [worker_state EXCEPT ![w] = Idle]
    /\ UNCHANGED <<shard_state, shard_dirty_count, shard_syncer,
                   job_queue, completed_jobs, unsafe_write_attempted,
                   job_shard, checkpoint_state,
                   checkpoint_synced_shards, checkpoint_target_shards,
                   last_saved_target_shards, last_saved_synced_shards,
                   global_checkpoint_saved>>

WorkerCheckAndDefer(w) ==
    \E j \in Jobs: WorkerCheckAndDeferJob(w, j)

(*
 * Worker processes job (writes to shard).
 * Only allowed if shard is NOT syncing.
 *)
WorkerProcessJob(w, j) ==
    /\ worker_state[w] = Processing
    /\ worker_job[w] = j
    /\ j \in Jobs
    /\ LET s == job_shard[j]
       IN
        /\ shard_state[s] # Syncing  \* Must not be syncing
        \* Write to shard: mark dirty, increment dirty count
        /\ shard_state' = [shard_state EXCEPT ![s] = Dirty]
        /\ shard_dirty_count' = [shard_dirty_count EXCEPT ![s] = @ + 1]
        \* Job complete, worker becomes idle
        /\ worker_job' = [worker_job EXCEPT ![w] = NONE]
        /\ worker_state' = [worker_state EXCEPT ![w] = Idle]
        /\ completed_jobs' = completed_jobs \cup {j}
        /\ unsafe_write_attempted' = unsafe_write_attempted \/ (shard_state[s] = Syncing)
        /\ global_checkpoint_saved' = FALSE
    /\ UNCHANGED <<shard_syncer, job_queue, deferred_queue, job_shard,
                   checkpoint_state, checkpoint_synced_shards,
                   checkpoint_target_shards, last_saved_target_shards,
                   last_saved_synced_shards>>

WorkerProcess(w) ==
    \E j \in Jobs: WorkerProcessJob(w, j)

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
                   worker_state, worker_job, completed_jobs, unsafe_write_attempted,
                   job_shard, checkpoint_state,
                   checkpoint_synced_shards, checkpoint_target_shards,
                   last_saved_target_shards, last_saved_synced_shards,
                   global_checkpoint_saved>>

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
                   worker_state, worker_job, job_queue, deferred_queue,
                   completed_jobs, unsafe_write_attempted, job_shard, last_saved_target_shards,
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
                   job_queue, deferred_queue, job_shard, checkpoint_state,
                   checkpoint_synced_shards, checkpoint_target_shards,
                   completed_jobs, unsafe_write_attempted, last_saved_target_shards,
                   last_saved_synced_shards, global_checkpoint_saved>>

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
                   completed_jobs, unsafe_write_attempted, job_shard, checkpoint_state,
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
                   job_queue, deferred_queue, job_shard, checkpoint_state,
                   checkpoint_synced_shards, checkpoint_target_shards,
                   completed_jobs, unsafe_write_attempted, last_saved_target_shards,
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
                   worker_state, worker_job, job_queue, deferred_queue,
                   completed_jobs, unsafe_write_attempted, job_shard, checkpoint_synced_shards,
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
                   job_queue, deferred_queue, completed_jobs, unsafe_write_attempted, job_shard,
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
                   worker_state, worker_job, job_queue, deferred_queue,
                   completed_jobs, unsafe_write_attempted, job_shard, checkpoint_synced_shards,
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
                   worker_state, worker_job, job_queue, deferred_queue,
                   completed_jobs, unsafe_write_attempted, job_shard, last_saved_target_shards,
                   last_saved_synced_shards, global_checkpoint_saved>>

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
          worker_job, job_queue, deferred_queue, completed_jobs,
          unsafe_write_attempted, job_shard,
          checkpoint_state, checkpoint_synced_shards, checkpoint_target_shards,
          last_saved_target_shards, last_saved_synced_shards,
          global_checkpoint_saved>>

Spec == Init /\ [][Next]_vars

\* Fairness specification for liveness properties
\* Workers will eventually pick up and either process or defer jobs
WorkerFairness ==
    /\ \A w \in Workers: WF_vars(WorkerPickJob(w))
    /\ \A w \in Workers: WF_vars(WorkerCheckAndDefer(w))
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

\* Deferred jobs will eventually return when shard is no longer syncing
DeferredFairness == \A j \in Jobs: SF_vars(DeferredJobReturns(j))

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
 * This is a sticky history invariant, so weakening WorkerProcess later will
 * be caught by TLC instead of making the property vacuous.
 *)
WorkersSafelyDefer ==
    unsafe_write_attempted = FALSE

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
        \/ j \in deferred_queue
        \/ j \in completed_jobs
        \/ \E w \in Workers: worker_job[w] = j

JobSetsDisjoint ==
    /\ job_queue \cap deferred_queue = {}
    /\ job_queue \cap completed_jobs = {}
    /\ deferred_queue \cap completed_jobs = {}

WorkerJobsDisjointFromSets ==
    \A w \in Workers:
        worker_job[w] # NONE =>
            /\ worker_job[w] \notin job_queue
            /\ worker_job[w] \notin deferred_queue
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
    /\ WorkersSafelyDefer
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

(*
 * Every queued job eventually completes under fair worker scheduling.
 *)
AllJobsEventuallyComplete ==
    <>(completed_jobs = Jobs)

=============================================================================
