------------------------- MODULE AsyncShardSyncProofs ------------------------
(*
 * TLAPS proof obligations for AsyncShardSync.tla.
 *
 * Safety is proved inductively: initialization establishes Safety, every
 * action preserves Safety, stuttering preserves Safety, and PTL lifts the
 * local obligations to Spec => []Safety.
 *
 * This is the lock-free model: workers write regardless of a shard's sync
 * state (WorkerProcessJob has no `# Syncing` guard), so there is no defer
 * queue, no WorkersSafelyDefer invariant, and no DeferredJobReturns action.
 * WorkerProcessJob's shard-state effect mirrors mark_dirty exactly --
 * Clean -> Dirty, identity otherwise.
 *)

EXTENDS AsyncShardSync, TLAPS

ModelConstantsOK ==
    /\ Workers \in SUBSET Workers
    /\ Shards \in SUBSET Shards
    /\ Jobs \in SUBSET Jobs
    /\ MaxSyncAttempts \in Nat
    /\ "checkpoint" \notin Workers
    /\ NONE \notin Workers
    /\ NONE \notin Jobs

THEOREM FunctionUpdateAt ==
    ASSUME NEW Domain, NEW Range, NEW f \in [Domain -> Range],
           NEW key \in Domain, NEW value \in Range
    PROVE  [f EXCEPT ![key] = value][key] = value
OBVIOUS

THEOREM FunctionUpdateOther ==
    ASSUME NEW Domain, NEW Range, NEW f \in [Domain -> Range],
           NEW key \in Domain, NEW other \in Domain, NEW value \in Range,
           other # key
    PROVE  [f EXCEPT ![key] = value][other] = f[other]
OBVIOUS

THEOREM FunctionUpdateTypeOK ==
    ASSUME NEW Domain, NEW Range, NEW f \in [Domain -> Range],
           NEW key \in Domain, NEW value \in Range
    PROVE  [f EXCEPT ![key] = value] \in [Domain -> Range]
BY SMT, FunctionUpdateAt, FunctionUpdateOther

THEOREM NatIncrementFunctionUpdateTypeOK ==
    ASSUME NEW Domain, NEW f \in [Domain -> Nat], NEW key \in Domain
    PROVE  [f EXCEPT ![key] = f[key] + 1] \in [Domain -> Nat]
BY SMT, FunctionUpdateAt, FunctionUpdateOther

THEOREM InitImpliesTypeOK ==
    ModelConstantsOK /\ Init => TypeOK
<1>1. ASSUME ModelConstantsOK, Init
      PROVE  shard_state \in [Shards -> {Clean, Dirty, Syncing, SyncFailed}]
  BY <1>1 DEF Init
<1>2. ASSUME ModelConstantsOK, Init
      PROVE  shard_dirty_count \in [Shards -> Nat]
  BY <1>2, SMT DEF ModelConstantsOK, Init
<1>3. ASSUME ModelConstantsOK, Init
      PROVE  shard_syncer \in [Shards -> Workers \cup {"checkpoint", NONE}]
  BY <1>3, SMT DEF ModelConstantsOK, Init, NONE
<1>4. ASSUME ModelConstantsOK, Init
      PROVE  worker_state \in [Workers -> {Idle, Processing}]
  BY <1>4 DEF Init
<1>5. ASSUME ModelConstantsOK, Init
      PROVE  worker_job \in [Workers -> Jobs \cup {NONE}]
  BY <1>5, SMT DEF ModelConstantsOK, Init, NONE
<1>6. ASSUME ModelConstantsOK, Init
      PROVE  job_queue \subseteq Jobs
  BY <1>6 DEF Init
<1>7. ASSUME ModelConstantsOK, Init
      PROVE  completed_jobs \subseteq Jobs
  BY <1>7 DEF Init
<1>8. ASSUME ModelConstantsOK, Init
      PROVE  job_shard \in [Jobs -> Shards]
  BY <1>8 DEF Init
<1>9. ASSUME ModelConstantsOK, Init
      PROVE  checkpoint_state \in {CkptIdle, CkptSyncing, CkptCheckpointing, CkptSaving}
  BY <1>9 DEF Init
<1>10. ASSUME ModelConstantsOK, Init
       PROVE  checkpoint_synced_shards \subseteq Shards
  BY <1>10 DEF Init
<1>11. ASSUME ModelConstantsOK, Init
       PROVE  checkpoint_target_shards \subseteq Shards
  BY <1>11 DEF Init
<1>12. ASSUME ModelConstantsOK, Init
       PROVE  last_saved_target_shards \subseteq Shards
  BY <1>12 DEF Init
<1>13. ASSUME ModelConstantsOK, Init
       PROVE  last_saved_synced_shards \subseteq Shards
  BY <1>13 DEF Init
<1>14. ASSUME ModelConstantsOK, Init
       PROVE  global_checkpoint_saved \in BOOLEAN
  BY <1>14 DEF Init
<1>. QED
  BY <1>1, <1>2, <1>3, <1>4, <1>5, <1>6, <1>7, <1>8,
     <1>9, <1>10, <1>11, <1>12, <1>13, <1>14
  DEF TypeOK

THEOREM InitImpliesAtMostOneSyncer ==
    ModelConstantsOK /\ Init => AtMostOneSyncer
BY SMT DEF ModelConstantsOK, Init, AtMostOneSyncer, Clean, Syncing, NONE

THEOREM InitImpliesCheckpointAtomicity ==
    ModelConstantsOK /\ Init => CheckpointAtomicity
BY DEF Init, CheckpointAtomicity

THEOREM InitImpliesSyncerConsistency ==
    ModelConstantsOK /\ Init => SyncerConsistency
BY SMT DEF ModelConstantsOK, Init, SyncerConsistency, Clean, Syncing, NONE

THEOREM InitImpliesCleanMeansZeroDirty ==
    ModelConstantsOK /\ Init => CleanMeansZeroDirty
BY SMT DEF Init, CleanMeansZeroDirty, Clean

THEOREM InitImpliesWorkerStateJobConsistency ==
    ModelConstantsOK /\ Init => WorkerStateJobConsistency
BY SMT DEF ModelConstantsOK, Init, WorkerStateJobConsistency, ProcessingHasJob,
           IdleHasNoJob, Idle, Processing, NONE

THEOREM InitImpliesJobPartition ==
    ModelConstantsOK /\ Init => JobPartition
BY SMT DEF ModelConstantsOK, Init, JobPartition, JobRepresented,
           JobSetsDisjoint, WorkerJobsDisjointFromSets, WorkerJobsUnique, NONE

THEOREM InitImpliesCheckpointReadyToSave ==
    ModelConstantsOK /\ Init => CheckpointReadyToSave
BY SMT DEF Init, CheckpointReadyToSave, CkptIdle, CkptCheckpointing, CkptSaving

THEOREM InitImpliesSafety ==
    ModelConstantsOK /\ Init => Safety
BY InitImpliesTypeOK,
   InitImpliesAtMostOneSyncer,
   InitImpliesCheckpointAtomicity,
   InitImpliesSyncerConsistency,
   InitImpliesCleanMeansZeroDirty,
   InitImpliesWorkerStateJobConsistency,
   InitImpliesJobPartition,
   InitImpliesCheckpointReadyToSave
DEF Safety

THEOREM WorkerPickJobByPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, NEW w \in Workers, NEW j \in Jobs,
           WorkerPickJobBy(w, j)
    PROVE  Safety'
<1>1. TypeOK'
  BY SMT DEF ModelConstantsOK, Safety, TypeOK, WorkerPickJobBy, Idle,
             Processing, NONE
<1>2. AtMostOneSyncer'
  BY SMT DEF Safety, AtMostOneSyncer, WorkerPickJobBy
<1>3. CheckpointAtomicity'
  BY SMT DEF Safety, CheckpointAtomicity, WorkerPickJobBy
<1>4. SyncerConsistency'
  BY SMT DEF Safety, SyncerConsistency, WorkerPickJobBy
<1>5. CleanMeansZeroDirty'
  BY SMT DEF Safety, CleanMeansZeroDirty, WorkerPickJobBy
<1>6. WorkerStateJobConsistency'
  BY SMT, FunctionUpdateAt, FunctionUpdateOther
     DEF ModelConstantsOK, Safety, TypeOK, WorkerStateJobConsistency,
         ProcessingHasJob, IdleHasNoJob, WorkerPickJobBy, Idle, Processing,
         NONE
<1>7. JobPartition'
  BY SMT, FunctionUpdateAt, FunctionUpdateOther
     DEF ModelConstantsOK, Safety, TypeOK, JobPartition, JobRepresented,
         JobSetsDisjoint, WorkerJobsDisjointFromSets, WorkerJobsUnique,
         WorkerPickJobBy, NONE
<1>8. CheckpointReadyToSave'
  BY SMT DEF Safety, CheckpointReadyToSave, WorkerPickJobBy
<1>. QED
  BY <1>1, <1>2, <1>3, <1>4, <1>5, <1>6, <1>7, <1>8 DEF Safety

THEOREM WorkerPickJobPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, NEW w \in Workers, WorkerPickJob(w)
    PROVE  Safety'
<1>1. PICK j \in job_queue: WorkerPickJobBy(w, j)
  BY DEF WorkerPickJob
<1>2. j \in Jobs
  BY <1>1, SMT DEF Safety, TypeOK, WorkerPickJobBy
<1>3. Safety'
  BY <1>1, <1>2, WorkerPickJobByPreservesSafety
<1>. QED
  BY <1>3

THEOREM WorkerProcessJobPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, NEW w \in Workers, NEW j \in Jobs,
           WorkerProcessJob(w, j)
    PROVE  Safety'
<1>1. TypeOK'
  <2>1. job_shard[j] \in Shards
    BY SMT DEF Safety, TypeOK
  <2>2. shard_state' \in [Shards -> {Clean, Dirty, Syncing, SyncFailed}]
    <3>1. shard_state \in [Shards -> {Clean, Dirty, Syncing, SyncFailed}]
      BY DEF Safety, TypeOK
    <3>2. (IF shard_state[job_shard[j]] = Clean THEN Dirty ELSE shard_state[job_shard[j]])
            \in {Clean, Dirty, Syncing, SyncFailed}
      BY <3>1, <2>1, SMT DEF Clean, Dirty, Syncing, SyncFailed
    <3>. QED
      BY <3>1, <3>2, <2>1, FunctionUpdateTypeOK DEF WorkerProcessJob
  <2>3. shard_dirty_count' \in [Shards -> Nat]
    BY <2>1, SMT, NatIncrementFunctionUpdateTypeOK
       DEF Safety, TypeOK, WorkerProcessJob
  <2>4. shard_syncer' \in [Shards -> Workers \cup {"checkpoint", NONE}]
    BY SMT DEF ModelConstantsOK, Safety, TypeOK, WorkerProcessJob, NONE
  <2>5. worker_state' \in [Workers -> {Idle, Processing}]
    BY SMT, FunctionUpdateTypeOK
       DEF Safety, TypeOK, WorkerProcessJob, Idle, Processing
  <2>6. worker_job' \in [Workers -> Jobs \cup {NONE}]
    BY SMT, FunctionUpdateTypeOK
       DEF ModelConstantsOK, Safety, TypeOK, WorkerProcessJob, NONE
  <2>7. job_queue' \subseteq Jobs
    BY SMT DEF Safety, TypeOK, WorkerProcessJob
  <2>8. completed_jobs' \subseteq Jobs
    BY SMT DEF Safety, TypeOK, WorkerProcessJob
  <2>9. job_shard' \in [Jobs -> Shards]
    BY SMT DEF Safety, TypeOK, WorkerProcessJob
  <2>10. checkpoint_state' \in {CkptIdle, CkptSyncing, CkptCheckpointing, CkptSaving}
    BY SMT DEF Safety, TypeOK, WorkerProcessJob, CkptIdle, CkptSyncing,
               CkptCheckpointing, CkptSaving
  <2>11. checkpoint_synced_shards' \subseteq Shards
    BY SMT DEF Safety, TypeOK, WorkerProcessJob
  <2>12. checkpoint_target_shards' \subseteq Shards
    BY SMT DEF Safety, TypeOK, WorkerProcessJob
  <2>13. last_saved_target_shards' \subseteq Shards
    BY SMT DEF Safety, TypeOK, WorkerProcessJob
  <2>14. last_saved_synced_shards' \subseteq Shards
    BY SMT DEF Safety, TypeOK, WorkerProcessJob
  <2>15. global_checkpoint_saved' = FALSE
    BY SMT DEF WorkerProcessJob
  <2>16. global_checkpoint_saved' \in BOOLEAN
    BY <2>15
  <2>. QED
    BY <2>2, <2>3, <2>4, <2>5, <2>6, <2>7, <2>8, <2>9,
       <2>10, <2>11, <2>12, <2>13, <2>14, <2>16
    DEF TypeOK
<1>2. AtMostOneSyncer'
  BY SMT DEF Safety, AtMostOneSyncer, WorkerProcessJob, Clean, Dirty, Syncing
<1>3. CheckpointAtomicity'
  BY SMT DEF Safety, CheckpointAtomicity, WorkerProcessJob
<1>4. SyncerConsistency'
  BY SMT DEF Safety, SyncerConsistency, WorkerProcessJob, Clean, Dirty, Syncing,
             NONE
<1>5. CleanMeansZeroDirty'
  BY SMT DEF ModelConstantsOK, Safety, TypeOK, CleanMeansZeroDirty,
             WorkerProcessJob, Clean, Dirty
<1>6. WorkerStateJobConsistency'
  BY SMT, FunctionUpdateAt, FunctionUpdateOther
     DEF ModelConstantsOK, Safety, TypeOK, WorkerStateJobConsistency,
         ProcessingHasJob, IdleHasNoJob, WorkerProcessJob, Idle, Processing,
         NONE
<1>7. JobPartition'
  BY SMT, FunctionUpdateAt, FunctionUpdateOther
     DEF ModelConstantsOK, Safety, TypeOK, JobPartition, JobRepresented,
         JobSetsDisjoint, WorkerJobsDisjointFromSets, WorkerJobsUnique,
         WorkerProcessJob, NONE
<1>8. CheckpointReadyToSave'
  BY SMT DEF Safety, CheckpointReadyToSave, WorkerProcessJob
<1>. QED
  BY <1>1, <1>2, <1>3, <1>4, <1>5, <1>6, <1>7, <1>8 DEF Safety

THEOREM WorkerProcessPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, NEW w \in Workers, WorkerProcess(w)
    PROVE  Safety'
<1>1. PICK j \in Jobs: WorkerProcessJob(w, j)
  BY DEF WorkerProcess
<1>2. Safety'
  BY <1>1, WorkerProcessJobPreservesSafety
<1>. QED
  BY <1>2

THEOREM CheckpointStartPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, CheckpointStart
    PROVE  Safety'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, AtMostOneSyncer,
           CheckpointAtomicity, SyncerConsistency,
           CleanMeansZeroDirty, WorkerStateJobConsistency, ProcessingHasJob,
           IdleHasNoJob, JobPartition, JobRepresented, JobSetsDisjoint,
           WorkerJobsDisjointFromSets, WorkerJobsUnique, CheckpointReadyToSave, CheckpointStart,
           Clean, Dirty, Syncing, SyncFailed, Idle, Processing,
           CkptIdle, CkptSyncing, CkptCheckpointing, CkptSaving, NONE

THEOREM CheckpointStartShardSyncPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, NEW s \in Shards, CheckpointStartShardSync(s)
    PROVE  Safety'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, AtMostOneSyncer,
           CheckpointAtomicity, SyncerConsistency,
           CleanMeansZeroDirty, WorkerStateJobConsistency, ProcessingHasJob,
           IdleHasNoJob, JobPartition, JobRepresented, JobSetsDisjoint,
           WorkerJobsDisjointFromSets, WorkerJobsUnique, CheckpointReadyToSave,
           CheckpointStartShardSync, Clean, Dirty, Syncing, SyncFailed,
           Idle, Processing, CkptIdle, CkptSyncing, CkptCheckpointing,
           CkptSaving, NONE

THEOREM CheckpointCompleteShardSyncPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, NEW s \in Shards, CheckpointCompleteShardSync(s)
    PROVE  Safety'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, AtMostOneSyncer,
           CheckpointAtomicity, SyncerConsistency,
           CleanMeansZeroDirty, WorkerStateJobConsistency, ProcessingHasJob,
           IdleHasNoJob, JobPartition, JobRepresented, JobSetsDisjoint,
           WorkerJobsDisjointFromSets, WorkerJobsUnique, CheckpointReadyToSave,
           CheckpointCompleteShardSync, Clean, Dirty, Syncing, SyncFailed,
           Idle, Processing, CkptIdle, CkptSyncing, CkptCheckpointing,
           CkptSaving, NONE

THEOREM CheckpointShardSyncFailsPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, NEW s \in Shards, CheckpointShardSyncFails(s)
    PROVE  Safety'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, AtMostOneSyncer,
           CheckpointAtomicity, SyncerConsistency,
           CleanMeansZeroDirty, WorkerStateJobConsistency, ProcessingHasJob,
           IdleHasNoJob, JobPartition, JobRepresented, JobSetsDisjoint,
           WorkerJobsDisjointFromSets, WorkerJobsUnique, CheckpointReadyToSave,
           CheckpointShardSyncFails, Clean, Dirty, Syncing, SyncFailed,
           Idle, Processing, CkptIdle, CkptSyncing, CkptCheckpointing,
           CkptSaving, NONE

THEOREM CheckpointAllSyncedPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, CheckpointAllSynced
    PROVE  Safety'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, AtMostOneSyncer,
           CheckpointAtomicity, SyncerConsistency,
           CleanMeansZeroDirty, WorkerStateJobConsistency, ProcessingHasJob,
           IdleHasNoJob, JobPartition, JobRepresented, JobSetsDisjoint,
           WorkerJobsDisjointFromSets, WorkerJobsUnique, CheckpointReadyToSave, CheckpointAllSynced,
           Clean, Dirty, Syncing, SyncFailed, Idle, Processing,
           CkptIdle, CkptSyncing, CkptCheckpointing, CkptSaving, NONE

THEOREM CheckpointAbortOnFailurePreservesSafety ==
    ASSUME ModelConstantsOK, Safety, CheckpointAbortOnFailure
    PROVE  Safety'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, AtMostOneSyncer,
           CheckpointAtomicity, SyncerConsistency,
           CleanMeansZeroDirty, WorkerStateJobConsistency, ProcessingHasJob,
           IdleHasNoJob, JobPartition, JobRepresented, JobSetsDisjoint,
           WorkerJobsDisjointFromSets, WorkerJobsUnique, CheckpointReadyToSave,
           CheckpointAbortOnFailure, Clean, Dirty, Syncing, SyncFailed,
           Idle, Processing, CkptIdle, CkptSyncing, CkptCheckpointing,
           CkptSaving, NONE

THEOREM CheckpointSaveGlobalPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, CheckpointSaveGlobal
    PROVE  Safety'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, AtMostOneSyncer,
           CheckpointAtomicity, SyncerConsistency,
           CleanMeansZeroDirty, WorkerStateJobConsistency, ProcessingHasJob,
           IdleHasNoJob, JobPartition, JobRepresented, JobSetsDisjoint,
           WorkerJobsDisjointFromSets, WorkerJobsUnique, CheckpointReadyToSave, CheckpointSaveGlobal,
           Clean, Dirty, Syncing, SyncFailed, Idle, Processing,
           CkptIdle, CkptSyncing, CkptCheckpointing, CkptSaving, NONE

THEOREM CheckpointCompletePreservesSafety ==
    ASSUME ModelConstantsOK, Safety, CheckpointComplete
    PROVE  Safety'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, AtMostOneSyncer,
           CheckpointAtomicity, SyncerConsistency,
           CleanMeansZeroDirty, WorkerStateJobConsistency, ProcessingHasJob,
           IdleHasNoJob, JobPartition, JobRepresented, JobSetsDisjoint,
           WorkerJobsDisjointFromSets, WorkerJobsUnique, CheckpointReadyToSave, CheckpointComplete,
           Clean, Dirty, Syncing, SyncFailed, Idle, Processing,
           CkptIdle, CkptSyncing, CkptCheckpointing, CkptSaving, NONE

THEOREM NextPreservesSafety ==
    (ModelConstantsOK /\ Safety /\ Next) => Safety'
BY WorkerPickJobPreservesSafety,
   WorkerProcessPreservesSafety,
   CheckpointStartPreservesSafety,
   CheckpointStartShardSyncPreservesSafety,
   CheckpointCompleteShardSyncPreservesSafety,
   CheckpointShardSyncFailsPreservesSafety,
   CheckpointAllSyncedPreservesSafety,
   CheckpointAbortOnFailurePreservesSafety,
   CheckpointSaveGlobalPreservesSafety,
   CheckpointCompletePreservesSafety
DEF Next

THEOREM StutterPreservesSafety ==
    Safety /\ UNCHANGED vars => Safety'
BY SMT DEF Safety, TypeOK, AtMostOneSyncer,
           CheckpointAtomicity, SyncerConsistency, CleanMeansZeroDirty,
           WorkerStateJobConsistency, ProcessingHasJob, IdleHasNoJob,
           JobPartition, JobRepresented, JobSetsDisjoint, WorkerJobsDisjointFromSets,
           WorkerJobsUnique, CheckpointReadyToSave, vars

THEOREM StepPreservesSafety ==
    (ModelConstantsOK /\ Safety /\ [Next]_vars) => Safety'
BY NextPreservesSafety, StutterPreservesSafety DEF vars

THEOREM InitImpliesSafetyUnderModelConstants ==
    ASSUME ModelConstantsOK
    PROVE  Init => Safety
BY InitImpliesSafety

THEOREM StepPreservesSafetyUnderModelConstants ==
    ASSUME ModelConstantsOK
    PROVE  Safety /\ [Next]_vars => Safety'
BY StepPreservesSafety

THEOREM SpecImpliesAlwaysSafety ==
    ASSUME ModelConstantsOK
    PROVE  Spec => []Safety
BY InitImpliesSafetyUnderModelConstants,
   StepPreservesSafetyUnderModelConstants,
   PTL DEF Spec

=============================================================================
