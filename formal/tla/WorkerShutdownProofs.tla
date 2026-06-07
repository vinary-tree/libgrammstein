------------------------ MODULE WorkerShutdownProofs ------------------------
(*
 * TLAPS proof obligations for WorkerShutdown.tla.
 *)

EXTENDS WorkerShutdown, TLAPS

ModelConstantsOK ==
    /\ Workers \in SUBSET Workers
    /\ Jobs \in SUBSET Jobs
    /\ MaxRetries \in Nat
    /\ NONE \notin Jobs

THEOREM InitImpliesTypeOK ==
    ModelConstantsOK /\ Init => TypeOK
<1>1. ASSUME ModelConstantsOK, Init
      PROVE  worker_state \in [Workers -> {Idle, PollingQueue, Processing,
                                           SendingResult, Exiting, Exited}]
  BY <1>1 DEF Init, Idle
<1>2. ASSUME ModelConstantsOK, Init
      PROVE  worker_job \in [Workers -> Jobs \cup {NONE}]
  BY <1>2, SMT DEF ModelConstantsOK, Init, NONE
<1>3. ASSUME ModelConstantsOK, Init
      PROVE  /\ job_queue \subseteq Jobs
             /\ results_pending \subseteq Jobs
             /\ results_received \subseteq Jobs
  BY <1>3 DEF Init
<1>4. ASSUME ModelConstantsOK, Init
      PROVE  shutdown_signaled \in BOOLEAN
  BY <1>4 DEF Init
<1>5. ASSUME ModelConstantsOK, Init
      PROVE  checkpoint_state \in {NotStarted, Draining, Checkpointing, Done,
                                    ForceQuitAborted, DrainAborted}
  BY <1>5 DEF Init, NotStarted
<1>. QED
  BY <1>1, <1>2, <1>3, <1>4, <1>5 DEF TypeOK

THEOREM InitImpliesProcessingHasJob ==
    ModelConstantsOK /\ Init => ProcessingHasJob
BY SMT DEF ModelConstantsOK, Init, ProcessingHasJob, Idle, Processing, NONE

THEOREM InitImpliesSendingHasJob ==
    ModelConstantsOK /\ Init => SendingHasJob
BY SMT DEF ModelConstantsOK, Init, SendingHasJob, Idle, SendingResult, NONE

THEOREM InitImpliesIdleNoJob ==
    ModelConstantsOK /\ Init => IdleNoJob
BY SMT DEF ModelConstantsOK, Init, IdleNoJob, Idle, PollingQueue, Exiting,
           Exited, NONE

THEOREM InitImpliesResultsDisjoint ==
    ModelConstantsOK /\ Init => ResultsDisjoint
BY DEF Init, ResultsDisjoint

THEOREM InitImpliesCheckpointAfterDrain ==
    ModelConstantsOK /\ Init => CheckpointAfterDrain
BY SMT DEF Init, CheckpointAfterDrain, NotStarted, Checkpointing, Done, ForceQuitAborted, DrainAborted

THEOREM InitImpliesCheckpointRequiresShutdown ==
    ModelConstantsOK /\ Init => CheckpointRequiresShutdown
BY SMT DEF Init, CheckpointRequiresShutdown, NotStarted, Draining,
           Checkpointing, Done, ForceQuitAborted, DrainAborted

THEOREM InitImpliesAbortStatesDoNotCheckpoint ==
    ModelConstantsOK /\ Init => AbortStatesDoNotCheckpoint
BY SMT DEF Init, AbortStatesDoNotCheckpoint, NotStarted, Checkpointing, Done,
           ForceQuitAborted, DrainAborted

THEOREM InitImpliesNoJobLost ==
    ModelConstantsOK /\ Init => NoJobLost
BY SMT DEF Init, NoJobLost

THEOREM InitImpliesJobUniqueOwnership ==
    ModelConstantsOK /\ Init => JobUniqueOwnership
BY SMT DEF ModelConstantsOK, Init, JobUniqueOwnership, NONE

THEOREM InitImpliesSafety ==
    ModelConstantsOK /\ Init => Safety
BY InitImpliesTypeOK,
   InitImpliesProcessingHasJob,
   InitImpliesSendingHasJob,
   InitImpliesIdleNoJob,
   InitImpliesResultsDisjoint,
   InitImpliesCheckpointAfterDrain,
   InitImpliesCheckpointRequiresShutdown,
   InitImpliesAbortStatesDoNotCheckpoint,
   InitImpliesNoJobLost,
   InitImpliesJobUniqueOwnership
DEF Safety

THEOREM WorkerCheckShutdownBeforePollPreservesSafety ==
    ASSUME ModelConstantsOK, Safety,
           NEW w \in Workers, WorkerCheckShutdownBeforePoll(w)
    PROVE  Safety'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, ProcessingHasJob, SendingHasJob,
           IdleNoJob, ResultsDisjoint, CheckpointAfterDrain, CheckpointRequiresShutdown, AbortStatesDoNotCheckpoint, NoJobLost,
           JobUniqueOwnership,
           WorkerCheckShutdownBeforePoll, Idle, PollingQueue, Processing,
           SendingResult, Exiting, Exited, NONE, NotStarted, Draining,
           Checkpointing, Done, ForceQuitAborted, DrainAborted

THEOREM WorkerStartPollingPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, NEW w \in Workers, WorkerStartPolling(w)
    PROVE  Safety'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, ProcessingHasJob, SendingHasJob,
           IdleNoJob, ResultsDisjoint, CheckpointAfterDrain, CheckpointRequiresShutdown, AbortStatesDoNotCheckpoint, NoJobLost,
           JobUniqueOwnership, WorkerStartPolling,
           Idle, PollingQueue, Processing, SendingResult, Exiting, Exited,
           NONE, NotStarted, Draining, Checkpointing, Done, ForceQuitAborted, DrainAborted

THEOREM WorkerShutdownWhileWaitingPreservesSafety ==
    ASSUME ModelConstantsOK, Safety,
           NEW w \in Workers, WorkerShutdownWhileWaiting(w)
    PROVE  Safety'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, ProcessingHasJob, SendingHasJob,
           IdleNoJob, ResultsDisjoint, CheckpointAfterDrain, CheckpointRequiresShutdown, AbortStatesDoNotCheckpoint, NoJobLost,
           JobUniqueOwnership,
           WorkerShutdownWhileWaiting, Idle, PollingQueue, Processing,
           SendingResult, Exiting, Exited, NONE, NotStarted, Draining,
           Checkpointing, Done, ForceQuitAborted, DrainAborted

THEOREM WorkerPickJobPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, NEW w \in Workers, WorkerPickJob(w)
    PROVE  Safety'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, ProcessingHasJob, SendingHasJob,
           IdleNoJob, ResultsDisjoint, CheckpointAfterDrain, CheckpointRequiresShutdown, AbortStatesDoNotCheckpoint, NoJobLost,
           JobUniqueOwnership, WorkerPickJob,
           Idle, PollingQueue, Processing, SendingResult, Exiting, Exited,
           NONE, NotStarted, Draining, Checkpointing, Done, ForceQuitAborted, DrainAborted

THEOREM WorkerQueueEmptyPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, NEW w \in Workers, WorkerQueueEmpty(w)
    PROVE  Safety'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, ProcessingHasJob, SendingHasJob,
           IdleNoJob, ResultsDisjoint, CheckpointAfterDrain, CheckpointRequiresShutdown, AbortStatesDoNotCheckpoint, NoJobLost,
           JobUniqueOwnership, WorkerQueueEmpty,
           Idle, PollingQueue, Processing, SendingResult, Exiting, Exited,
           NONE, NotStarted, Draining, Checkpointing, Done, ForceQuitAborted, DrainAborted

THEOREM WorkerFinishProcessingPreservesSafety ==
    ASSUME ModelConstantsOK, Safety,
           NEW w \in Workers, WorkerFinishProcessing(w)
    PROVE  Safety'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, ProcessingHasJob, SendingHasJob,
           IdleNoJob, ResultsDisjoint, CheckpointAfterDrain, CheckpointRequiresShutdown, AbortStatesDoNotCheckpoint, NoJobLost,
           JobUniqueOwnership, WorkerFinishProcessing,
           Idle, PollingQueue, Processing, SendingResult, Exiting, Exited,
           NONE, NotStarted, Draining, Checkpointing, Done, ForceQuitAborted, DrainAborted

THEOREM WorkerSendResultPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, NEW w \in Workers, WorkerSendResult(w)
    PROVE  Safety'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, ProcessingHasJob, SendingHasJob,
           IdleNoJob, ResultsDisjoint, CheckpointAfterDrain, CheckpointRequiresShutdown, AbortStatesDoNotCheckpoint, NoJobLost,
           JobUniqueOwnership, WorkerSendResult,
           Idle, PollingQueue, Processing, SendingResult, Exiting, Exited,
           NONE, NotStarted, Draining, Checkpointing, Done, ForceQuitAborted, DrainAborted

THEOREM WorkerExitPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, NEW w \in Workers, WorkerExit(w)
    PROVE  Safety'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, ProcessingHasJob, SendingHasJob,
           IdleNoJob, ResultsDisjoint, CheckpointAfterDrain, CheckpointRequiresShutdown, AbortStatesDoNotCheckpoint, NoJobLost,
           JobUniqueOwnership, WorkerExit,
           Idle, PollingQueue, Processing, SendingResult, Exiting, Exited,
           NONE, NotStarted, Draining, Checkpointing, Done, ForceQuitAborted, DrainAborted

THEOREM SignalShutdownPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, SignalShutdown
    PROVE  Safety'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, ProcessingHasJob, SendingHasJob,
           IdleNoJob, ResultsDisjoint, CheckpointAfterDrain, CheckpointRequiresShutdown, AbortStatesDoNotCheckpoint, NoJobLost,
           JobUniqueOwnership, SignalShutdown,
           Idle, PollingQueue, Processing, SendingResult, Exiting, Exited,
           NONE, NotStarted, Draining, Checkpointing, Done, ForceQuitAborted, DrainAborted

THEOREM ReceiveResultPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, ReceiveResult
    PROVE  Safety'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, ProcessingHasJob, SendingHasJob,
           IdleNoJob, ResultsDisjoint, CheckpointAfterDrain, CheckpointRequiresShutdown, AbortStatesDoNotCheckpoint, NoJobLost,
           JobUniqueOwnership, ReceiveResult,
           Idle, PollingQueue, Processing, SendingResult, Exiting, Exited,
           NONE, NotStarted, Draining, Checkpointing, Done, ForceQuitAborted, DrainAborted

THEOREM StartCheckpointPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, StartCheckpoint
    PROVE  Safety'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, ProcessingHasJob, SendingHasJob,
           IdleNoJob, ResultsDisjoint, CheckpointAfterDrain, CheckpointRequiresShutdown, AbortStatesDoNotCheckpoint, NoJobLost,
           JobUniqueOwnership, StartCheckpoint,
           Idle, PollingQueue, Processing, SendingResult, Exiting, Exited,
           NONE, NotStarted, Draining, Checkpointing, Done, ForceQuitAborted, DrainAborted

THEOREM CompleteCheckpointPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, CompleteCheckpoint
    PROVE  Safety'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, ProcessingHasJob, SendingHasJob,
           IdleNoJob, ResultsDisjoint, CheckpointAfterDrain, CheckpointRequiresShutdown, AbortStatesDoNotCheckpoint, NoJobLost,
           JobUniqueOwnership, CompleteCheckpoint,
           Idle, PollingQueue, Processing, SendingResult, Exiting, Exited,
           NONE, NotStarted, Draining, Checkpointing, Done, ForceQuitAborted, DrainAborted

THEOREM ForceQuitPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, ForceQuit
    PROVE  Safety'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, ProcessingHasJob, SendingHasJob,
           IdleNoJob, ResultsDisjoint, CheckpointAfterDrain, CheckpointRequiresShutdown, AbortStatesDoNotCheckpoint, NoJobLost,
           JobUniqueOwnership, ForceQuit,
           Idle, PollingQueue, Processing, SendingResult, Exiting, Exited,
           NONE, NotStarted, Draining, Checkpointing, Done, ForceQuitAborted, DrainAborted

THEOREM AbortDrainWithoutCheckpointPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, AbortDrainWithoutCheckpoint
    PROVE  Safety'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, ProcessingHasJob, SendingHasJob,
           IdleNoJob, ResultsDisjoint, CheckpointAfterDrain, CheckpointRequiresShutdown, AbortStatesDoNotCheckpoint, NoJobLost,
           JobUniqueOwnership, AbortDrainWithoutCheckpoint,
           Idle, PollingQueue, Processing, SendingResult, Exiting, Exited,
           NONE, NotStarted, Draining, Checkpointing, Done, ForceQuitAborted, DrainAborted

THEOREM NextPreservesSafety ==
    (ModelConstantsOK /\ Safety /\ Next) => Safety'
BY WorkerCheckShutdownBeforePollPreservesSafety,
   WorkerStartPollingPreservesSafety,
   WorkerShutdownWhileWaitingPreservesSafety,
   WorkerPickJobPreservesSafety,
   WorkerQueueEmptyPreservesSafety,
   WorkerFinishProcessingPreservesSafety,
   WorkerSendResultPreservesSafety,
   WorkerExitPreservesSafety,
   SignalShutdownPreservesSafety,
   ReceiveResultPreservesSafety,
   StartCheckpointPreservesSafety,
   CompleteCheckpointPreservesSafety,
   ForceQuitPreservesSafety,
   AbortDrainWithoutCheckpointPreservesSafety
DEF Next

THEOREM ForceQuitSkipsCheckpoint ==
    ASSUME ForceQuit
    PROVE  checkpoint_state' = ForceQuitAborted
BY DEF ForceQuit

THEOREM DrainFailureSkipsCheckpoint ==
    ASSUME AbortDrainWithoutCheckpoint
    PROVE  checkpoint_state' = DrainAborted
BY DEF AbortDrainWithoutCheckpoint

THEOREM StutterPreservesSafety ==
    Safety /\ UNCHANGED vars => Safety'
BY SMT DEF Safety, TypeOK, ProcessingHasJob, SendingHasJob, IdleNoJob,
           ResultsDisjoint, CheckpointAfterDrain, CheckpointRequiresShutdown, AbortStatesDoNotCheckpoint, NoJobLost,
           JobUniqueOwnership, vars

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
