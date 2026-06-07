------------------------ MODULE ImporterLifecycleProofs ------------------------
(*
 * TLAPS proof obligations for ImporterLifecycle.tla.
 *)

EXTENDS ImporterLifecycle, TLAPS

THEOREM TrueIsBoolean ==
    TRUE \in BOOLEAN
OBVIOUS

THEOREM FalseIsBoolean ==
    FALSE \in BOOLEAN
OBVIOUS

THEOREM FalseNotTrue ==
    FALSE # TRUE
OBVIOUS

THEOREM InitImpliesTypeOK ==
    Init => TypeOK
<1>1. ASSUME Init
      PROVE  phase \in Phases
  BY <1>1 DEF Init, Phases, Collecting
<1>2. ASSUME Init
      PROVE  /\ workers_drained \in BOOLEAN
             /\ all_results_received \in BOOLEAN
             /\ cleanup_done \in BOOLEAN
             /\ checkpoint_saved \in BOOLEAN
             /\ stats_computed \in BOOLEAN
             /\ merge_done \in BOOLEAN
             /\ completed_event \in BOOLEAN
             /\ terminal_error \in BOOLEAN
  BY <1>2, FalseIsBoolean DEF Init
<1>. QED
  BY <1>1, <1>2 DEF TypeOK

THEOREM InitImpliesCompletedRequiresAllWork ==
    Init => CompletedRequiresAllWork
BY SMT, FalseNotTrue DEF Init, CompletedRequiresAllWork

THEOREM InitImpliesNoFinalizeBeforeCheckpoint ==
    Init => NoFinalizeBeforeCheckpoint
BY SMT, FalseNotTrue
DEF Init, NoFinalizeBeforeCheckpoint, Collecting, ComputingStats, Merging,
    Completed

THEOREM InitImpliesEarlyTerminalSkipsFinalize ==
    Init => EarlyTerminalSkipsFinalize
BY SMT DEF Init, EarlyTerminalSkipsFinalize, Collecting, Cancelled, ForceQuit

THEOREM InitImpliesForceQuitSkipsCheckpoint ==
    Init => ForceQuitSkipsCheckpoint
BY SMT DEF Init, ForceQuitSkipsCheckpoint, Collecting, ForceQuit

THEOREM InitImpliesFailedNeverCompletes ==
    Init => FailedNeverCompletes
BY SMT DEF Init, FailedNeverCompletes, Collecting, Failed

THEOREM InitImpliesTerminalErrorNeverCompletes ==
    Init => TerminalErrorNeverCompletes
BY SMT, FalseNotTrue DEF Init, TerminalErrorNeverCompletes

THEOREM InitImpliesSafety ==
    Init => Safety
BY InitImpliesTypeOK,
   InitImpliesCompletedRequiresAllWork,
   InitImpliesNoFinalizeBeforeCheckpoint,
   InitImpliesEarlyTerminalSkipsFinalize,
   InitImpliesForceQuitSkipsCheckpoint,
   InitImpliesFailedNeverCompletes,
   InitImpliesTerminalErrorNeverCompletes
DEF Safety

THEOREM WorkerResultsCompletePreservesSafety ==
    ASSUME Safety, WorkerResultsComplete
    PROVE  Safety'
BY SMT, TrueIsBoolean, FalseIsBoolean
DEF Safety, TypeOK, Phases, CompletedRequiresAllWork,
           NoFinalizeBeforeCheckpoint, EarlyTerminalSkipsFinalize,
           ForceQuitSkipsCheckpoint, FailedNeverCompletes,
           TerminalErrorNeverCompletes, WorkerResultsComplete,
           Collecting, CleaningUp, SavingFinalCheckpoint, ComputingStats,
           Merging, Completed, Cancelled, ForceQuit, Failed

THEOREM CleanupCompletePreservesSafety ==
    ASSUME Safety, CleanupComplete
    PROVE  Safety'
BY SMT, TrueIsBoolean, FalseIsBoolean
DEF Safety, TypeOK, Phases, CompletedRequiresAllWork,
           NoFinalizeBeforeCheckpoint, EarlyTerminalSkipsFinalize,
           ForceQuitSkipsCheckpoint, FailedNeverCompletes,
           TerminalErrorNeverCompletes, CleanupComplete,
           Collecting, CleaningUp, SavingFinalCheckpoint, ComputingStats,
           Merging, Completed, Cancelled, ForceQuit, Failed

THEOREM FinalCheckpointSavedPreservesSafety ==
    ASSUME Safety, FinalCheckpointSaved
    PROVE  Safety'
BY SMT, TrueIsBoolean, FalseIsBoolean
DEF Safety, TypeOK, Phases, CompletedRequiresAllWork,
           NoFinalizeBeforeCheckpoint, EarlyTerminalSkipsFinalize,
           ForceQuitSkipsCheckpoint, FailedNeverCompletes,
           TerminalErrorNeverCompletes, FinalCheckpointSaved,
           Collecting, CleaningUp, SavingFinalCheckpoint, ComputingStats,
           Merging, Completed, Cancelled, ForceQuit, Failed

THEOREM FinalCheckpointFailsPreservesSafety ==
    ASSUME Safety, FinalCheckpointFails
    PROVE  Safety'
BY SMT, TrueIsBoolean, FalseIsBoolean
DEF Safety, TypeOK, Phases, CompletedRequiresAllWork,
           NoFinalizeBeforeCheckpoint, EarlyTerminalSkipsFinalize,
           ForceQuitSkipsCheckpoint, FailedNeverCompletes,
           TerminalErrorNeverCompletes, FinalCheckpointFails,
           Collecting, CleaningUp, SavingFinalCheckpoint, ComputingStats,
           Merging, Completed, Cancelled, ForceQuit, Failed

THEOREM StatsCompletePreservesSafety ==
    ASSUME Safety, StatsComplete
    PROVE  Safety'
BY SMT, TrueIsBoolean, FalseIsBoolean
DEF Safety, TypeOK, Phases, CompletedRequiresAllWork,
           NoFinalizeBeforeCheckpoint, EarlyTerminalSkipsFinalize,
           ForceQuitSkipsCheckpoint, FailedNeverCompletes,
           TerminalErrorNeverCompletes, StatsComplete,
           Collecting, CleaningUp, SavingFinalCheckpoint, ComputingStats,
           Merging, Completed, Cancelled, ForceQuit, Failed

THEOREM StatsFailsPreservesSafety ==
    ASSUME Safety, StatsFails
    PROVE  Safety'
BY SMT, TrueIsBoolean, FalseIsBoolean
DEF Safety, TypeOK, Phases, CompletedRequiresAllWork,
           NoFinalizeBeforeCheckpoint, EarlyTerminalSkipsFinalize,
           ForceQuitSkipsCheckpoint, FailedNeverCompletes,
           TerminalErrorNeverCompletes, StatsFails,
           Collecting, CleaningUp, SavingFinalCheckpoint, ComputingStats,
           Merging, Completed, Cancelled, ForceQuit, Failed

THEOREM MergeCompletePreservesSafety ==
    ASSUME Safety, MergeComplete
    PROVE  Safety'
BY SMT, TrueIsBoolean, FalseIsBoolean
DEF Safety, TypeOK, Phases, CompletedRequiresAllWork,
           NoFinalizeBeforeCheckpoint, EarlyTerminalSkipsFinalize,
           ForceQuitSkipsCheckpoint, FailedNeverCompletes,
           TerminalErrorNeverCompletes, MergeComplete,
           Collecting, CleaningUp, SavingFinalCheckpoint, ComputingStats,
           Merging, Completed, Cancelled, ForceQuit, Failed

THEOREM MergeFailsPreservesSafety ==
    ASSUME Safety, MergeFails
    PROVE  Safety'
BY SMT, TrueIsBoolean, FalseIsBoolean
DEF Safety, TypeOK, Phases, CompletedRequiresAllWork,
           NoFinalizeBeforeCheckpoint, EarlyTerminalSkipsFinalize,
           ForceQuitSkipsCheckpoint, FailedNeverCompletes,
           TerminalErrorNeverCompletes, MergeFails,
           Collecting, CleaningUp, SavingFinalCheckpoint, ComputingStats,
           Merging, Completed, Cancelled, ForceQuit, Failed

THEOREM GracefulCancelPreservesSafety ==
    ASSUME Safety, GracefulCancel
    PROVE  Safety'
BY SMT, TrueIsBoolean, FalseIsBoolean
DEF Safety, TypeOK, Phases, CompletedRequiresAllWork,
           NoFinalizeBeforeCheckpoint, EarlyTerminalSkipsFinalize,
           ForceQuitSkipsCheckpoint, FailedNeverCompletes,
           TerminalErrorNeverCompletes, GracefulCancel,
           Collecting, CleaningUp, SavingFinalCheckpoint, ComputingStats,
           Merging, Completed, Cancelled, ForceQuit, Failed

THEOREM ForceQuitAbortPreservesSafety ==
    ASSUME Safety, ForceQuitAbort
    PROVE  Safety'
<1>1. TypeOK'
<2>1. phase' \in Phases
  BY SMT DEF ForceQuitAbort, Phases, ForceQuit
<2>2. /\ cleanup_done' \in BOOLEAN
       /\ checkpoint_saved' \in BOOLEAN
  BY TrueIsBoolean, FalseIsBoolean DEF ForceQuitAbort
<2>3. /\ workers_drained' \in BOOLEAN
       /\ all_results_received' \in BOOLEAN
       /\ stats_computed' \in BOOLEAN
       /\ merge_done' \in BOOLEAN
       /\ completed_event' \in BOOLEAN
       /\ terminal_error' \in BOOLEAN
  BY SMT DEF Safety, TypeOK, ForceQuitAbort
<2>. QED
  BY <2>1, <2>2, <2>3 DEF TypeOK
<1>2. CompletedRequiresAllWork'
  BY SMT DEF CompletedRequiresAllWork, ForceQuitAbort, Completed
<1>3. NoFinalizeBeforeCheckpoint'
  BY SMT, FalseNotTrue
     DEF NoFinalizeBeforeCheckpoint, ForceQuitAbort, ComputingStats,
         Merging, Completed, ForceQuit
<1>4. EarlyTerminalSkipsFinalize'
  BY SMT DEF EarlyTerminalSkipsFinalize, ForceQuitAbort, Cancelled, ForceQuit
<1>5. ForceQuitSkipsCheckpoint'
  BY SMT DEF ForceQuitSkipsCheckpoint, ForceQuitAbort, ForceQuit
<1>6. FailedNeverCompletes'
  BY SMT DEF FailedNeverCompletes, ForceQuitAbort, Failed
<1>7. TerminalErrorNeverCompletes'
  BY SMT DEF TerminalErrorNeverCompletes, ForceQuitAbort
<1>. QED
  BY <1>1, <1>2, <1>3, <1>4, <1>5, <1>6, <1>7 DEF Safety

THEOREM MissingResultsAbortPreservesSafety ==
    ASSUME Safety, MissingResultsAbort
    PROVE  Safety'
<1>1. TypeOK'
<2>1. phase' \in Phases
  BY SMT DEF MissingResultsAbort, Phases, Failed
<2>2. /\ workers_drained' \in BOOLEAN
       /\ all_results_received' \in BOOLEAN
       /\ cleanup_done' \in BOOLEAN
       /\ terminal_error' \in BOOLEAN
  BY TrueIsBoolean, FalseIsBoolean DEF MissingResultsAbort
<2>3. checkpoint_saved' \in BOOLEAN
  BY DEF MissingResultsAbort
<2>4. /\ stats_computed' \in BOOLEAN
       /\ merge_done' \in BOOLEAN
       /\ completed_event' \in BOOLEAN
  BY SMT DEF Safety, TypeOK, MissingResultsAbort
<2>. QED
  BY <2>1, <2>2, <2>3, <2>4 DEF TypeOK
<1>2. CompletedRequiresAllWork'
  BY SMT DEF CompletedRequiresAllWork, MissingResultsAbort, Completed
<1>3. NoFinalizeBeforeCheckpoint'
  BY SMT, FalseNotTrue
     DEF NoFinalizeBeforeCheckpoint, MissingResultsAbort, ComputingStats,
         Merging, Completed, Failed
<1>4. EarlyTerminalSkipsFinalize'
  BY SMT DEF EarlyTerminalSkipsFinalize, MissingResultsAbort, Cancelled,
             ForceQuit
<1>5. ForceQuitSkipsCheckpoint'
  BY SMT DEF ForceQuitSkipsCheckpoint, MissingResultsAbort, ForceQuit, Failed
<1>6. FailedNeverCompletes'
  BY SMT DEF FailedNeverCompletes, MissingResultsAbort, Failed
<1>7. TerminalErrorNeverCompletes'
  BY SMT DEF TerminalErrorNeverCompletes, MissingResultsAbort
<1>. QED
  BY <1>1, <1>2, <1>3, <1>4, <1>5, <1>6, <1>7 DEF Safety

THEOREM NextPreservesSafety ==
    Safety /\ Next => Safety'
BY WorkerResultsCompletePreservesSafety,
   CleanupCompletePreservesSafety,
   FinalCheckpointSavedPreservesSafety,
   FinalCheckpointFailsPreservesSafety,
   StatsCompletePreservesSafety,
   StatsFailsPreservesSafety,
   MergeCompletePreservesSafety,
   MergeFailsPreservesSafety,
   GracefulCancelPreservesSafety,
   ForceQuitAbortPreservesSafety,
   MissingResultsAbortPreservesSafety
DEF Next

THEOREM StutterPreservesSafety ==
    Safety /\ UNCHANGED vars => Safety'
BY SMT DEF Safety, TypeOK, Phases, CompletedRequiresAllWork,
           NoFinalizeBeforeCheckpoint, EarlyTerminalSkipsFinalize,
           ForceQuitSkipsCheckpoint, FailedNeverCompletes,
           TerminalErrorNeverCompletes, vars

THEOREM StepPreservesSafety ==
    Safety /\ [Next]_vars => Safety'
BY NextPreservesSafety, StutterPreservesSafety DEF vars

THEOREM SpecImpliesAlwaysSafety ==
    Spec => []Safety
BY InitImpliesSafety, StepPreservesSafety, PTL DEF Spec

=============================================================================
