--------------------------- MODULE ImporterLifecycle ---------------------------
(*
 * Formal verification of the Google Books importer lifecycle after worker
 * processing has reached a terminal condition.
 *
 * Target:
 *   src/sources/google_books/importer/import_ops.rs
 *   src/sources/google_books/state_machine.rs
 *
 * The worker-level safety details are specified in WorkerShutdown.tla. This
 * module verifies the outer lifecycle ordering:
 *   - normal completion requires cleanup, final checkpoint, stats, and merge
 *   - graceful cancellation checkpoints and cleans up without finalization
 *   - force quit cleans up without checkpointing
 *   - missing worker results cannot fall through to finalization/completion
 *)

EXTENDS Naturals, TLC

VARIABLES
    \* @type: Int;
    phase,
    \* @type: Bool;
    workers_drained,
    \* @type: Bool;
    all_results_received,
    \* @type: Bool;
    cleanup_done,
    \* @type: Bool;
    checkpoint_saved,
    \* @type: Bool;
    stats_computed,
    \* @type: Bool;
    merge_done,
    \* @type: Bool;
    completed_event,
    \* @type: Bool;
    terminal_error

Collecting == 0
CleaningUp == 1
SavingFinalCheckpoint == 2
ComputingStats == 3
Merging == 4
Completed == 5
Cancelled == 6
ForceQuit == 7
Failed == 8

Phases == {Collecting, CleaningUp, SavingFinalCheckpoint, ComputingStats,
           Merging, Completed, Cancelled, ForceQuit, Failed}

TerminalPhases == {Completed, Cancelled, ForceQuit, Failed}

TypeOK ==
    /\ phase \in Phases
    /\ workers_drained \in BOOLEAN
    /\ all_results_received \in BOOLEAN
    /\ cleanup_done \in BOOLEAN
    /\ checkpoint_saved \in BOOLEAN
    /\ stats_computed \in BOOLEAN
    /\ merge_done \in BOOLEAN
    /\ completed_event \in BOOLEAN
    /\ terminal_error \in BOOLEAN

Init ==
    /\ phase = Collecting
    /\ workers_drained = FALSE
    /\ all_results_received = FALSE
    /\ cleanup_done = FALSE
    /\ checkpoint_saved = FALSE
    /\ stats_computed = FALSE
    /\ merge_done = FALSE
    /\ completed_event = FALSE
    /\ terminal_error = FALSE

(*
 * Normal path: all workers and results are accounted for, then the importer
 * executes deterministic cleanup before saving the final checkpoint.
 *)
WorkerResultsComplete ==
    /\ phase = Collecting
    /\ ~workers_drained
    /\ ~all_results_received
    /\ ~cleanup_done
    /\ ~checkpoint_saved
    /\ ~stats_computed
    /\ ~merge_done
    /\ ~completed_event
    /\ ~terminal_error
    /\ workers_drained' = TRUE
    /\ all_results_received' = TRUE
    /\ phase' = CleaningUp
    /\ UNCHANGED <<cleanup_done, checkpoint_saved, stats_computed, merge_done,
                   completed_event, terminal_error>>

CleanupComplete ==
    /\ phase = CleaningUp
    /\ workers_drained
    /\ all_results_received
    /\ ~cleanup_done
    /\ ~checkpoint_saved
    /\ ~stats_computed
    /\ ~merge_done
    /\ ~completed_event
    /\ ~terminal_error
    /\ cleanup_done' = TRUE
    /\ phase' = SavingFinalCheckpoint
    /\ UNCHANGED <<workers_drained, all_results_received, checkpoint_saved,
                   stats_computed, merge_done, completed_event, terminal_error>>

FinalCheckpointSaved ==
    /\ phase = SavingFinalCheckpoint
    /\ workers_drained
    /\ all_results_received
    /\ cleanup_done
    /\ ~checkpoint_saved
    /\ ~stats_computed
    /\ ~merge_done
    /\ ~completed_event
    /\ ~terminal_error
    /\ checkpoint_saved' = TRUE
    /\ phase' = ComputingStats
    /\ UNCHANGED <<workers_drained, all_results_received, cleanup_done,
                   stats_computed, merge_done, completed_event, terminal_error>>

FinalCheckpointFails ==
    /\ phase = SavingFinalCheckpoint
    /\ workers_drained
    /\ all_results_received
    /\ cleanup_done
    /\ ~stats_computed
    /\ ~merge_done
    /\ ~completed_event
    /\ ~terminal_error
    /\ terminal_error' = TRUE
    /\ phase' = Failed
    /\ UNCHANGED <<workers_drained, all_results_received, cleanup_done,
                   checkpoint_saved, stats_computed, merge_done, completed_event>>

StatsComplete ==
    /\ phase = ComputingStats
    /\ workers_drained
    /\ all_results_received
    /\ cleanup_done
    /\ checkpoint_saved
    /\ ~stats_computed
    /\ ~merge_done
    /\ ~completed_event
    /\ ~terminal_error
    /\ stats_computed' = TRUE
    /\ phase' = Merging
    /\ UNCHANGED <<workers_drained, all_results_received, cleanup_done,
                   checkpoint_saved, merge_done, completed_event, terminal_error>>

StatsFails ==
    /\ phase = ComputingStats
    /\ workers_drained
    /\ all_results_received
    /\ cleanup_done
    /\ checkpoint_saved
    /\ ~merge_done
    /\ ~completed_event
    /\ ~terminal_error
    /\ terminal_error' = TRUE
    /\ phase' = Failed
    /\ UNCHANGED <<workers_drained, all_results_received, cleanup_done,
                   checkpoint_saved, stats_computed, merge_done, completed_event>>

MergeComplete ==
    /\ phase = Merging
    /\ workers_drained
    /\ all_results_received
    /\ cleanup_done
    /\ checkpoint_saved
    /\ stats_computed
    /\ ~merge_done
    /\ ~completed_event
    /\ ~terminal_error
    /\ merge_done' = TRUE
    /\ completed_event' = TRUE
    /\ phase' = Completed
    /\ UNCHANGED <<workers_drained, all_results_received, cleanup_done,
                   checkpoint_saved, stats_computed, terminal_error>>

MergeFails ==
    /\ phase = Merging
    /\ workers_drained
    /\ all_results_received
    /\ cleanup_done
    /\ checkpoint_saved
    /\ stats_computed
    /\ ~completed_event
    /\ ~terminal_error
    /\ terminal_error' = TRUE
    /\ phase' = Failed
    /\ UNCHANGED <<workers_drained, all_results_received, cleanup_done,
                   checkpoint_saved, stats_computed, merge_done, completed_event>>

(*
 * Graceful cancellation occurs after WorkerShutdown has established the safe
 * checkpoint precondition. It checkpoints and cleans up, but does not finalize.
 *)
GracefulCancel ==
    /\ phase = Collecting
    /\ ~workers_drained
    /\ ~cleanup_done
    /\ ~checkpoint_saved
    /\ ~stats_computed
    /\ ~merge_done
    /\ ~completed_event
    /\ ~terminal_error
    /\ workers_drained' = TRUE
    /\ cleanup_done' = TRUE
    /\ checkpoint_saved' = TRUE
    /\ phase' = Cancelled
    /\ UNCHANGED <<all_results_received, stats_computed, merge_done,
                   completed_event, terminal_error>>

(*
 * Force quit avoids checkpointing. Worker tasks may be aborted, but importer
 * resources are still cleaned up before returning.
 *)
ForceQuitAbort ==
    /\ phase = Collecting
    /\ ~cleanup_done
    /\ ~checkpoint_saved
    /\ ~stats_computed
    /\ ~merge_done
    /\ ~completed_event
    /\ ~terminal_error
    /\ cleanup_done' = TRUE
    /\ checkpoint_saved' = FALSE
    /\ phase' = ForceQuit
    /\ UNCHANGED <<workers_drained, all_results_received, stats_computed,
                   merge_done, completed_event, terminal_error>>

(*
 * If workers exit before every result is accounted for, the importer may save
 * an emergency checkpoint but must return without finalization or completion.
 *)
MissingResultsAbort ==
    /\ phase = Collecting
    /\ ~workers_drained
    /\ ~all_results_received
    /\ ~cleanup_done
    /\ ~stats_computed
    /\ ~merge_done
    /\ ~completed_event
    /\ ~terminal_error
    /\ workers_drained' = TRUE
    /\ all_results_received' = FALSE
    /\ cleanup_done' = TRUE
    /\ checkpoint_saved' \in BOOLEAN
    /\ terminal_error' = TRUE
    /\ phase' = Failed
    /\ UNCHANGED <<stats_computed, merge_done, completed_event>>

Next ==
    \/ WorkerResultsComplete
    \/ CleanupComplete
    \/ FinalCheckpointSaved
    \/ FinalCheckpointFails
    \/ StatsComplete
    \/ StatsFails
    \/ MergeComplete
    \/ MergeFails
    \/ GracefulCancel
    \/ ForceQuitAbort
    \/ MissingResultsAbort

vars == <<phase, workers_drained, all_results_received, cleanup_done,
          checkpoint_saved, stats_computed, merge_done, completed_event,
          terminal_error>>

Spec == Init /\ [][Next]_vars

Fairness ==
    /\ WF_vars(WorkerResultsComplete)
    /\ WF_vars(CleanupComplete)
    /\ WF_vars(FinalCheckpointSaved)
    /\ WF_vars(FinalCheckpointFails)
    /\ WF_vars(StatsComplete)
    /\ WF_vars(StatsFails)
    /\ WF_vars(MergeComplete)
    /\ WF_vars(MergeFails)
    /\ WF_vars(GracefulCancel)
    /\ WF_vars(ForceQuitAbort)
    /\ WF_vars(MissingResultsAbort)

FairSpec == Spec /\ Fairness

CompletedRequiresAllWork ==
    completed_event = TRUE =>
        /\ phase = Completed
        /\ workers_drained = TRUE
        /\ all_results_received = TRUE
        /\ cleanup_done = TRUE
        /\ checkpoint_saved = TRUE
        /\ stats_computed = TRUE
        /\ merge_done = TRUE
        /\ terminal_error = FALSE

PhaseOrder ==
    /\ (phase = Collecting =>
            /\ ~workers_drained
            /\ ~all_results_received
            /\ ~cleanup_done
            /\ ~checkpoint_saved
            /\ ~stats_computed
            /\ ~merge_done
            /\ ~completed_event
            /\ ~terminal_error)
    /\ (phase = CleaningUp =>
            /\ workers_drained
            /\ all_results_received
            /\ ~cleanup_done
            /\ ~checkpoint_saved
            /\ ~stats_computed
            /\ ~merge_done
            /\ ~completed_event
            /\ ~terminal_error)
    /\ (phase = SavingFinalCheckpoint =>
            /\ workers_drained
            /\ all_results_received
            /\ cleanup_done
            /\ ~checkpoint_saved
            /\ ~stats_computed
            /\ ~merge_done
            /\ ~completed_event
            /\ ~terminal_error)
    /\ (phase = ComputingStats =>
            /\ workers_drained
            /\ all_results_received
            /\ cleanup_done
            /\ checkpoint_saved
            /\ ~stats_computed
            /\ ~merge_done
            /\ ~completed_event
            /\ ~terminal_error)
    /\ (phase = Merging =>
            /\ workers_drained
            /\ all_results_received
            /\ cleanup_done
            /\ checkpoint_saved
            /\ stats_computed
            /\ ~merge_done
            /\ ~completed_event
            /\ ~terminal_error)
    /\ (phase = Completed =>
            /\ workers_drained
            /\ all_results_received
            /\ cleanup_done
            /\ checkpoint_saved
            /\ stats_computed
            /\ merge_done
            /\ completed_event
            /\ ~terminal_error)
    /\ (phase = Cancelled =>
            /\ workers_drained
            /\ cleanup_done
            /\ checkpoint_saved
            /\ ~stats_computed
            /\ ~merge_done
            /\ ~completed_event
            /\ ~terminal_error)
    /\ (phase = ForceQuit =>
            /\ cleanup_done
            /\ ~checkpoint_saved
            /\ ~stats_computed
            /\ ~merge_done
            /\ ~completed_event
            /\ ~terminal_error)
    /\ (phase = Failed =>
            /\ cleanup_done
            /\ terminal_error
            /\ ~completed_event)

NoFinalizeBeforeCheckpoint ==
    (phase \in {ComputingStats, Merging, Completed} \/ stats_computed = TRUE \/
     merge_done = TRUE \/ completed_event = TRUE) => checkpoint_saved = TRUE

EarlyTerminalSkipsFinalize ==
    phase \in {Cancelled, ForceQuit} =>
        /\ cleanup_done = TRUE
        /\ stats_computed = FALSE
        /\ merge_done = FALSE
        /\ completed_event = FALSE

ForceQuitSkipsCheckpoint ==
    phase = ForceQuit => checkpoint_saved = FALSE

FailedNeverCompletes ==
    phase = Failed => completed_event = FALSE

TerminalErrorNeverCompletes ==
    terminal_error = TRUE => completed_event = FALSE

Safety ==
    /\ TypeOK
    /\ CompletedRequiresAllWork
    /\ NoFinalizeBeforeCheckpoint
    /\ EarlyTerminalSkipsFinalize
    /\ ForceQuitSkipsCheckpoint
    /\ FailedNeverCompletes
    /\ TerminalErrorNeverCompletes

LifecycleEventuallyTerminates ==
    phase = Collecting ~> phase \in TerminalPhases

=============================================================================
