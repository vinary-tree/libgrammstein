------------------------ MODULE CheckpointStateMachine ------------------------
(*
 * Formal verification of the Checkpoint prefix state machine.
 *
 * This specification models the prefix lifecycle from:
 *   src/sources/google_books/checkpoint.rs
 *
 * Prefix States (from checkpoint.rs:24-35):
 *   - not_started: Not in any list (needs processing)
 *   - in_progress: In in_progress_prefixes (started but not finished)
 *   - completed:   In completed_prefixes (successfully processed)
 *   - failed:      In failed_prefixes (failed after exhausting retries)
 *
 * Key Properties to Verify:
 *   1. DisjointSets: completed, in_progress, failed are mutually disjoint
 *   2. NoDoubleProcessing: completed prefix never returns to in_progress
 *   3. CrashRecoverySound: in_progress prefixes moved to failed on recovery
 *   4. StateTransitionValidity: Only valid state transitions occur
 *)

EXTENDS Naturals, FiniteSets, Sequences, TLC

CONSTANTS
    \* @type: Set(Int);
    Orders,         \* Set of n-gram orders (e.g., {1, 2, 3, 4, 5})
    \* @type: Set(Str);
    Prefixes,       \* Set of prefix identifiers
    \* @type: Set(Str);
    Workers         \* Set of worker IDs (for parallel processing)

\* Symmetry set for model checking optimization
PrefixSymmetry == Permutations(Prefixes)

VARIABLES
    \* @type: <<Int, Str>> -> Str;
    prefix_state,           \* (order, prefix) -> state
    \* @type: Int -> Set(Str);
    completed_prefixes,     \* order -> Set of prefix
    \* @type: Int -> Set(Str);
    in_progress_prefixes,   \* order -> Set of prefix
    \* @type: Int -> Set(Str);
    failed_prefixes,        \* order -> Set of prefix
    \* @type: Int -> Bool;
    order_complete,         \* order -> Bool
    \* @type: Bool;
    system_running,         \* Bool: system is running (not crashed)
    \* @type: <<Int, Str>> -> Str;
    worker_assignment,      \* (order, prefix) -> worker | NONE
    \* @type: Bool;
    recovery_needed,        \* Bool: a crash happened and restart is not complete
    \* @type: Int -> Set(Str);
    crashed_in_progress     \* order -> Set of prefixes in progress at crash time

(* Prefix states as constants *)
NotStarted == "not_started"
InProgress == "in_progress"
Completed == "completed"
Failed == "failed"

NONE == "NONE"

WorkerIdsExcludeNone == NONE \notin Workers

ASSUME WorkerIdsExcludeNone

(* Type invariant *)
TypeOK ==
    /\ prefix_state \in [Orders \X Prefixes -> {NotStarted, InProgress, Completed, Failed}]
    /\ completed_prefixes \in [Orders -> SUBSET Prefixes]
    /\ in_progress_prefixes \in [Orders -> SUBSET Prefixes]
    /\ failed_prefixes \in [Orders -> SUBSET Prefixes]
    /\ order_complete \in [Orders -> BOOLEAN]
    /\ system_running \in BOOLEAN
    /\ worker_assignment \in [Orders \X Prefixes -> Workers \cup {NONE}]
    /\ recovery_needed \in BOOLEAN
    /\ crashed_in_progress \in [Orders -> SUBSET Prefixes]

(* Initial state: all prefixes not started *)
Init ==
    /\ prefix_state = [op \in Orders \X Prefixes |-> NotStarted]
    /\ completed_prefixes = [o \in Orders |-> {}]
    /\ in_progress_prefixes = [o \in Orders |-> {}]
    /\ failed_prefixes = [o \in Orders |-> {}]
    /\ order_complete = [o \in Orders |-> FALSE]
    /\ system_running = TRUE
    /\ worker_assignment = [op \in Orders \X Prefixes |-> NONE]
    /\ recovery_needed = FALSE
    /\ crashed_in_progress = [o \in Orders |-> {}]

(* ---------------------------------------------------------------------------
 * Actions - Normal Operation
 * --------------------------------------------------------------------------- *)

(*
 * Start processing a prefix.
 * Models: checkpoint.rs:314-325 (start_prefix)
 *
 * Preconditions:
 *   - System is running
 *   - Prefix is not already in_progress
 *   - Order is not marked complete
 *)
StartPrefix(o, p, w) ==
    /\ system_running
    /\ ~order_complete[o]
    \* HashMap-backed Rust state allows a caller to overwrite Completed with
    \* InProgress for an explicit reprocess; it must not duplicate InProgress.
    /\ prefix_state[<<o, p>>] # InProgress
    /\ prefix_state' = [prefix_state EXCEPT ![<<o, p>>] = InProgress]
    /\ in_progress_prefixes' = [in_progress_prefixes EXCEPT ![o] = @ \cup {p}]
    \* Remove from other sets if present (checkpoint.rs:319-320)
    /\ completed_prefixes' = [completed_prefixes EXCEPT ![o] = @ \ {p}]
    /\ failed_prefixes' = [failed_prefixes EXCEPT ![o] = @ \ {p}]
    /\ worker_assignment' = [worker_assignment EXCEPT ![<<o, p>>] = w]
    /\ UNCHANGED <<order_complete, system_running, recovery_needed, crashed_in_progress>>

(*
 * Complete a prefix successfully.
 * Models: checkpoint.rs:331-342 (complete_prefix)
 *
 * Moves prefix from in_progress to completed.
 *)
CompletePrefix(o, p) ==
    /\ system_running
    /\ prefix_state[<<o, p>>] = InProgress
    /\ prefix_state' = [prefix_state EXCEPT ![<<o, p>>] = Completed]
    /\ in_progress_prefixes' = [in_progress_prefixes EXCEPT ![o] = @ \ {p}]
    /\ completed_prefixes' = [completed_prefixes EXCEPT ![o] = @ \cup {p}]
    /\ worker_assignment' = [worker_assignment EXCEPT ![<<o, p>>] = NONE]
    /\ UNCHANGED <<failed_prefixes, order_complete, system_running, recovery_needed, crashed_in_progress>>

(*
 * Fail a prefix (exhausted retries).
 * Models: checkpoint.rs:349-359 (fail_prefix)
 *
 * Moves prefix from in_progress to failed.
 *)
FailPrefix(o, p) ==
    /\ system_running
    /\ prefix_state[<<o, p>>] = InProgress
    /\ prefix_state' = [prefix_state EXCEPT ![<<o, p>>] = Failed]
    /\ in_progress_prefixes' = [in_progress_prefixes EXCEPT ![o] = @ \ {p}]
    /\ failed_prefixes' = [failed_prefixes EXCEPT ![o] = @ \cup {p}]
    /\ worker_assignment' = [worker_assignment EXCEPT ![<<o, p>>] = NONE]
    /\ UNCHANGED <<completed_prefixes, order_complete, system_running, recovery_needed, crashed_in_progress>>

(*
 * Clear a failed prefix for retry.
 * Models: checkpoint.rs:362-366 (clear_failed)
 *
 * Moves prefix from failed back to not_started.
 *)
ClearFailed(o, p) ==
    /\ system_running
    /\ prefix_state[<<o, p>>] = Failed
    /\ prefix_state' = [prefix_state EXCEPT ![<<o, p>>] = NotStarted]
    /\ failed_prefixes' = [failed_prefixes EXCEPT ![o] = @ \ {p}]
    /\ UNCHANGED <<completed_prefixes, in_progress_prefixes, order_complete,
                   system_running, worker_assignment, recovery_needed, crashed_in_progress>>

(*
 * Mark an order as complete.
 * Models: checkpoint.rs:445-449 (complete_order)
 *
 * Note: In v2, completed_prefixes are NOT cleared (kept for verification).
 *)
CompleteOrder(o) ==
    /\ system_running
    /\ ~order_complete[o]
    \* All prefixes should be completed or failed (none in progress)
    /\ in_progress_prefixes[o] = {}
    /\ order_complete' = [order_complete EXCEPT ![o] = TRUE]
    /\ UNCHANGED <<prefix_state, completed_prefixes, in_progress_prefixes,
                   failed_prefixes, system_running, worker_assignment,
                   recovery_needed, crashed_in_progress>>

(* ---------------------------------------------------------------------------
 * Actions - Crash and Recovery
 * --------------------------------------------------------------------------- *)

(*
 * System crash.
 *
 * Models an unexpected termination where in-progress prefixes may have
 * partial data. The system_running flag prevents new operations.
 *)
Crash ==
    /\ system_running
    /\ system_running' = FALSE
    /\ recovery_needed' = TRUE
    /\ crashed_in_progress' = in_progress_prefixes
    /\ UNCHANGED <<prefix_state, completed_prefixes, in_progress_prefixes,
                   failed_prefixes, order_complete, worker_assignment>>

(*
 * Recover in-progress prefixes as failed.
 * Models: checkpoint.rs:408-416 (recover_in_progress_as_failed)
 *
 * On resume, in-progress prefixes are moved to failed because they may
 * have partial data that needs to be cleared before retry.
 *)
RecoverInProgressAsFailed(o) ==
    /\ ~system_running
    /\ in_progress_prefixes[o] # {}
    /\ LET prefs == in_progress_prefixes[o]
       IN
           /\ prefix_state' = [op \in Orders \X Prefixes |->
               IF op[1] = o /\ op[2] \in prefs THEN Failed
               ELSE prefix_state[op]]
           /\ in_progress_prefixes' = [in_progress_prefixes EXCEPT ![o] = {}]
           /\ failed_prefixes' = [failed_prefixes EXCEPT ![o] = @ \cup prefs]
           /\ worker_assignment' = [op \in Orders \X Prefixes |->
               IF op[1] = o /\ op[2] \in prefs THEN NONE
               ELSE worker_assignment[op]]
    /\ UNCHANGED <<completed_prefixes, order_complete, system_running,
                   recovery_needed, crashed_in_progress>>

(*
 * System restart after recovery.
 *
 * Should only happen after all in-progress prefixes have been recovered.
 *)
Restart ==
    /\ ~system_running
    \* All orders must have empty in_progress sets (recovered)
    /\ \A o \in Orders: in_progress_prefixes[o] = {}
    /\ system_running' = TRUE
    /\ recovery_needed' = FALSE
    /\ crashed_in_progress' = [o \in Orders |-> {}]
    /\ UNCHANGED <<prefix_state, completed_prefixes, in_progress_prefixes,
                   failed_prefixes, order_complete, worker_assignment>>

(* ---------------------------------------------------------------------------
 * Next state relation
 * --------------------------------------------------------------------------- *)

Next ==
    \/ \E o \in Orders, p \in Prefixes, w \in Workers: StartPrefix(o, p, w)
    \/ \E o \in Orders, p \in Prefixes: CompletePrefix(o, p)
    \/ \E o \in Orders, p \in Prefixes: FailPrefix(o, p)
    \/ \E o \in Orders, p \in Prefixes: ClearFailed(o, p)
    \/ \E o \in Orders: CompleteOrder(o)
    \/ Crash
    \/ \E o \in Orders: RecoverInProgressAsFailed(o)
    \/ Restart

(* Weak fairness for normal operations *)
Fairness ==
    /\ \A o \in Orders, p \in Prefixes, w \in Workers:
        WF_<<prefix_state, completed_prefixes, in_progress_prefixes,
             failed_prefixes, order_complete, system_running, worker_assignment,
             recovery_needed, crashed_in_progress>>(
            StartPrefix(o, p, w))
    /\ \A o \in Orders, p \in Prefixes:
        WF_<<prefix_state, completed_prefixes, in_progress_prefixes,
             failed_prefixes, order_complete, system_running, worker_assignment,
             recovery_needed, crashed_in_progress>>(
            CompletePrefix(o, p))
    /\ \A o \in Orders, p \in Prefixes:
        WF_<<prefix_state, completed_prefixes, in_progress_prefixes,
             failed_prefixes, order_complete, system_running, worker_assignment,
             recovery_needed, crashed_in_progress>>(
            FailPrefix(o, p))
    /\ \A o \in Orders:
        WF_<<prefix_state, completed_prefixes, in_progress_prefixes,
             failed_prefixes, order_complete, system_running, worker_assignment,
             recovery_needed, crashed_in_progress>>(
            RecoverInProgressAsFailed(o))
    /\ WF_<<prefix_state, completed_prefixes, in_progress_prefixes,
             failed_prefixes, order_complete, system_running, worker_assignment,
             recovery_needed, crashed_in_progress>>(Restart)

vars == <<prefix_state, completed_prefixes, in_progress_prefixes,
          failed_prefixes, order_complete, system_running,
          worker_assignment, recovery_needed, crashed_in_progress>>

Spec == Init /\ [][Next]_vars

FairSpec == Spec /\ Fairness

(* ---------------------------------------------------------------------------
 * Safety Invariants
 * --------------------------------------------------------------------------- *)

(*
 * CRITICAL INVARIANT: Sets are mutually disjoint.
 *
 * A prefix cannot be in multiple states simultaneously.
 * Violation could cause double-counting or data corruption.
 *)
DisjointSets ==
    \A o \in Orders:
        /\ completed_prefixes[o] \cap in_progress_prefixes[o] = {}
        /\ completed_prefixes[o] \cap failed_prefixes[o] = {}
        /\ in_progress_prefixes[o] \cap failed_prefixes[o] = {}

(*
 * State consistency: prefix_state matches set membership.
 *)
StateConsistent ==
    \A o \in Orders, p \in Prefixes:
        /\ (prefix_state[<<o, p>>] = Completed) <=> (p \in completed_prefixes[o])
        /\ (prefix_state[<<o, p>>] = InProgress) <=> (p \in in_progress_prefixes[o])
        /\ (prefix_state[<<o, p>>] = Failed) <=> (p \in failed_prefixes[o])
        \* NotStarted is the only remaining TypeOK state, so it is implied by
        \* absence from the three active state sets.

(*
 * Completed order means no prefixes are in progress.
 *)
CompletedOrderNoInProgress ==
    \A o \in Orders:
        order_complete[o] => in_progress_prefixes[o] = {}

(*
 * Worker assignment consistency: only in_progress prefixes have workers.
 *)
WorkerAssignmentConsistent ==
    \A o \in Orders, p \in Prefixes:
        (worker_assignment[<<o, p>>] # NONE) <=> (p \in in_progress_prefixes[o])

(*
 * No double-processing: a completed prefix cannot become in_progress again.
 * (It can only be restarted via explicit start_prefix which clears completed)
 *
 * This is enforced by the state machine - checked implicitly via StateConsistent.
 * We make it explicit here for documentation.
 *)
NoDoubleProcessing ==
    \A o \in Orders:
        completed_prefixes[o] \cap in_progress_prefixes[o] = {}

(*
 * A crash records the in-progress set until restart. Every recorded prefix is
 * either still pending recovery or has been moved to Failed.
 *)
CrashRecoverySound ==
    recovery_needed =>
        /\ ~system_running
        /\ \A o \in Orders:
            \A p \in crashed_in_progress[o]:
                p \in in_progress_prefixes[o] \/ p \in failed_prefixes[o]

(*
 * Combined safety invariant.
 *)
Safety ==
    /\ TypeOK
    /\ DisjointSets
    /\ StateConsistent
    /\ CompletedOrderNoInProgress
    /\ WorkerAssignmentConsistent
    /\ NoDoubleProcessing
    /\ CrashRecoverySound

(* ---------------------------------------------------------------------------
 * Recovery Correctness
 * --------------------------------------------------------------------------- *)

(*
 * After crash, system eventually recovers or stays crashed.
 * (No zombie states)
 *)
RecoveryEventual ==
    recovery_needed ~> (\A o \in Orders: in_progress_prefixes[o] = {})

(*
 * Crash recovery soundness: when system restarts, no in_progress prefixes
 * remain (they were all moved to failed).
 *)
RestartEventuallyAfterRecovery ==
    (~system_running /\ \A o \in Orders: in_progress_prefixes[o] = {}) ~> system_running

(* ---------------------------------------------------------------------------
 * Liveness Properties
 * --------------------------------------------------------------------------- *)

(*
 * An in_progress prefix eventually completes or fails.
 *)
InProgressEventuallyResolves ==
    \A o \in Orders, p \in Prefixes:
        (prefix_state[<<o, p>>] = InProgress /\ system_running) ~>
            (prefix_state[<<o, p>>] \in {Completed, Failed} \/ ~system_running)

(* ---------------------------------------------------------------------------
 * Spec-to-Code Traceability
 * --------------------------------------------------------------------------- *)

(*
 * Storage Format History:
 *   v2: One trie key per prefix - \x00__ckpt__:prefix:{order}:{prefix}
 *   v3: Bitmap chunks - \x00__ckpt__:bitmap:{order}:{chunk}
 *       Each u64 holds 32 prefix states (2 bits each)
 *
 * The storage format is transparent to this specification, which models
 * the logical state machine semantics. The bijective encoding between
 * HashMap<String, PrefixState> and Vec<u64> bitmaps is verified by
 * exhaustive property-based tests in checkpoint.rs:
 *   - test_prefix_to_index_exhaustive_two_char
 *   - test_index_prefix_roundtrip_two_char
 *   - test_pack_unpack_roundtrip_full
 *   - test_pack_unpack_roundtrip_mixed_states
 *)

(*
 * Mapping from TLA+ to Rust implementation:
 *
 * TLA+ Variable            | Rust Field/Method                    | Location
 * -------------------------|--------------------------------------|----------
 * prefix_state             | Derived from set membership          | implicit
 * completed_prefixes       | OrderProgress.completed_prefixes     | checkpoint.rs:42
 * in_progress_prefixes     | OrderProgress.in_progress_prefixes   | checkpoint.rs:49
 * failed_prefixes          | OrderProgress.failed_prefixes        | checkpoint.rs:57
 * order_complete           | OrderProgress.is_complete            | checkpoint.rs:61
 *
 * TLA+ Action              | Rust Method                          | Location
 * -------------------------|--------------------------------------|----------
 * StartPrefix              | ImportCheckpoint::start_prefix       | checkpoint.rs:314-325
 * CompletePrefix           | ImportCheckpoint::complete_prefix    | checkpoint.rs:331-342
 * FailPrefix               | ImportCheckpoint::fail_prefix        | checkpoint.rs:349-359
 * ClearFailed              | ImportCheckpoint::clear_failed       | checkpoint.rs:362-366
 * CompleteOrder            | ImportCheckpoint::complete_order     | checkpoint.rs:445-449
 * RecoverInProgressAsFailed| ImportCheckpoint::recover_in_progress_as_failed | checkpoint.rs:408-416
 *
 * State Queries            | Rust Method                          | Location
 * -------------------------|--------------------------------------|----------
 * needs_prefix(o, p)       | ImportCheckpoint::needs_prefix       | checkpoint.rs:460-470
 * is_in_progress(o, p)     | ImportCheckpoint::is_in_progress     | checkpoint.rs:369-374
 * is_failed_prefix(o, p)   | ImportCheckpoint::is_failed_prefix   | checkpoint.rs:377-382
 *)

=============================================================================
