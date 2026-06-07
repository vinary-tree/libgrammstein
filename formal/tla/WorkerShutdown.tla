---------------------------- MODULE WorkerShutdown ----------------------------
(*
 * Formal verification of the Worker Shutdown Protocol.
 *
 * This specification models the worker lifecycle and graceful shutdown
 * protocol from:
 *   src/sources/google_books/importer/import_ops.rs
 *   src/sources/google_books/importer/worker_pool.rs
 *
 * Worker States:
 *   - Idle: No job, waiting to poll queue
 *   - PollingQueue: Attempting to dequeue a job
 *   - Processing: Executing a job
 *   - SendingResult: Sending job result to main task
 *   - Exiting: Worker has decided to exit
 *   - Exited: Worker has terminated
 *
 * Key Properties to Verify:
 *   1. WorkersEventuallyTerminate: All workers exit after shutdown signal
 *   2. NoJobLost: Every started job sends a result before worker exits
 *   3. CheckpointAfterDrain: Checkpoint happens only after result channel drained
 *   4. ProcessingHasJob: Worker in Processing state always has a job
 *   5. ForceQuitSkipsCheckpoint: Force quit aborts without checkpointing
 *
 * Spec-to-Code Traceability:
 *
 * TLA+ Variable            | Rust Implementation                  | Location
 * -------------------------|--------------------------------------|----------
 * shutdown_signaled        | shutdown_rx (watch channel)          | worker_pool.rs
 * worker_state             | worker_task loop state               | worker_pool.rs
 * worker_job               | Job from job_rx.recv()               | worker_pool.rs
 * job_queue                | async_channel<Job>                   | import_ops.rs
 * results_channel          | mpsc::channel<JobResult>             | import_ops.rs
 * results_received         | results_received counter             | import_ops.rs
 * checkpoint_state         | cancellation/checkpoint control      | import_ops.rs
 *
 * TLA+ Action              | Rust Location                        | Lines
 * -------------------------|--------------------------------------|------
 * WorkerCheckShutdownBeforePoll | if *shutdown_rx.borrow()        | 647-651
 * WorkerShutdownWhileWaiting    | tokio::select! shutdown branch  | 656-661
 * WorkerPickJob                 | job_rx.recv()                   | 663
 * WorkerFinishProcessing        | process_single_attempt() return | 799-800
 * WorkerSendResult              | result_tx.send(job_result)      | 820
 * SignalShutdown                | signal_all_shutdown()
 * ReceiveResult                 | result_rx.recv() in drain loop
 * StartCheckpoint               | after all workers exit
 * ForceQuit                     | ForceQuit command path
 * AbortDrainWithoutCheckpoint   | worker-exit tracking failure path
 *)

EXTENDS Naturals, FiniteSets, TLC

CONSTANTS
    \* @type: Set(Str);
    Workers,        \* Set of worker IDs (e.g., {w1, w2, w3})
    \* @type: Set(Str);
    Jobs,           \* Set of job identifiers
    \* @type: Int;
    MaxRetries      \* Maximum retry attempts per job

\* Symmetry sets for model checking optimization
WorkerSymmetry == Permutations(Workers)
JobSymmetry == Permutations(Jobs)

VARIABLES
    \* Worker state
    \* @type: Str -> Str;
    worker_state,       \* Worker -> {Idle, PollingQueue, Processing, SendingResult, Exiting, Exited}
    \* @type: Str -> Str;
    worker_job,         \* Worker -> Job | NONE (current job being processed)

    \* Job queue abstraction. FIFO ordering is irrelevant to the shutdown
    \* safety and liveness properties verified here.
    \* @type: Set(Str);
    job_queue,          \* Set of jobs waiting to be processed

    \* Result channel
    \* @type: Set(Str);
    results_pending,    \* Set of jobs whose results are in the channel (sent but not received)
    \* @type: Set(Str);
    results_received,   \* Set of jobs whose results have been received by main task

    \* Shutdown coordination
    \* @type: Bool;
    shutdown_signaled,  \* Boolean: shutdown signal has been sent

    \* Checkpoint state
    \* @type: Str;
    checkpoint_state    \* {NotStarted, Draining, Checkpointing, Done,
                            \* ForceQuitAborted, DrainAborted}

(* State constants *)
Idle == "Idle"
PollingQueue == "PollingQueue"
Processing == "Processing"
SendingResult == "SendingResult"
Exiting == "Exiting"
Exited == "Exited"

NONE == "NONE"

NotStarted == "NotStarted"
Draining == "Draining"
Checkpointing == "Checkpointing"
Done == "Done"
ForceQuitAborted == "ForceQuitAborted"
DrainAborted == "DrainAborted"

(* Type invariant *)
TypeOK ==
    /\ worker_state \in [Workers -> {Idle, PollingQueue, Processing, SendingResult, Exiting, Exited}]
    /\ worker_job \in [Workers -> Jobs \cup {NONE}]
    /\ job_queue \subseteq Jobs
    /\ results_pending \subseteq Jobs
    /\ results_received \subseteq Jobs
    /\ shutdown_signaled \in BOOLEAN
    /\ checkpoint_state \in {NotStarted, Draining, Checkpointing, Done,
                             ForceQuitAborted, DrainAborted}

(* Initial state: all workers idle, jobs queued, no shutdown *)
Init ==
    /\ worker_state = [w \in Workers |-> Idle]
    /\ worker_job = [w \in Workers |-> NONE]
    /\ job_queue = Jobs
    /\ results_pending = {}
    /\ results_received = {}
    /\ shutdown_signaled = FALSE
    /\ checkpoint_state = NotStarted

(* ---------------------------------------------------------------------------
 * Actions - Normal Operation (before shutdown)
 * --------------------------------------------------------------------------- *)

(*
 * Worker checks shutdown before polling for work.
 * Models: importer.rs:647-651
 *
 * If shutdown is signaled, transition directly to Exiting.
 *)
WorkerCheckShutdownBeforePoll(w) ==
    /\ worker_state[w] = Idle
    /\ shutdown_signaled
    /\ worker_state' = [worker_state EXCEPT ![w] = Exiting]
    /\ UNCHANGED <<worker_job, job_queue, results_pending, results_received,
                   shutdown_signaled, checkpoint_state>>

(*
 * Worker starts polling for a job.
 * Models: importer.rs:654-663 (entering tokio::select!)
 *)
WorkerStartPolling(w) ==
    /\ worker_state[w] = Idle
    /\ ~shutdown_signaled  \* Would have taken Exiting path if shutdown
    /\ worker_state' = [worker_state EXCEPT ![w] = PollingQueue]
    /\ UNCHANGED <<worker_job, job_queue, results_pending, results_received,
                   shutdown_signaled, checkpoint_state>>

(*
 * Worker receives shutdown signal while waiting for job.
 * Models: importer.rs:656-661 (shutdown_rx.changed() branch)
 *)
WorkerShutdownWhileWaiting(w) ==
    /\ worker_state[w] = PollingQueue
    /\ shutdown_signaled
    /\ worker_state' = [worker_state EXCEPT ![w] = Exiting]
    /\ UNCHANGED <<worker_job, job_queue, results_pending, results_received,
                   shutdown_signaled, checkpoint_state>>

(*
 * Worker successfully dequeues a job.
 * Models: importer.rs:663 (job_rx.recv() returns Some(job))
 *)
WorkerPickJob(w) ==
    /\ worker_state[w] = PollingQueue
    /\ job_queue # {}
    /\ \E job \in job_queue:
        /\ worker_state' = [worker_state EXCEPT ![w] = Processing]
        /\ worker_job' = [worker_job EXCEPT ![w] = job]
        /\ job_queue' = job_queue \ {job}
    /\ UNCHANGED <<results_pending, results_received, shutdown_signaled, checkpoint_state>>

(*
 * Worker finds queue empty and exits.
 * Models: importer.rs:666-677 (job_rx.recv() returns None / queue closed)
 *)
WorkerQueueEmpty(w) ==
    /\ worker_state[w] = PollingQueue
    /\ job_queue = {}
    /\ ~shutdown_signaled  \* If shutdown, WorkerShutdownWhileWaiting takes priority
    /\ worker_state' = [worker_state EXCEPT ![w] = Exiting]
    /\ UNCHANGED <<worker_job, job_queue, results_pending, results_received,
                   shutdown_signaled, checkpoint_state>>

(*
 * Worker finishes processing a job.
 * Models: importer.rs:799-800 (process_single_attempt returns)
 *
 * Note: Worker transitions to SendingResult before checking shutdown again.
 * This ensures the job's result is sent before worker considers exiting.
 *)
WorkerFinishProcessing(w) ==
    /\ worker_state[w] = Processing
    /\ worker_job[w] # NONE
    /\ worker_state' = [worker_state EXCEPT ![w] = SendingResult]
    /\ UNCHANGED <<worker_job, job_queue, results_pending, results_received,
                   shutdown_signaled, checkpoint_state>>

(*
 * Worker sends job result to result channel.
 * Models: importer.rs:820 (result_tx.send(job_result))
 *)
WorkerSendResult(w) ==
    /\ worker_state[w] = SendingResult
    /\ worker_job[w] # NONE
    /\ results_pending' = results_pending \cup {worker_job[w]}
    /\ worker_job' = [worker_job EXCEPT ![w] = NONE]
    /\ worker_state' = [worker_state EXCEPT ![w] = Idle]
    /\ UNCHANGED <<job_queue, results_received, shutdown_signaled, checkpoint_state>>

(*
 * Worker transitions from Exiting to Exited.
 * Models: End of worker_task function
 *)
WorkerExit(w) ==
    /\ worker_state[w] = Exiting
    /\ worker_state' = [worker_state EXCEPT ![w] = Exited]
    /\ UNCHANGED <<worker_job, job_queue, results_pending, results_received,
                   shutdown_signaled, checkpoint_state>>

(* ---------------------------------------------------------------------------
 * Actions - Shutdown and Checkpoint
 * --------------------------------------------------------------------------- *)

(*
 * Main task signals shutdown to all workers.
 * Models: signal_all_shutdown().
 *)
SignalShutdown ==
    /\ ~shutdown_signaled
    /\ shutdown_signaled' = TRUE
    /\ checkpoint_state' = Draining
    /\ UNCHANGED <<worker_state, worker_job, job_queue, results_pending, results_received>>

(*
 * Main task receives a result during drain phase.
 * Models: result_rx.recv() in the cancellation drain loop.
 *)
ReceiveResult ==
    /\ checkpoint_state = Draining
    /\ results_pending # {}
    /\ \E job \in results_pending:
           /\ results_pending' = results_pending \ {job}
           /\ results_received' = results_received \cup {job}
    /\ UNCHANGED <<worker_state, worker_job, job_queue, shutdown_signaled, checkpoint_state>>

(*
 * Main task starts checkpoint after all results drained.
 * Models: after cancellation observes every worker exit, before save_checkpoint.
 *
 * Precondition: No results pending AND all workers have exited.
 * This ensures no more results will be produced.
 *)
StartCheckpoint ==
    /\ checkpoint_state = Draining
    /\ results_pending = {}
    /\ \A w \in Workers: worker_state[w] = Exited
    /\ checkpoint_state' = Checkpointing
    /\ UNCHANGED <<worker_state, worker_job, job_queue, results_pending,
                   results_received, shutdown_signaled>>

(*
 * Checkpoint completes.
 * Models: save_checkpoint() returns.
 *)
CompleteCheckpoint ==
    /\ checkpoint_state = Checkpointing
    /\ checkpoint_state' = Done
    /\ UNCHANGED <<worker_state, worker_job, job_queue, results_pending,
                   results_received, shutdown_signaled>>

(*
 * Force quit interrupts without writing a checkpoint. It still signals workers
 * to stop, but the main task returns before waiting for checkpoint safety.
 *)
ForceQuit ==
    /\ checkpoint_state \in {NotStarted, Draining}
    /\ shutdown_signaled' = TRUE
    /\ checkpoint_state' = ForceQuitAborted
    /\ UNCHANGED <<worker_state, worker_job, job_queue, results_pending,
                   results_received>>

(*
 * If worker-exit tracking fails during graceful cancellation, the main task
 * aborts rather than checkpointing from an unproved worker state.
 *)
AbortDrainWithoutCheckpoint ==
    /\ checkpoint_state = Draining
    /\ shutdown_signaled
    /\ checkpoint_state' = DrainAborted
    /\ UNCHANGED <<worker_state, worker_job, job_queue, results_pending,
                   results_received, shutdown_signaled>>

(* ---------------------------------------------------------------------------
 * Next state relation
 * --------------------------------------------------------------------------- *)

Next ==
    \/ \E w \in Workers:
        \/ WorkerCheckShutdownBeforePoll(w)
        \/ WorkerStartPolling(w)
        \/ WorkerShutdownWhileWaiting(w)
        \/ WorkerPickJob(w)
        \/ WorkerQueueEmpty(w)
        \/ WorkerFinishProcessing(w)
        \/ WorkerSendResult(w)
        \/ WorkerExit(w)
    \/ SignalShutdown
    \/ ReceiveResult
    \/ StartCheckpoint
    \/ CompleteCheckpoint
    \/ ForceQuit
    \/ AbortDrainWithoutCheckpoint

(* Fairness: all enabled actions eventually happen *)
vars == <<worker_state, worker_job, job_queue, results_pending, results_received,
          shutdown_signaled, checkpoint_state>>

Fairness ==
    /\ \A w \in Workers:
        /\ WF_vars(WorkerCheckShutdownBeforePoll(w))
        /\ WF_vars(WorkerStartPolling(w))
        /\ WF_vars(WorkerShutdownWhileWaiting(w))
        /\ WF_vars(WorkerPickJob(w))
        /\ WF_vars(WorkerQueueEmpty(w))
        /\ WF_vars(WorkerFinishProcessing(w))
        /\ WF_vars(WorkerSendResult(w))
        /\ WF_vars(WorkerExit(w))
    /\ WF_vars(SignalShutdown)
    /\ WF_vars(ReceiveResult)
    /\ WF_vars(StartCheckpoint)
    /\ WF_vars(CompleteCheckpoint)

Spec == Init /\ [][Next]_vars

FairSpec == Spec /\ Fairness

(* ---------------------------------------------------------------------------
 * Safety Invariants
 * --------------------------------------------------------------------------- *)

(*
 * CRITICAL INVARIANT: Worker in Processing state has a job.
 *)
ProcessingHasJob ==
    \A w \in Workers:
        worker_state[w] = Processing => worker_job[w] # NONE

(*
 * Worker in SendingResult state has a job.
 *)
SendingHasJob ==
    \A w \in Workers:
        worker_state[w] = SendingResult => worker_job[w] # NONE

(*
 * Idle, Exiting, and Exited workers have no job.
 *)
IdleNoJob ==
    \A w \in Workers:
        worker_state[w] \in {Idle, PollingQueue, Exiting, Exited} => worker_job[w] = NONE

(*
 * Results sets are disjoint.
 *)
ResultsDisjoint ==
    results_pending \cap results_received = {}

(*
 * Checkpoint only starts after shutdown and drain complete.
 *)
CheckpointAfterDrain ==
    checkpoint_state \in {Checkpointing, Done} =>
        /\ shutdown_signaled
        /\ results_pending = {}
        /\ \A w \in Workers: worker_state[w] = Exited

CheckpointRequiresShutdown ==
    checkpoint_state \in {Draining, Checkpointing, Done,
                          ForceQuitAborted, DrainAborted} => shutdown_signaled

AbortStatesDoNotCheckpoint ==
    checkpoint_state \in {ForceQuitAborted, DrainAborted} =>
        checkpoint_state \notin {Checkpointing, Done}

(*
 * No job is lost: a job that was picked up has its result in pending or received.
 * More precisely: jobs not in queue must have their result somewhere.
 *)
NoJobLost ==
    \A j \in Jobs:
        (j \notin job_queue /\ ~(\E w \in Workers: worker_job[w] = j)) =>
            (j \in results_pending \/ j \in results_received)

(*
 * No job is represented in more than one active place. This is stronger than
 * NoJobLost: it prevents duplicate queue entries and two workers owning the
 * same job concurrently.
 *)
JobUniqueOwnership ==
    /\ job_queue \cap results_pending = {}
    /\ job_queue \cap results_received = {}
    /\ \A w \in Workers:
        worker_job[w] # NONE =>
            /\ worker_job[w] \notin job_queue
            /\ worker_job[w] \notin results_pending
            /\ worker_job[w] \notin results_received
    /\ \A w1, w2 \in Workers:
        worker_job[w1] # NONE /\ worker_job[w1] = worker_job[w2] => w1 = w2

(*
 * Combined safety invariant.
 *)
Safety ==
    /\ TypeOK
    /\ ProcessingHasJob
    /\ SendingHasJob
    /\ IdleNoJob
    /\ ResultsDisjoint
    /\ CheckpointAfterDrain
    /\ CheckpointRequiresShutdown
    /\ AbortStatesDoNotCheckpoint
    /\ NoJobLost
    /\ JobUniqueOwnership

(* ---------------------------------------------------------------------------
 * Liveness Properties
 * --------------------------------------------------------------------------- *)

(*
 * After shutdown is signaled, all workers eventually terminate.
 *)
WorkersEventuallyTerminate ==
    shutdown_signaled ~> \A w \in Workers: worker_state[w] = Exited

(*
 * A worker processing a job eventually sends its result.
 *)
ProcessingEventuallySendsResult ==
    \A w \in Workers:
        (worker_state[w] = Processing) ~>
            (worker_state[w] \in {Idle, Exiting, Exited})

(*
 * Checkpoint eventually either completes after graceful shutdown or reaches an
 * explicit no-checkpoint abort state after force quit / drain failure.
 *)
CheckpointEventuallyCompletes ==
    shutdown_signaled ~>
        checkpoint_state \in {Done, ForceQuitAborted, DrainAborted}

(*
 * All jobs eventually complete (result received).
 *)
AllJobsComplete ==
    <>(\A j \in Jobs: j \in results_received)

(* ---------------------------------------------------------------------------
 * Auxiliary Invariants for Debugging
 * --------------------------------------------------------------------------- *)

(*
 * Count of jobs currently being processed.
 *)
JobsInFlight == Cardinality({w \in Workers: worker_job[w] # NONE})

(*
 * System is quiescent: all workers idle or exited, queue empty.
 *)
Quiescent ==
    /\ \A w \in Workers: worker_state[w] \in {Idle, Exited}
    /\ job_queue = {}

=============================================================================
