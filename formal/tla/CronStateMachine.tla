--------------------------- MODULE CronStateMachine ---------------------------
(*
 * Formal verification of the Cron State Machine scheduler.
 *
 * This specification models:
 *   1. CronThread state machine (CheckEvents/DrainChannel/Sleeping/ExecutingTask/Terminated)
 *   2. TestThread behavior (spawn, wait_ready, schedule_tasks, request_shutdown)
 *   3. Ready signal synchronization mechanism
 *   4. Task execution and panic handling
 *
 * Key Properties to Verify:
 *   - TasksEventuallyExecute: Scheduled tasks execute before termination (under fairness)
 *   - PanicSafety: A panicking task does not prevent subsequent tasks
 *   - TerminationSafety: Scheduler eventually terminates when requested
 *   - NoDeadlock: System cannot reach a deadlocked state
 *   - ReadySignalCorrectness: Tasks scheduled after ready signal are processed
 *)

EXTENDS Naturals, FiniteSets, TLC

CONSTANTS
    \* @type: Int;
    MaxTasks,         \* Maximum number of tasks to model
    \* @type: Int;
    MaxTime           \* Maximum simulated time steps

\* Task types
NORMAL == "normal"
PANICKING == "panicking"

\* Cron thread states
CheckEvents == "check_events"
DrainChannel == "drain_channel"
Sleeping == "sleeping"
ExecutingTask == "executing_task"
Terminated == "terminated"

\* Test thread states
TestInit == "test_init"
TestWaitingReady == "test_waiting_ready"
TestScheduling == "test_scheduling"
TestWaitingTasks == "test_waiting_tasks"
TestRequestingShutdown == "test_requesting_shutdown"
TestJoining == "test_joining"
TestDone == "test_done"

VARIABLES
    \* Cron thread state
    \* @type: Str;
    cron_state,

    \* Whether ready signal has been sent by cron thread
    \* @type: Bool;
    ready_signal_sent,

    \* Whether ready signal has been received by test thread
    \* @type: Bool;
    ready_signal_received,

    \* Termination flag (atomic boolean in implementation)
    \* @type: Bool;
    terminating,

    \* Channel abstraction: set of tasks waiting to be received.
    \* FIFO ordering is irrelevant to the verified safety/liveness properties.
    \* Each task is a record [id: Nat, type: {NORMAL, PANICKING}, due_time: Nat]
    \* @type: Set([id: Int, type: Str, due_time: Int]);
    channel,

    \* Whether the sender side of the task channel is still open
    \* @type: Bool;
    channel_open,

    \* Task queue: set of tasks in the priority queue
    \* @type: Set([id: Int, type: Str, due_time: Int]);
    task_queue,

    \* Current simulated time
    \* @type: Int;
    current_time,

    \* Set of executed task IDs
    \* @type: Set(Int);
    executed_tasks,

    \* Set of panicked task IDs
    \* @type: Set(Int);
    panicked_tasks,

    \* Test thread state
    \* @type: Str;
    test_state,

    \* Tasks the test wants to schedule: set of [id, type, due_time]
    \* @type: Set([id: Int, type: Str, due_time: Int]);
    tasks_to_schedule,

    \* Whether the cron thread has been spawned
    \* @type: Bool;
    cron_spawned,

    \* ID for the next task
    \* @type: Int;
    next_task_id

TaskUniverse == [id: 1..MaxTasks, type: {NORMAL, PANICKING}, due_time: 0..MaxTime]

(* Type invariant *)
TypeOK ==
    /\ cron_state \in {CheckEvents, DrainChannel, Sleeping, ExecutingTask, Terminated}
    /\ ready_signal_sent \in BOOLEAN
    /\ ready_signal_received \in BOOLEAN
    /\ terminating \in BOOLEAN
    /\ channel_open \in BOOLEAN
    /\ channel \subseteq TaskUniverse
    /\ task_queue \subseteq TaskUniverse
    /\ current_time \in 0..MaxTime
    /\ executed_tasks \subseteq 1..MaxTasks
    /\ panicked_tasks \subseteq 1..MaxTasks
    /\ test_state \in {TestInit, TestWaitingReady, TestScheduling,
                       TestWaitingTasks, TestRequestingShutdown, TestJoining, TestDone}
    /\ tasks_to_schedule \subseteq TaskUniverse
    /\ cron_spawned \in BOOLEAN
    /\ next_task_id \in 1..(MaxTasks + 1)

(* Helper: Check if any task is due *)
\* @type: (Set([id: Int, type: Str, due_time: Int]), Int) => Bool;
HasDueTask(queue, time) ==
    \E t \in queue: t.due_time <= time

\* @type: ([id: Int, type: Str, due_time: Int]) => Bool;
EarliestPendingTask(task) ==
    \A other \in tasks_to_schedule: task.due_time <= other.due_time

\* @type: ([id: Int, type: Str, due_time: Int]) => Bool;
EarliestDueTask(task) ==
    /\ task \in task_queue
    /\ task.due_time <= current_time
    /\ \A other \in task_queue:
        (other.due_time <= current_time) => task.due_time <= other.due_time

(* Initial state *)
Init ==
    /\ cron_state = CheckEvents  \* Will be set properly when cron spawns
    /\ ready_signal_sent = FALSE
    /\ ready_signal_received = FALSE
    /\ terminating = FALSE
    /\ channel = {}
    /\ channel_open = TRUE
    /\ task_queue = {}
    /\ current_time = 0
    /\ executed_tasks = {}
    /\ panicked_tasks = {}
    /\ test_state = TestInit
    /\ tasks_to_schedule = {}
    /\ cron_spawned = FALSE
    /\ next_task_id = 1

(* ---------------------------------------------------------------------------
 * Test Thread Actions
 * --------------------------------------------------------------------------- *)

(*
 * Test spawns the cron thread.
 *)
TestSpawnCron ==
    /\ test_state = TestInit
    /\ ~cron_spawned
    /\ cron_spawned' = TRUE
    /\ test_state' = TestWaitingReady
    /\ UNCHANGED <<cron_state, ready_signal_sent, ready_signal_received,
                   terminating, channel, channel_open, task_queue, current_time,
                   executed_tasks, panicked_tasks, tasks_to_schedule, next_task_id>>

(*
 * Test waits for ready signal.
 *)
TestReceiveReady ==
    /\ test_state = TestWaitingReady
    /\ ready_signal_sent = TRUE
    /\ ready_signal_received' = TRUE
    /\ test_state' = TestScheduling
    \* Set up tasks to schedule: panicking task at time 0, normal task at time 50
    /\ tasks_to_schedule' = {[id |-> 1, type |-> PANICKING, due_time |-> 0],
                              [id |-> 2, type |-> NORMAL, due_time |-> 50]}
    /\ next_task_id' = 3
    /\ UNCHANGED <<cron_state, ready_signal_sent, terminating, channel, channel_open,
                   task_queue, current_time, executed_tasks, panicked_tasks, cron_spawned>>

(*
 * Test schedules a task (sends to channel).
 *)
TestScheduleTask ==
    /\ test_state = TestScheduling
    /\ tasks_to_schedule # {}
    /\ channel_open = TRUE
    /\ \E task \in tasks_to_schedule:
        /\ EarliestPendingTask(task)
        /\ channel' = channel \cup {task}
        /\ tasks_to_schedule' = tasks_to_schedule \ {task}
        /\ test_state' =
            IF tasks_to_schedule \ {task} = {}
            THEN TestWaitingTasks
            ELSE TestScheduling
    /\ UNCHANGED <<cron_state, ready_signal_sent, ready_signal_received,
                   terminating, channel_open, task_queue, current_time, executed_tasks,
                   panicked_tasks, cron_spawned, next_task_id>>

(*
 * Test waits for tasks to execute (simulated by time advancing).
 * Transitions to requesting shutdown when normal task has executed.
 *)
TestCheckTasksDone ==
    /\ test_state = TestWaitingTasks
    /\ 2 \in executed_tasks  \* Normal task (id=2) has executed
    /\ test_state' = TestRequestingShutdown
    /\ UNCHANGED <<cron_state, ready_signal_sent, ready_signal_received,
                   terminating, channel, channel_open, task_queue, current_time,
                   executed_tasks, panicked_tasks, tasks_to_schedule,
                   cron_spawned, next_task_id>>

(*
 * Test requests shutdown.
 *)
TestRequestShutdown ==
    /\ test_state = TestRequestingShutdown
    /\ terminating' = TRUE
    /\ test_state' = TestJoining
    /\ UNCHANGED <<cron_state, ready_signal_sent, ready_signal_received,
                   channel, channel_open, task_queue, current_time, executed_tasks,
                   panicked_tasks, tasks_to_schedule, cron_spawned, next_task_id>>

(*
 * Test can also drop the sending handle. This models channel close as a
 * separate shutdown path from setting the termination flag.
 *)
TestDropHandle ==
    /\ test_state = TestRequestingShutdown
    /\ channel_open' = FALSE
    /\ test_state' = TestJoining
    /\ UNCHANGED <<cron_state, ready_signal_sent, ready_signal_received,
                   terminating, channel, task_queue, current_time,
                   executed_tasks, panicked_tasks, tasks_to_schedule,
                   cron_spawned, next_task_id>>

(*
 * Test joins cron thread (waits for termination).
 *)
TestJoin ==
    /\ test_state = TestJoining
    /\ cron_state = Terminated
    /\ test_state' = TestDone
    /\ UNCHANGED <<cron_state, ready_signal_sent, ready_signal_received,
                   terminating, channel, channel_open, task_queue, current_time,
                   executed_tasks, panicked_tasks, tasks_to_schedule,
                   cron_spawned, next_task_id>>

(*
 * Done state - allows stuttering to avoid deadlock at terminal state.
 *)
Done ==
    /\ test_state = TestDone
    /\ cron_state = Terminated
    /\ UNCHANGED <<cron_state, ready_signal_sent, ready_signal_received, terminating,
                   channel, channel_open, task_queue, current_time, executed_tasks, panicked_tasks,
                   test_state, tasks_to_schedule, cron_spawned, next_task_id>>

(* ---------------------------------------------------------------------------
 * Cron Thread Actions
 * --------------------------------------------------------------------------- *)

(*
 * Cron thread sends ready signal at start of run().
 * This models Fix 1: ready signal sent INSIDE the event loop.
 *)
CronSendReady ==
    /\ cron_spawned = TRUE
    /\ ready_signal_sent = FALSE
    /\ ready_signal_sent' = TRUE
    /\ UNCHANGED <<cron_state, ready_signal_received, terminating, channel, channel_open,
                   task_queue, current_time, executed_tasks, panicked_tasks,
                   test_state, tasks_to_schedule, cron_spawned, next_task_id>>

(*
 * Cron checks for due tasks BEFORE checking termination.
 * This models Fix 2: due tasks have priority over termination check.
 *)
CronCheckDueTasks ==
    /\ cron_spawned = TRUE
    /\ ready_signal_sent = TRUE
    /\ cron_state = CheckEvents
    /\ HasDueTask(task_queue, current_time)
    /\ cron_state' = ExecutingTask
    /\ UNCHANGED <<ready_signal_sent, ready_signal_received, terminating,
                   channel, channel_open, task_queue, current_time, executed_tasks,
                   panicked_tasks, test_state, tasks_to_schedule,
                   cron_spawned, next_task_id>>

(*
 * Cron checks for termination (only if no due tasks).
 *)
CronCheckTermination ==
    /\ cron_spawned = TRUE
    /\ ready_signal_sent = TRUE
    /\ cron_state = CheckEvents
    /\ ~HasDueTask(task_queue, current_time)
    /\ terminating = TRUE
    /\ cron_state' = Terminated
    /\ UNCHANGED <<ready_signal_sent, ready_signal_received, terminating,
                   channel, channel_open, task_queue, current_time, executed_tasks,
                   panicked_tasks, test_state, tasks_to_schedule,
                   cron_spawned, next_task_id>>

(*
 * Cron checks channel for new tasks.
 *)
CronCheckChannel ==
    /\ cron_spawned = TRUE
    /\ ready_signal_sent = TRUE
    /\ cron_state = CheckEvents
    /\ ~HasDueTask(task_queue, current_time)
    /\ terminating = FALSE
    /\ channel # {}
    /\ cron_state' = DrainChannel
    /\ UNCHANGED <<ready_signal_sent, ready_signal_received, terminating,
                   channel, channel_open, task_queue, current_time, executed_tasks,
                   panicked_tasks, test_state, tasks_to_schedule,
                   cron_spawned, next_task_id>>

(*
 * Cron observes the task channel has closed and no queued work remains.
 *)
CronCheckChannelClosed ==
    /\ cron_spawned = TRUE
    /\ ready_signal_sent = TRUE
    /\ cron_state = CheckEvents
    /\ ~HasDueTask(task_queue, current_time)
    /\ terminating = FALSE
    /\ channel_open = FALSE
    /\ channel = {}
    /\ task_queue = {}
    /\ cron_state' = Terminated
    /\ UNCHANGED <<ready_signal_sent, ready_signal_received, terminating,
                   channel, channel_open, task_queue, current_time, executed_tasks,
                   panicked_tasks, test_state, tasks_to_schedule,
                   cron_spawned, next_task_id>>

(*
 * Cron has no events - go to sleep.
 *)
CronNoEvents ==
    /\ cron_spawned = TRUE
    /\ ready_signal_sent = TRUE
    /\ cron_state = CheckEvents
    /\ ~HasDueTask(task_queue, current_time)
    /\ terminating = FALSE
    /\ channel = {}
    /\ (channel_open = TRUE \/ task_queue # {})
    /\ cron_state' = Sleeping
    /\ UNCHANGED <<ready_signal_sent, ready_signal_received, terminating,
                   channel, channel_open, task_queue, current_time, executed_tasks,
                   panicked_tasks, test_state, tasks_to_schedule,
                   cron_spawned, next_task_id>>

(*
 * Cron drains one task from channel to queue.
 *)
CronDrainTask ==
    /\ cron_spawned = TRUE
    /\ cron_state = DrainChannel
    /\ channel # {}
    /\ \E task \in channel:
        /\ task_queue' = task_queue \cup {task}
        /\ channel' = channel \ {task}
    /\ UNCHANGED <<cron_state, ready_signal_sent, ready_signal_received,
                   terminating, channel_open, current_time, executed_tasks, panicked_tasks,
                   test_state, tasks_to_schedule, cron_spawned, next_task_id>>

(*
 * Cron finishes draining - check for due tasks or return to check events.
 *)
CronFinishDrain ==
    /\ cron_spawned = TRUE
    /\ cron_state = DrainChannel
    /\ channel = {}
    /\ IF HasDueTask(task_queue, current_time)
       THEN cron_state' = ExecutingTask
       ELSE cron_state' = Sleeping
    /\ UNCHANGED <<ready_signal_sent, ready_signal_received, terminating,
                   channel, channel_open, task_queue, current_time, executed_tasks,
                   panicked_tasks, test_state, tasks_to_schedule,
                   cron_spawned, next_task_id>>

(*
 * Cron executes a due task.
 *)
CronExecutePanickingTask(task) ==
    /\ cron_spawned = TRUE
    /\ cron_state = ExecutingTask
    /\ EarliestDueTask(task)
    /\ task.id \in 1..MaxTasks
    /\ task.type = PANICKING
    /\ task_queue' = task_queue \ {task}
    /\ task.id \notin executed_tasks
    /\ panicked_tasks' = panicked_tasks \cup {task.id}
    /\ UNCHANGED executed_tasks
    /\ cron_state' = CheckEvents
    /\ UNCHANGED <<ready_signal_sent, ready_signal_received, terminating,
                   channel, channel_open, current_time, test_state, tasks_to_schedule,
                   cron_spawned, next_task_id>>

CronExecuteNormalTask(task) ==
    /\ cron_spawned = TRUE
    /\ cron_state = ExecutingTask
    /\ EarliestDueTask(task)
    /\ task.id \in 1..MaxTasks
    /\ task.type = NORMAL
    /\ task_queue' = task_queue \ {task}
    /\ task.id \notin panicked_tasks
    /\ executed_tasks' = executed_tasks \cup {task.id}
    /\ UNCHANGED panicked_tasks
    /\ cron_state' = CheckEvents
    /\ UNCHANGED <<ready_signal_sent, ready_signal_received, terminating,
                   channel, channel_open, current_time, test_state, tasks_to_schedule,
                   cron_spawned, next_task_id>>

CronExecuteTaskBy(task) ==
    \/ CronExecutePanickingTask(task)
    \/ CronExecuteNormalTask(task)

CronExecuteTask ==
    \E task \in task_queue: CronExecuteTaskBy(task)

(*
 * Cron wakes from sleep - time advances.
 *)
CronWakeUp ==
    /\ cron_spawned = TRUE
    /\ cron_state = Sleeping
    /\ (channel_open = TRUE \/ channel # {} \/ task_queue # {})
    /\ current_time' =
        IF current_time < MaxTime
        THEN current_time + 1
        ELSE MaxTime
    /\ cron_state' = CheckEvents
    /\ UNCHANGED <<ready_signal_sent, ready_signal_received, terminating,
                   channel, channel_open, task_queue, executed_tasks, panicked_tasks,
                   test_state, tasks_to_schedule, cron_spawned, next_task_id>>

(*
 * Cron detects termination while sleeping.
 * In the real implementation, poll_event() is called after sleep returns,
 * which checks termination. This models that behavior.
 *)
CronTerminateFromSleep ==
    /\ cron_spawned = TRUE
    /\ cron_state = Sleeping
    /\ terminating = TRUE
    /\ cron_state' = Terminated
    /\ UNCHANGED <<ready_signal_sent, ready_signal_received, terminating,
                   channel, channel_open, task_queue, current_time, executed_tasks,
                   panicked_tasks, test_state, tasks_to_schedule,
                   cron_spawned, next_task_id>>

(*
 * Cron wakes and sees that the channel closed with no remaining queued work.
 *)
CronTerminateClosedFromSleep ==
    /\ cron_spawned = TRUE
    /\ cron_state = Sleeping
    /\ terminating = FALSE
    /\ channel_open = FALSE
    /\ channel = {}
    /\ task_queue = {}
    /\ cron_state' = Terminated
    /\ UNCHANGED <<ready_signal_sent, ready_signal_received, terminating,
                   channel, channel_open, task_queue, current_time, executed_tasks,
                   panicked_tasks, test_state, tasks_to_schedule,
                   cron_spawned, next_task_id>>

(* ---------------------------------------------------------------------------
 * Next state relation
 * --------------------------------------------------------------------------- *)

Next ==
    \* Test thread actions
    \/ TestSpawnCron
    \/ TestReceiveReady
    \/ TestScheduleTask
    \/ TestCheckTasksDone
    \/ TestRequestShutdown
    \/ TestDropHandle
    \/ TestJoin
    \* Cron thread actions
    \/ CronSendReady
    \/ CronCheckDueTasks
    \/ CronCheckTermination
    \/ CronCheckChannel
    \/ CronCheckChannelClosed
    \/ CronNoEvents
    \/ CronDrainTask
    \/ CronFinishDrain
    \/ CronExecuteTask
    \/ CronWakeUp
    \/ CronTerminateFromSleep
    \/ CronTerminateClosedFromSleep
    \* Terminal state (avoids deadlock detection)
    \/ Done

vars == <<cron_state, ready_signal_sent, ready_signal_received, terminating,
          channel, channel_open, task_queue, current_time, executed_tasks, panicked_tasks,
          test_state, tasks_to_schedule, cron_spawned, next_task_id>>

Spec == Init /\ [][Next]_vars

(* ---------------------------------------------------------------------------
 * Fairness Constraints
 * --------------------------------------------------------------------------- *)

\* Cron thread actions are fair (will eventually happen when enabled)
CronFairness ==
    /\ WF_vars(CronSendReady)
    /\ WF_vars(CronCheckDueTasks)
    /\ WF_vars(CronCheckTermination)
    /\ WF_vars(CronCheckChannel)
    /\ WF_vars(CronCheckChannelClosed)
    /\ WF_vars(CronNoEvents)
    /\ WF_vars(CronDrainTask)
    /\ WF_vars(CronFinishDrain)
    /\ WF_vars(CronExecuteTask)
    /\ WF_vars(CronWakeUp)
    /\ WF_vars(CronTerminateFromSleep)
    /\ WF_vars(CronTerminateClosedFromSleep)

\* Test thread actions are fair
TestFairness ==
    /\ WF_vars(TestSpawnCron)
    /\ WF_vars(TestReceiveReady)
    /\ WF_vars(TestScheduleTask)
    /\ WF_vars(TestCheckTasksDone)
    /\ WF_vars(TestRequestShutdown)
    /\ WF_vars(TestDropHandle)
    /\ WF_vars(TestJoin)

FairSpec == Spec /\ CronFairness /\ TestFairness

(* ---------------------------------------------------------------------------
 * Safety Invariants
 * --------------------------------------------------------------------------- *)

(*
 * Ready signal is sent only once and before test can proceed.
 *)
ReadySignalSafety ==
    ready_signal_received => ready_signal_sent

(*
 * Panicked tasks don't prevent other tasks from being tracked.
 * The sets of executed and panicked tasks are disjoint.
 *)
PanicIsolation ==
    executed_tasks \cap panicked_tasks = {}

PendingTasks ==
    channel \cup task_queue \cup tasks_to_schedule

PendingTasksConsistentWithResults ==
    \A task \in PendingTasks:
        /\ task.type = NORMAL => task.id \notin panicked_tasks
        /\ task.type = PANICKING => task.id \notin executed_tasks

PendingTaskIdsTypeConsistent ==
    \A t1, t2 \in PendingTasks:
        t1.id = t2.id => t1.type = t2.type

TaskIdConsistency ==
    /\ PendingTasksConsistentWithResults
    /\ PendingTaskIdsTypeConsistent

(*
 * Test completes only after normal task has executed.
 *)
TestCompletionRequiresExecution ==
    test_state = TestDone => 2 \in executed_tasks

(*
 * Once the test has observed task completion and moves into shutdown/join,
 * task 2 remains recorded as executed.
 *)
TestProgressRequiresExecution ==
    test_state \in {TestRequestingShutdown, TestJoining, TestDone} =>
        2 \in executed_tasks

(*
 * Cron terminates only when termination is requested or the task channel closes.
 *)
TerminationRequiresRequest ==
    cron_state = Terminated => (terminating = TRUE \/ channel_open = FALSE)

NormalTaskWaitsForEarlierPanic ==
    2 \in executed_tasks => 1 \in panicked_tasks

(*
 * Combined safety invariant.
 *)
Safety ==
    /\ TypeOK
    /\ ReadySignalSafety
    /\ PanicIsolation
    /\ TestCompletionRequiresExecution
    /\ TestProgressRequiresExecution
    /\ TerminationRequiresRequest

(* ---------------------------------------------------------------------------
 * Liveness Properties (under fairness)
 * --------------------------------------------------------------------------- *)

(*
 * CRITICAL: If test starts and fair scheduling, test eventually completes.
 *)
TestEventuallyCompletes ==
    (test_state = TestInit) ~> (test_state = TestDone)

(*
 * Cron eventually terminates after shutdown is requested.
 *)
CronEventuallyTerminates ==
    (terminating = TRUE) ~> (cron_state = Terminated)

(*
 * Normal task eventually executes (even after a panicking task).
 * This is the key property for test_panic_safety.
 *)
NormalTaskExecutesAfterPanic ==
    (1 \in panicked_tasks) ~> (2 \in executed_tasks)

(*
 * Ready signal is eventually received after being sent.
 *)
ReadySignalEventuallyReceived ==
    ready_signal_sent ~> ready_signal_received

=============================================================================
