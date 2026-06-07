------------------------ MODULE CronStateMachineProofs -----------------------
(*
 * TLAPS proof obligations for CronStateMachine.tla.
 *)

EXTENDS CronStateMachine, TLAPS

ModelConstantsOK ==
    /\ MaxTasks \in Nat
    /\ MaxTime \in Nat
    /\ MaxTasks >= 2
    /\ MaxTime >= 50

THEOREM OneInTaskIdRange ==
    ModelConstantsOK => 1 \in 1..(MaxTasks + 1)
BY SMT DEF ModelConstantsOK

THEOREM ZeroInTimeRange ==
    ModelConstantsOK => 0 \in 0..MaxTime
BY SMT DEF ModelConstantsOK

THEOREM TrueIsBoolean ==
    TRUE \in BOOLEAN
OBVIOUS

THEOREM FalseIsBoolean ==
    FALSE \in BOOLEAN
OBVIOUS

THEOREM InsertIntoSubset ==
    ASSUME NEW S, NEW U, NEW x \in U, S \subseteq U
    PROVE  S \cup {x} \subseteq U
BY SMT

THEOREM InitImpliesTypeOK ==
    ModelConstantsOK /\ Init => TypeOK
<1>1. ASSUME ModelConstantsOK, Init
      PROVE  cron_state \in {CheckEvents, DrainChannel, Sleeping, ExecutingTask, Terminated}
  BY <1>1 DEF Init, CheckEvents
<1>2. ASSUME ModelConstantsOK, Init
      PROVE  /\ ready_signal_sent \in BOOLEAN
             /\ ready_signal_received \in BOOLEAN
             /\ terminating \in BOOLEAN
             /\ channel_open \in BOOLEAN
             /\ cron_spawned \in BOOLEAN
  BY <1>2, TrueIsBoolean, FalseIsBoolean DEF Init
<1>3. ASSUME ModelConstantsOK, Init
      PROVE  /\ channel \subseteq TaskUniverse
             /\ task_queue \subseteq TaskUniverse
             /\ tasks_to_schedule \subseteq TaskUniverse
  BY <1>3 DEF Init
<1>4. ASSUME ModelConstantsOK, Init
      PROVE  current_time \in 0..MaxTime
  BY <1>4, ZeroInTimeRange DEF Init
<1>5. ASSUME ModelConstantsOK, Init
      PROVE  /\ executed_tasks \subseteq 1..MaxTasks
             /\ panicked_tasks \subseteq 1..MaxTasks
  BY <1>5 DEF Init
<1>6. ASSUME ModelConstantsOK, Init
      PROVE  test_state \in {TestInit, TestWaitingReady, TestScheduling,
                             TestWaitingTasks, TestRequestingShutdown,
                             TestJoining, TestDone}
  BY <1>6 DEF Init, TestInit
<1>7. ASSUME ModelConstantsOK, Init
      PROVE  next_task_id \in 1..(MaxTasks + 1)
  BY <1>7, OneInTaskIdRange DEF Init
<1>. QED
  BY <1>1, <1>2, <1>3, <1>4, <1>5, <1>6, <1>7 DEF TypeOK

THEOREM InitImpliesReadySignalSafety ==
    ModelConstantsOK /\ Init => ReadySignalSafety
BY DEF Init, ReadySignalSafety

THEOREM InitImpliesPanicIsolation ==
    ModelConstantsOK /\ Init => PanicIsolation
BY DEF Init, PanicIsolation

THEOREM InitImpliesTestCompletionRequiresExecution ==
    ModelConstantsOK /\ Init => TestCompletionRequiresExecution
BY SMT DEF Init, TestCompletionRequiresExecution, TestDone, TestInit

THEOREM InitImpliesTestProgressRequiresExecution ==
    ModelConstantsOK /\ Init => TestProgressRequiresExecution
BY SMT DEF Init, TestProgressRequiresExecution, TestRequestingShutdown,
           TestJoining, TestDone, TestInit

THEOREM InitImpliesTerminationRequiresRequest ==
    ModelConstantsOK /\ Init => TerminationRequiresRequest
BY SMT DEF Init, TerminationRequiresRequest, CheckEvents, Terminated

THEOREM InitImpliesSafety ==
    ModelConstantsOK /\ Init => Safety
BY InitImpliesTypeOK,
   InitImpliesReadySignalSafety,
   InitImpliesPanicIsolation,
   InitImpliesTestCompletionRequiresExecution,
   InitImpliesTestProgressRequiresExecution,
   InitImpliesTerminationRequiresRequest
DEF Safety

THEOREM TestSpawnCronPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, TestSpawnCron
    PROVE  Safety'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, TaskUniverse, ReadySignalSafety,
           PanicIsolation, TestCompletionRequiresExecution,
           TestProgressRequiresExecution, TerminationRequiresRequest,
           TestSpawnCron, NORMAL, PANICKING,
           CheckEvents, DrainChannel, Sleeping, ExecutingTask, Terminated,
           TestInit, TestWaitingReady, TestScheduling, TestWaitingTasks,
           TestRequestingShutdown, TestJoining, TestDone

THEOREM TestReceiveReadyPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, TestReceiveReady
    PROVE  Safety'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, TaskUniverse, ReadySignalSafety,
           PanicIsolation, TestCompletionRequiresExecution,
           TestProgressRequiresExecution, TerminationRequiresRequest,
           TestReceiveReady, NORMAL, PANICKING,
           CheckEvents, DrainChannel, Sleeping, ExecutingTask, Terminated,
           TestInit, TestWaitingReady, TestScheduling, TestWaitingTasks,
           TestRequestingShutdown, TestJoining, TestDone

THEOREM TestScheduleTaskPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, TestScheduleTask
    PROVE  Safety'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, TaskUniverse, ReadySignalSafety,
           PanicIsolation, TestCompletionRequiresExecution,
           TestProgressRequiresExecution, TerminationRequiresRequest,
           TestScheduleTask, NORMAL, PANICKING,
           CheckEvents, DrainChannel, Sleeping, ExecutingTask, Terminated,
           TestInit, TestWaitingReady, TestScheduling, TestWaitingTasks,
           TestRequestingShutdown, TestJoining, TestDone

THEOREM TestCheckTasksDonePreservesSafety ==
    ASSUME ModelConstantsOK, Safety, TestCheckTasksDone
    PROVE  Safety'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, TaskUniverse, ReadySignalSafety,
           PanicIsolation, TestCompletionRequiresExecution,
           TestProgressRequiresExecution, TerminationRequiresRequest,
           TestCheckTasksDone, NORMAL, PANICKING,
           CheckEvents, DrainChannel, Sleeping, ExecutingTask, Terminated,
           TestInit, TestWaitingReady, TestScheduling, TestWaitingTasks,
           TestRequestingShutdown, TestJoining, TestDone

THEOREM TestRequestShutdownPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, TestRequestShutdown
    PROVE  Safety'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, TaskUniverse, ReadySignalSafety,
           PanicIsolation, TestCompletionRequiresExecution,
           TestProgressRequiresExecution, TerminationRequiresRequest,
           TestRequestShutdown, NORMAL, PANICKING,
           CheckEvents, DrainChannel, Sleeping, ExecutingTask, Terminated,
           TestInit, TestWaitingReady, TestScheduling, TestWaitingTasks,
           TestRequestingShutdown, TestJoining, TestDone

THEOREM TestDropHandlePreservesSafety ==
    ASSUME ModelConstantsOK, Safety, TestDropHandle
    PROVE  Safety'
<1>1. TypeOK'
  <2>1. /\ cron_state' \in {CheckEvents, DrainChannel, Sleeping, ExecutingTask, Terminated}
         /\ ready_signal_sent' \in BOOLEAN
         /\ ready_signal_received' \in BOOLEAN
         /\ terminating' \in BOOLEAN
         /\ channel' \subseteq TaskUniverse
         /\ task_queue' \subseteq TaskUniverse
         /\ current_time' \in 0..MaxTime
         /\ executed_tasks' \subseteq 1..MaxTasks
         /\ panicked_tasks' \subseteq 1..MaxTasks
         /\ tasks_to_schedule' \subseteq TaskUniverse
         /\ cron_spawned' \in BOOLEAN
         /\ next_task_id' \in 1..(MaxTasks + 1)
    BY SMT DEF ModelConstantsOK, Safety, TypeOK, TaskUniverse, TestDropHandle,
               CheckEvents, DrainChannel, Sleeping, ExecutingTask, Terminated
  <2>2. channel_open' = FALSE
    BY SMT DEF TestDropHandle
  <2>3. channel_open' \in BOOLEAN
    BY <2>2, FalseIsBoolean
  <2>4. test_state' \in {TestInit, TestWaitingReady, TestScheduling,
                         TestWaitingTasks, TestRequestingShutdown,
                         TestJoining, TestDone}
    BY SMT DEF TestDropHandle, TestInit, TestWaitingReady, TestScheduling,
               TestWaitingTasks, TestRequestingShutdown, TestJoining, TestDone
  <2>. QED
    BY <2>1, <2>3, <2>4 DEF TypeOK
<1>2. ReadySignalSafety'
  BY SMT DEF Safety, ReadySignalSafety, TestDropHandle
<1>3. PanicIsolation'
  BY SMT DEF Safety, PanicIsolation, TestDropHandle
<1>4. TestCompletionRequiresExecution'
  BY SMT DEF Safety, TestCompletionRequiresExecution, TestDropHandle,
             TestJoining, TestDone
<1>5. TestProgressRequiresExecution'
  BY SMT DEF Safety, TestProgressRequiresExecution, TestDropHandle,
             TestRequestingShutdown, TestJoining, TestDone
<1>6. TerminationRequiresRequest'
  BY SMT DEF Safety, TerminationRequiresRequest, TestDropHandle
<1>. QED
  BY <1>1, <1>2, <1>3, <1>4, <1>5, <1>6 DEF Safety

THEOREM TestJoinPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, TestJoin
    PROVE  Safety'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, TaskUniverse, ReadySignalSafety,
           PanicIsolation, TestCompletionRequiresExecution,
           TestProgressRequiresExecution, TerminationRequiresRequest,
           TestJoin, NORMAL, PANICKING,
           CheckEvents, DrainChannel, Sleeping, ExecutingTask, Terminated,
           TestInit, TestWaitingReady, TestScheduling, TestWaitingTasks,
           TestRequestingShutdown, TestJoining, TestDone

THEOREM CronSendReadyPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, CronSendReady
    PROVE  Safety'
BY SMT, TrueIsBoolean DEF ModelConstantsOK, Safety, TypeOK, TaskUniverse,
           ReadySignalSafety, PanicIsolation, TestCompletionRequiresExecution,
           TestProgressRequiresExecution, TerminationRequiresRequest,
           CronSendReady, NORMAL, PANICKING, CheckEvents, DrainChannel,
           Sleeping, ExecutingTask, Terminated, TestInit, TestWaitingReady,
           TestScheduling, TestWaitingTasks, TestRequestingShutdown,
           TestJoining, TestDone

THEOREM CronCheckDueTasksPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, CronCheckDueTasks
    PROVE  Safety'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, TaskUniverse, ReadySignalSafety,
           PanicIsolation, TestCompletionRequiresExecution,
           TestProgressRequiresExecution, TerminationRequiresRequest,
           CronCheckDueTasks, HasDueTask, NORMAL, PANICKING, CheckEvents,
           DrainChannel, Sleeping, ExecutingTask, Terminated, TestInit,
           TestWaitingReady, TestScheduling, TestWaitingTasks,
           TestRequestingShutdown, TestJoining, TestDone

THEOREM CronCheckTerminationPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, CronCheckTermination
    PROVE  Safety'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, TaskUniverse, ReadySignalSafety,
           PanicIsolation, TestCompletionRequiresExecution,
           TestProgressRequiresExecution, TerminationRequiresRequest,
           CronCheckTermination, HasDueTask, NORMAL, PANICKING, CheckEvents,
           DrainChannel, Sleeping, ExecutingTask, Terminated, TestInit,
           TestWaitingReady, TestScheduling, TestWaitingTasks,
           TestRequestingShutdown, TestJoining, TestDone

THEOREM CronCheckChannelPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, CronCheckChannel
    PROVE  Safety'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, TaskUniverse, ReadySignalSafety,
           PanicIsolation, TestCompletionRequiresExecution,
           TestProgressRequiresExecution, TerminationRequiresRequest,
           CronCheckChannel, HasDueTask, NORMAL, PANICKING, CheckEvents,
           DrainChannel, Sleeping, ExecutingTask, Terminated, TestInit,
           TestWaitingReady, TestScheduling, TestWaitingTasks,
           TestRequestingShutdown, TestJoining, TestDone

THEOREM CronCheckChannelClosedPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, CronCheckChannelClosed
    PROVE  Safety'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, TaskUniverse, ReadySignalSafety,
           PanicIsolation, TestCompletionRequiresExecution,
           TestProgressRequiresExecution, TerminationRequiresRequest,
           CronCheckChannelClosed, HasDueTask, NORMAL, PANICKING, CheckEvents,
           DrainChannel, Sleeping, ExecutingTask, Terminated, TestInit,
           TestWaitingReady, TestScheduling, TestWaitingTasks,
           TestRequestingShutdown, TestJoining, TestDone

THEOREM CronNoEventsPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, CronNoEvents
    PROVE  Safety'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, TaskUniverse, ReadySignalSafety,
           PanicIsolation, TestCompletionRequiresExecution,
           TestProgressRequiresExecution, TerminationRequiresRequest,
           CronNoEvents, HasDueTask, NORMAL, PANICKING, CheckEvents,
           DrainChannel, Sleeping, ExecutingTask, Terminated, TestInit,
           TestWaitingReady, TestScheduling, TestWaitingTasks,
           TestRequestingShutdown, TestJoining, TestDone

THEOREM CronDrainTaskPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, CronDrainTask
    PROVE  Safety'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, TaskUniverse, ReadySignalSafety,
           PanicIsolation, TestCompletionRequiresExecution,
           TestProgressRequiresExecution, TerminationRequiresRequest,
           CronDrainTask, NORMAL, PANICKING, CheckEvents, DrainChannel,
           Sleeping, ExecutingTask, Terminated, TestInit, TestWaitingReady,
           TestScheduling, TestWaitingTasks, TestRequestingShutdown,
           TestJoining, TestDone

THEOREM CronFinishDrainPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, CronFinishDrain
    PROVE  Safety'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, TaskUniverse, ReadySignalSafety,
           PanicIsolation, TestCompletionRequiresExecution,
           TestProgressRequiresExecution, TerminationRequiresRequest,
           CronFinishDrain, HasDueTask, NORMAL, PANICKING, CheckEvents,
           DrainChannel, Sleeping, ExecutingTask, Terminated, TestInit,
           TestWaitingReady, TestScheduling, TestWaitingTasks,
           TestRequestingShutdown, TestJoining, TestDone

THEOREM CronExecutePanickingTaskPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, NEW task \in TaskUniverse,
           CronExecutePanickingTask(task)
    PROVE  Safety'
<1>1. TypeOK'
  <2>1. /\ cron_state' \in {CheckEvents, DrainChannel, Sleeping, ExecutingTask, Terminated}
         /\ ready_signal_sent' \in BOOLEAN
         /\ ready_signal_received' \in BOOLEAN
         /\ terminating' \in BOOLEAN
         /\ channel_open' \in BOOLEAN
         /\ channel' \subseteq TaskUniverse
         /\ task_queue' \subseteq TaskUniverse
         /\ current_time' \in 0..MaxTime
         /\ executed_tasks' \subseteq 1..MaxTasks
         /\ test_state' \in {TestInit, TestWaitingReady, TestScheduling,
                             TestWaitingTasks, TestRequestingShutdown,
                             TestJoining, TestDone}
         /\ tasks_to_schedule' \subseteq TaskUniverse
         /\ cron_spawned' \in BOOLEAN
         /\ next_task_id' \in 1..(MaxTasks + 1)
    BY SMT DEF ModelConstantsOK, Safety, TypeOK, TaskUniverse,
               CronExecutePanickingTask, CheckEvents, DrainChannel, Sleeping,
               ExecutingTask, Terminated, TestInit, TestWaitingReady,
               TestScheduling, TestWaitingTasks, TestRequestingShutdown,
               TestJoining, TestDone
  <2>2. task.id \in 1..MaxTasks
    BY SMT DEF CronExecutePanickingTask
  <2>3. panicked_tasks \subseteq 1..MaxTasks
    BY SMT DEF Safety, TypeOK
  <2>4. panicked_tasks' = panicked_tasks \cup {task.id}
    BY SMT DEF CronExecutePanickingTask
  <2>5. panicked_tasks' \subseteq 1..MaxTasks
    BY <2>2, <2>3, <2>4, InsertIntoSubset
  <2>. QED
    BY <2>1, <2>5 DEF TypeOK
<1>2. ReadySignalSafety'
  BY SMT DEF Safety, ReadySignalSafety, CronExecutePanickingTask
<1>3. PanicIsolation'
  BY SMT DEF Safety, PanicIsolation, CronExecutePanickingTask
<1>4. TestCompletionRequiresExecution'
  BY SMT DEF Safety, TestCompletionRequiresExecution, CronExecutePanickingTask
<1>5. TestProgressRequiresExecution'
  BY SMT DEF Safety, TestProgressRequiresExecution, CronExecutePanickingTask
<1>6. TerminationRequiresRequest'
  BY SMT DEF Safety, TerminationRequiresRequest, CronExecutePanickingTask,
             CheckEvents, Terminated
<1>. QED
  BY <1>1, <1>2, <1>3, <1>4, <1>5, <1>6 DEF Safety

THEOREM CronExecuteNormalTaskPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, NEW task \in TaskUniverse,
           CronExecuteNormalTask(task)
    PROVE  Safety'
<1>1. TypeOK'
  <2>1. /\ cron_state' \in {CheckEvents, DrainChannel, Sleeping, ExecutingTask, Terminated}
         /\ ready_signal_sent' \in BOOLEAN
         /\ ready_signal_received' \in BOOLEAN
         /\ terminating' \in BOOLEAN
         /\ channel_open' \in BOOLEAN
         /\ channel' \subseteq TaskUniverse
         /\ task_queue' \subseteq TaskUniverse
         /\ current_time' \in 0..MaxTime
         /\ panicked_tasks' \subseteq 1..MaxTasks
         /\ test_state' \in {TestInit, TestWaitingReady, TestScheduling,
                             TestWaitingTasks, TestRequestingShutdown,
                             TestJoining, TestDone}
         /\ tasks_to_schedule' \subseteq TaskUniverse
         /\ cron_spawned' \in BOOLEAN
         /\ next_task_id' \in 1..(MaxTasks + 1)
    BY SMT DEF ModelConstantsOK, Safety, TypeOK, TaskUniverse,
               CronExecuteNormalTask, CheckEvents, DrainChannel, Sleeping,
               ExecutingTask, Terminated, TestInit, TestWaitingReady,
               TestScheduling, TestWaitingTasks, TestRequestingShutdown,
               TestJoining, TestDone
  <2>2. task.id \in 1..MaxTasks
    BY SMT DEF CronExecuteNormalTask
  <2>3. executed_tasks \subseteq 1..MaxTasks
    BY SMT DEF Safety, TypeOK
  <2>4. executed_tasks' = executed_tasks \cup {task.id}
    BY SMT DEF CronExecuteNormalTask
  <2>5. executed_tasks' \subseteq 1..MaxTasks
    BY <2>2, <2>3, <2>4, InsertIntoSubset
  <2>. QED
    BY <2>1, <2>5 DEF TypeOK
<1>2. ReadySignalSafety'
  BY SMT DEF Safety, ReadySignalSafety, CronExecuteNormalTask
<1>3. PanicIsolation'
  BY SMT DEF Safety, PanicIsolation, CronExecuteNormalTask
<1>4. TestCompletionRequiresExecution'
  BY SMT DEF Safety, TestCompletionRequiresExecution, CronExecuteNormalTask
<1>5. TestProgressRequiresExecution'
  BY SMT DEF Safety, TestProgressRequiresExecution, CronExecuteNormalTask
<1>6. TerminationRequiresRequest'
  BY SMT DEF Safety, TerminationRequiresRequest, CronExecuteNormalTask,
             CheckEvents, Terminated
<1>. QED
  BY <1>1, <1>2, <1>3, <1>4, <1>5, <1>6 DEF Safety

THEOREM CronExecuteTaskByPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, NEW task \in TaskUniverse,
           CronExecuteTaskBy(task)
    PROVE  Safety'
BY CronExecutePanickingTaskPreservesSafety,
   CronExecuteNormalTaskPreservesSafety
DEF CronExecuteTaskBy

THEOREM CronExecuteTaskPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, CronExecuteTask
    PROVE  Safety'
<1>1. PICK task \in task_queue: CronExecuteTaskBy(task)
  BY DEF CronExecuteTask
<1>2. task \in TaskUniverse
  BY <1>1, SMT DEF Safety, TypeOK, CronExecuteTaskBy
<1>3. Safety'
  BY <1>1, <1>2, CronExecuteTaskByPreservesSafety
<1>. QED
  BY <1>3

THEOREM CronWakeUpPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, CronWakeUp
    PROVE  Safety'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, TaskUniverse, ReadySignalSafety,
           PanicIsolation, TestCompletionRequiresExecution,
           TestProgressRequiresExecution, TerminationRequiresRequest,
           CronWakeUp, NORMAL, PANICKING, CheckEvents, DrainChannel,
           Sleeping, ExecutingTask, Terminated, TestInit, TestWaitingReady,
           TestScheduling, TestWaitingTasks, TestRequestingShutdown,
           TestJoining, TestDone

THEOREM CronTerminateFromSleepPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, CronTerminateFromSleep
    PROVE  Safety'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, TaskUniverse, ReadySignalSafety,
           PanicIsolation, TestCompletionRequiresExecution,
           TestProgressRequiresExecution, TerminationRequiresRequest,
           CronTerminateFromSleep, NORMAL, PANICKING, CheckEvents, DrainChannel,
           Sleeping, ExecutingTask, Terminated, TestInit, TestWaitingReady,
           TestScheduling, TestWaitingTasks, TestRequestingShutdown,
           TestJoining, TestDone

THEOREM CronTerminateClosedFromSleepPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, CronTerminateClosedFromSleep
    PROVE  Safety'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, TaskUniverse, ReadySignalSafety,
           PanicIsolation, TestCompletionRequiresExecution,
           TestProgressRequiresExecution, TerminationRequiresRequest,
           CronTerminateClosedFromSleep, NORMAL, PANICKING, CheckEvents,
           DrainChannel, Sleeping, ExecutingTask, Terminated, TestInit,
           TestWaitingReady, TestScheduling, TestWaitingTasks,
           TestRequestingShutdown, TestJoining, TestDone

THEOREM DonePreservesSafety ==
    ASSUME ModelConstantsOK, Safety, Done
    PROVE  Safety'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, TaskUniverse, ReadySignalSafety,
           PanicIsolation, TestCompletionRequiresExecution,
           TestProgressRequiresExecution, TerminationRequiresRequest,
           Done, NORMAL, PANICKING, CheckEvents, DrainChannel, Sleeping,
           ExecutingTask, Terminated, TestInit, TestWaitingReady,
           TestScheduling, TestWaitingTasks, TestRequestingShutdown,
           TestJoining, TestDone

THEOREM CronActionsPreserveSafety ==
    ASSUME ModelConstantsOK, Safety,
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
           \/ Done
    PROVE  Safety'
BY CronSendReadyPreservesSafety,
   CronCheckDueTasksPreservesSafety,
   CronCheckTerminationPreservesSafety,
   CronCheckChannelPreservesSafety,
   CronCheckChannelClosedPreservesSafety,
   CronNoEventsPreservesSafety,
   CronDrainTaskPreservesSafety,
   CronFinishDrainPreservesSafety,
   CronExecuteTaskPreservesSafety,
   CronWakeUpPreservesSafety,
   CronTerminateFromSleepPreservesSafety,
   CronTerminateClosedFromSleepPreservesSafety,
   DonePreservesSafety

THEOREM NextPreservesSafety ==
    (ModelConstantsOK /\ Safety /\ Next) => Safety'
BY TestSpawnCronPreservesSafety,
   TestReceiveReadyPreservesSafety,
   TestScheduleTaskPreservesSafety,
   TestCheckTasksDonePreservesSafety,
   TestRequestShutdownPreservesSafety,
   TestDropHandlePreservesSafety,
   TestJoinPreservesSafety,
   CronActionsPreserveSafety
DEF Next

THEOREM StutterPreservesSafety ==
    Safety /\ UNCHANGED vars => Safety'
BY SMT DEF Safety, TypeOK, ReadySignalSafety, PanicIsolation,
           TestCompletionRequiresExecution, TestProgressRequiresExecution,
           TerminationRequiresRequest, vars

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
