-------------------- MODULE CheckpointStateMachineProofs ---------------------
(*
 * TLAPS proof obligations for CheckpointStateMachine.tla.
 *
 * The proof is intentionally decomposed by safety clause.  That keeps each
 * obligation small enough for automated backends and mirrors the invariants
 * checked by TLC.
 *)

EXTENDS CheckpointStateMachine, FiniteSetTheorems, TLAPS

ModelConstantsOK ==
    /\ Orders \in SUBSET Nat
    /\ Prefixes \in SUBSET Prefixes
    /\ Workers \in SUBSET Workers
    /\ NONE \notin Workers

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

THEOREM InitImpliesTypeOK ==
    ModelConstantsOK /\ Init => TypeOK
<1>1. ASSUME ModelConstantsOK, Init
      PROVE  prefix_state \in [Orders \X Prefixes -> {NotStarted, InProgress, Completed, Failed}]
  BY <1>1, SMT DEF ModelConstantsOK, Init, NotStarted, InProgress, Completed, Failed
<1>2. ASSUME ModelConstantsOK, Init
      PROVE  completed_prefixes \in [Orders -> SUBSET Prefixes]
  BY <1>2, SMT DEF ModelConstantsOK, Init
<1>3. ASSUME ModelConstantsOK, Init
      PROVE  in_progress_prefixes \in [Orders -> SUBSET Prefixes]
  BY <1>3, SMT DEF ModelConstantsOK, Init
<1>4. ASSUME ModelConstantsOK, Init
      PROVE  failed_prefixes \in [Orders -> SUBSET Prefixes]
  BY <1>4, SMT DEF ModelConstantsOK, Init
<1>5. ASSUME ModelConstantsOK, Init
      PROVE  order_complete \in [Orders -> BOOLEAN]
  BY <1>5 DEF Init
<1>6. ASSUME ModelConstantsOK, Init
      PROVE  system_running \in BOOLEAN
  BY <1>6 DEF Init
<1>7. ASSUME ModelConstantsOK, Init
      PROVE  worker_assignment \in [Orders \X Prefixes -> Workers \cup {NONE}]
  BY <1>7, SMT DEF ModelConstantsOK, Init, NONE
<1>8. ASSUME ModelConstantsOK, Init
      PROVE  recovery_needed \in BOOLEAN
  BY <1>8 DEF Init
<1>9. ASSUME ModelConstantsOK, Init
      PROVE  crashed_in_progress \in [Orders -> SUBSET Prefixes]
  BY <1>9, SMT DEF ModelConstantsOK, Init
<1>. QED
  BY <1>1, <1>2, <1>3, <1>4, <1>5, <1>6, <1>7, <1>8, <1>9
  DEF TypeOK

THEOREM InitImpliesDisjointSets ==
    ModelConstantsOK /\ Init => DisjointSets
BY DEF Init, DisjointSets

THEOREM InitImpliesStateConsistent ==
    ModelConstantsOK /\ Init => StateConsistent
BY DEF Init, StateConsistent, NotStarted, InProgress, Completed, Failed

THEOREM InitImpliesCompletedOrderNoInProgress ==
    ModelConstantsOK /\ Init => CompletedOrderNoInProgress
BY DEF Init, CompletedOrderNoInProgress

THEOREM InitImpliesWorkerAssignmentConsistent ==
    ModelConstantsOK /\ Init => WorkerAssignmentConsistent
BY DEF Init, WorkerAssignmentConsistent, NotStarted, InProgress, NONE

THEOREM InitImpliesNoDoubleProcessing ==
    ModelConstantsOK /\ Init => NoDoubleProcessing
BY DEF Init, NoDoubleProcessing, NotStarted, Completed

THEOREM InitImpliesCrashRecoverySound ==
    ModelConstantsOK /\ Init => CrashRecoverySound
BY DEF Init, CrashRecoverySound

THEOREM InitImpliesSafety ==
    ModelConstantsOK /\ Init => Safety
BY InitImpliesTypeOK,
   InitImpliesDisjointSets,
   InitImpliesStateConsistent,
   InitImpliesCompletedOrderNoInProgress,
   InitImpliesWorkerAssignmentConsistent,
   InitImpliesNoDoubleProcessing,
   InitImpliesCrashRecoverySound
DEF Safety

THEOREM StartPrefixPreservesSafety ==
    ASSUME ModelConstantsOK, Safety,
           NEW o \in Orders, NEW p \in Prefixes, NEW w \in Workers,
           StartPrefix(o, p, w)
    PROVE  Safety'
<1>1. TypeOK'
  BY SMT DEF ModelConstantsOK, Safety, TypeOK, StartPrefix,
             NotStarted, InProgress, Completed, Failed, NONE
<1>2. DisjointSets'
  BY SMT DEF ModelConstantsOK, Safety, TypeOK, DisjointSets, StateConsistent,
             StartPrefix, NotStarted, InProgress, Completed, Failed
<1>3. StateConsistent'
  <2>1. ASSUME NEW oo \in Orders, NEW pp \in Prefixes
        PROVE  /\ (prefix_state'[<<oo, pp>>] = Completed) <=> (pp \in completed_prefixes'[oo])
               /\ (prefix_state'[<<oo, pp>>] = InProgress) <=> (pp \in in_progress_prefixes'[oo])
               /\ (prefix_state'[<<oo, pp>>] = Failed) <=> (pp \in failed_prefixes'[oo])
    <3>1. CASE oo = o /\ pp = p
      BY <3>1, FunctionUpdateAt, SMT
      DEF ModelConstantsOK, Safety, TypeOK, StateConsistent, StartPrefix,
          NotStarted, InProgress, Completed, Failed
    <3>2. CASE oo = o /\ pp # p
      BY <3>2, FunctionUpdateAt, FunctionUpdateOther, SMT
      DEF ModelConstantsOK, Safety, TypeOK, StateConsistent, StartPrefix,
          NotStarted, InProgress, Completed, Failed
    <3>3. CASE oo # o
      BY <3>3, FunctionUpdateOther, SMT
      DEF ModelConstantsOK, Safety, TypeOK, StateConsistent, StartPrefix,
          NotStarted, InProgress, Completed, Failed
    <3>. QED
      BY <3>1, <3>2, <3>3
  <2>. QED
    BY <2>1 DEF StateConsistent
<1>4. CompletedOrderNoInProgress'
  BY SMT DEF ModelConstantsOK, Safety, CompletedOrderNoInProgress, StartPrefix
<1>5. WorkerAssignmentConsistent'
  <2>1. ASSUME NEW oo \in Orders, NEW pp \in Prefixes
        PROVE  (worker_assignment'[<<oo, pp>>] # NONE) <=> (pp \in in_progress_prefixes'[oo])
    <3>1. CASE oo = o /\ pp = p
      BY <3>1, FunctionUpdateAt, SMT
      DEF ModelConstantsOK, Safety, TypeOK, WorkerAssignmentConsistent,
          StartPrefix, InProgress, NONE
    <3>2. CASE oo = o /\ pp # p
      BY <3>2, FunctionUpdateAt, FunctionUpdateOther, SMT
      DEF ModelConstantsOK, Safety, TypeOK, WorkerAssignmentConsistent,
          StartPrefix, InProgress, NONE
    <3>3. CASE oo # o
      BY <3>3, FunctionUpdateOther, SMT
      DEF ModelConstantsOK, Safety, TypeOK, WorkerAssignmentConsistent,
          StartPrefix, InProgress, NONE
    <3>. QED
      BY <3>1, <3>2, <3>3
  <2>. QED
    BY <2>1 DEF WorkerAssignmentConsistent
<1>6. NoDoubleProcessing'
  <2>1. ASSUME NEW oo \in Orders
        PROVE  completed_prefixes'[oo] \cap in_progress_prefixes'[oo] = {}
    <3>1. CASE oo = o
      BY <3>1, FunctionUpdateAt, SMT
      DEF ModelConstantsOK, Safety, TypeOK, StateConsistent, NoDoubleProcessing,
          StartPrefix, Completed, InProgress
    <3>2. CASE oo # o
      BY <3>2, FunctionUpdateOther, SMT
      DEF ModelConstantsOK, Safety, TypeOK, StateConsistent, NoDoubleProcessing,
          StartPrefix, Completed, InProgress
    <3>. QED
      BY <3>1, <3>2
  <2>. QED
    BY <2>1 DEF NoDoubleProcessing
<1>7. CrashRecoverySound'
  BY SMT DEF ModelConstantsOK, Safety, CrashRecoverySound, StartPrefix
<1>. QED
  BY <1>1, <1>2, <1>3, <1>4, <1>5, <1>6, <1>7 DEF Safety

THEOREM CompletePrefixPreservesSafety ==
    ASSUME ModelConstantsOK, Safety,
           NEW o \in Orders, NEW p \in Prefixes,
           CompletePrefix(o, p)
    PROVE  Safety'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, DisjointSets, StateConsistent,
           CompletedOrderNoInProgress, WorkerAssignmentConsistent,
           NoDoubleProcessing, CrashRecoverySound, CompletePrefix,
           NotStarted, InProgress, Completed, Failed, NONE

THEOREM FailPrefixPreservesSafety ==
    ASSUME ModelConstantsOK, Safety,
           NEW o \in Orders, NEW p \in Prefixes,
           FailPrefix(o, p)
    PROVE  Safety'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, DisjointSets, StateConsistent,
           CompletedOrderNoInProgress, WorkerAssignmentConsistent,
           NoDoubleProcessing, CrashRecoverySound, FailPrefix,
           NotStarted, InProgress, Completed, Failed, NONE

THEOREM ClearFailedPreservesSafety ==
    ASSUME ModelConstantsOK, Safety,
           NEW o \in Orders, NEW p \in Prefixes,
           ClearFailed(o, p)
    PROVE  Safety'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, DisjointSets, StateConsistent,
           CompletedOrderNoInProgress, WorkerAssignmentConsistent,
           NoDoubleProcessing, CrashRecoverySound, ClearFailed,
           NotStarted, InProgress, Completed, Failed, NONE

THEOREM CompleteOrderPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, NEW o \in Orders, CompleteOrder(o)
    PROVE  Safety'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, DisjointSets, StateConsistent,
           CompletedOrderNoInProgress, WorkerAssignmentConsistent,
           NoDoubleProcessing, CrashRecoverySound, CompleteOrder,
           NotStarted, InProgress, Completed, Failed, NONE

THEOREM CrashPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, Crash
    PROVE  Safety'
<1>1. TypeOK'
  BY IsaM("auto") DEF ModelConstantsOK, Safety, TypeOK, Crash
<1>2. DisjointSets'
  BY SMT DEF ModelConstantsOK, Safety, DisjointSets, Crash
<1>3. StateConsistent'
  BY IsaM("auto") DEF ModelConstantsOK, Safety, StateConsistent, Crash
<1>4. CompletedOrderNoInProgress'
  BY SMT DEF ModelConstantsOK, Safety, CompletedOrderNoInProgress, Crash
<1>5. WorkerAssignmentConsistent'
  BY IsaM("auto") DEF ModelConstantsOK, Safety, WorkerAssignmentConsistent, Crash
<1>6. NoDoubleProcessing'
  BY IsaM("auto") DEF ModelConstantsOK, Safety, NoDoubleProcessing, Crash
<1>7. CrashRecoverySound'
  BY SMT DEF ModelConstantsOK, Safety, CrashRecoverySound, Crash
<1>. QED
  BY <1>1, <1>2, <1>3, <1>4, <1>5, <1>6, <1>7 DEF Safety

THEOREM RecoverInProgressAsFailedPreservesSafety ==
    ASSUME ModelConstantsOK, Safety,
           NEW o \in Orders, RecoverInProgressAsFailed(o)
    PROVE  Safety'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, DisjointSets, StateConsistent,
           CompletedOrderNoInProgress, WorkerAssignmentConsistent,
           NoDoubleProcessing, CrashRecoverySound, RecoverInProgressAsFailed,
           NotStarted, InProgress, Completed, Failed, NONE

THEOREM RestartPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, Restart
    PROVE  Safety'
<1>1. TypeOK'
  BY IsaM("auto") DEF ModelConstantsOK, Safety, TypeOK, Restart
<1>2. DisjointSets'
  BY SMT DEF ModelConstantsOK, Safety, DisjointSets, Restart
<1>3. StateConsistent'
  BY IsaM("auto") DEF ModelConstantsOK, Safety, StateConsistent, Restart
<1>4. CompletedOrderNoInProgress'
  BY SMT DEF ModelConstantsOK, Safety, CompletedOrderNoInProgress, Restart
<1>5. WorkerAssignmentConsistent'
  BY IsaM("auto") DEF ModelConstantsOK, Safety, WorkerAssignmentConsistent, Restart
<1>6. NoDoubleProcessing'
  BY IsaM("auto") DEF ModelConstantsOK, Safety, NoDoubleProcessing, Restart
<1>7. CrashRecoverySound'
  BY SMT DEF ModelConstantsOK, Safety, CrashRecoverySound, Restart
<1>. QED
  BY <1>1, <1>2, <1>3, <1>4, <1>5, <1>6, <1>7 DEF Safety

THEOREM NextPreservesSafety ==
    (ModelConstantsOK /\ Safety /\ Next) => Safety'
BY StartPrefixPreservesSafety,
   CompletePrefixPreservesSafety,
   FailPrefixPreservesSafety,
   ClearFailedPreservesSafety,
   CompleteOrderPreservesSafety,
   CrashPreservesSafety,
   RecoverInProgressAsFailedPreservesSafety,
   RestartPreservesSafety
DEF Next

THEOREM StutterPreservesSafety ==
    Safety /\ UNCHANGED vars => Safety'
BY SMT DEF Safety, TypeOK, DisjointSets, StateConsistent,
           CompletedOrderNoInProgress, WorkerAssignmentConsistent,
           NoDoubleProcessing, CrashRecoverySound, vars

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
