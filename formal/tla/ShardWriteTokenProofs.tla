------------------------ MODULE ShardWriteTokenProofs ------------------------
(*
 * TLAPS proof obligations for ShardWriteToken.tla.
 *
 * These lemmas keep the deductive proof surface small: prove the initial state
 * establishes each safety clause independently, then compose the clauses into
 * the model's Safety operator.
 *)

EXTENDS ShardWriteToken, TLAPS

ModelConstantsOK ==
    /\ MaxGeneration \in Nat
    /\ Workers \in SUBSET Workers
    /\ Shards \in SUBSET Shards

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
      PROVE  write_locked \in [Shards -> BOOLEAN]
  BY <1>1 DEF Init
<1>2. ASSUME ModelConstantsOK, Init
      PROVE  write_holder \in [Shards -> SUBSET Workers]
  BY <1>2 DEF Init
<1>3. ASSUME ModelConstantsOK, Init, NEW s \in Shards
      PROVE  \A w1, w2 \in write_holder[s]: w1 = w2
  BY <1>3 DEF Init
<1>4. ASSUME ModelConstantsOK, Init
      PROVE  write_generation \in [Shards -> 0..MaxGeneration]
  BY <1>4, SMT DEF Init, ModelConstantsOK
<1>5. ASSUME ModelConstantsOK, Init
      PROVE  max_generation_seen \in [Shards -> 0..MaxGeneration]
  BY <1>5, SMT DEF Init, ModelConstantsOK
<1>6. ASSUME ModelConstantsOK, Init
      PROVE  tokens \in [Workers \X Shards -> [exists: BOOLEAN, gen: 0..MaxGeneration, valid: BOOLEAN]]
  BY <1>6, SMT DEF Init, NoToken, ModelConstantsOK
<1>7. ASSUME ModelConstantsOK, Init
      PROVE  pc \in [Workers -> {Idle, WantLock, HaveLock, Releasing}]
  BY <1>7 DEF Init
<1>. QED
  BY <1>1, <1>2, <1>3, <1>4, <1>5, <1>6, <1>7
  DEF TypeOK, WriteLockedTypeOK, WriteHolderTypeOK, WriteHolderUnique,
      WriteGenerationTypeOK, MaxGenerationSeenTypeOK, TokensTypeOK, PcTypeOK

THEOREM InitImpliesAtMostOneWriter ==
    ModelConstantsOK /\ Init => AtMostOneWriter
<1>1. ASSUME ModelConstantsOK, Init, NEW s \in Shards
      PROVE  \A w1, w2 \in ValidTokenHolders(s): w1 = w2
  <2>1. ValidTokenHolders(s) = {}
    BY <1>1, Zenon DEF Init, NoToken, ValidTokenHolders
  <2>. QED
    BY <2>1
<1>. QED
  BY <1>1 DEF AtMostOneWriter

THEOREM InitImpliesLockedImpliesHolder ==
    ModelConstantsOK /\ Init => LockedImpliesHolder
BY DEF Init, LockedImpliesHolder

THEOREM InitImpliesUnlockedImpliesNoHolder ==
    ModelConstantsOK /\ Init => UnlockedImpliesNoHolder
BY DEF Init, UnlockedImpliesNoHolder

THEOREM InitImpliesGenerationMonotonic ==
    ModelConstantsOK /\ Init => GenerationMonotonic
BY DEF Init, GenerationMonotonic

THEOREM InitImpliesValidTokenGenerationMatch ==
    ModelConstantsOK /\ Init => ValidTokenGenerationMatch
BY DEF Init, ValidTokenGenerationMatch, NoToken

THEOREM InitImpliesValidTokenImpliesHolder ==
    ModelConstantsOK /\ Init => ValidTokenImpliesHolder
BY DEF Init, ValidTokenImpliesHolder, NoToken

THEOREM InitImpliesSafety ==
    ModelConstantsOK /\ Init => Safety
BY InitImpliesTypeOK,
   InitImpliesAtMostOneWriter,
   InitImpliesLockedImpliesHolder,
   InitImpliesUnlockedImpliesNoHolder,
   InitImpliesGenerationMonotonic,
   InitImpliesValidTokenGenerationMatch,
   InitImpliesValidTokenImpliesHolder
DEF Safety

THEOREM RequestLockPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, NEW w \in Workers, RequestLock(w)
    PROVE  Safety'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, AtMostOneWriter,
           WriteLockedTypeOK, WriteHolderTypeOK, WriteHolderUnique,
           WriteGenerationTypeOK, MaxGenerationSeenTypeOK, TokensTypeOK, PcTypeOK,
           LockedImpliesHolder, UnlockedImpliesNoHolder,
           GenerationMonotonic, ValidTokenGenerationMatch,
           ValidTokenImpliesHolder, ValidTokenHolders, RequestLock

THEOREM WantReleasePreservesSafety ==
    ASSUME ModelConstantsOK, Safety, NEW w \in Workers, WantRelease(w)
    PROVE  Safety'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, AtMostOneWriter,
           WriteLockedTypeOK, WriteHolderTypeOK, WriteHolderUnique,
           WriteGenerationTypeOK, MaxGenerationSeenTypeOK, TokensTypeOK, PcTypeOK,
           LockedImpliesHolder, UnlockedImpliesNoHolder,
           GenerationMonotonic, ValidTokenGenerationMatch,
           ValidTokenImpliesHolder, ValidTokenHolders, WantRelease

THEOREM TokenInvalidationPreservesTypeOK ==
    ASSUME ModelConstantsOK, Safety,
           NEW w \in Workers, NEW s \in Shards, TokenInvalidation(w, s)
    PROVE  TypeOK'
BY SMT DEF ModelConstantsOK, Safety, TypeOK,
           WriteLockedTypeOK, WriteHolderTypeOK, WriteHolderUnique,
           WriteGenerationTypeOK, MaxGenerationSeenTypeOK, TokensTypeOK, PcTypeOK,
           TokenInvalidation, InvalidToken

THEOREM TokenInvalidationPreservesAtMostOneWriter ==
    ASSUME ModelConstantsOK, Safety,
           NEW w \in Workers, NEW s \in Shards, TokenInvalidation(w, s)
    PROVE  AtMostOneWriter'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, TokensTypeOK, AtMostOneWriter,
           ValidTokenGenerationMatch, ValidTokenHolders, TokenInvalidation,
           InvalidToken

THEOREM TokenInvalidationPreservesUnchangedSafetyClauses ==
    ASSUME ModelConstantsOK, Safety,
           NEW w \in Workers, NEW s \in Shards, TokenInvalidation(w, s)
    PROVE  /\ LockedImpliesHolder'
           /\ UnlockedImpliesNoHolder'
           /\ GenerationMonotonic'
           /\ ValidTokenGenerationMatch'
           /\ ValidTokenImpliesHolder'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, TokensTypeOK,
           LockedImpliesHolder, UnlockedImpliesNoHolder,
           GenerationMonotonic, ValidTokenGenerationMatch,
           ValidTokenImpliesHolder, TokenInvalidation, InvalidToken

THEOREM TokenInvalidationPreservesSafety ==
    ASSUME ModelConstantsOK, Safety,
           NEW w \in Workers, NEW s \in Shards, TokenInvalidation(w, s)
    PROVE  Safety'
BY TokenInvalidationPreservesTypeOK,
   TokenInvalidationPreservesAtMostOneWriter,
   TokenInvalidationPreservesUnchangedSafetyClauses
DEF Safety

THEOREM ReleaseValidPreservesTypeOK ==
    ASSUME ModelConstantsOK, Safety,
           NEW w \in Workers, NEW s \in Shards, ReleaseValid(w, s)
    PROVE  TypeOK'
BY SMT DEF ModelConstantsOK, Safety, TypeOK,
           WriteLockedTypeOK, WriteHolderTypeOK, WriteHolderUnique,
           WriteGenerationTypeOK, MaxGenerationSeenTypeOK, TokensTypeOK,
           PcTypeOK, ReleaseValid, NoToken

THEOREM ReleaseInvalidPreservesTypeOK ==
    ASSUME ModelConstantsOK, Safety,
           NEW w \in Workers, NEW s \in Shards, ReleaseInvalid(w, s)
    PROVE  TypeOK'
BY SMT DEF ModelConstantsOK, Safety, TypeOK,
           WriteLockedTypeOK, WriteHolderTypeOK, WriteHolderUnique,
           WriteGenerationTypeOK, MaxGenerationSeenTypeOK, TokensTypeOK,
           PcTypeOK, ReleaseInvalid, NoToken

THEOREM ReleasePreservesTypeOK ==
    ASSUME ModelConstantsOK, Safety,
           NEW w \in Workers, NEW s \in Shards, Release(w, s)
    PROVE  TypeOK'
BY ReleaseValidPreservesTypeOK, ReleaseInvalidPreservesTypeOK DEF Release

THEOREM ReleaseValidPreservesSafetyClauses ==
    ASSUME ModelConstantsOK, Safety,
           NEW w \in Workers, NEW s \in Shards, ReleaseValid(w, s)
    PROVE  /\ AtMostOneWriter'
           /\ LockedImpliesHolder'
           /\ UnlockedImpliesNoHolder'
           /\ GenerationMonotonic'
           /\ ValidTokenGenerationMatch'
           /\ ValidTokenImpliesHolder'
<1>1. AtMostOneWriter'
  BY SMT DEF ModelConstantsOK, Safety, TypeOK, WriteHolderTypeOK,
             TokensTypeOK, AtMostOneWriter, ValidTokenGenerationMatch,
             ValidTokenImpliesHolder, ValidTokenHolders, ReleaseValid, NoToken
<1>2. LockedImpliesHolder'
  <2>1. ASSUME NEW shard \in Shards
        PROVE  write_locked'[shard] = TRUE =>
               \E holder \in Workers: write_holder'[shard] = {holder}
    <3>1. CASE shard = s
      BY <3>1, FunctionUpdateAt, SMT
      DEF ModelConstantsOK, Safety, TypeOK, WriteLockedTypeOK, ReleaseValid
    <3>2. CASE shard # s
      BY <3>2, FunctionUpdateOther, SMT
      DEF ModelConstantsOK, Safety, TypeOK, WriteLockedTypeOK,
          LockedImpliesHolder, ReleaseValid
    <3>. QED
      BY <3>1, <3>2
  <2>. QED
    BY <2>1 DEF LockedImpliesHolder
<1>3. UnlockedImpliesNoHolder'
  BY SMT DEF ModelConstantsOK, Safety, TypeOK, WriteHolderTypeOK,
             UnlockedImpliesNoHolder, ReleaseValid
<1>4. GenerationMonotonic'
  BY SMT DEF ModelConstantsOK, Safety, GenerationMonotonic, ReleaseValid
<1>5. ValidTokenGenerationMatch'
  BY SMT DEF ModelConstantsOK, Safety, TypeOK, TokensTypeOK,
             ValidTokenGenerationMatch, ValidTokenHolders, AtMostOneWriter,
             ReleaseValid, NoToken
<1>6. ValidTokenImpliesHolder'
  BY SMT DEF ModelConstantsOK, Safety, TypeOK, WriteHolderTypeOK,
             TokensTypeOK, ValidTokenImpliesHolder, ValidTokenHolders,
             AtMostOneWriter, ReleaseValid, NoToken
<1>. QED
  BY <1>1, <1>2, <1>3, <1>4, <1>5, <1>6

THEOREM ReleaseInvalidPreservesSafetyClauses ==
    ASSUME ModelConstantsOK, Safety,
           NEW w \in Workers, NEW s \in Shards, ReleaseInvalid(w, s)
    PROVE  /\ AtMostOneWriter'
           /\ LockedImpliesHolder'
           /\ UnlockedImpliesNoHolder'
           /\ GenerationMonotonic'
           /\ ValidTokenGenerationMatch'
           /\ ValidTokenImpliesHolder'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, WriteHolderTypeOK,
           TokensTypeOK, AtMostOneWriter, LockedImpliesHolder,
           UnlockedImpliesNoHolder, GenerationMonotonic,
           ValidTokenGenerationMatch, ValidTokenImpliesHolder,
           ValidTokenHolders, ReleaseInvalid, NoToken

THEOREM ReleasePreservesSafetyClauses ==
    ASSUME ModelConstantsOK, Safety,
           NEW w \in Workers, NEW s \in Shards, Release(w, s)
    PROVE  /\ AtMostOneWriter'
           /\ LockedImpliesHolder'
           /\ UnlockedImpliesNoHolder'
           /\ GenerationMonotonic'
           /\ ValidTokenGenerationMatch'
           /\ ValidTokenImpliesHolder'
BY ReleaseValidPreservesSafetyClauses, ReleaseInvalidPreservesSafetyClauses
DEF Release

THEOREM ReleasePreservesSafety ==
    ASSUME ModelConstantsOK, Safety,
           NEW w \in Workers, NEW s \in Shards, Release(w, s)
    PROVE  Safety'
BY ReleasePreservesTypeOK, ReleasePreservesSafetyClauses DEF Safety

THEOREM TryAcquireSuccessPreservesTypeOK ==
    ASSUME ModelConstantsOK, Safety,
           NEW w \in Workers, NEW s \in Shards, TryAcquireSuccess(w, s)
    PROVE  TypeOK'
<1>1. write_locked' \in [Shards -> BOOLEAN]
  BY SMT DEF Safety, TypeOK, WriteLockedTypeOK, TryAcquireSuccess
<1>2. write_holder' \in [Shards -> SUBSET Workers]
  BY SMT DEF Safety, TypeOK, WriteHolderTypeOK, TryAcquireSuccess
<1>3. \A shard \in Shards: \A w1, w2 \in write_holder'[shard]: w1 = w2
  BY SMT DEF Safety, TypeOK, WriteHolderUnique, TryAcquireSuccess
<1>4. write_generation' \in [Shards -> 0..MaxGeneration]
  BY SMT DEF ModelConstantsOK, Safety, TypeOK, WriteGenerationTypeOK,
             TryAcquireSuccess
<1>5. max_generation_seen' \in [Shards -> 0..MaxGeneration]
  BY SMT DEF ModelConstantsOK, Safety, TypeOK, GenerationMonotonic,
             WriteGenerationTypeOK, MaxGenerationSeenTypeOK, TryAcquireSuccess
<1>6. tokens' \in [Workers \X Shards -> [exists: BOOLEAN, gen: 0..MaxGeneration, valid: BOOLEAN]]
  BY SMT DEF ModelConstantsOK, Safety, TypeOK, WriteGenerationTypeOK,
             TokensTypeOK, TryAcquireSuccess, Token
<1>7. pc' \in [Workers -> {Idle, WantLock, HaveLock, Releasing}]
  BY SMT DEF Safety, TypeOK, PcTypeOK, TryAcquireSuccess
<1>. QED
  BY <1>1, <1>2, <1>3, <1>4, <1>5, <1>6, <1>7
  DEF TypeOK, WriteLockedTypeOK, WriteHolderTypeOK, WriteHolderUnique,
      WriteGenerationTypeOK, MaxGenerationSeenTypeOK, TokensTypeOK, PcTypeOK

THEOREM TryAcquireFailurePreservesTypeOK ==
    ASSUME ModelConstantsOK, Safety,
           NEW w \in Workers, NEW s \in Shards, TryAcquireFailure(w, s)
    PROVE  TypeOK'
BY SMT DEF ModelConstantsOK, Safety, TypeOK,
           WriteLockedTypeOK, WriteHolderTypeOK, WriteHolderUnique,
           WriteGenerationTypeOK, MaxGenerationSeenTypeOK, TokensTypeOK,
           PcTypeOK, TryAcquireFailure, Token

THEOREM TryAcquirePreservesTypeOK ==
    ASSUME ModelConstantsOK, Safety,
           NEW w \in Workers, NEW s \in Shards, TryAcquire(w, s)
    PROVE  TypeOK'
BY TryAcquireSuccessPreservesTypeOK, TryAcquireFailurePreservesTypeOK
DEF TryAcquire

THEOREM TryAcquireSuccessPreservesSafetyClauses ==
    ASSUME ModelConstantsOK, Safety,
           NEW w \in Workers, NEW s \in Shards, TryAcquireSuccess(w, s)
    PROVE  /\ AtMostOneWriter'
           /\ LockedImpliesHolder'
           /\ UnlockedImpliesNoHolder'
           /\ GenerationMonotonic'
           /\ ValidTokenGenerationMatch'
           /\ ValidTokenImpliesHolder'
<1>1. AtMostOneWriter'
  BY SMT DEF ModelConstantsOK, Safety, TypeOK, WriteHolderTypeOK,
             TokensTypeOK, AtMostOneWriter, UnlockedImpliesNoHolder,
             ValidTokenImpliesHolder, ValidTokenHolders, TryAcquireSuccess, Token
<1>2. LockedImpliesHolder'
  <2>1. ASSUME NEW shard \in Shards
        PROVE  write_locked'[shard] = TRUE =>
               \E holder \in Workers: write_holder'[shard] = {holder}
    <3>1. CASE shard = s
      BY <3>1, FunctionUpdateAt, SMT
      DEF ModelConstantsOK, Safety, TypeOK, WriteLockedTypeOK,
          WriteHolderTypeOK, TryAcquireSuccess
    <3>2. CASE shard # s
      BY <3>2, FunctionUpdateOther, SMT
      DEF ModelConstantsOK, Safety, TypeOK, WriteLockedTypeOK,
          WriteHolderTypeOK, LockedImpliesHolder, TryAcquireSuccess
    <3>. QED
      BY <3>1, <3>2
  <2>. QED
    BY <2>1 DEF LockedImpliesHolder
<1>3. UnlockedImpliesNoHolder'
  <2>1. ASSUME NEW shard \in Shards
        PROVE  write_locked'[shard] = FALSE => write_holder'[shard] = {}
    <3>1. CASE shard = s
      BY <3>1, FunctionUpdateAt, SMT
      DEF ModelConstantsOK, Safety, TypeOK, WriteLockedTypeOK, TryAcquireSuccess
    <3>2. CASE shard # s
      BY <3>2, FunctionUpdateOther, SMT
      DEF ModelConstantsOK, Safety, TypeOK, WriteLockedTypeOK,
          WriteHolderTypeOK, UnlockedImpliesNoHolder, TryAcquireSuccess
    <3>. QED
      BY <3>1, <3>2
  <2>. QED
    BY <2>1 DEF UnlockedImpliesNoHolder
<1>4. GenerationMonotonic'
  <2>1. ASSUME NEW shard \in Shards
        PROVE  write_generation'[shard] = max_generation_seen'[shard]
    <3>1. CASE shard = s
      BY <3>1, FunctionUpdateAt, SMT
      DEF ModelConstantsOK, Safety, TypeOK, WriteGenerationTypeOK,
          MaxGenerationSeenTypeOK, GenerationMonotonic, TryAcquireSuccess
    <3>2. CASE shard # s
      BY <3>2, FunctionUpdateOther, SMT
      DEF ModelConstantsOK, Safety, TypeOK, WriteGenerationTypeOK,
          MaxGenerationSeenTypeOK, GenerationMonotonic, TryAcquireSuccess
    <3>. QED
      BY <3>1, <3>2
  <2>. QED
    BY <2>1 DEF GenerationMonotonic
<1>5. ValidTokenGenerationMatch'
  BY SMT DEF ModelConstantsOK, Safety, TypeOK, WriteGenerationTypeOK,
             TokensTypeOK, UnlockedImpliesNoHolder, ValidTokenImpliesHolder,
             ValidTokenGenerationMatch, TryAcquireSuccess, Token
<1>6. ValidTokenImpliesHolder'
  BY SMT DEF ModelConstantsOK, Safety, TypeOK, WriteHolderTypeOK,
             TokensTypeOK, UnlockedImpliesNoHolder, ValidTokenImpliesHolder,
             TryAcquireSuccess, Token
<1>. QED
  BY <1>1, <1>2, <1>3, <1>4, <1>5, <1>6

THEOREM TryAcquireFailurePreservesSafetyClauses ==
    ASSUME ModelConstantsOK, Safety,
           NEW w \in Workers, NEW s \in Shards, TryAcquireFailure(w, s)
    PROVE  /\ AtMostOneWriter'
           /\ LockedImpliesHolder'
           /\ UnlockedImpliesNoHolder'
           /\ GenerationMonotonic'
           /\ ValidTokenGenerationMatch'
           /\ ValidTokenImpliesHolder'
BY SMT DEF ModelConstantsOK, Safety, TypeOK, WriteHolderTypeOK,
           TokensTypeOK, AtMostOneWriter, LockedImpliesHolder,
           UnlockedImpliesNoHolder, GenerationMonotonic,
           ValidTokenGenerationMatch, ValidTokenImpliesHolder,
           ValidTokenHolders, TryAcquireFailure, Token

THEOREM TryAcquirePreservesSafetyClauses ==
    ASSUME ModelConstantsOK, Safety,
           NEW w \in Workers, NEW s \in Shards, TryAcquire(w, s)
    PROVE  /\ AtMostOneWriter'
           /\ LockedImpliesHolder'
           /\ UnlockedImpliesNoHolder'
           /\ GenerationMonotonic'
           /\ ValidTokenGenerationMatch'
           /\ ValidTokenImpliesHolder'
BY TryAcquireSuccessPreservesSafetyClauses,
   TryAcquireFailurePreservesSafetyClauses
DEF TryAcquire

THEOREM TryAcquirePreservesSafety ==
    ASSUME ModelConstantsOK, Safety,
           NEW w \in Workers, NEW s \in Shards, TryAcquire(w, s)
    PROVE  Safety'
BY TryAcquirePreservesTypeOK, TryAcquirePreservesSafetyClauses DEF Safety

THEOREM NextPreservesSafety ==
    (ModelConstantsOK /\ Safety /\ Next) => Safety'
BY RequestLockPreservesSafety, WantReleasePreservesSafety,
   TokenInvalidationPreservesSafety, ReleasePreservesSafety,
   TryAcquirePreservesSafety
DEF Next

THEOREM StutterPreservesSafety ==
    Safety /\ UNCHANGED vars => Safety'
BY SMT DEF Safety, TypeOK, AtMostOneWriter, ValidTokenHolders,
           WriteLockedTypeOK, WriteHolderTypeOK, WriteHolderUnique,
           WriteGenerationTypeOK, MaxGenerationSeenTypeOK, TokensTypeOK, PcTypeOK,
           LockedImpliesHolder, UnlockedImpliesNoHolder,
           GenerationMonotonic, ValidTokenGenerationMatch,
           ValidTokenImpliesHolder, vars

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

THEOREM ModelSpecImpliesAlwaysSafety ==
    (ModelConstantsOK /\ Spec) => []Safety
BY SpecImpliesAlwaysSafety

=============================================================================
