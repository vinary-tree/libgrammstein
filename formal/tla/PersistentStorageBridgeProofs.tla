--------------------- MODULE PersistentStorageBridgeProofs ---------------------
(*
 * TLAPS safety proof for PersistentStorageBridge.tla.
 *)

EXTENDS PersistentStorageBridge, TLAPS

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
    ModelConstantsOK /\ Init => TypeOK
BY TrueIsBoolean, FalseIsBoolean DEF ModelConstantsOK, Init, TypeOK

THEOREM InitImpliesClaimRequiresDurableEvidence ==
    ModelConstantsOK /\ Init => ClaimRequiresDurableEvidence
BY SMT, FalseNotTrue DEF ModelConstantsOK, Init, ClaimRequiresDurableEvidence

THEOREM InitImpliesRecoveredPrefixHasCheckpointEvidence ==
    ModelConstantsOK /\ Init => RecoveredPrefixHasCheckpointEvidence
BY SMT, FalseNotTrue
DEF ModelConstantsOK, Init, RecoveredPrefixHasCheckpointEvidence

THEOREM InitImpliesFailedSyncCannotClaimWithoutDurableData ==
    ModelConstantsOK /\ Init => FailedSyncCannotClaimWithoutDurableData
BY SMT, FalseNotTrue
DEF ModelConstantsOK, Init, FailedSyncCannotClaimWithoutDurableData

THEOREM InitImpliesForceQuitPublishesNoNewClaim ==
    ModelConstantsOK /\ Init => ForceQuitPublishesNoNewClaim
BY SMT, FalseNotTrue DEF ModelConstantsOK, Init, ForceQuitPublishesNoNewClaim

THEOREM InitImpliesDrainFailurePublishesNoNewClaim ==
    ModelConstantsOK /\ Init => DrainFailurePublishesNoNewClaim
BY SMT, FalseNotTrue DEF ModelConstantsOK, Init, DrainFailurePublishesNoNewClaim

THEOREM InitImpliesCheckpointAfterWorkerDrain ==
    ModelConstantsOK /\ Init => CheckpointAfterWorkerDrain
BY SMT, FalseNotTrue DEF ModelConstantsOK, Init, CheckpointAfterWorkerDrain

THEOREM InitImpliesMetadataFailureDoesNotClaim ==
    ModelConstantsOK /\ Init => MetadataFailureDoesNotClaim
BY SMT, FalseNotTrue DEF ModelConstantsOK, Init, MetadataFailureDoesNotClaim

THEOREM InitImpliesDataDurableRequiresVisible ==
    ModelConstantsOK /\ Init => DataDurableRequiresVisible
BY SMT, FalseNotTrue DEF ModelConstantsOK, Init, DataDurableRequiresVisible

THEOREM InitImpliesVocabularyDurableRequiresVisible ==
    ModelConstantsOK /\ Init => VocabularyDurableRequiresVisible
BY SMT, FalseNotTrue DEF ModelConstantsOK, Init, VocabularyDurableRequiresVisible

THEOREM InitImpliesVocabularyStableWhenRecovered ==
    ModelConstantsOK /\ Init => VocabularyStableWhenRecovered
BY SMT, FalseNotTrue DEF ModelConstantsOK, Init, VocabularyStableWhenRecovered

THEOREM InitImpliesSingleTrieClaimSharesBoundary ==
    ModelConstantsOK /\ Init => SingleTrieClaimSharesBoundary
BY SMT, FalseNotTrue DEF ModelConstantsOK, Init, SingleTrieClaimSharesBoundary

THEOREM InitImpliesShardedClaimHasAuxiliaryMetadata ==
    ModelConstantsOK /\ Init => ShardedClaimHasAuxiliaryMetadata
BY SMT, FalseNotTrue DEF ModelConstantsOK, Init, ShardedClaimHasAuxiliaryMetadata

THEOREM InitImpliesSafety ==
    ModelConstantsOK /\ Init => Safety
BY InitImpliesTypeOK, InitImpliesClaimRequiresDurableEvidence,
   InitImpliesRecoveredPrefixHasCheckpointEvidence,
   InitImpliesFailedSyncCannotClaimWithoutDurableData,
   InitImpliesForceQuitPublishesNoNewClaim,
   InitImpliesDrainFailurePublishesNoNewClaim,
   InitImpliesCheckpointAfterWorkerDrain,
   InitImpliesMetadataFailureDoesNotClaim,
   InitImpliesDataDurableRequiresVisible,
   InitImpliesVocabularyDurableRequiresVisible,
   InitImpliesVocabularyStableWhenRecovered,
   InitImpliesSingleTrieClaimSharesBoundary,
   InitImpliesShardedClaimHasAuxiliaryMetadata
DEF Safety

THEOREM CommitPrefixOkPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, CommitPrefixOk
    PROVE  Safety'
BY SMT, TrueIsBoolean, FalseIsBoolean
DEF ModelConstantsOK, Safety, TypeOK, ClaimRequiresDurableEvidence,
    RecoveredPrefixHasCheckpointEvidence, FailedSyncCannotClaimWithoutDurableData,
    ForceQuitPublishesNoNewClaim, DrainFailurePublishesNoNewClaim,
    CheckpointAfterWorkerDrain, MetadataFailureDoesNotClaim,
    DataDurableRequiresVisible, VocabularyDurableRequiresVisible,
    VocabularyStableWhenRecovered, SingleTrieClaimSharesBoundary,
    ShardedClaimHasAuxiliaryMetadata, Active, DependencyEvidenceReady,
    CommitPrefixOk

THEOREM SyncDataOkPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, SyncDataOk
    PROVE  Safety'
BY SMT, TrueIsBoolean, FalseIsBoolean
DEF ModelConstantsOK, Safety, TypeOK, ClaimRequiresDurableEvidence,
    RecoveredPrefixHasCheckpointEvidence, FailedSyncCannotClaimWithoutDurableData,
    ForceQuitPublishesNoNewClaim, DrainFailurePublishesNoNewClaim,
    CheckpointAfterWorkerDrain, MetadataFailureDoesNotClaim,
    DataDurableRequiresVisible, VocabularyDurableRequiresVisible,
    VocabularyStableWhenRecovered, SingleTrieClaimSharesBoundary,
    ShardedClaimHasAuxiliaryMetadata, Active, DependencyEvidenceReady,
    SyncDataOk

THEOREM SyncDataFailPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, SyncDataFail
    PROVE  Safety'
BY SMT, TrueIsBoolean, FalseIsBoolean
DEF ModelConstantsOK, Safety, TypeOK, ClaimRequiresDurableEvidence,
    RecoveredPrefixHasCheckpointEvidence, FailedSyncCannotClaimWithoutDurableData,
    ForceQuitPublishesNoNewClaim, DrainFailurePublishesNoNewClaim,
    CheckpointAfterWorkerDrain, MetadataFailureDoesNotClaim,
    DataDurableRequiresVisible, VocabularyDurableRequiresVisible,
    VocabularyStableWhenRecovered, SingleTrieClaimSharesBoundary,
    ShardedClaimHasAuxiliaryMetadata, Active, DependencyEvidenceReady,
    SyncDataFail

THEOREM CheckpointVocabOkPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, CheckpointVocabOk
    PROVE  Safety'
BY SMT, TrueIsBoolean, FalseIsBoolean
DEF ModelConstantsOK, Safety, TypeOK, ClaimRequiresDurableEvidence,
    RecoveredPrefixHasCheckpointEvidence, FailedSyncCannotClaimWithoutDurableData,
    ForceQuitPublishesNoNewClaim, DrainFailurePublishesNoNewClaim,
    CheckpointAfterWorkerDrain, MetadataFailureDoesNotClaim,
    DataDurableRequiresVisible, VocabularyDurableRequiresVisible,
    VocabularyStableWhenRecovered, SingleTrieClaimSharesBoundary,
    ShardedClaimHasAuxiliaryMetadata, Active, DependencyEvidenceReady,
    CheckpointVocabOk

THEOREM DrainWorkersPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, DrainWorkers
    PROVE  Safety'
BY SMT, TrueIsBoolean, FalseIsBoolean
DEF ModelConstantsOK, Safety, TypeOK, ClaimRequiresDurableEvidence,
    RecoveredPrefixHasCheckpointEvidence, FailedSyncCannotClaimWithoutDurableData,
    ForceQuitPublishesNoNewClaim, DrainFailurePublishesNoNewClaim,
    CheckpointAfterWorkerDrain, MetadataFailureDoesNotClaim,
    DataDurableRequiresVisible, VocabularyDurableRequiresVisible,
    VocabularyStableWhenRecovered, SingleTrieClaimSharesBoundary,
    ShardedClaimHasAuxiliaryMetadata, Active, DependencyEvidenceReady,
    DrainWorkers

THEOREM SaveCheckpointOkPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, SaveCheckpointOk
    PROVE  Safety'
BY SMT, TrueIsBoolean, FalseIsBoolean
DEF ModelConstantsOK, Safety, TypeOK, ClaimRequiresDurableEvidence,
    RecoveredPrefixHasCheckpointEvidence, FailedSyncCannotClaimWithoutDurableData,
    ForceQuitPublishesNoNewClaim, DrainFailurePublishesNoNewClaim,
    CheckpointAfterWorkerDrain, MetadataFailureDoesNotClaim,
    DataDurableRequiresVisible, VocabularyDurableRequiresVisible,
    VocabularyStableWhenRecovered, SingleTrieClaimSharesBoundary,
    ShardedClaimHasAuxiliaryMetadata, Active, DependencyEvidenceReady,
    SaveCheckpointOk

THEOREM SaveCheckpointFailPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, SaveCheckpointFail
    PROVE  Safety'
BY SMT, TrueIsBoolean, FalseIsBoolean
DEF ModelConstantsOK, Safety, TypeOK, ClaimRequiresDurableEvidence,
    RecoveredPrefixHasCheckpointEvidence, FailedSyncCannotClaimWithoutDurableData,
    ForceQuitPublishesNoNewClaim, DrainFailurePublishesNoNewClaim,
    CheckpointAfterWorkerDrain, MetadataFailureDoesNotClaim,
    DataDurableRequiresVisible, VocabularyDurableRequiresVisible,
    VocabularyStableWhenRecovered, SingleTrieClaimSharesBoundary,
    ShardedClaimHasAuxiliaryMetadata, Active, DependencyEvidenceReady,
    SaveCheckpointFail

THEOREM GracefulCancelCheckpointPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, GracefulCancelCheckpoint
    PROVE  Safety'
BY SMT, TrueIsBoolean, FalseIsBoolean
DEF ModelConstantsOK, Safety, TypeOK, ClaimRequiresDurableEvidence,
    RecoveredPrefixHasCheckpointEvidence, FailedSyncCannotClaimWithoutDurableData,
    ForceQuitPublishesNoNewClaim, DrainFailurePublishesNoNewClaim,
    CheckpointAfterWorkerDrain, MetadataFailureDoesNotClaim,
    DataDurableRequiresVisible, VocabularyDurableRequiresVisible,
    VocabularyStableWhenRecovered, SingleTrieClaimSharesBoundary,
    ShardedClaimHasAuxiliaryMetadata, Active, DependencyEvidenceReady,
    GracefulCancelCheckpoint

THEOREM ForceQuitAbortPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, ForceQuitAbort
    PROVE  Safety'
BY SMT, TrueIsBoolean, FalseIsBoolean
DEF ModelConstantsOK, Safety, TypeOK, ClaimRequiresDurableEvidence,
    RecoveredPrefixHasCheckpointEvidence, FailedSyncCannotClaimWithoutDurableData,
    ForceQuitPublishesNoNewClaim, DrainFailurePublishesNoNewClaim,
    CheckpointAfterWorkerDrain, MetadataFailureDoesNotClaim,
    DataDurableRequiresVisible, VocabularyDurableRequiresVisible,
    VocabularyStableWhenRecovered, SingleTrieClaimSharesBoundary,
    ShardedClaimHasAuxiliaryMetadata, Active, DependencyEvidenceReady,
    ForceQuitAbort

THEOREM DrainFailureAbortPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, DrainFailureAbort
    PROVE  Safety'
BY SMT, TrueIsBoolean, FalseIsBoolean
DEF ModelConstantsOK, Safety, TypeOK, ClaimRequiresDurableEvidence,
    RecoveredPrefixHasCheckpointEvidence, FailedSyncCannotClaimWithoutDurableData,
    ForceQuitPublishesNoNewClaim, DrainFailurePublishesNoNewClaim,
    CheckpointAfterWorkerDrain, MetadataFailureDoesNotClaim,
    DataDurableRequiresVisible, VocabularyDurableRequiresVisible,
    VocabularyStableWhenRecovered, SingleTrieClaimSharesBoundary,
    ShardedClaimHasAuxiliaryMetadata, Active, DependencyEvidenceReady,
    DrainFailureAbort

THEOREM CrashAndReopenPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, CrashAndReopen
    PROVE  Safety'
BY SMT, TrueIsBoolean, FalseIsBoolean
DEF ModelConstantsOK, Safety, TypeOK, ClaimRequiresDurableEvidence,
    RecoveredPrefixHasCheckpointEvidence, FailedSyncCannotClaimWithoutDurableData,
    ForceQuitPublishesNoNewClaim, DrainFailurePublishesNoNewClaim,
    CheckpointAfterWorkerDrain, MetadataFailureDoesNotClaim,
    DataDurableRequiresVisible, VocabularyDurableRequiresVisible,
    VocabularyStableWhenRecovered, SingleTrieClaimSharesBoundary,
    ShardedClaimHasAuxiliaryMetadata, Active, DependencyEvidenceReady,
    CrashAndReopen

THEOREM NextPreservesSafety ==
    ModelConstantsOK /\ Safety /\ Next => Safety'
BY CommitPrefixOkPreservesSafety, SyncDataOkPreservesSafety,
   SyncDataFailPreservesSafety, CheckpointVocabOkPreservesSafety,
   DrainWorkersPreservesSafety, SaveCheckpointOkPreservesSafety,
   SaveCheckpointFailPreservesSafety, GracefulCancelCheckpointPreservesSafety,
   ForceQuitAbortPreservesSafety, DrainFailureAbortPreservesSafety,
   CrashAndReopenPreservesSafety
DEF Next

THEOREM StutterPreservesSafety ==
    Safety /\ UNCHANGED vars => Safety'
BY SMT
DEF ModelConstantsOK, Safety, TypeOK, ClaimRequiresDurableEvidence,
    RecoveredPrefixHasCheckpointEvidence, FailedSyncCannotClaimWithoutDurableData,
    ForceQuitPublishesNoNewClaim, DrainFailurePublishesNoNewClaim,
    CheckpointAfterWorkerDrain, MetadataFailureDoesNotClaim,
    DataDurableRequiresVisible, VocabularyDurableRequiresVisible,
    VocabularyStableWhenRecovered, SingleTrieClaimSharesBoundary,
    ShardedClaimHasAuxiliaryMetadata, vars

THEOREM StepPreservesSafety ==
    ModelConstantsOK /\ Safety /\ [Next]_vars => Safety'
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
