----------------------- MODULE QuerySemanticsBridgeProofs ----------------------
(*
 * TLAPS safety proof for QuerySemanticsBridge.tla.
 *)

EXTENDS QuerySemanticsBridge, TLAPS

THEOREM InitImpliesSafety ==
    ModelConstantsOK /\ Init => Safety
BY SMT
DEF ModelConstantsOK, Init, Safety, TypeOK, MetadataHiddenFromRoot,
    VisibleDataPreserved, ValueYieldingQueryCorrect, NoMetadataValueYielded,
    OovReadsDoNotAllocateVocabulary, AggregatedLookupCorrect, DataTerms,
    StorageTerms, VocabularyTerms, LookupTerms, AllTerms, Values,
    VocabularySize, TermValue, BackendRootEdges, WrappedRootEdges,
    ResultRecord, MetadataTerm,
    FirstTerm, SecondTerm, OovTerm, FirstValue, SecondValue, MetadataValue,
    NoValue

THEOREM RefreshRootViewPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, RefreshRootView
    PROVE  Safety'
BY SMT
DEF ModelConstantsOK, Safety, TypeOK, MetadataHiddenFromRoot,
    VisibleDataPreserved, ValueYieldingQueryCorrect, NoMetadataValueYielded,
    OovReadsDoNotAllocateVocabulary, AggregatedLookupCorrect, DataTerms,
    StorageTerms, VocabularyTerms, LookupTerms, AllTerms, Values,
    VocabularySize, TermValue, BackendRootEdges, WrappedRootEdges,
    ResultRecord, MetadataTerm,
    FirstTerm, SecondTerm, OovTerm, FirstValue, SecondValue, MetadataValue,
    NoValue, RefreshRootView

THEOREM YieldVisibleValuesPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, YieldVisibleValues
    PROVE  Safety'
BY SMT
DEF ModelConstantsOK, Safety, TypeOK, MetadataHiddenFromRoot,
    VisibleDataPreserved, ValueYieldingQueryCorrect, NoMetadataValueYielded,
    OovReadsDoNotAllocateVocabulary, AggregatedLookupCorrect, DataTerms,
    StorageTerms, VocabularyTerms, LookupTerms, AllTerms, Values,
    VocabularySize, TermValue, BackendRootEdges, WrappedRootEdges,
    ResultRecord, ExpectedValueResults, MetadataTerm, FirstTerm, SecondTerm,
    OovTerm, FirstValue, SecondValue, MetadataValue, NoValue,
    YieldVisibleValues

THEOREM ReadOovWithoutAllocatingPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, ReadOovWithoutAllocating
    PROVE  Safety'
BY SMT
DEF ModelConstantsOK, Safety, TypeOK, MetadataHiddenFromRoot,
    VisibleDataPreserved, ValueYieldingQueryCorrect, NoMetadataValueYielded,
    OovReadsDoNotAllocateVocabulary, AggregatedLookupCorrect, DataTerms,
    StorageTerms, VocabularyTerms, LookupTerms, AllTerms, Values,
    VocabularySize, TermValue, BackendRootEdges, WrappedRootEdges,
    ResultRecord, MetadataTerm,
    FirstTerm, SecondTerm, OovTerm, FirstValue, SecondValue, MetadataValue,
    NoValue, ReadOovWithoutAllocating

THEOREM LookupExistingTermPreservesSafety ==
    ASSUME ModelConstantsOK, Safety, LookupExistingTerm
    PROVE  Safety'
BY SMT
DEF ModelConstantsOK, Safety, TypeOK, MetadataHiddenFromRoot,
    VisibleDataPreserved, ValueYieldingQueryCorrect, NoMetadataValueYielded,
    OovReadsDoNotAllocateVocabulary, AggregatedLookupCorrect, DataTerms,
    StorageTerms, VocabularyTerms, LookupTerms, AllTerms, Values,
    VocabularySize, TermValue, BackendRootEdges, WrappedRootEdges,
    ResultRecord, MetadataTerm,
    FirstTerm, SecondTerm, OovTerm, FirstValue, SecondValue, MetadataValue,
    NoValue, LookupExistingTerm

THEOREM NextPreservesSafety ==
    ModelConstantsOK /\ Safety /\ Next => Safety'
BY RefreshRootViewPreservesSafety, YieldVisibleValuesPreservesSafety,
   ReadOovWithoutAllocatingPreservesSafety, LookupExistingTermPreservesSafety
DEF Next

THEOREM StutterPreservesSafety ==
    Safety /\ UNCHANGED vars => Safety'
BY SMT
DEF Safety, TypeOK, MetadataHiddenFromRoot, VisibleDataPreserved,
    ValueYieldingQueryCorrect, NoMetadataValueYielded,
    OovReadsDoNotAllocateVocabulary, AggregatedLookupCorrect, vars

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
