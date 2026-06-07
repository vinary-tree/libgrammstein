-------------------------- MODULE QuerySemanticsBridge -------------------------
(*
 * Bridge model for libgrammstein dictionary/query wrapper semantics.
 *
 * Target source:
 *   src/ngram/vocabulary_indexed.rs
 *   src/ngram/metadata_filtering_zipper.rs
 *   src/dictionary/vocabulary_backed.rs
 *   src/aggregated/mod.rs
 *
 * Imported dependency evidence:
 *   liblevenshtein trusted dictionary/search proof islands stay authoritative
 *   for edit-distance and MSM query semantics. This local model verifies only
 *   the libgrammstein adapter obligations around vocabulary encoding, metadata
 *   hiding, value-yielding traversal, and sharded lookup routing.
 *)

EXTENDS Naturals, TLC

CONSTANT
    \* @type: Bool;
    MetadataPresent

ModelConstantsOK ==
    MetadataPresent \in BOOLEAN

MetadataTerm == 0
FirstTerm == 1
SecondTerm == 2
OovTerm == 3

DataTerms == {FirstTerm, SecondTerm}
StorageTerms == DataTerms
VocabularyTerms == DataTerms
LookupTerms == DataTerms \cup {OovTerm}
AllTerms == DataTerms \cup {MetadataTerm, OovTerm}

FirstValue == 10
SecondValue == 20
MetadataValue == 99
NoValue == 0

Values == {FirstValue, SecondValue, MetadataValue, NoValue}
VocabularySize == 2

TermValue(t) ==
    CASE t = MetadataTerm -> MetadataValue
      [] t = FirstTerm -> FirstValue
      [] t = SecondTerm -> SecondValue
      [] OTHER -> NoValue

BackendRootEdges ==
    DataTerms \cup (IF MetadataPresent THEN {MetadataTerm} ELSE {})

WrappedRootEdges ==
    BackendRootEdges \ {MetadataTerm}

ResultRecord ==
    [term : AllTerms, value : Values]

ExpectedValueResults ==
    {[term |-> t, value |-> TermValue(t)] : t \in WrappedRootEdges \cap StorageTerms}

VARIABLES
    \* @type: Set(Int);
    root_view,
    \* @type: Set([term: Int, value: Int]);
    query_results,
    \* @type: Int;
    vocab_size,
    \* @type: Int;
    lookup_term,
    \* @type: Int;
    lookup_value

vars == <<root_view, query_results, vocab_size, lookup_term, lookup_value>>

Init ==
    /\ root_view = WrappedRootEdges
    /\ query_results = {}
    /\ vocab_size = VocabularySize
    /\ lookup_term = OovTerm
    /\ lookup_value = NoValue

RefreshRootView ==
    /\ root_view' = WrappedRootEdges
    /\ UNCHANGED <<query_results, vocab_size, lookup_term, lookup_value>>

YieldVisibleValues ==
    /\ query_results' = ExpectedValueResults
    /\ UNCHANGED <<root_view, vocab_size, lookup_term, lookup_value>>

ReadOovWithoutAllocating ==
    /\ lookup_term' = OovTerm
    /\ lookup_value' = NoValue
    /\ vocab_size' = vocab_size
    /\ UNCHANGED <<root_view, query_results>>

LookupExistingTerm ==
    \E t \in DataTerms :
        /\ lookup_term' = t
        /\ lookup_value' = TermValue(t)
        /\ vocab_size' = vocab_size
        /\ UNCHANGED <<root_view, query_results>>

Next ==
    \/ RefreshRootView
    \/ YieldVisibleValues
    \/ ReadOovWithoutAllocating
    \/ LookupExistingTerm

Spec ==
    Init /\ [][Next]_vars

TypeOK ==
    /\ root_view \subseteq AllTerms
    /\ query_results \subseteq ResultRecord
    /\ vocab_size \in Nat
    /\ lookup_term \in LookupTerms
    /\ lookup_value \in Values

MetadataHiddenFromRoot ==
    MetadataTerm \notin root_view

VisibleDataPreserved ==
    StorageTerms \subseteq root_view

ValueYieldingQueryCorrect ==
    \A r \in query_results :
        /\ r.term \in StorageTerms
        /\ r.value = TermValue(r.term)

NoMetadataValueYielded ==
    \A r \in query_results : r.value # MetadataValue

OovReadsDoNotAllocateVocabulary ==
    vocab_size = VocabularySize

AggregatedLookupCorrect ==
    /\ lookup_term \in DataTerms => lookup_value = TermValue(lookup_term)
    /\ lookup_term = OovTerm => lookup_value = NoValue

Safety ==
    /\ TypeOK
    /\ MetadataHiddenFromRoot
    /\ VisibleDataPreserved
    /\ ValueYieldingQueryCorrect
    /\ NoMetadataValueYielded
    /\ OovReadsDoNotAllocateVocabulary
    /\ AggregatedLookupCorrect

THEOREM ModelConstantsOK => (Spec => []Safety)

=============================================================================
