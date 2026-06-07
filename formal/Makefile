# Formal verification entrypoint for libgrammstein.

SHELL := /bin/sh

TLA_DIR := tla
ROCQ_DIR := rocq

TLC ?= tlc
APALACHE ?= apalache-mc
TLAPM ?= tlapm
CARGO ?= cargo
CARGO_MANIFEST ?= ../Cargo.toml
FIND ?= find
GIT ?= git
JQ ?= jq
RUSTFMT ?= rustfmt
TIMEOUT ?= timeout
XARGS ?= xargs
TLC_JAVA_OPTS ?= -XX:+UseParallelGC -Xmx768m
TLC_OPTS ?=
TLC_WORKERS ?= 1
TLC_STRESS_WORKERS ?= 1
TLA_MODEL_TIMEOUT ?= 60s
DEPENDENCY_TLA_MODEL_TIMEOUT ?= 120s
APALACHE_JVM_ARGS ?= -Xmx1536m
APALACHE_TIMEOUT ?= 60s
TLAPS_TIMEOUT ?= 120s
MIRI_TOOLCHAIN ?= nightly
MIRI_FLAGS ?= -Zmiri-strict-provenance -Zmiri-disable-isolation
RG ?= rg

TLC_METADIR ?= /tmp/libgrammstein-tlc
APALACHE_OUT ?= /tmp/libgrammstein-apalache
STRESS_TIMEOUT ?= 60s
LIBDICTENSTEIN_DIR ?= ../../libdictenstein
LIBLEVENSHTEIN_DIR ?= ../../liblevenshtein-rust

LIBDICTENSTEIN_IMPORTED_TLA_MODULES := StorageSyscallOutcome \
	AsyncWalGroupCommit PublicDurabilityPolicy SharedPersistentConcurrency \
	ConcurrentCheckpointPublication PersistentEndToEndTrace \
	PersistentTransactionIncrementRecovery

LIBDICTENSTEIN_MIRI_FORMAL_TESTS := \
	vocab_child_remove_transfers_box_ownership_once \
	vocab_insert_child_replaces_without_aliasing_old_box \
	vocab_clone_deep_copies_child_boxes \
	vocab_get_or_create_child_mutation_keeps_unique_raw_borrow \
	char_child_remove_transfers_box_ownership_once \
	char_insert_child_replaces_without_aliasing_old_box \
	char_clone_deep_copies_child_boxes \
	char_get_or_create_child_mutation_keeps_unique_raw_borrow \
	swizzled_pointer_raw_extraction_is_gated_by_in_memory_state \
	swizzled_pointer_losing_lazy_load_candidate_can_be_reclaimed_once

TLC_RUN = TLC_JAVA_OPTS="$(TLC_JAVA_OPTS)" TLC_OPTS="$(TLC_OPTS)" $(TIMEOUT) $(TLA_MODEL_TIMEOUT) $(TLC) -workers $(TLC_WORKERS)
TLC_STRESS_RUN = TLC_JAVA_OPTS="$(TLC_JAVA_OPTS)" TLC_OPTS="$(TLC_OPTS)" $(TIMEOUT) $(STRESS_TIMEOUT) $(TLC) -workers $(TLC_STRESS_WORKERS)
DEPENDENCY_TLC_RUN = TLC_JAVA_OPTS="$(TLC_JAVA_OPTS)" TLC_OPTS="$(TLC_OPTS)" $(TIMEOUT) $(DEPENDENCY_TLA_MODEL_TIMEOUT) $(TLC) -workers $(TLC_WORKERS)
APALACHE_RUN = JVM_ARGS="$(APALACHE_JVM_ARGS)" $(TIMEOUT) $(APALACHE_TIMEOUT) $(APALACHE)
TLAPS_RUN = $(TIMEOUT) $(TLAPS_TIMEOUT) $(TLAPM)

# ShardWriteTokenProofs.tla retired with the WriteToken mechanism (lock-free
# overlay migration); see the DEPRECATED banner in ShardWriteToken.tla. Its
# successor AsyncShardSyncProofs.tla remains in the gate below.
TLAPS_FILES := CheckpointStateMachineProofs.tla \
	AsyncShardSyncProofs.tla CronStateMachineProofs.tla WorkerShutdownProofs.tla \
	ImporterLifecycleProofs.tla PersistentStorageBridgeProofs.tla \
	QuerySemanticsBridgeProofs.tla

.PHONY: check complete complete-with-dependencies dependency-contracts strict-dependency-contracts dependency-imported-tlc dependency-io-uring dependency-miri-unsafe dependency-warning-hygiene rocq tlaps tla-safety tla-liveness apalache rust-alignment source-hygiene source-hygiene-fast source-hygiene-full local-warning-hygiene stress clean

check: source-hygiene-fast rocq tla-safety tla-liveness apalache rust-alignment

complete: check stress tlaps source-hygiene-full

complete-with-dependencies: dependency-contracts complete

strict-dependency-contracts: dependency-contracts dependency-imported-tlc dependency-io-uring dependency-miri-unsafe

dependency-contracts:
	cd $(LIBDICTENSTEIN_DIR) && scripts/verify-formal-correspondence.sh
	cd $(LIBLEVENSHTEIN_DIR) && scripts/verify-formal.sh trusted

dependency-imported-tlc:
	@set -e; for module in $(LIBDICTENSTEIN_IMPORTED_TLA_MODULES); do \
		echo "Checking libdictenstein $$module with TLC"; \
		(cd $(LIBDICTENSTEIN_DIR)/formal-verification/tla+ && \
			$(DEPENDENCY_TLC_RUN) \
				-metadir $(TLC_METADIR)/libdictenstein-$$module \
				-config $$module.cfg $$module.tla); \
	done

dependency-io-uring:
	cd $(LIBDICTENSTEIN_DIR) && $(CARGO) test --features "persistent-artrie io-uring-backend" --lib io_uring_completion_
	cd $(LIBDICTENSTEIN_DIR) && $(CARGO) test --features "persistent-artrie io-uring-backend" --test persistent_artrie_storage_correspondence

dependency-warning-hygiene:
	@command -v $(JQ) >/dev/null 2>&1 || { echo "$(JQ) is required for dependency warning hygiene"; exit 127; }
	@tmp="$$(mktemp)"; \
	if ! $(CARGO) check --manifest-path $(CARGO_MANIFEST) --all-targets --all-features --message-format=json > "$$tmp"; then \
		rm -f "$$tmp"; \
		exit 1; \
	fi; \
	dependency_warnings="$$( \
		$(JQ) -r 'select(.reason == "compiler-message" and .message.level == "warning" and ((.package_id | contains("/libgrammstein#")) | not)) | .package_id' "$$tmp" \
		| sort | uniq -c \
	)"; \
	if [ -n "$$dependency_warnings" ]; then \
		echo "ERROR: dependency warnings remain:"; \
		echo "$$dependency_warnings"; \
		rm -f "$$tmp"; \
		exit 1; \
	fi; \
	rm -f "$$tmp"

dependency-miri-unsafe:
	@if ! $(CARGO) +$(MIRI_TOOLCHAIN) miri --version >/dev/null 2>&1; then \
		echo "cargo +$(MIRI_TOOLCHAIN) miri is not available; install the miri component for that toolchain."; \
		exit 127; \
	fi
	@set -e; export MIRIFLAGS="$(MIRI_FLAGS) $${MIRIFLAGS:-}"; \
	for test_name in $(LIBDICTENSTEIN_MIRI_FORMAL_TESTS); do \
		echo "Checking libdictenstein Miri unsafe-boundary test $$test_name"; \
		(cd $(LIBDICTENSTEIN_DIR) && \
			$(CARGO) +$(MIRI_TOOLCHAIN) miri test \
				--features persistent-artrie \
				--test persistent_artrie_formal_correspondence \
				$$test_name); \
	done
	@export MIRIFLAGS="$(MIRI_FLAGS) $${MIRIFLAGS:-}"; \
	cd $(LIBDICTENSTEIN_DIR) && $(CARGO) +$(MIRI_TOOLCHAIN) miri test \
		--features persistent-artrie --lib persistent_artrie_core::swizzled_ptr::tests
	@export MIRIFLAGS="$(MIRI_FLAGS) $${MIRIFLAGS:-}"; \
	cd $(LIBDICTENSTEIN_DIR) && $(CARGO) +$(MIRI_TOOLCHAIN) miri test \
		--features persistent-artrie --lib \
		persistent_vocab_artrie::tests::vocab_leaf_eviction_invalidates_node_map_entry_before_drop
	@export MIRIFLAGS="$(MIRI_FLAGS) $${MIRIFLAGS:-}"; \
	cd $(LIBDICTENSTEIN_DIR) && $(CARGO) +$(MIRI_TOOLCHAIN) miri test \
		--features persistent-artrie --lib \
		persistent_vocab_artrie::tests::vocab_leaf_eviction_keeps_sibling_queries_on_live_nodes
	@export MIRIFLAGS="$(MIRI_FLAGS) $${MIRIFLAGS:-}"; \
	cd $(LIBDICTENSTEIN_DIR) && $(CARGO) +$(MIRI_TOOLCHAIN) miri test \
		--features persistent-artrie --lib \
		persistent_vocab_artrie::tests::vocab_heap_node_map_parent_chain_tracks_live_nodes
	@export MIRIFLAGS="$(MIRI_FLAGS) $${MIRIFLAGS:-}"; \
	cd $(LIBDICTENSTEIN_DIR) && $(CARGO) +$(MIRI_TOOLCHAIN) miri test \
		--features persistent-artrie --lib \
		persistent_artrie_core::buffer_manager::tests::fixed_buffer_registration_covers_write_guard_mutation_and_flush

rocq:
	$(MAKE) -C $(ROCQ_DIR) check

tlaps:
	@if ! command -v $(TLAPM) >/dev/null 2>&1; then \
		echo "TLAPS proof manager '$(TLAPM)' was not found. Put tlapm on PATH or pass TLAPM=/path/to/tlapm."; \
		exit 127; \
	fi
	@tlaps_lib="$$(PATH=$$(dirname "$$(command -v $(TLAPM))"):$$PATH $(TLAPM) --where)"; \
	set -e; for proof in $(TLAPS_FILES); do \
		echo "Checking $$proof with TLAPS"; \
		(cd $(TLA_DIR) && PATH="$$tlaps_lib/bin:$$PATH" $(TLAPS_RUN) $$proof); \
	done

tla-safety:
	cd $(TLA_DIR) && $(TLC_RUN) -metadir $(TLC_METADIR)/checkpoint-safety -config MC_CheckpointStateMachine.cfg CheckpointStateMachine.tla
	cd $(TLA_DIR) && $(TLC_RUN) -metadir $(TLC_METADIR)/async-safety -config AsyncShardSync.cfg AsyncShardSync.tla
	cd $(TLA_DIR) && $(TLC_RUN) -metadir $(TLC_METADIR)/cron-safety -config CronStateMachine.cfg CronStateMachine.tla
	cd $(TLA_DIR) && $(TLC_RUN) -metadir $(TLC_METADIR)/worker-safety -config MC_WorkerShutdown_Safety.cfg WorkerShutdown.tla
	cd $(TLA_DIR) && $(TLC_RUN) -metadir $(TLC_METADIR)/importer-lifecycle-safety -config ImporterLifecycle.cfg ImporterLifecycle.tla
	cd $(TLA_DIR) && $(TLC_RUN) -metadir $(TLC_METADIR)/persistent-storage-bridge-safety -config PersistentStorageBridge.cfg PersistentStorageBridge.tla
	cd $(TLA_DIR) && $(TLC_RUN) -metadir $(TLC_METADIR)/persistent-storage-bridge-sharded-safety -config PersistentStorageBridge_Sharded.cfg PersistentStorageBridge.tla
	cd $(TLA_DIR) && $(TLC_RUN) -metadir $(TLC_METADIR)/query-semantics-bridge-safety -config QuerySemanticsBridge.cfg QuerySemanticsBridge.tla
	cd $(TLA_DIR) && $(TLC_RUN) -metadir $(TLC_METADIR)/query-semantics-bridge-no-metadata-safety -config QuerySemanticsBridge_NoMetadata.cfg QuerySemanticsBridge.tla

tla-liveness:
	cd $(TLA_DIR) && $(TLC_RUN) -metadir $(TLC_METADIR)/checkpoint-liveness -config MC_CheckpointStateMachine_Liveness.cfg CheckpointStateMachine.tla
	cd $(TLA_DIR) && $(TLC_RUN) -metadir $(TLC_METADIR)/async-liveness -config AsyncShardSync_Liveness.cfg AsyncShardSync.tla
	cd $(TLA_DIR) && $(TLC_RUN) -metadir $(TLC_METADIR)/cron-liveness -config CronStateMachine_Liveness.cfg CronStateMachine.tla
	cd $(TLA_DIR) && $(TLC_RUN) -metadir $(TLC_METADIR)/worker-liveness -config MC_WorkerShutdown_NoSymmetry.cfg WorkerShutdown.tla
	cd $(TLA_DIR) && $(TLC_RUN) -metadir $(TLC_METADIR)/importer-lifecycle-liveness -config ImporterLifecycle_Liveness.cfg ImporterLifecycle.tla

apalache:
	cd $(TLA_DIR) && $(APALACHE_RUN) --features=no-rows --out-dir=$(APALACHE_OUT)/checkpoint typecheck CheckpointStateMachine.tla
	cd $(TLA_DIR) && $(APALACHE_RUN) --features=no-rows --out-dir=$(APALACHE_OUT)/async typecheck AsyncShardSync.tla
	cd $(TLA_DIR) && $(APALACHE_RUN) --features=no-rows --out-dir=$(APALACHE_OUT)/cron typecheck CronStateMachine.tla
	cd $(TLA_DIR) && $(APALACHE_RUN) --features=no-rows --out-dir=$(APALACHE_OUT)/worker typecheck WorkerShutdown.tla
	cd $(TLA_DIR) && $(APALACHE_RUN) --features=no-rows --out-dir=$(APALACHE_OUT)/importer-lifecycle typecheck ImporterLifecycle.tla
	cd $(TLA_DIR) && $(APALACHE_RUN) --features=no-rows --out-dir=$(APALACHE_OUT)/persistent-storage-bridge typecheck PersistentStorageBridge.tla
	cd $(TLA_DIR) && $(APALACHE_RUN) --features=no-rows --out-dir=$(APALACHE_OUT)/query-semantics-bridge typecheck QuerySemanticsBridge.tla

rust-alignment:
	$(CARGO) test --features google-books sources::google_books::sharding::shard::tests::test_sync_tracked_marks_dirty
	$(CARGO) test --features google-books sources::google_books::sharding::mkn
	$(CARGO) test --features google-books sources::google_books::importer::tests::shutdown_checkpoint_safety
	$(CARGO) test --features google-books persistent_storage_bridge
	$(CARGO) test --features google-books vocabulary_query
	$(CARGO) test metadata_filtering_zipper
	$(CARGO) test --features google-books sources::google_books::state_machine::tests
	$(CARGO) test util::cron::tests
	$(CARGO) test --features loom-tests --test loom_formal_alignment

source-hygiene: source-hygiene-fast source-hygiene-full

source-hygiene-fast:
	$(FIND) ../src ../tests ../benches -name '*.rs' -print0 | $(XARGS) -0 $(RUSTFMT) --edition 2021 --check
	$(GIT) -C .. diff --check
	@command -v $(RG) >/dev/null 2>&1 || { echo "$(RG) is required for source marker hygiene"; exit 127; }
	@! $(RG) -n '#\[[i]gnore|T[O]DO|F[I]XME|H[A]CK|not yet implement[e]d|unimplement[e]d!\(|t[o]do!\(' \
		../src ../tests ../benches ../docs ../Cargo.toml ../README.md README.md TASKS.md tla rocq dependencies \
		--glob '!target/**' --glob '!.tlacache/**' --glob '!Cargo.lock'
	@! $(RG) -n 'placehold[e]r|st[u]b' \
		../src ../tests ../Cargo.toml ../README.md README.md TASKS.md tla rocq dependencies \
		--glob '!target/**' --glob '!.tlacache/**' --glob '!Cargo.lock'

source-hygiene-full: local-warning-hygiene
	$(CARGO) test --manifest-path $(CARGO_MANIFEST) --lib --all-features -- --test-threads=1

local-warning-hygiene:
	@command -v $(JQ) >/dev/null 2>&1 || { echo "$(JQ) is required for local warning hygiene"; exit 127; }
	@tmp="$$(mktemp)"; \
	if ! $(CARGO) check --manifest-path $(CARGO_MANIFEST) --all-targets --all-features --message-format=json > "$$tmp"; then \
		rm -f "$$tmp"; \
		exit 1; \
	fi; \
	if $(JQ) -e 'select(.reason == "compiler-message" and .message.level == "warning" and (.package_id | contains("/libgrammstein#")))' "$$tmp" >/dev/null; then \
		echo "ERROR: local libgrammstein warnings found"; \
		$(JQ) -r 'select(.reason == "compiler-message" and .message.level == "warning" and (.package_id | contains("/libgrammstein#"))) | (.message.rendered // .message.message)' "$$tmp"; \
		rm -f "$$tmp"; \
		exit 1; \
	fi; \
	dependency_warnings="$$( \
		$(JQ) -r 'select(.reason == "compiler-message" and .message.level == "warning" and ((.package_id | contains("/libgrammstein#")) | not)) | .package_id' "$$tmp" \
		| sort | uniq -c \
	)"; \
	if [ -n "$$dependency_warnings" ]; then \
		echo "Dependency warnings observed (non-fatal for libgrammstein local gate):"; \
		echo "$$dependency_warnings"; \
	fi; \
	rm -f "$$tmp"

stress:
	cd $(TLA_DIR) && $(TLC_STRESS_RUN) -metadir $(TLC_METADIR)/checkpoint-stress -config MC_CheckpointStateMachine_Stress.cfg CheckpointStateMachine.tla
	cd $(TLA_DIR) && $(TLC_STRESS_RUN) -metadir $(TLC_METADIR)/async-stress -config AsyncShardSync_Stress.cfg AsyncShardSync.tla
	cd $(TLA_DIR) && $(TLC_STRESS_RUN) -metadir $(TLC_METADIR)/cron-stress -config CronStateMachine_Stress.cfg CronStateMachine.tla
	cd $(TLA_DIR) && $(TLC_STRESS_RUN) -metadir $(TLC_METADIR)/worker-stress -config MC_WorkerShutdown_Stress.cfg WorkerShutdown.tla
	cd $(TLA_DIR) && $(TLC_STRESS_RUN) -metadir $(TLC_METADIR)/importer-lifecycle-stress -config ImporterLifecycle_Stress.cfg ImporterLifecycle.tla
	cd $(TLA_DIR) && $(TLC_STRESS_RUN) -metadir $(TLC_METADIR)/persistent-storage-bridge-stress -config PersistentStorageBridge_Stress.cfg PersistentStorageBridge.tla
	cd $(TLA_DIR) && $(TLC_STRESS_RUN) -metadir $(TLC_METADIR)/query-semantics-bridge-stress -config QuerySemanticsBridge_Stress.cfg QuerySemanticsBridge.tla

clean:
	$(MAKE) -C $(ROCQ_DIR) clean
	rm -rf $(TLA_DIR)/.tlacache
