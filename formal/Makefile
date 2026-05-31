# Formal verification entrypoint for libgrammstein.

SHELL := /bin/sh

TLA_DIR := tla
ROCQ_DIR := rocq

TLC ?= tlc
APALACHE ?= apalache-mc
TLAPM ?= tlapm
CARGO ?= cargo
TIMEOUT ?= timeout

TLC_METADIR ?= /tmp/libgrammstein-tlc
APALACHE_OUT ?= /tmp/libgrammstein-apalache
STRESS_TIMEOUT ?= 180s
TLAPS_FILES := ShardWriteTokenProofs.tla CheckpointStateMachineProofs.tla

.PHONY: check complete rocq tlaps tla-safety tla-liveness apalache rust-alignment stress clean

check: rocq tla-safety tla-liveness apalache rust-alignment

complete: check tlaps

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
		(cd $(TLA_DIR) && PATH="$$tlaps_lib/bin:$$PATH" $(TLAPM) $$proof); \
	done

tla-safety:
	cd $(TLA_DIR) && $(TLC) -metadir $(TLC_METADIR)/shard-safety -config MC_ShardWriteToken.cfg ShardWriteToken.tla
	cd $(TLA_DIR) && $(TLC) -metadir $(TLC_METADIR)/checkpoint-safety -config MC_CheckpointStateMachine.cfg CheckpointStateMachine.tla
	cd $(TLA_DIR) && $(TLC) -metadir $(TLC_METADIR)/async-safety -config AsyncShardSync.cfg AsyncShardSync.tla
	cd $(TLA_DIR) && $(TLC) -metadir $(TLC_METADIR)/cron-safety -config CronStateMachine.cfg CronStateMachine.tla
	cd $(TLA_DIR) && $(TLC) -metadir $(TLC_METADIR)/worker-safety -config MC_WorkerShutdown_NoSymmetry.cfg WorkerShutdown.tla

tla-liveness:
	cd $(TLA_DIR) && $(TLC) -metadir $(TLC_METADIR)/shard-liveness -config MC_ShardWriteToken_Liveness.cfg ShardWriteToken.tla
	cd $(TLA_DIR) && $(TLC) -metadir $(TLC_METADIR)/checkpoint-liveness -config MC_CheckpointStateMachine_Liveness.cfg CheckpointStateMachine.tla
	cd $(TLA_DIR) && $(TLC) -metadir $(TLC_METADIR)/async-liveness -config AsyncShardSync_Liveness.cfg AsyncShardSync.tla
	cd $(TLA_DIR) && $(TLC) -metadir $(TLC_METADIR)/cron-liveness -config CronStateMachine_Liveness.cfg CronStateMachine.tla

apalache:
	cd $(TLA_DIR) && $(APALACHE) --features=no-rows --out-dir=$(APALACHE_OUT)/shard typecheck ShardWriteToken.tla
	cd $(TLA_DIR) && $(APALACHE) --features=no-rows --out-dir=$(APALACHE_OUT)/checkpoint typecheck CheckpointStateMachine.tla
	cd $(TLA_DIR) && $(APALACHE) --features=no-rows --out-dir=$(APALACHE_OUT)/async typecheck AsyncShardSync.tla
	cd $(TLA_DIR) && $(APALACHE) --features=no-rows --out-dir=$(APALACHE_OUT)/cron typecheck CronStateMachine.tla
	cd $(TLA_DIR) && $(APALACHE) --features=no-rows --out-dir=$(APALACHE_OUT)/worker typecheck WorkerShutdown.tla

rust-alignment:
	$(CARGO) test --features google-books sources::google_books::sharding::shard::tests::test_write_token_generation_does_not_wrap
	$(CARGO) test --features google-books sources::google_books::sharding::mkn

stress:
	cd $(TLA_DIR) && $(TIMEOUT) $(STRESS_TIMEOUT) $(TLC) -metadir $(TLC_METADIR)/shard-stress -config MC_ShardWriteToken_Stress.cfg ShardWriteToken.tla || status=$$?; if [ "$${status:-0}" = "124" ]; then echo "ShardWriteToken stress timed out after $(STRESS_TIMEOUT)"; elif [ "$${status:-0}" != "0" ]; then exit "$$status"; fi
	cd $(TLA_DIR) && $(TIMEOUT) $(STRESS_TIMEOUT) $(TLC) -metadir $(TLC_METADIR)/checkpoint-stress -config MC_CheckpointStateMachine_Stress.cfg CheckpointStateMachine.tla || status=$$?; if [ "$${status:-0}" = "124" ]; then echo "CheckpointStateMachine stress timed out after $(STRESS_TIMEOUT)"; elif [ "$${status:-0}" != "0" ]; then exit "$$status"; fi
	cd $(TLA_DIR) && $(TIMEOUT) $(STRESS_TIMEOUT) $(TLC) -metadir $(TLC_METADIR)/async-stress -config AsyncShardSync_Stress.cfg AsyncShardSync.tla || status=$$?; if [ "$${status:-0}" = "124" ]; then echo "AsyncShardSync stress timed out after $(STRESS_TIMEOUT)"; elif [ "$${status:-0}" != "0" ]; then exit "$$status"; fi
	cd $(TLA_DIR) && $(TIMEOUT) $(STRESS_TIMEOUT) $(TLC) -metadir $(TLC_METADIR)/cron-stress -config CronStateMachine_Stress.cfg CronStateMachine.tla || status=$$?; if [ "$${status:-0}" = "124" ]; then echo "CronStateMachine stress timed out after $(STRESS_TIMEOUT)"; elif [ "$${status:-0}" != "0" ]; then exit "$$status"; fi
	cd $(TLA_DIR) && $(TIMEOUT) $(STRESS_TIMEOUT) $(TLC) -metadir $(TLC_METADIR)/worker-stress -config MC_WorkerShutdown_Stress.cfg WorkerShutdown.tla || status=$$?; if [ "$${status:-0}" = "124" ]; then echo "WorkerShutdown stress timed out after $(STRESS_TIMEOUT)"; elif [ "$${status:-0}" != "0" ]; then exit "$$status"; fi

clean:
	$(MAKE) -C $(ROCQ_DIR) clean
	rm -rf $(TLA_DIR)/.tlacache
