# Makefile for Rocq/Coq formal verification proofs
#
# Usage:
#   make          - Build all proofs
#   make clean    - Remove generated files
#   make check    - Verify proofs compile without proof holes
#
# For memory-intensive proofs, use resource limits:
#   systemd-run --user --scope -p MemoryMax=16G -p CPUQuota=400% make

COQMAKEFILE := CoqMakefile

.PHONY: all clean check

all: $(COQMAKEFILE)
	$(MAKE) -f $(COQMAKEFILE)

$(COQMAKEFILE): _CoqProject
	coq_makefile -f _CoqProject -o $(COQMAKEFILE)

clean:
	@if [ -f $(COQMAKEFILE) ]; then $(MAKE) -f $(COQMAKEFILE) clean; fi
	rm -f $(COQMAKEFILE) $(COQMAKEFILE).conf
	rm -f *.vo *.vok *.vos *.glob .*.aux *.crashcoqide

# Check for incomplete or trusted proof holes
check: all
	@echo "Checking for proof holes and trusted declarations..."
	@if grep -nE '\b(Admitted|admit|Axiom|Parameter|Abort)\b' *.v 2>/dev/null | grep -v '^[^:]*:[[:space:]]*(\*' | grep -v '^Makefile:'; then \
		echo "ERROR: Found proof holes or trusted declarations!"; \
		exit 1; \
	else \
		echo "All proofs complete (no proof holes or trusted declarations found)"; \
	fi
