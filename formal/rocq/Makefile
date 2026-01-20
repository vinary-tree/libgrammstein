# Makefile for Rocq/Coq formal verification proofs
#
# Usage:
#   make          - Build all proofs
#   make clean    - Remove generated files
#   make check    - Verify proofs compile without admitted lemmas
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

# Check for admitted lemmas
check: all
	@echo "Checking for admitted lemmas..."
	@if grep -r "Admitted" *.v 2>/dev/null | grep -v "^#" | grep -v "(*"; then \
		echo "ERROR: Found admitted lemmas!"; \
		exit 1; \
	else \
		echo "All proofs complete (no Admitted lemmas found)"; \
	fi
