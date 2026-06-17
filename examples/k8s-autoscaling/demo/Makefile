# Convenience targets for running the K8s autoscaling agent demo.
# Run from this directory: `make demo`, `make reset`, etc.

SCRIPTS := ./scripts

.PHONY: demo reset clean infra

# One command to start a demo session: reset the scenarios to their broken
# state, then port-forward the orchestrator, open the browser, and launch the
# tmux log dashboard.
demo:
	$(SCRIPTS)/reset-demo.sh
	$(SCRIPTS)/demo-terminal.sh

# Re-break both remediation scenarios so you can demo the fixes again.
reset:
	$(SCRIPTS)/reset-demo.sh

# Remove the demo scenario resources entirely.
clean:
	$(SCRIPTS)/reset-demo.sh --clean

# Live infra view (CPU-only nodes + demo pods) in a separate terminal.
infra:
	$(SCRIPTS)/show-infra.sh
