# Convenience targets for running the K8s autoscaling agent demo.
# Run from this directory: `make demo`, `make reset`, etc.

SCRIPTS := ./scripts

.PHONY: demo reset clean infra eval scoreboard triage-acc recall

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

# Run the eval against the orchestrator (port-forwarded to localhost:8000 by
# default; override with ORCHESTRATOR_URL) and print the scoreboard row for the
# active seat configuration: quality, Bedrock $/query, p50 latency. REPEAT=1 is
# a quick smoke check; the default 4 is what the noise level needs.
REPEAT ?= 4
eval:
	python3 eval/run_eval.py --repeat $(REPEAT)

# Deterministic per-seat quality, no judge. triage-acc scores whoever holds the
# TRIAGE seat (set TRIAGE=llm|classifier) on the 43-query held-out set;
# recall scores retrieval three ways (bedrock, slemify, slemify+rerank) on the
# recall set. Both need the relevant services port-forwarded; see the scripts.
triage-acc:
	python3 eval/triage_acc.py

recall:
	python3 eval/recall.py $(if $(MODE),--mode $(MODE),)

# Print the live scoreboard from /stats: per seat configuration, queries seen,
# Bedrock $/query, p50/p95 latency, escalation rate, per-step p50.
scoreboard:
	@curl -s $${ORCHESTRATOR_URL:-http://localhost:8000}/stats | python3 -c \
	  'import json,sys; d=json.load(sys.stdin); print(json.dumps({"seats": d["seats"], "slm_first_pass_rate": d["slm_first_pass_rate"], "scoreboard": d["scoreboard"]}, indent=2))'
