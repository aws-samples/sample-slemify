# Convenience targets for running the K8s autoscaling agent demo.
# Run from this directory: `make demo`, `make reset`, etc.

SCRIPTS := ./scripts

.PHONY: demo reset clean infra eval eval-incluster scoreboard scoreboard-save triage-acc recall index-tuned check-infra swap forward forward-check

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
# active step configuration: quality, Bedrock $/query, p50 latency. REPEAT=1 is
# a quick smoke check; the default 4 is what the noise level needs.
REPEAT ?= 4
eval:
	python3 eval/run_eval.py --repeat $(REPEAT)

# Same eval, run as a Job next to the orchestrator. Use this once the analyst
# step is on CPU: a query then takes 25 to 75 seconds and a local port-forward
# does not reliably survive the run. The scorecard is copied back to
# eval/results/ when the Job finishes.
eval-incluster:
	bash scripts/run-eval-incluster.sh --repeat $(REPEAT)

# Deterministic per-step quality, no judge. triage-acc scores whichever model runs the
# TRIAGE step (set TRIAGE=llm|classifier) on the 43-query held-out set;
# recall scores retrieval three ways (bedrock, slemify, slemify+rerank) on the
# recall set. Both need the relevant services port-forwarded; see the scripts.
triage-acc:
	python3 eval/triage_acc.py

recall:
	python3 eval/recall.py $(if $(MODE),--mode $(MODE),)

# Build the tuned knowledge index against the retriever you just fine-tuned.
# The index stores vectors from the tuned encoder, so it can only be built once
# that encoder is serving (module 2). `make recall` and EMBED=slemify read it.
index-tuned:
	bash scripts/index-tuned.sh

# Check the two module-3 assets that provisioning finishes after the event is
# marked ready (S3 CSI addon, analyst GGUF) and name the repair if one is
# missing. REPAIR=1 starts the idempotent provisioning build to fix it.
check-infra:
	REPAIR_FLAG=$(REPAIR) bash scripts/check-infra.sh

# Print the live scoreboard from /stats: per step configuration, queries seen,
# Bedrock $/query, p50/p95 latency, escalation rate, per-step p50.
scoreboard:
	@curl -s $${ORCHESTRATOR_URL:-http://localhost:8000}/stats | python3 -c \
	  'import json,sys; d=json.load(sys.stdin); print(json.dumps({"steps": d["steps"], "slm_first_pass_rate": d["slm_first_pass_rate"], "scoreboard": d["scoreboard"]}, indent=2))'

# Swap the model in one or more steps and re-establish the port-forward.
# Usage: make swap TRIAGE=classifier | make swap EMBED=slemify RERANK=on
#        make swap ANALYST=slm | make swap GATE=llm
# Sets the given step variables on the orchestrator Deployment, waits for the
# rollout, restarts the local 8000 port-forward the rollout killed, and prints
# the resulting steps block.
SWAP_VARS := $(foreach v,TRIAGE EMBED RERANK ANALYST GATE,$(if $($(v)),$(v)=$($(v))))
swap:
	@test -n "$(SWAP_VARS)" || { echo "usage: make swap TRIAGE=classifier|llm|off EMBED=bedrock|slemify RERANK=on|off ANALYST=llm|slm GATE=llm|off"; exit 1; }
	kubectl set env deployment/k8s-autoscaling-orchestrator -n slemify $(SWAP_VARS)
	kubectl rollout status deployment/k8s-autoscaling-orchestrator -n slemify
	@# The rollout killed the 8000 forward; re-establish it by delegating to the
	@# `forward` target, which detaches the forward (setsid+nohup) and kills any
	@# stale one precisely. Done in a sub-make so no stray job-control "Terminated"
	@# line from a backgrounded child reaches this recipe's output.
	@$(MAKE) --no-print-directory forward >/dev/null 2>&1 || true
	@sleep 1; curl -s localhost:8000/stats | python3 -c 'import sys,json; print(json.load(sys.stdin)["steps"])'

# (Re)start every port-forward the lab needs, idempotently. Always safe to run,
# any number of times: for each service it kills any stale forward on that local
# port before starting a fresh one, so you never end up with two bound to the
# same port. It covers the two long-lived forwards (orchestrator 8000, OpenSearch
# 9200) AND each per-model forward (triage 8081, analyst 8082, retriever 8083,
# reranker 8084). The long-lived two are always forwarded; a per-model forward is
# started only once that model has a ready pod behind its Service, so it is
# correct at every stage of the lab and does not forward a model you have not
# deployed yet (or one scaled to zero, like the reranker before module 2). If
# anything cannot connect, run this; it is the one command for "a forward died,"
# whichever forward it was.
#
# Each entry is "localPort:svcName:remotePort". The first two are always present;
# the model services get a forward as you deploy (or scale up) each one.
FORWARDS_CORE := 8000:k8s-autoscaling-orchestrator:80 9200:opensearch-cluster-master:9200
FORWARDS_MODEL := \
	8081:k8s-autoscaling-triage-inference:8080 \
	8082:k8s-autoscaling-analyst-inference:8080 \
	8083:k8s-autoscaling-retriever-inference:8080 \
	8084:k8s-autoscaling-reranker:80
# A model forward is active only if the Service has at least one ready endpoint
# address (a running pod); the forward targets check that with `kubectl get
# endpoints` inline below.
forward:
	@for f in $(FORWARDS_CORE); do \
	  lp=$${f%%:*}; rest=$${f#*:}; svc=$${rest%%:*}; rp=$${rest##*:}; \
	  pkill -f "port-forward.*$$lp:$$rp" >/dev/null 2>&1 || true; sleep 1; \
	  setsid nohup kubectl port-forward -n slemify svc/$$svc $$lp:$$rp > /tmp/pf-$$lp.log 2>&1 < /dev/null & disown 2>/dev/null || true; \
	  echo "forwarding $$lp -> $$svc"; \
	done
	@for f in $(FORWARDS_MODEL); do \
	  lp=$${f%%:*}; rest=$${f#*:}; svc=$${rest%%:*}; rp=$${rest##*:}; \
	  ready=$$(kubectl get endpoints -n slemify $$svc -o jsonpath='{.subsets[*].addresses[*].ip}' 2>/dev/null); \
	  if [ -n "$$ready" ]; then \
	    pkill -f "port-forward.*$$lp:$$rp" >/dev/null 2>&1 || true; sleep 1; \
	    setsid nohup kubectl port-forward -n slemify svc/$$svc $$lp:$$rp > /tmp/pf-$$lp.log 2>&1 < /dev/null & disown 2>/dev/null || true; \
	    echo "forwarding $$lp -> $$svc"; \
	  fi; \
	done
	@sleep 2; $(MAKE) --no-print-directory forward-check

# Print the HTTP status of each forward that should be up at this stage. The two
# long-lived forwards always print; a model row prints only once that model has
# a ready pod (so a not-yet-deployed or scaled-to-zero model is simply omitted,
# not shown as DOWN). 200/4xx means the forward is up; 000 means not reachable.
forward-check:
	@for f in $(FORWARDS_CORE); do \
	  lp=$${f%%:*}; rest=$${f#*:}; svc=$${rest%%:*}; \
	  path=$$([ "$$lp" = "8000" ] && echo "/stats" || echo "/"); \
	  code=$$(curl -s -o /dev/null -w "%{http_code}" localhost:$$lp$$path); \
	  if [ "$$code" = "000" ]; then state="DOWN (not reachable)"; else state="up (HTTP $$code)"; fi; \
	  printf "%-5s %-36s %s\n" "$$lp" "$$svc" "$$state"; \
	done
	@for f in $(FORWARDS_MODEL); do \
	  lp=$${f%%:*}; rest=$${f#*:}; svc=$${rest%%:*}; \
	  ready=$$(kubectl get endpoints -n slemify $$svc -o jsonpath='{.subsets[*].addresses[*].ip}' 2>/dev/null); \
	  if [ -n "$$ready" ]; then \
	    code=$$(curl -s -o /dev/null -w "%{http_code}" localhost:$$lp/); \
	    if [ "$$code" = "000" ]; then state="DOWN (not reachable)"; else state="up (HTTP $$code)"; fi; \
	    printf "%-5s %-36s %s\n" "$$lp" "$$svc" "$$state"; \
	  fi; \
	done

# Append the active configuration's scoreboard row to scoreboard.tsv in this
# directory, so a swap (which clears the orchestrator's in-memory rows for the
# previous configuration) can never erase a number you already measured. Run it
# right after each `make eval` / `make eval-incluster`, before the next swap.
# The file is yours to keep; it is git-ignored.
SCOREBOARD_FILE ?= scoreboard.tsv
scoreboard-save:
	@curl -s $${ORCHESTRATOR_URL:-http://localhost:8000}/stats | python3 -c \
	  'import json,sys,os,time; \
d=json.load(sys.stdin); s=d["steps"]; \
cfg=",".join("%s=%s"%(k,s[k]) for k in sorted(s)); \
sb=d.get("scoreboard",{}).get(cfg,{}); \
f=os.environ.get("SCOREBOARD_FILE","scoreboard.tsv"); \
new=not os.path.exists(f); \
h="when\tconfig\tqueries\tfrontier_usd_p50\ttotal_ms_p50\tescalation_rate\tfirst_pass_rate\n"; \
row="%s\t%s\t%s\t%s\t%s\t%s\t%s\n"%(time.strftime("%Y-%m-%d %H:%M"), cfg, sb.get("queries",""), sb.get("frontier_usd_per_query_p50",""), sb.get("total_ms_p50",""), sb.get("escalation_rate",""), d.get("slm_first_pass_rate","")); \
open(f,"a").write((h if new else "")+row); \
print("saved row for ["+cfg+"] to "+f)'
