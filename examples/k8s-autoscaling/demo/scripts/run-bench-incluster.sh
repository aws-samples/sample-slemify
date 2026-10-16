#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Price-performance benchmark for a model served on CPU, run IN-CLUSTER as a Job.
#
# The Job sends the held-out messages to the model's Service at several
# concurrency levels and reads the model pod's /stats (CPU seconds used,
# requests served) around each level, so the cost is priced on the CPU the
# model actually used. This script looks up the node the model runs on and
# its on-demand price, so the Job needs no AWS or Kubernetes permissions.
#
# Usage:
#   ./run-bench-incluster.sh triage                      # levels 1,2,4,8, 10 s each
#   LEVELS=1,4 DURATION=5 ./run-bench-incluster.sh triage
#
# If eval/results/ holds a TRIAGE=llm run of eval/triage_acc.py, the result is
# printed next to the frontier model's cost and latency for the same step.
set -euo pipefail

MODEL="${1:-triage}"
NAMESPACE="${NAMESPACE:-slemify}"
LEVELS="${LEVELS:-1,2,4,8}"
DURATION="${DURATION:-10}"
HERE="$(cd "$(dirname "$0")" && pwd)"
EVAL_DIR="$(cd "$HERE/../eval" && pwd)"
RESULTS_DIR="$EVAL_DIR/results"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
JOB_NAME="k8s-autoscaling-bench-${MODEL}-$(echo "$STAMP" | tr '[:upper:]' '[:lower:]')"

case "$MODEL" in
  triage)
    PROJECT="k8s-autoscaling-triage"
    SVC="k8s-autoscaling-triage-inference"; PORT=8080
    QUERY_SET="$EVAL_DIR/triage-heldout.yaml" ;;
  *)
    echo "unknown model '$MODEL' (supported: triage)" >&2; exit 2 ;;
esac

POD="$(kubectl get pods -n "$NAMESPACE" -l "slemify.io/project=$PROJECT,slemify.io/stage=serving" \
  --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
if [ -z "$POD" ]; then
  echo "No running $MODEL pod in $NAMESPACE. Deploy it first (slemify deploy --config $MODEL/expert.yaml)." >&2
  exit 1
fi
NODE="$(kubectl get pod "$POD" -n "$NAMESPACE" -o jsonpath='{.spec.nodeName}')"
INSTANCE_TYPE="$(kubectl get node "$NODE" -o jsonpath='{.metadata.labels.node\.kubernetes\.io/instance-type}')"
REGION="$(kubectl get node "$NODE" -o jsonpath='{.metadata.labels.topology\.kubernetes\.io/region}')"
VCPUS="$(kubectl get node "$NODE" -o jsonpath='{.status.capacity.cpu}')"
CPU_REQUEST="$(kubectl get pod "$POD" -n "$NAMESPACE" -o jsonpath='{.spec.containers[0].resources.requests.cpu}')"
case "$CPU_REQUEST" in *m) CPU_REQUEST="$(python3 -c "print(${CPU_REQUEST%m}/1000)")" ;; esac

# On-demand Linux list price of the node (the Price List API lives in us-east-1).
USD_PER_HOUR="$(aws pricing get-products --region us-east-1 --service-code AmazonEC2 \
  --filters Type=TERM_MATCH,Field=instanceType,Value="$INSTANCE_TYPE" \
            Type=TERM_MATCH,Field=regionCode,Value="$REGION" \
            Type=TERM_MATCH,Field=operatingSystem,Value=Linux \
            Type=TERM_MATCH,Field=tenancy,Value=Shared \
            Type=TERM_MATCH,Field=preInstalledSw,Value=NA \
            Type=TERM_MATCH,Field=capacitystatus,Value=Used \
  --query 'PriceList[0]' --output text 2>/dev/null | python3 -c '
import json, sys
try:
    d = json.loads(sys.stdin.read())
    for t in d["terms"]["OnDemand"].values():
        for p in t["priceDimensions"].values():
            print(p["pricePerUnit"]["USD"]); raise SystemExit
except (ValueError, KeyError, TypeError):
    pass
print(0)' || echo 0)"
if [ "$USD_PER_HOUR" = "0" ]; then
  echo "(Could not read the on-demand price of $INSTANCE_TYPE; the \$ columns will show n/a.)" >&2
fi

QUERIES_JSON="$(mktemp)"
python3 - "$QUERY_SET" > "$QUERIES_JSON" <<'PY'
import json, os, sys, yaml
path = sys.argv[1]
items = yaml.safe_load(open(path))
here = os.path.dirname(path)
print(json.dumps([open(os.path.join(here, it["file"])).read() if "file" in it else it["query"]
                  for it in items]))
PY

IMAGE="$(kubectl get deployment k8s-autoscaling-orchestrator -n "$NAMESPACE" \
  -o jsonpath='{.spec.template.spec.containers[0].image}')"
echo "=== Bench Job: $JOB_NAME ($MODEL on $INSTANCE_TYPE) ==="

kubectl create configmap k8s-autoscaling-bench-src -n "$NAMESPACE" \
  --from-file=bench.py="$EVAL_DIR/bench.py" --from-file=queries.json="$QUERIES_JSON" \
  --dry-run=client -o yaml | kubectl apply -f - >/dev/null
rm -f "$QUERIES_JSON"

kubectl apply -f - >/dev/null <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: $JOB_NAME
  namespace: $NAMESPACE
  labels:
    app: k8s-autoscaling-bench
    slemify.io/component: demo
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 3600
  template:
    metadata:
      labels:
        app: k8s-autoscaling-bench
    spec:
      restartPolicy: Never
      automountServiceAccountToken: false
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        runAsGroup: 1000
        seccompProfile:
          type: RuntimeDefault
      nodeSelector:
        slemify.io/workload: slm
      tolerations:
        - key: slemify.io/slm
          operator: Exists
          effect: NoSchedule
      # Prefer a different node from the model, so the load generator does not
      # compete with it for cores. On a single-node pool it shares the node.
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              podAffinityTerm:
                topologyKey: kubernetes.io/hostname
                labelSelector:
                  matchLabels:
                    slemify.io/project: $PROJECT
      containers:
        - name: bench
          image: $IMAGE
          workingDir: /bench-src
          command: ["python3", "-u", "bench.py"]
          args: ["--url", "http://$SVC.$NAMESPACE:$PORT", "--name", "$MODEL",
                 "--queries", "queries.json", "--levels", "$LEVELS", "--duration", "$DURATION",
                 "--instance-type", "$INSTANCE_TYPE", "--vcpus", "$VCPUS",
                 "--usd-per-hour", "$USD_PER_HOUR", "--cpu-request", "$CPU_REQUEST"]
          resources:
            requests: {cpu: "1", memory: 256Mi}
            limits: {cpu: "2", memory: 512Mi}
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities:
              drop: ["ALL"]
          volumeMounts:
            - {name: bench-src, mountPath: /bench-src}
            - {name: tmp, mountPath: /tmp}
      volumes:
        - name: bench-src
          configMap:
            name: k8s-autoscaling-bench-src
        - name: tmp
          emptyDir:
            sizeLimit: 16Mi
EOF

kubectl wait --for=condition=ready pod -l job-name="$JOB_NAME" -n "$NAMESPACE" --timeout=180s >/dev/null 2>&1 || true
LOG_FILE="$(mktemp)"
kubectl logs -f "job/$JOB_NAME" -n "$NAMESPACE" | sed '/===BENCH_JSON_BEGIN===/,/===BENCH_JSON_END===/d' || true
kubectl wait --for=condition=complete "job/$JOB_NAME" -n "$NAMESPACE" --timeout=300s >/dev/null 2>&1 || true
kubectl logs "job/$JOB_NAME" -n "$NAMESPACE" > "$LOG_FILE" 2>/dev/null || true

mkdir -p "$RESULTS_DIR"
OUT="$RESULTS_DIR/bench-$MODEL-$STAMP.json"
if sed -n '/===BENCH_JSON_BEGIN===/,/===BENCH_JSON_END===/p' "$LOG_FILE" | sed '1d;$d' > "$OUT" && [ -s "$OUT" ]; then
  echo "Saved: eval/results/$(basename "$OUT")"
else
  echo "=== WARNING: no benchmark result in the Job log (kubectl logs job/$JOB_NAME -n $NAMESPACE) ===" >&2
  rm -f "$OUT"; rm -f "$LOG_FILE"; exit 1
fi
rm -f "$LOG_FILE"

# Put the result next to the frontier model doing the same step, if measured.
if [ "$MODEL" = "triage" ]; then
  python3 - "$OUT" "$RESULTS_DIR" <<'PY'
import glob, json, os, sys
bench = json.load(open(sys.argv[1]))
runs = sorted(glob.glob(os.path.join(sys.argv[2], "triage-llm-*.json")))
llm = next((json.load(open(p)) for p in reversed(runs)
            if json.load(open(p)).get("usd_per_call_mean")), None)
if not llm:
    print("\n(Run 'TRIAGE=llm make triage-acc' to put the frontier model's cost next to this.)")
    raise SystemExit
f_usd, f_ms = llm["usd_per_call_mean"], llm["latency_p50_ms"]
print("\nThe same triage step, both ways:")
print(f"  {'':22} {'p50':>9} {'$ per 1M':>11}")
print(f"  {'frontier model':22} {f_ms:>7.0f}ms {f_usd * 1e6:>11,.2f}")
c_usd = bench.get("usd_per_1m")
c_txt = f"{c_usd:,.2f}" if c_usd is not None else "n/a"
print(f"  {'classifier on CPU':22} {bench['p50_ms_single']:>7.0f}ms {c_txt:>11}")
if c_usd:
    print(f"  the classifier is about {f_usd * 1e6 / c_usd:,.0f}x cheaper and "
          f"{f_ms / max(bench['p50_ms_single'], 0.1):,.0f}x faster per message")
pod_usd = bench.get("pod_usd_per_hour")
if pod_usd:
    print(f"  one classifier pod (${pod_usd:.3f}/hour) costs less than frontier triage above "
          f"{pod_usd / f_usd:,.0f} messages per hour")
PY
fi
echo "=== Done. Job $JOB_NAME will self-clean in 1h ==="
