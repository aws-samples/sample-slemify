#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Run the eval scorecard IN-CLUSTER as a Kubernetes Job.
#
# Why: the hard cases (escalate + abstain) take multiple minutes each; local
# kubectl port-forwards systematically drop mid-run and get scored as errors.
# Running next to the orchestrator removes the flaky hop entirely.
#
# How it works:
#   1. Packs eval/run_eval.py + eval/cases.yaml into a ConfigMap.
#   2. Launches a Job on the orchestrator's image (has all Python deps) and
#      service account (has Bedrock access for the judge), with in-cluster
#      service URLs.
#   3. Streams the Job logs, then extracts the scorecard JSON from the log
#      markers into eval/results/ locally.
#
# Usage:
#   ./run-eval-incluster.sh                          # all cases, repeat=4
#   ./run-eval-incluster.sh --repeat 1               # quick smoke check
#   ./run-eval-incluster.sh --only drift,minvalues-valid
#   ./run-eval-incluster.sh --save-baseline          # saves baseline locally too
set -euo pipefail

NAMESPACE="${NAMESPACE:-slemify}"
HERE="$(cd "$(dirname "$0")" && pwd)"
EVAL_DIR="$HERE/../eval"
RESULTS_DIR="$EVAL_DIR/results"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
JOB_NAME="k8s-autoscaling-eval-$(echo "$STAMP" | tr '[:upper:]' '[:lower:]')"

# Pass through run_eval.py args; --save-baseline is handled locally after.
SAVE_BASELINE=false
ARGS=()
for a in "$@"; do
  if [ "$a" = "--save-baseline" ]; then SAVE_BASELINE=true; else ARGS+=("$a"); fi
done
# JSON array of args for the Job spec (empty list when no args).
ARGS_JSON="[]"
if [ "${#ARGS[@]}" -gt 0 ]; then
  ARGS_JSON="$(printf '%s\n' "${ARGS[@]}" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().splitlines()))')"
fi

# The orchestrator's running image (already in ECR, has all deps).
IMAGE="$(kubectl get deployment k8s-autoscaling-orchestrator -n "$NAMESPACE" \
  -o jsonpath='{.spec.template.spec.containers[0].image}')"
echo "=== Eval Job: $JOB_NAME (image: $IMAGE) ==="

kubectl create configmap k8s-autoscaling-eval-src -n "$NAMESPACE" \
  --from-file=run_eval.py="$EVAL_DIR/run_eval.py" \
  --from-file=cases.yaml="$EVAL_DIR/cases.yaml" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: $JOB_NAME
  namespace: $NAMESPACE
  labels:
    app: k8s-autoscaling-eval
    slemify.io/component: demo
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 3600
  template:
    metadata:
      labels:
        app: k8s-autoscaling-eval
    spec:
      restartPolicy: Never
      serviceAccountName: k8s-autoscaling-orchestrator
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
      containers:
        - name: eval
          image: $IMAGE
          workingDir: /eval-src
          command: ["python3", "-u", "run_eval.py"]
          args: $ARGS_JSON
          env:
            - name: AWS_REGION
              value: "eu-west-1"
            - name: AWS_DEFAULT_REGION
              value: "eu-west-1"
            - name: ORCHESTRATOR_URL
              value: "http://k8s-autoscaling-orchestrator.$NAMESPACE"
            - name: EMBEDDING_URL
              value: "http://k8s-autoscaling-retriever-inference.$NAMESPACE:8080"
            - name: KNOWLEDGE_URL
              value: "http://opensearch-cluster-master.$NAMESPACE:9200/k8s-autoscaling-knowledge"
            - name: SCORECARD_STDOUT
              value: "1"
            - name: HOME
              value: "/tmp"
          resources:
            requests:
              cpu: 100m
              memory: 256Mi
            limits:
              cpu: 500m
              memory: 512Mi
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities:
              drop:
                - ALL
          volumeMounts:
            - name: eval-src
              mountPath: /eval-src
            - name: results
              mountPath: /eval-src/results
            - name: tmp
              mountPath: /tmp
      volumes:
        - name: eval-src
          configMap:
            name: k8s-autoscaling-eval-src
        - name: results
          emptyDir:
            sizeLimit: 64Mi
        - name: tmp
          emptyDir:
            sizeLimit: 64Mi
EOF

echo "=== Waiting for pod, then streaming logs ==="
kubectl wait --for=condition=ready pod -l job-name="$JOB_NAME" -n "$NAMESPACE" --timeout=300s || true
LOG_FILE="$(mktemp)"
kubectl logs -f "job/$JOB_NAME" -n "$NAMESPACE" | tee "$LOG_FILE" || true

# Wait for terminal state (logs can end before status settles).
kubectl wait --for=condition=complete "job/$JOB_NAME" -n "$NAMESPACE" --timeout=120s 2>/dev/null || true

# Extract the scorecard from the log markers into local results/.
mkdir -p "$RESULTS_DIR"
OUT="$RESULTS_DIR/scorecard-$STAMP.json"
if sed -n '/===SCORECARD_JSON_BEGIN===/,/===SCORECARD_JSON_END===/p' "$LOG_FILE" \
    | sed '1d;$d' > "$OUT" && [ -s "$OUT" ]; then
  echo "=== Scorecard saved: $OUT ==="
  if [ "$SAVE_BASELINE" = true ]; then
    cp "$OUT" "$RESULTS_DIR/baseline.json"
    echo "=== Baseline saved: $RESULTS_DIR/baseline.json ==="
  fi
else
  echo "=== WARNING: no scorecard found in Job logs (job failed early?) ===" >&2
  rm -f "$OUT"
fi
rm -f "$LOG_FILE"

FAILED="$(kubectl get "job/$JOB_NAME" -n "$NAMESPACE" -o jsonpath='{.status.failed}' 2>/dev/null || echo '')"
if [ -n "$FAILED" ] && [ "$FAILED" != "0" ]; then
  echo "=== Job reported failure (exit code reflects eval failures or an error) ===" >&2
fi
echo "=== Done. Job $JOB_NAME will self-clean in 1h (ttlSecondsAfterFinished) ==="
