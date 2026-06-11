#!/bin/bash
# Launch the full demo environment: port-forward, tmux dashboard, and browser.
#
# Layout:
#   ┌─────────────────────────────────────┐
#   │  Pane 0: Orchestrator logs          │
#   │  (shows routing decisions + timing) │
#   ├──────────────────┬──────────────────┤
#   │  Pane 1: Triage  │  Pane 2: Auditor │
#   │  SLM logs        │  SLM logs        │
#   └──────────────────┴──────────────────┘
#
# Also opens http://localhost:8000 in the default browser.
#
# Usage:
#   ./demo-terminal.sh

SESSION="slemify-demo"
PORT=8000

# Kill any existing port-forward on the target port
if lsof -i ":${PORT}" -t >/dev/null 2>&1; then
  echo "Killing existing process on port ${PORT}..."
  kill $(lsof -i ":${PORT}" -t) 2>/dev/null
  sleep 1
fi

# Kill existing tmux session if any
tmux kill-session -t "$SESSION" 2>/dev/null

# Start port-forward in background
echo "Starting port-forward (svc/k8s-autoscaling-orchestrator -> localhost:${PORT})..."
kubectl port-forward -n slemify svc/k8s-autoscaling-orchestrator "${PORT}:80" &
PF_PID=$!
sleep 2

# Verify port-forward is running
if ! kill -0 "$PF_PID" 2>/dev/null; then
  echo "Error: port-forward failed to start"
  exit 1
fi

# Open browser
echo "Opening browser at http://localhost:${PORT}..."
open "http://localhost:${PORT}" 2>/dev/null || xdg-open "http://localhost:${PORT}" 2>/dev/null

# Create tmux session
tmux new-session -d -s "$SESSION" -n "demo"

# Top pane: orchestrator logs
tmux send-keys -t "$SESSION" \
  'echo "=== Orchestrator (routing + RAG + streaming) ===" && kubectl logs -n slemify deployment/k8s-autoscaling-orchestrator -f --tail=5' Enter

# Split horizontally for bottom row
tmux split-window -t "$SESSION" -v -p 40

# Bottom-left: triage SLM
tmux send-keys -t "$SESSION" \
  'echo "=== Triage SLM (4B, CPU) ===" && kubectl logs -n slemify -l slemify.io/project=k8s-autoscaling-triage -f --tail=1' Enter

# Split bottom row vertically for auditor
tmux split-window -t "$SESSION" -h -p 50

# Bottom-right: auditor SLM
tmux send-keys -t "$SESSION" \
  'echo "=== Auditor SLM (8B, CPU) ===" && kubectl logs -n slemify -l slemify.io/project=k8s-autoscaling-auditor -f --tail=1' Enter

# Select top pane
tmux select-pane -t "$SESSION:0.0"

# Attach (port-forward runs in background, killed when script exits)
trap "kill $PF_PID 2>/dev/null" EXIT
tmux attach -t "$SESSION"
