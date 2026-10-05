# Authoritative facts

Curated, self-contained statements of facts the agent is asked about, written as
prose so a single retrieved chunk answers the question completely. The upstream
docs are the source of truth; these snippets exist because the upstream pages
split a fact across several headings or a wide table, which the size-based chunker
can break apart so that no one retrieved chunk holds the whole answer. Each fact
below is kept short enough to survive chunking intact and phrased to match how the
question is actually asked.

## Karpenter consolidationPolicy valid values

The `spec.disruption.consolidationPolicy` field on a Karpenter v1 NodePool accepts
exactly three valid values: `WhenEmpty`, `WhenEmptyOrUnderutilized`, and `Balanced`.

- `WhenEmpty`: Karpenter considers a node for consolidation only when it is empty,
  meaning it runs only pods with no disruption cost (such as daemonsets or pods the
  user annotated as cheap to disrupt). This is the most conservative policy.
- `WhenEmptyOrUnderutilized`: Karpenter considers any node that can be removed or
  replaced to reduce cost, including underutilized nodes, not just empty ones. This
  is the default when `consolidationPolicy` is not set.
- `Balanced`: Karpenter consolidates a node only when the cost saved outweighs the
  disruption to the running pods, scoring each candidate action and acting only when
  the savings are large enough relative to the disruption.

There is no `WhenUnderutilized` value in Karpenter v1 and `WhenEmptyOrUnderutilized`
did not rename or replace a separate `WhenUnderutilized` policy; the three values
above are the complete, current set.

## Karpenter minValues and the min-values policy

`minValues` in a Karpenter NodePool requirement sets the minimum number of distinct
values the scheduler must consider for that requirement key, so the scheduler keeps
a floor of instance-type (or other key) flexibility when it provisions a node. When
the same key is given `minValues` more than once, the largest value applies.

What happens when `minValues` cannot be satisfied is governed by the controller-wide
min-values policy, set with the `--min-values-policy` flag or the `MIN_VALUES_POLICY`
environment variable. `Strict` fails the scheduling loop for that NodePool (Karpenter
falls back to another NodePool that fits, or the pod stays unschedulable). `BestEffort`
relaxes `minValues` until they can be met so scheduling proceeds. The default is
`Strict`, which preserves the original behavior.

## KEDA scale to zero

To let a KEDA-managed workload scale to zero when there are no events, set
`minReplicaCount: 0` on the ScaledObject. KEDA then removes all replicas while the
trigger reports no activity and scales back up from zero when events arrive. The
scaler's `activationThreshold` defaults to `0`, and that default is correct for
scale-to-zero: a positive `activationThreshold` is not required, and the workload
stays at zero while the activating metric is at or below the threshold.
