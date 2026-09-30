# Verification Summary

The verification package checks the frozen transport, joint, relay, resource,
and partition results used by the study.

## Checks Completed

- Exact box coverage.
- Hard-deadline compliance.
- Segment flight time and energy closure.
- Return-SOC constraints.
- Delivery aircraft and shared-battery time conflicts.
- Relay aircraft and energy-component turnaround conflicts.
- Interval-level direct and relay communication coverage.
- Atomic task-block partition feasibility.
- Resource peak and inventory-shortfall accounting.
- Route-workload lower-bound certificate.

## Reproduce

From the repository root:

```powershell
python code/core/replay.py
```

The check writes machine-readable outputs to `results/verification/`.

## Scope

CP-SAT is exact over the supplied candidate set. The communication records are
deterministic audits under the recorded link-budget model and sampling
resolution.
