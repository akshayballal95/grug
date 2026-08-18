# Incident Report: Ledger Lag Regression

On 2026-02-11 the streaming ledger fell behind the event bus by up to 47 seconds.
It is important to note that no invoices were lost and no customer was billed
twice. The backlog drained on its own within 18 minutes of the trigger being
removed, and it did not require a manual replay.

## Root cause

A configuration change raised the batch flush interval from 200 ms to 5,000 ms
in order to reduce write amplification on the primary. In practice this was not
a safe change: the folding step assumes it will be woken at least once per
second, and without that wakeup it accumulates events in memory rather than
committing them. Throughput did not drop, so none of the existing dashboards
showed a problem until the lag alert fired 12 minutes later.

The check that should have caught this is a simple one:

```python
def flush_interval_is_safe(interval_ms: int, fold_deadline_ms: int = 1000) -> bool:
    """The folder must be woken at least once per deadline window."""
    if interval_ms <= 0:
        raise ValueError("interval must be positive")
    return interval_ms <= fold_deadline_ms
```

## Remediation

We reverted the interval to 200 ms at 14:32 UTC. Acme Corporation and Globex
Industries were the only tenants above the alerting threshold, and neither
reported a downstream failure. The Platform Reliability team has added the
guard above to the config validator, so an unsafe interval is now rejected at
deploy time rather than at runtime.

Do not raise the flush interval above 1,000 ms without a load test. If you need
a larger batch for a backfill, use the `--offline-batch` flag instead, which
bypasses the folding path entirely and is documented at
https://example.com/runbooks/backfill.
