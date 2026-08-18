# Migrating the Billing Pipeline

The billing pipeline has been rewritten to run on the streaming ingest service.
It is important to note that the migration is not automatic: accounts on the
legacy monthly plan must be moved by hand before the cutover date of 2026-03-15.
Please note that no data is deleted during the migration, and no invoices are
regenerated.

## What changed

The previous pipeline batched invoice lines once per night. The new one folds
each line into a running total as it arrives, which means the ledger is never
more than 3 seconds behind the event stream. In practice we measured a median
lag of 1.2 seconds across 4,800 accounts during the trial at Acme Corporation,
and a p99 lag of 9.6 seconds. Globex Industries ran the same trial and saw no
regression in throughput.

The key economic point is easy to state and easy to get wrong: bills scale with
volume, not price. Doubling the unit price of a plan does not double the
pipeline cost, because the pipeline charges per event rather than per dollar.
Teams that budget on revenue rather than event count will therefore overestimate
their spend by roughly 40%.

## Rollout

We are rolling out in three waves. Wave 1 covers internal tenants only. Wave 2
adds customers under 500 seats, and Wave 3 covers everyone else. Each wave is
gated on the previous one running clean for 72 hours, so the whole rollout takes
at least 9 days and should not be compressed further without a sign-off from the
Platform Reliability team.

To check which wave a tenant is in, call the status endpoint:

```python
def wave_for(tenant_id: str) -> int:
    """Return the rollout wave for a tenant, or 0 if it is not enrolled."""
    response = client.get(f"/v2/tenants/{tenant_id}/rollout")
    if response.status_code == 404:
        return 0
    return response.json()["wave"]
```

The endpoint is rate limited to 100 requests per minute per token. Do not poll
it in a tight loop; use the webhook instead. If you need the historical wave
assignment, that is not exposed through the API and you should ask the Platform
Reliability team directly.

## Known issues

Invoices created between 2026-01-01 and 2026-01-14 have a rounding error of at
most 0.02 in the tax line. These are being corrected in a background job that
runs nightly. No customer action is required, and no refunds are needed, since
the error never favours the vendor.

## Migration checklist

Before you start, confirm that the tenant has no open disputes, that the last
successful export is less than 24 hours old, and that the finance contact on
file is current. Migrations that begin without these three checks have failed
about 15% of the time, almost always because a dispute was still open and the
ledger could not be frozen. Run the preflight command with the `--dry-run` flag
first; it prints the same report without writing anything. If the report shows
any line item marked `UNRESOLVED`, stop and escalate rather than forcing the
migration through, because a forced migration cannot be rolled back once the
ledger has been frozen and the new pipeline has accepted its first event.
