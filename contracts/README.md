# Message contracts

The wire format between **GoldenAI-Video/web-app** (producer of jobs, consumer of events) and the
workers (consumers of jobs, producers of events): **GoldenAI-Video/Research** and
**character_swap**.

This folder is **vendored byte-identical into every repo**. There is no package to publish and no
registry to run: the copies are the contract, and a test in each repo fails if they drift.

## Tasks and streams (1.2)

A job's `task` decides which worker pool gets it. Each task has its own job stream and consumer
group, so one pool never steals another's work; the event streams are shared, since every event
carries its `job_id`.

| task | job stream | group | worker | `project_id` / `shot_id` |
|---|---|---|---|---|
| `generate` (default, omitted on the wire) | `jobs.generate` | `research-workers` | Research `pipeline.worker` | project folder / shot file stem |
| `character_swap` | `jobs.character_swap` | `swap-workers` | character_swap `worker --role render` | `demoN` / `all` or a shot number |
| `character_swap_util` | `jobs.character_swap_util` | `swap-util-workers` | character_swap `worker --role util` | `demoN` / shot number (`action: plan_edit`), or `characters` / name (`action: add_character`) |

`ProviderCapabilities.task` says which task a provider serves, and the web app validates a job's
overrides only against providers of that task. `ParameterSpec.group` splits one provider's
parameters into several forms (character swap: `create` options and per-`shot` edits).

**Cancellation.** `BrokerClient.request_cancel(job_id)` sets `<prefix>.cancel:<job_id>`; a worker
that polls `cancel_requested(job_id)` stops and publishes a FailedEvent with code `cancelled`.
Advisory: a worker that does not poll finishes normally.

```
contracts/
├── VERSION               contract version, bumped on every wire-visible change
├── messages.schema.json  canonical JSON Schema for every message
├── checksum.txt          sha256 of every file here except itself
├── python/               client, typed messages, capability specs (both backends)
└── typescript/           types + capability helpers (frontend)
```

## Capabilities: why most changes need no web-app edit

Generation parameters are **not** enumerated in this contract, and not in the web app
either. `JobRequest.overrides` and `CompletedEvent.output` are open maps, and Research
workers advertise what they accept as a `ProviderCapabilities` document
(`python/capabilities.py`) on the Redis key `<prefix>.capabilities:<provider>`, refreshed
on a heartbeat with a TTL.

The web app reads those documents to build its form and to validate a submitted override.
So adding a model or a parameter is:

1. declare a `ParameterSpec` (or `ModelSpec`) in the provider's `capabilities()`
2. restart the worker

No schema bump, no contract sync, no web-app change. `ParameterSpec` carries the type,
bounds, allowed values, label and help text, which is enough for a UI to render a control
and for a server to reject a bad value with a reason.

Capabilities are **advisory**. With nothing advertising — no worker running, or one
restarting — the web app forwards overrides unchecked under generic ceilings rather than
refusing to queue work.

This is why the contract version moves far less often than the pipeline does: bump it only
for a change to the *message envelope*, not for a parameter a provider gained.

## Changing a contract

1. Edit the files in **one** repo.
2. Bump `VERSION` (minor for an additive/optional field, major for anything that breaks a reader).
3. Regenerate the checksum: `python contracts/python/checksum.py --write`
4. Copy into the other repo: `scripts/sync-contracts.sh ../<other-repo>`
5. Commit in both repos, referencing each other's commit in the message.

`scripts/check-contracts.sh` verifies the local copy matches `checksum.txt`; the backend test suite
runs the same check, so an un-synced edit fails CI rather than production.

## Compatibility rule

Readers must ignore unknown fields and must not fail on a higher `contract_version` with the same
major. Writers must never remove or retype an existing field within a major version.

Two places deliberately go further and *preserve* what they do not recognise rather than
merely tolerating it: `overrides` on the way in, and `output` on the way out. A field
neither side names still reaches the provider, and still reaches the browser.
