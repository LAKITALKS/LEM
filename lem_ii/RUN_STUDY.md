# Full-study execution authorized 8 September 2026

Author: Lazaros Varvatis.

The 8 September execution instruction authorizes development, validation,
freeze, confirmation collection and confirmation analysis in that order. It
supersedes the earlier smoke-only and development-only execution permissions.
It does not change the scientific protocol or `config/study.json`. Only its six
designated execution gates change in private runtime copies. No further routine
approval is required between these phases. The repository defaults remain closed
to prevent an accidental run by merely checking out the code.

## Frozen implementation and private data

The production controller is `launch_study.py`, and its Modal entry point is
`modal_study.py`. The old smoke launcher remains exclusively technical.
Model adapter, model/tokenizer revision, BF16 precision, simulator, seeds,
features, ablations and statistical definitions are unchanged. The collection
extension only adds a persistence callback. Runtime-prefix accounting supports
explicit continuation of validated partial sessions; it changes no measurement.

Before initialization, commit all uploaded code. The launcher obtains the real
clean Git commit and SHA-256 hashes. The container verifies every uploaded file,
including the complete study source set, against that frozen attestation. It
does not pretend to run Git in a container without `.git`. The attestation and
its verification method are stored in each run manifest. All phases use the
same attestation; post-collection source edits cannot silently rebind records.

The single local study root is `.runtime/full-study-20260908`. Its fixed study ID
maps to one task-owned Modal Volume. Source manifests, private authorization
configuration, cumulative reservations, original failures, data and exports stay
there, outside Git. The existing technical budget ledger is preserved separately
and never reset. A local exclusive controller lock and a durable remote active
claim prevent concurrent studies. Each consumed invocation claim is permanent.

## Resource and budget bounds

Each serial phase has one L4, hard limits of two physical CPU cores and 16 GiB
RAM, a 10,800-second provider function timeout, a 900-second startup/build bound,
zero application retries and no warm container. A separate client watchdog stops
the owned app. The controller explicitly verifies provider termination before
accepting and importing a phase result. CPU analysis runs locally with that GPU
stopped.

Each attempt permanently reserves $4.50. The initial three allocations are
$13.50; at most two investigated continuations can consume another $9.00. A
separate $1.50 storage/other reserve makes the new-use ceiling $24.00. At the
verified public rates, a full resource-time envelope plus $1.00 per-attempt
noncompute margin is below $4.50. Failed reservations are not refunded, and the
remaining phases' allocations cannot be consumed by development. These bounds
also fit the previous $25 internal study allowance. The live preflight must
verify the authorized $27 gross workspace limit, current usage with rounding
headroom, active workspace/environment and absence of another running app.
Neither this controller nor a resumed phase changes provider budget settings.

Volume checkpoints are committed after every successful turn and after every
complete-dialogue archive. The Volume holds no model weights, only study files;
each phase loads the same fixed model once. Downloads are immutable session
exports plus the shared run records. Storage is budgeted independently of GPU
termination; the source-backed local delivery remains permanent.

## Execution sequence

Use the existing tested Python 3.11 environment. Private preflight JSON contains
`verified_at_utc`, `workspace`, `environment: main`,
`workspace_gross_usage_limit_usd: 27`, `workspace_usage_upper_usd` and `rates`
(`L4: 0.000222`, `CPU_core: 0.0000131`, `RAM_GiB: 0.00000222`). Evidence must be
under two hours old and must be refreshed from the actual provider state.

```sh
python -m lem_ii.launch_study development --preflight .runtime/private-study-preflight.json
python -m lem_ii.launch_study validation --preflight .runtime/private-study-preflight.json
```

Each command validates and skips all complete compatible sessions before model
loading or a new reservation. A phase completes only with exactly 36 validated
24-turn sessions. Earlier phase inventories are checked remotely as well.
The full 24-turn technical smoke had 24 `max_new_tokens` stops: every answer
reached its prespecified 128-token cap. Retain that cap and report all study
length/stop distributions; do not optimize it after observing outcomes.

On the unchanged local CPU environment, load the preserved base configuration,
derive its development/validation gate copy with `production.phase_config`, and
call `study.analyze_development_validation(data_root, config, freeze_directory)`.
This reads exactly 72 complete records, fits only development data, applies the
fixed validation selection, and uses all 199 label permutations and 2,000
bootstrap repetitions. It saves the four actual fitted endpoint artifacts.
Then call `production.seal_freeze` with that directory, configuration and the
original source attestation. The seal binds data, artifacts, code, software and
time; its files are made read-only locally. No confirmation data may exist yet.

```sh
python -m lem_ii.launch_study confirmation --preflight .runtime/private-study-preflight.json
```

The remote worker verifies the uploaded four-artifact freeze and its timestamp
before generating any confirmation dialogue. After exactly 36 confirmed complete
sessions are locally validated, `study.confirm` uses the same CPU environment,
the confirmation-only gate copy and the frozen directory. This path never fits
or selects anything. Its predictions and all prespecified contrasts are saved.

## Explicit recovery, never a new confirmation trial

If a collection invocation fails, preserve its errors and budget reservation.
`launch_study.reconcile_stopped_attempt(attempt_id)` verifies the owned app is
stopped, downloads a separate durable recovery snapshot and reconciles the
original attempt. It may release only that stopped attempt's active claim, never
its consumed invocation claim. A subsequent collection command with
`--resume-inspected-partial` uses a new paid reservation for unfinished work.
Existing full dialogues are skipped; partial histories, bindings and checksums
are validated and replayed before any next response. Corrupt checkpoints fail
closed and are not overwritten or replaced by fabricated observations.

An interrupted development/validation analysis can use `resume=True` only with
identical validated inputs, source and software. Already frozen endpoint models
are loaded and verified, never refitted or replaced.

An interrupted `study.confirm` can use `resume=True` only under its original
permanent `confirmation_grid.lock`, original trial ID, input fingerprints,
artifact hashes and output path. Predictions are checkpointed by endpoint.
Finished endpoints are reused; deterministic uncertainty calculations may be
continued on those same predictions. A process lock excludes a concurrent call.
Changing records, artifacts, configuration, software or output path is rejected.
Never delete a confirmation lock or select another output path to try again.

The final report must separate descriptive development/validation results from
confirmation results. It reports every planned method, including negative
comparisons, and describes only the fixed synthetic profile/topic experiment.

Operational references checked before launch:
[Modal prices](https://modal.com/pricing),
[provider timeouts](https://modal.com/docs/guide/timeouts),
[durable Volumes and explicit commits](https://modal.com/docs/guide/volumes).
