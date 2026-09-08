# LEM-II-Light

**Execution update, 8 September 2026:** the full 108-dialogue study, freeze and
independent confirmation are authorized by the new execution instruction. Follow
[RUN_STUDY.md](RUN_STUDY.md) for the bounded production path and recovery rules.
That dated authorization supersedes the earlier smoke-only execution scope below;
the original scientific configuration and historical smoke evidence are preserved.

Author: Lazaros Varvatis. This directory contains the executable pipeline for a
**prespecified protocol**, not a publicly preregistered or completed study.
The original LEM-I manuscripts, figures, correction and experiments are separate.

**Verified 7 September 2026:** the real Qwen3B Modal smoke passed 24+2 turns,
including actual state extraction, session reset and local feature recomputation.
See the [public technical verification record](TECHNICAL_SMOKE_2026-09-07.md).
The scientific study remains unexecuted.

Read [PROTOCOL.md](PROTOCOL.md) for the scientific decisions and
[ANALYSIS.md](ANALYSIS.md) for the analysis contract. The matching machine-readable
specification is [config/study.json](config/study.json).

## Scope and current authorization

The target is three **observable interaction regimes**, instantiated by 18 known
synthetic profiles. The planned 108 sessions contain 24 user/assistant turns each.
Each profile appears in all six topics: two development, two validation and two
confirmation topics. Topic, formulation family and phase change together. The
target generalization is to the held-out topics/formulations within this known
synthetic profile set. No inference about humans, identity, private cognition,
post-reset memory, attractors or additional information in an information-theoretic
sense is warranted.

The three paired comparisons share cases, classifier family and the same three-C
tuning budget: text alone versus text plus static activations/geometry; that
static basis versus added time/delay features; the same complete basis versus
added topology. Turn24 is primary;12/16/20/24 form the separate fixed-window curve.

This task authorizes only local implementation/fixtures and at most $5 gross
additional Modal technical usage **across all attempts**. The remainder of the
$30 study budget is not released. All study and confirmation execution gates
remain false. Nothing here schedules, deploys or automatically continues a run.

## Local setup and validation

From the repository root, use Python 3.11 and an isolated environment:

```sh
python3.11 -m venv .venv-lem-ii
.venv-lem-ii/bin/python -m pip install -r lem_ii/requirements.txt
```

Direct dependencies are pinned. The technical evidence records the full local
and remote package sets; Linux CUDA wheels and macOS CPU wheels differ. No
global installation or ASCR environment is required.

Cache the small, pinned tokenizer for the real local extraction test. This
downloads tokenizer files only, not3B model weights:

```sh
.venv-lem-ii/bin/python -c 'from transformers import AutoTokenizer; AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", revision="aa8e72537993ba99e69dfaafa59ed015b17504d1", cache_dir=".runtime/hf-cache", trust_remote_code=False)'
.venv-lem-ii/bin/python -m unittest discover -s lem_ii/tests -v
.venv-lem-ii/bin/python -m unittest discover -s experiments/tests -v
```

Without that cached tokenizer, the local Qwen test explicitly skips and extraction
must be reported as unverified. With it, the production adapter is exercised on
a tiny randomly initialized Qwen2 model, including24+2 turns and session reset.
These are synthetic software fixtures, not scientific model observations.

Exercise all four analysis endpoints on clearly marked fixtures:

```sh
.venv-lem-ii/bin/python -m lem_ii.evaluation --output .runtime/fixture-analysis --diagnostic-permutations 3 --diagnostic-bootstrap 100
```

The reduced repetition counts are logged diagnostic overrides. Omitting them
uses the predeclared199 label randomizations and2000 profile bootstrap samples.
Neither variant collects or evaluates confirmation data. Fixture scores are
software outputs and must never be reported as LLM research results.

## Exact model measurement

Model and tokenizer are Qwen/Qwen2.5-3B-Instruct at immutable revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1`, unquantized BF16, SDPA attention.
No training or weight updates occur. Greedy generation is explicit and limited
to128 new tokens. The8192-token cap reserves space for the answer and rejects
overflow without truncation.

The measured token is the final newline token198 of the assistant generation
prefix `[151644,77091,198]`, before the current answer. Actual template tokens
and the UTF-8 template hash are asserted. Hidden-state index0 is the embedding
output,18 the middle block and36 the final normalized block output;36 is primary.
Only those three last-nonpadding vectors are saved as float32, without pooling
or raw normalization. `logits_to_keep=1` avoids materializing unused token logits
and leaves hidden-state extraction unchanged.

Each turn has one separate cache-free extraction prefill plus generation's own
prefill. Generation uses a transient cache within that single answer. Each turn
rebuilds its entire visible history; no cache/state is carried into another
turn or session. A uniform system prompt is used. Profile/regime/seed/policy
metadata go only to separate labels/audit files. The text baseline gets exactly
the rendered pre-answer context, never the current or future answer.

## Bounded Modal smoke

Prepare a **private** `.runtime/private-preflight.json` using the structure in
[config/preflight.example.json](config/preflight.example.json). The public example
contains placeholders and cannot authorize or start a run. Independently verify
the active Modal profile/workspace, environment, current public tariffs, applicable
gross usage limit and current usage immediately before execution. The private
proof must be under two hours old, and its workspace must match the active profile.
A spend limit after credits is not a gross usage limit; do not increase an account
limit to pass this controller. The configured guard expects a $5 gross usage limit
and enough remaining headroom for its conservative reservations. The controller
refuses live apps, invalid costs or uncommitted executable inputs.

```sh
.venv-lem-ii/bin/python -m lem_ii.launch_smoke --preflight .runtime/private-preflight.json
```

This private path is also the controller default. Keep account identifiers, live
preflight evidence, provider usage reports, attempt ledgers and billing-based
projections outside Git. No live account or invoice evidence is supplied by this
public repository.

The fixed smoke is one developmental-topic, answer-responsive24-turn session and
a two-turn repetition with the same seed/first prompt to verify reset. They have
new smoke-only IDs and are permanently ineligible for scientific analysis.
No class prediction or topology advantage is an acceptance requirement.

All invocations of this task must use the same preserved
`.runtime/technical-budget.json`. **Never delete, relocate or reset it to obtain
more attempts; do not launch from a second checkout with a fresh ledger.** Each
attempt permanently consumes a$2 reservation even after failure. At most two
attempts are allowed, one at a time, leaving at least$1 additional task reserve.
The estimated per-attempt envelope is $1.76661144, including:

- one L4 at $0.000222/second;
- CPU limit2 physical cores and RAM limit16GiB, together$0.00006172/second;
- at most900 seconds startup plus1800 seconds execution and2 seconds idle;
- $0.50 CPU image-build allowance and$0.50 price/termination/storage reserve.

The client watchdog includes image construction and stops an overlong build at
900 seconds. An absolute2700-second attempt deadline, remote1800-second function
timeout,26-turn/3328-output-token/400000-two-pass-prefill-token guards and zero
application retries bound compute. An atomic Modal Dict claim prevents a
preempted input from repeating model work. A failed termination check leaves
the ledger active for investigation. Termination retries never repeat inference.
No persistent model/output Volume, scheduled function, minimum warm container,
region premium or non-preemptible premium is used. The small task-owned claim
dictionary remains as replay protection. Estimates are not cent-exact billing
guarantees; provider metering can lag.

Raw output is returned into `.runtime/<attempt-id>/results/` and excluded from
Git. Python failures return partial checkpoints; an infrastructure crash can lose
ephemeral partial output and remains an explicitly failed attempt. No automatic
resume invents missing observations. Manifests bind source commit/files,
configuration, tokenizer/template, layers, seeds, generation, hardware and
versions. Every turn has checksummed visible text, separate policy audit and
state array. Incomplete/corrupt/mixed checkpoints fail validation.

```sh
.venv-lem-ii/bin/python -m lem_ii.summarize_smoke .runtime/ATTEMPT_ID/results --billing-report .runtime/private-billing.json --output .runtime/private-technical-summary.json
```

The full summary and its console output may contain private provider data; keep
them local. The checked-in public technical record contains only explicitly
selected technical measurements and source hashes.

This independently validates saved checkpoints and recomputes real-state static,
time, geometry and topology features locally. This smoke-only computation uses
full-dimensional per-vector L2 normalization; it is explicitly distinct from
the development-fitted scientific PCA8/scaling path tested on fixtures. It fits
no scientific transform or classifier on smoke observations.

## Later study execution

The prepared plan contains metadata and seeds only:

```sh
.venv-lem-ii/bin/python -m lem_ii.study plan --output .runtime/study-plan.json
.venv-lem-ii/bin/python -m lem_ii.study --help
```

After a separate authorization and cost review, `study collect` operates on an
already provisioned compatible CUDA GPU with explicit run limits; it does not
provision Modal resources. It requires an explicit externally enforced provider
resource timeout at or below the run deadline plus60 seconds. That provider stop
must be independently verified before a future run; Python's alarm cannot
guarantee interruption of a hung CUDA call or release a rented GPU. The currently
tested Modal smoke has its own enforced function timeout and app-stop controller.
`analyze-development-validation` reads the complete
validated72-session set, executes the same analysis path and freezes all four
endpoint artifacts. `confirm` consumes all36 reserved confirmation sessions and
all four frozen artifacts once, without fitting or tuning. Current gates fail
before model/data access. Enabling a gate is an authorization record, not itself
a permission grant. The smoke controller can never launch those study modes.

Order and label controls reuse the saved study contexts/states and add no GPU
dialogues. The two-turn reset control belongs to this technical budget. The
private planning projection counts108 model loads conservatively,2592 extra
extraction prefills, full measured24-turn context growth, and a separate CPU
analysis/build allowance. Private measured usage and account-specific projections
are not published here. One smoke topic cannot establish an across-topic cost
distribution.

Sources checked before execution: [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct),
[Modal prices](https://modal.com/pricing), [Modal budgets](https://modal.com/docs/guide/budgets).
