# Public technical verification — 7 September 2026

The real **Qwen/Qwen2.5-3B-Instruct** technical smoke passed. This record contains
technical measurements only. Private account identifiers, provider reports,
attempt records and account-specific financial calculations remain local.

- Model/tokenizer revision: `aa8e72537993ba99e69dfaafa59ed015b17504d1`.
- Unchanged, unquantized **BF16** weights; **SDPA** attention; NVIDIA L4.
- **26 turns:** one answer-responsive 24-turn smoke dialogue and a separate
  two-turn session reset check. Smoke observations are ineligible for study analysis.
- Full 24-turn extraction plus generation: **144.8 seconds**; worker duration:
  **220.6 seconds**.
- Input context grew from **82 to 4,237 tokens**, within the 8,192-token cap
  including the reserved answer. No context truncation occurred.
- Measured the last token of the assistant generation prefix before each answer.
  Saved hidden-state indices **0, 18, 36**, each with **2,048 coordinates**;
  index **36** is the prespecified primary layer.
- Session reset: identical first context, answer tokens and extracted states.
- Saved real states passed all static, geometric, temporal and topological feature
  calculations and independent local recomputation. Technical feature checking
  used the explicitly labelled smoke-only normalization; no classifier or
  scientific preprocessing was fitted to these observations.
- Local validation of the execution/review snapshot: **66 pipeline tests passed**,
  **0 skipped**, plus the **5 existing LEM-I regression tests**. These are local
  checks, not GitHub CI.

The scientific study and its independent confirmation phase were **not executed**.
No prediction advantage, personality, identity, hidden cognition or attractor
result is claimed.

The [technical JSON](evidence/technical-smoke.json) supplies detailed measurements,
model/software metadata and the original executed source hashes for the preserved
core files. Its execution reference
`cf9e6e00eb3f9589e301de248b852df61a2ca8a1` identifies a **private local execution
snapshot**, not a commit in the public publication ancestry. Private records and
that history are retained locally. Publication changes generalize workspace
configuration and test fixtures and remove private operational evidence; the
model adapter, simulator, scientific configuration, feature extraction, collection
and evaluation code remain unchanged.

The next scientific execution requires separate authorization and a private review
of current costs, account limits and an independently enforced GPU resource timeout.
The public protocol retains its prespecified $5 technical and $30 total planning
bounds and public tariffs; this record publishes no actual invoice or usage values.
