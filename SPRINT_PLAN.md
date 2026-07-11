# Dirty-run postmortem and diagnostic sprint plan

Status: frozen 2026-07-10, after the v3 no-go recorded in `POC_RESULTS.md`.
Drafted with Claude from a full read of this repo; decisions and thresholds
owned by Haining. This document is the driving spec for the next execution
round; a coding agent (Codex) should be able to implement the sprint from
Sections 5–8 without further context.

Non-goals: this repo does not touch biology or security; the ACL design doc
(`silent_committee_design.md`, kept outside this repo) is the parent program.

---

## 1. What happened

We ran the narrowed proof of concept from `DIRTY_RUN_PLAN.md` on
`Qwen/Qwen2.5-7B-Instruct` (pinned) with a 100-prompt lens on one H100
(~40 min total GPU). Full numbers, hashes, and job IDs are in
`POC_RESULTS.md`; the load-bearing facts are:

- Upstream directional reproduction passed (J-lens > logit lens on the pinned
  multihop and association sets, especially at pass@5–50). The lens works in
  its native regime.
- At our read point, J-lens predicted `iris` for all 1,500 examples. Its test
  top-1 (0.0633) is the balanced label frequency; its MRR (0.2136) equals the
  random-rank expectation for 16 candidates (H16/16 = 0.2113). Logit lens MRR
  0.2112 — also exactly random.
- The supervised raw residual probe: 0.539 train / 0.080 test top-1 (chance
  0.0625) under cross-template generalization. The probe is the information
  ceiling for every linear readout at that (position, layer); the ceiling is
  at chance.
- Next-token logits: 0.140 top-1 (~2.2x chance) — the only model-side readout
  with real signal.
- Text-only behavioral probe: 0.843 — the model can compute the bridge when
  asked.
- Utility smoke: generic clarification already 0.84 on omitted items; oracle
  gate missed by 0.0019; J-targeted (constant wrong label) actively hurt
  (-0.167). Gates were kept as written. No-go.

## 2. Diagnosis, ranked by certainty

**D1 (certain, code-level): the read point is a chat-template token.**
`render_chat(..., add_generation_prompt=True)` ends the prompt with the
assistant-header template; `capture_pre_speech` reads `activations[layer][0, -1]`
— the last template token, nearly identical across all examples. The residual
there is dominated by token identity + position, so any linear map of it is
near-constant across examples.

**D2 (certain, code-level): no prior calibration.** `unembed(transport(h))`
raw logits are restricted to candidates and argmaxed. Candidate logits carry a
context-independent prior (unembedding geometry / frequency). Near-constant h
(D1) times an uncalibrated static prior = one winner forever. That winner was
`iris`. Both lens MRRs sitting exactly at the random expectation is the
signature of zero contextual correlation, not weak correlation.

**D3 (near-certain, task-level): the task never forces Agent A to compute the
bridge before speaking.** B decodes the clue itself (generic clarification is
84% accurate), so A's optimal policy is to relay the clue; the label reduction
(clue -> "amber") is legitimately deferred to B. "Poised to say" presupposes
the computation has occurred; here it hadn't. Evidence: the supervised probe
ceiling at the read cell is chance. This is not "J-lens lost"; it is "there
was nothing at that cell to read."

**D4 (secondary, do not buy before fixing D1–D3):** single canonical layer;
100-prompt lens; 7B scale (latent multihop precomputation strengthens with
scale); plain-text fitting distribution vs chat-formatted application.

**Positive residue:** next-token 0.14 says signal appears at verbalization
time (some A messages do name the bridge; final-layer logits at the template
tail see imminent words). This points at decode-time reading as the viable
regime.

## 3. What this run does and does not refute

Refuted: single-layer, uncalibrated, template-tail, pre-speech readout on a
7B model in a task that does not force pre-speech computation.

Not refuted: the program premise ("agent workspaces are readable around the
moment of speech"). The dirty run was effectively an instrument-validation
(E0b) attempt carrying two confounds of its own (unforcing task, template
read point) plus a scoring bug-equivalent (no calibration). A clean negative
requires the sprint below to fail.

## 4. Sprint objective, hard rules, and preregistered gates

**Objective.** Determine whether any (position, layer, calibration) cell
carries linearly decodable bridge information, and whether calibrated J-lens
approaches the supervised-probe ceiling at the best cell.

**Hard rules (Codex: these are inviolable):**
1. Train/dev only. The v3 300-item test split and the fixed 20-item smoke are
   development-contaminated; never evaluate on them again. A fresh held-out
   split (new item IDs, new fingerprint) is generated only after all method
   choices are frozen.
2. Thresholds below are frozen at the commit that adds this file. No edits
   after seeing results.
3. Never persist full-vocabulary logits or full generation trajectories
   (existing storage contract). Respect `storage:` guards in configs.
4. Keep the repo's provenance discipline: pinned revisions, sha256 of every
   artifact, manifests via `provenance.py` patterns, atomic writes.
5. No jobs during the 2026-07-25 → 2026-08-02 Tempest maintenance. Slurm
   account `group-jasonclark`. Copy sbatch header conventions from
   `runs/03_extract_score_readouts.sbatch`.

**Gates (dev split, 16-way, chance = 0.0625):**
- **G-INFO:** max over cells of probe dev top-1 >= 0.25 (4x chance).
- **G-CONST:** after calibration, no single label accounts for > 50% of
  J-lens predictions, and prediction entropy > 1.5 bits.
- **G-LENS:** at the best G-INFO cell, calibrated J-lens dev top-1 >=
  0.5 x probe dev top-1 at that cell, and > calibrated logit lens.
- **G-FORCE:** on the forcing variant (S3), probe ceiling >= 0.25 at the
  post-computation read point.

## 5. Sprint tasks and scaffolding

Execution order: S0 -> S2-offline -> S1 -> (S3 if S1 fails or for
confirmation) -> S4 only if S1+S3 both fail. S0 and S2-offline need no GPU.

### S0 — Offline calibration sanity on existing v3 artifacts (30 min, CPU)

The v3 artifacts already store per-candidate raw scores for jlens /
logit_lens / next_token. Before any new capture:

- New script `scripts/recalibrate_v3.py`: load stored candidate scores,
  subtract per-candidate means estimated on the **train** split only
  (train-mean calibration is a cheap stand-in for null-prompt calibration),
  re-score dev with the existing metrics.
- Expected: constant-`iris` disappears (G-CONST); accuracy likely stays near
  chance (because of D1/D3). If accuracy jumps >= 0.25 on dev, the whole
  failure was scoring, and we go straight to a calibrated bounded smoke.
- Output: `analysis/recalibrated_v3_summary.json` with both raw and
  calibrated dev metrics per method.

### S1 — Position x layer sweep with the probe as information detector

**New modules:**

```
src/jlens_panel/sweep/
  __init__.py
  positions.py     # position resolvers, pure functions + fail-closed errors
  capture.py       # multi-layer, multi-position capture incl. decode steps
  probe_sweep.py   # per-cell probes and the results table
scripts/sweep_positions.py
runs/05_position_sweep.sbatch      # 1x H100, expect <= 2 h wall
config/sprint.yaml
```

**Position resolver contract (`positions.py`).** Given the tokenizer, the
example, and the rendered prompt, return integer token indices for:

| name | definition |
|---|---|
| `template_tail` | last position of the `add_generation_prompt=True` render (current read point; kept as the negative control) |
| `content_last` | last token of the raw user content, i.e. the position immediately before the first `<|im_end|>` that closes the user turn |
| `clue_last` | last token of the gold agent_a_fact string (locate its character span in the rendered prompt, map via `return_offsets_mapping=True`; the fact string is unique within the prompt — fail closed if 0 or >1 matches) |
| `meanpool_content8` | mean of residuals over the last 8 user-content token positions (a pooled feature, not an index; implement in capture) |
| `decode_t` for t in {1, 2, 4, 8} | residual at the t-th generated token during greedy decoding (max_new_tokens = 8, temperature 0) |

Add unit tests in `tests/test_positions.py` covering: exact index vs a
hand-tokenized fixture, the fail-closed paths, and a Qwen chat-template
regression fixture (store the rendered string, not the tokenizer, in the
test).

**Capture (`capture.py`).** One forward pass per example captures **all
fitted `lens.source_layers`** (not one canonical layer) at all static
positions via `jlens.hooks.ActivationRecorder`; decode positions come from a
stepwise greedy loop with the recorder active (8 steps). Store fp16 CPU
residuals per (example, position, layer) plus candidate-restricted raw scores
for jlens and logit lens per cell. Budget check: 1,200 train+dev examples x
~28 layers x 9 position slots x 3584 dims x 2 bytes ≈ 2.2 GB — within the
20 GB run limit, but verify with `RunByteBudget` before the loop and fail
closed.

**Probes (`probe_sweep.py`).** Per cell: reuse `RawResidualProbeReadout`
(LogisticRegression, C=1.0 default plus C in {0.01, 0.1} as a regularization
mini-grid; pick on dev). Fit on train, evaluate on dev. Output one tidy CSV:
`analysis/sweep_results.csv` with columns
`position,layer,method,C,dev_top1,dev_mrr,dev_logloss,n_train,n_dev` where
method ∈ {probe, jlens_raw, jlens_calibrated, logitlens_raw,
logitlens_calibrated}. Also emit `analysis/sweep_heatmap.json`
(position x layer -> probe dev_top1) for plotting.

### S2 — Null calibration as a first-class artifact

- New module `src/jlens_panel/calibration.py`:

```python
@dataclass(frozen=True)
class NullCalibration:
    method: str                 # "jlens" | "logit_lens"
    position_type: str          # one of the resolver names
    layer: int
    candidate_means: dict[str, float]
    candidate_stds: dict[str, float]
    n_null_prompts: int
    provenance: Mapping[str, object]
    # save()/load() with sha256, atomic write, fail-closed schema checks

def calibrate(scores: Mapping[str, float], cal: NullCalibration,
              *, mode: str = "center") -> dict[str, float]:
    # "center": score - mean ; "zscore": (score - mean) / std (std floor 1e-6)
```

- Null prompts: 200 neutral prompts drawn with a fixed seed from the same
  corpus loader `scripts/fit_lens.py` uses, rendered per position type
  (chat-wrapped for `template_tail`/`decode_*`, raw text for content types).
  Build with `scripts/calibrate_null.py`; artifact under
  `artifacts/calibration/` with a manifest.
- Wire into `probe_sweep.py` so every lens cell reports raw and calibrated
  variants. Calibration is estimated once and never refit per split.

### S3 — Forcing variant (the corrected E0b), schema `synthetic-bridge-v4`

New module `src/jlens_panel/data/forcing_bridge.py`, mirroring the v3
generator's validation style (frozen dataclasses, leakage audits via
`contains_candidate_word`, disjoint split resources — reuse by import, do not
copy-paste).

Protocol per item (**compute-then-speak**):
1. Turn 1 (private scratchpad): A answers the existing
   `agent_a_probe_question` — "reply with only the concept". Behavioral
   verification: the reply must match the gold bridge (case-folded); items
   where it does not are excluded from the primary analysis and counted.
2. Turn 2: A writes the handoff sentence under the added constraint "do not
   use the key term itself". Lexical audit stratifies overt vs silent
   handoffs.
3. Read points: `post_scratchpad` (last content token of A's own turn-1
   answer, before turn 2 begins) and `decode_t` of the turn-2 handoff.
   At `post_scratchpad` the computation has provably occurred — if calibrated
   J-lens cannot read the concept even there, the lens (not the task) is the
   limit.

Sizes: 800 train / 200 dev, same 16 candidates, seeds via the v3 derivation
pattern with a new schema tag. Script `scripts/run_forcing_sweep.py` reuses
S1 capture and probes on the two new read points x all layers.
`runs/06_forcing_sweep.sbatch`, 1x H100.

Note the confound to keep honest in analysis: in v4 turn-2, the gold concept
token exists earlier in A's own context (its scratchpad answer), so
`post_scratchpad` reads test "can the lens see a recently computed, contextually
present concept" — that is exactly the instrument question, but it is a weaker
claim than v3's derived-and-absent setting. Report both framings.

### S4 — Scale check (conditional; run only if S1 and S3 both fail G-INFO/G-FORCE)

- Config `config/sprint_32b.yaml`: `Qwen/Qwen2.5-32B-Instruct` (pin the
  revision at run time), refit lens with the same recipe (100 prompts is
  acceptable given the 7B upstream check passed; 250 if the fit job is cheap),
  rerun S1 on the top-5 cells by 7B probe ordering plus `post_scratchpad`.
- One 4x H100 node, single job, hard-capped. This is the last lever before
  the kill branch.

## 6. Decision tree

```
S0 calibrated dev top1 >= 0.25 ──yes──> scoring was the failure; calibrated
        │ no                             bounded smoke, then back to main plan
S1 G-INFO pass? ──yes──> G-LENS pass? ──yes──> GO: adopt best read point
        │ no                    │ no            (likely content/decode-time),
        │                       │               freeze, fresh held-out split,
        │                       │               bounded confirmatory smoke
        │                       └────> lens-limited: information exists but
        │                              J-lens cannot read it here. Options:
        │                              dialogue-distribution lens refit (S5,
        │                              stretch) or report as a mechanism-
        │                              identified negative.
S3 G-FORCE pass? ──yes──> rerun G-LENS at post_scratchpad (same branch above)
        │ no
S4 32B: any cell >= 0.25? ──yes──> scale-limited: program continues on 30B+
        │ no
KILL: pre/mid-speech linear readability of derived concepts fails on this
model family with calibration, forcing, and a full sweep. Pivot to the
preregistered negative-methods paper; downgrade J-lens components in the
clinical (jd) project accordingly.
```

## 7. Utility-task redesign (parked until a readout passes)

The v3 communication task has no headroom (generic 0.84, oracle 1.00). When
the readout question is settled, revise the stressor before any utility rerun:
- Hard channel budget for A (<= 6 words), making clue relay infeasible and
  the label the natural compression.
- Distractors 12–15; and/or require two-clue composition for chain selection.
- Headroom targets to re-gate: generic <= 0.60; oracle - generic >= 0.25.
- Add the matched direct-to-B control for predicted labels (already flagged
  in `DIRTY_RUN_PLAN.md`'s causal contract) to separate relation recovery
  from label echo.

## 8. Codex execution notes

- Setup: `pip install -e '.[analysis]'`; run `pytest` before and after each
  task; `pre-commit run --all-files` must pass.
- Match house style: frozen dataclasses, fail-closed validation with typed
  errors, lazy GPU imports (`import torch` only inside functions that need
  it), dependency-light core modules importable without ML runtimes.
- Every artifact: sha256 + manifest via existing `provenance.py` helpers;
  atomic writes via the existing patterns in `readouts/artifacts.py`.
- Definition of done, per task:
  - S0: script + summary JSON + one paragraph appended to this file under
    "## 9. Results log" with raw vs calibrated dev numbers.
  - S1: modules + tests green + `sweep_results.csv` + heatmap JSON + sbatch
    that ran to completion on Tempest + results-log paragraph.
  - S2: calibration artifacts + integration + G-CONST check in the log.
  - S3: v4 generator + tests (leakage, verification, strata) + sweep on the
    two read points + log paragraph with the verified-scratchpad rate.
  - S4: only via explicit human approval after S1+S3 results are logged.
- Do not: touch `test.jsonl` or the 20-item smoke; alter any gate; persist
  vocab-sized tensors; exceed the 20 GB run budget; schedule across the
  maintenance window; select thresholds or cells on anything but dev.

## 9. Results log

(Append one dated paragraph per completed task; never edit prior entries.)
