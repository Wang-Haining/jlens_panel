# Dirty-run proof of concept

## Question

Can a pre-speech readout identify a task-critical bridge concept that an agent
does not include in its initial message, and can targeted clarification improve a
second agent's final answer?

This run is exploratory. It decides whether a fresh, preregistered confirmatory
study is warranted; its test split is not a paper test set.

## Scope

- Model: `Qwen/Qwen2.5-7B-Instruct`, pinned to revision
  `a09a35458c702b33eeacc393d103063234e8bc28`.
- Data: deterministic synthetic A-to-B bridge chains with 16 balanced,
  tokenizer-validated single-token concepts. Agent A derives each bridge from
  a split-specific clue; no candidate string is present in A-visible input.
- Read point: last input position at assistant turn start.
- Readouts: J-lens, logit lens, raw residual probe, next-token logits, and a
  separate text-only classification prompt.
- Clarification conditions: generic, J-targeted, best non-J, and oracle.
- Seeds: 17, 29, and 43 for generated messages and receiver answers.

Personas, 30B models, natural benchmarks, activation editing, routing, tools,
and multi-round committees are out of scope.

## Execution order

1. Refit the lens from 100 neutral prompts using the pinned upstream commit.
2. Reproduce upstream multihop and association readout evaluations.
3. Generate 1,000/200/300 train/dev/test bridge examples.
4. Verify every configured bridge string is a unique single tokenizer token.
5. Extract one canonical-layer residual and candidate-only scores per example.
6. Fit the raw probe on train only; select the best non-J method on dev only.
7. Run a 20-item end-to-end smoke on one H100.
8. Run the dirty test only after inspecting smoke outputs and disk usage.

## Causal contract

Every clarification branch reuses the same cached initial Agent A message. The
primary omitted-concept subset is defined by preregistered lexical matching on
that common, pre-treatment message before clarification condition assignment.
This dirty-run audit is not a semantic-absence claim. All-items
intention-to-treat results are reported alongside the subset analysis.

The targeted arms explicitly present a readout-selected label to Agent A. Their
utility estimand is therefore **readout-guided query routing**: whether asking A
about that label elicits a relation that helps B. It is not evidence that the
label was recovered as private knowledge, and it does not separate relation
recovery from label echo. A confirmatory utility study would add a direct-to-B
label-injection control and score clarification relation correctness.

## Go/no-go gates

- Upstream reproduction and deterministic no-op checks must pass.
- Sender direct bridge accuracy must be at least 0.70.
- At least five unique smoke items must omit the gold bridge; otherwise revise
  the communication stressor before running the dirty test.
- Oracle clarification must improve receiver accuracy by at least 0.15.
- J-targeted clarification must improve over generic clarification by at least
  0.05 to justify a confirmatory utility study.
- J-specific superiority is claimed only if J-lens beats strong zero-shot and
  supervised non-J baselines on the locked split.

## Resource guardrails

- Slurm account: `group-jasonclark`.
- Start with one H100; use one four-H100 node only after smoke review.
- A second four-H100 node is reserved for seed/condition parallelism.
- Run artifacts have a 20 GB hard target; full-vocabulary logits and full
  generation trajectories are never persisted.
- Tempest filesystem checks run before model loading and during long loops.
- No job may span the scheduled 2026-07-25 through 2026-08-02 maintenance.
