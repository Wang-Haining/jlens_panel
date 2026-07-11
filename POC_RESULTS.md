# Dirty-run proof-of-concept results

Run date: 2026-07-10. This document records an exploratory development run,
not a confirmatory paper result. The fixed 20-item smoke and the 300-item test
split are now development-contaminated and must not be reused as a final test
set after method changes.

## Reproducibility anchors

- Code for the corrected v3 run: `11b42188b5314d9e29199c7f74670d81bbe52def`.
- Model: `Qwen/Qwen2.5-7B-Instruct` at
  `a09a35458c702b33eeacc393d103063234e8bc28`.
- Upstream Jacobian Lens: `581d398613e5602a5af361e1c34d3a92ea82ba8e`.
- Fitted lens SHA-256:
  `a3385c81a9672c1dfb71c27ab7e2082b4d60e2fa0f4216e6dfe555480528bf9c`.
- v3 dataset SHA-256:
  `515426e1f04a33e858095fe709265f3f83827d574a8100686f2c1652258aa727`.
- v3 score bundle SHA-256:
  `c0c586c077bd5e6b85e21de920f36b5010cdcae42fd930e33f0f644f9f141da5`.
- v3 readout summary SHA-256:
  `9e74f9b1fc53ea12c9da273844ef59feb8df2ce8dd2d4ccbc732234aa921f5a5`.
- Corrected outcome/analysis SHA-256:
  `3cbbc02f02f45e0b062f02ec3c02fa49a9aa0ae88ca6f0c325d18aa2d59507bb`
  and `1d442ec6fda757993e6482decb77119c186f04b49af2215c5d0bf4bffbf84d1e`.
- Tempest storage after the run: about 2.17 GB for the project and 3.1 TB
  free on the home filesystem.

The relevant Slurm jobs all used one H100:

| Stage | Job | State | Elapsed |
|---|---:|---|---:|
| Environment check | 4210575 | completed | 00:00:10 |
| Fit 100-prompt lens | 4210576 | completed | 00:28:27 |
| Upstream evaluations | 4210577 | completed | 00:01:06 |
| Initial v2 readouts | 4210704 | completed | 00:07:17 |
| Invalid v2 smoke | 4210751 | cancelled after scoring bug | 00:01:11 |
| Corrected v3 readouts | 4210784 | completed | 00:06:18 |
| Corrected v3 smoke | 4210785 | completed | 00:01:44 |

The v2 smoke is retained only as debugging evidence. Its answer-surface metric
rejected semantically correct typed identifiers, so none of its treatment
estimates are valid.

## Upstream method sanity check

The 100-prompt lens is a directional method reproduction, not an exact
paper-number replication. J-lens exceeded logit lens on both pinned upstream
sets, especially at larger k.

| Evaluation | Method | pass@1 | pass@5 | pass@10 | pass@50 |
|---|---|---:|---:|---:|---:|
| Multihop | J-lens | 0.1882 | 0.3853 | 0.4857 | 0.7061 |
| Multihop | Logit lens | 0.1828 | 0.3280 | 0.3799 | 0.5502 |
| Association | J-lens | 0.0098 | 0.0784 | 0.1176 | 0.2255 |
| Association | Logit lens | 0.0000 | 0.0098 | 0.0196 | 0.0784 |

Nine of 103 multihop targets and three of 102 association targets had no tested
single-token Qwen representation. They were conservatively counted as failures
for both methods. All 16 dirty-run candidates were separately verified as
distinct single tokens.

## Five-readout result

The corrected v3 test metrics were:

| Readout | Top-1 accuracy | MRR | Log loss |
|---|---:|---:|---:|
| J-lens | 0.0633 | 0.2136 | 3.6566 |
| Logit lens | 0.0667 | 0.2112 | 4.1440 |
| Raw residual probe | 0.0800 | 0.2314 | 3.3085 |
| Next-token logits | 0.1400 | 0.3072 | 4.3731 |
| Text-only behavioral probe | 0.8433 | 0.9091 | 1.2832 |

The supervised raw probe reached 0.539 top-1 accuracy on train but only 0.080
on test. The dev-selected strongest non-J method was the text-only probe; test
outcomes were not used for that selection.

J-lens predicted `iris` for all 1,500 train/dev/test examples. Its 0.0633 test
accuracy is therefore the balanced-label frequency of `iris`, not a contextual
signal. Its MRR is also approximately the random-rank expectation. The primary
J-lens readout hypothesis failed in this setup.

## Corrected clarification smoke

The corrected protocol used one cached initial message per item and seed,
finite typed/bare answer aliases, compact structured answer IDs, and a
deterministic direct-gold oracle. Clustered bootstrap intervals resampled the
20 items rather than treating the three seeds as independent items.

| Subset | Generic | J-targeted | Best non-J targeted | Direct oracle |
|---|---:|---:|---:|---:|
| ITT: 20 items, 60 item-seeds | 0.8667 | 0.7167 | 0.8667 | 1.0000 |
| Omitted: 18 items, 50 item-seeds | 0.8400 | 0.6600 | 0.8400 | 1.0000 |

On the omitted subset:

- J-targeted minus best non-J was -0.1667, with a 95% item-clustered
  bootstrap interval of [-0.3333, -0.0185].
- J-targeted minus generic was -0.1667, with interval [-0.3519, 0.0000].
- Best non-J minus generic was 0.0000, with interval [-0.0556, 0.0556].
- Direct oracle minus generic was +0.1481, with interval [0.0185, 0.3148].

## Gate decision

| Gate | Result | Decision |
|---|---:|---|
| Directional upstream method check | J-lens better at pass@5-50 | pass |
| Text-only bridge competence >= 0.70 | 0.8433 | pass |
| At least 5 omitted smoke items | 18 | pass |
| Oracle minus generic >= 0.15 | 0.1481 | **fail** |
| J-targeted minus generic >= 0.05 | -0.1667 | **fail** |

The thresholds are kept as written. The oracle gate misses by about 0.0019;
it is not rounded up or changed after seeing the result.

## Decision and next experiment

This proof of concept is a **no-go for a full dirty run and a no-go for an ACL
claim in its current form**. No second node should be used.

Two different problems need train/dev-only repair before any new held-out test:

1. The readout needs diagnosis. Leading hypotheses are a strong candidate-token
   prior, the single canonical-layer choice, and a mismatch between the generic
   128-token fitting distribution and the longer task/chat read point. A small
   layer sweep and null-prompt prior calibration are cheaper first checks than
   fitting a larger model or launching a full experiment.
2. The communication task leaves too little headroom: generic clarification is
   already 84% accurate on omitted cases, and the strong text-only target does
   not improve it. The stressor must create a meaningful generic-to-oracle gap
   without making the task artificial or leaking the answer.

If either component is revised, freeze the choice on train/dev, generate a
fresh held-out split with new item IDs, and rerun only a bounded smoke before
considering confirmatory scale.
