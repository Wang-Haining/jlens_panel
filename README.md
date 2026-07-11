# J-Lens Panel

Proof-of-concept experiments for measuring information loss at the natural-language
handoff between independently executed language-model agents.

The first milestone is deliberately narrow:

1. reproduce the public Jacobian Lens readout on a pinned 7B model;
2. compare five pre-speech readouts on a controlled bridge-concept task;
3. test whether targeted clarification improves a receiver agent's answer.

The dirty run is exploratory. Its test split is a go/no-go instrument and must not
be reused as the confirmatory paper dataset. The full protocol and stop rules are
in [DIRTY_RUN_PLAN.md](DIRTY_RUN_PLAN.md).
The completed proof-of-concept metrics, artifact hashes, and no-go decision are
recorded in [POC_RESULTS.md](POC_RESULTS.md).

The synthetic task is deliberately nontrivial: Agent A sees descriptions from
which a bridge concept must be derived, while the 16 bridge strings themselves
are absent from all A-visible input. Agent B sees only literal bridge-to-answer
relations. This prevents a context-echo baseline from passing by copying the
target word.

## Development

```bash
python -m pip install -e '.[dev,analysis]'
pre-commit install
pre-commit run --all-files
pytest
```

GPU and Tempest dependencies are isolated in the `gpu` extra. Large artifacts are
never committed. Every run must write an immutable manifest and pass the disk-space
guard before model loading.

For a CPU-only data check:

```bash
python scripts/generate_dirty_data.py \
  --output-dir data/generated/dirty_v3 \
  --train-size 16 --dev-size 16 --test-size 16
```

Tempest jobs are intentionally staged rather than submitted as one automatic
pipeline; see [runs/README.md](runs/README.md). Each gate is reviewed before the
next H100 allocation.

## Upstream method

The reference Jacobian Lens implementation is pinned to Anthropic commit
`581d398613e5602a5af361e1c34d3a92ea82ba8e`.
The dirty run uses Qwen2.5-7B-Instruct revision
`a09a35458c702b33eeacc393d103063234e8bc28`, resolved from the Tempest cache.
