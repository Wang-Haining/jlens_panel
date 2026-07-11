# Tempest jobs

All jobs use the `group-jasonclark` Slurm account and fail before model loading
when the Tempest disk guard is crossed. The dirty run starts on one GPU and may
scale to one four-H100 node only after the 20-item smoke passes. A second node is
reserved for seed/condition parallelism after review of the first outputs.

No job should be submitted across the scheduled Tempest maintenance window from
2026-07-25 through 2026-08-02.

The initial proof of concept reuses the already validated `~/jd/.venv` without
modifying it. `PYTHONPATH` points at this repository. A project-local environment
will be frozen before any confirmatory run.

Bootstrap from the tracked GitHub branch before any submission:

```bash
git clone --branch codex/dirty-run-poc \
  git@github.com:Wang-Haining/jlens_panel.git /home/g91p721/jlens_panel
cd /home/g91p721/jlens_panel
bash runs/provision_tempest.sh
```

Provisioning pins the upstream evaluation checkout, copies the already vetted
fit corpus without changing the shared `~/jd/.venv`, and creates `logs/` before
Slurm tries to open `#SBATCH --output`. Submit jobs through
`bash runs/submit_sbatch.sh runs/NN_job.sbatch` so the log precondition is
always satisfied.

Run stages in order, stopping to inspect each gate:

1. `00_check_environment.sbatch` validates CUDA, pinned package/model revisions,
   and all 16 single-token candidates.
2. `01_fit_lens.sbatch` performs the clean 100-prompt fit.
3. `02_evaluate_upstream.sbatch` runs the official multihop and association
   evaluations. Do not submit stage 3 until this reproduction is reviewed.
4. `03_extract_score_readouts.sbatch` generates the frozen data, extracts the
   compact states, fits the train-only probe, and freezes the best non-J method
   on development log loss. Review the sender competence and readout metrics.
5. `04_clarification_smoke.sbatch` runs only the first 20 test items under the
   four paired conditions and three seeds, then computes clustered intervals.
6. `05_position_sweep.sbatch` is the current diagnostic sprint: it builds
   resumable 200-prompt null calibrations, then captures and analyzes only the
   frozen v3 train/dev splits across all registered positions and layers.

Stage 4 is development-contaminated historical scaffolding and must not be
rerun. Stage 5 never reads it or the v3 test split. A fresh held-out evaluation
is generated only after the diagnostic method choices are frozen.
