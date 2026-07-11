# Contributing

All changes must pass the repository-provided pre-commit configuration and the
unit test suite before commit:

```bash
pre-commit run --all-files
pytest
python -m compileall -q src scripts tests
```

Keep heavyweight scientific imports behind execution boundaries so data,
configuration, and CLI help remain usable without a GPU environment. Never
commit model weights, generated datasets, residuals, logs, or run outputs.

Experiment outputs must be append-only or atomically replaced, keyed for resume,
and accompanied by an immutable manifest containing configuration and Git hashes.
