---
name: ml-code-workflow
description: "Use when implementing or reviewing reproducible code for the CrossSell extra NPV project: data generation, models, policy layer, evaluation, notebook, tests, artifacts, and command-line demo."
---

# ML Code Workflow

## Workflow

1. Read `README.md`, `run_experiment.py`, `demo.py`, `utils/`, and `tests/` before editing.
2. Keep the public code synthetic and deterministic by default: use explicit seeds and small demo sizes.
3. Preserve separation of responsibilities:
   - `data.py`: synthetic data, event sequences, randomized logs;
   - `models.py`: response/uplift/transformer/stacking models;
   - `policies.py`: score-to-top-N selection with `group_id` constraints;
   - `evaluation.py`: category-level and set-level IPS/SNIPS plus diagnostics;
   - `metrics.py`: model diagnostics;
   - `run_experiment.py` and `demo.py`: orchestration and artifact saving.
4. Keep public written artifacts consistent with the repo boundary:
   - `article/`: public CrossSell article and PDF, not the contest-only employer-specific version;
   - `docs/`: methodology and NDA boundary;
   - `notebooks/demo.ipynb`: executed notebook that calls shared pipeline code.
5. After changes, run:

```bash
python demo.py --bootstrap 10 --artifacts-dir artifacts/smoke
pytest
```

If `pytest` is unavailable, call the test functions directly and state that formal pytest execution was blocked by the environment.

## Acceptance Criteria

- `demo.py` completes from a clean checkout and writes readable artifacts.
- `notebooks/demo.ipynb` calls the shared pipeline; it must not duplicate hidden notebook-only logic.
- `pytest` passes without warnings when possible.
- `.gitignore` excludes generated artifacts, caches, and local notebook output.
- Final metrics are reproducible for the default seed and documented in `docs/final_demo_results.md`.
- All public files remain NDA-safe under a grep pass for internal names and forbidden terms.
- The repository remains reviewable after deleting `artifacts/`, `__pycache__`, `.pytest_cache`, and LaTeX build files.

## Review Checklist

- `data.py` owns the simulator and leakage checks.
- `models.py` owns only model fitting/scoring, not policy decisions.
- `policies.py` owns greedy top-N selection and `group_id` constraints.
- `evaluation.py` owns category-level and set-level policy evaluation.
- `run_experiment.py` orchestrates, saves artifacts, and avoids production assumptions.

## Review Rules

- Do not add heavy dependencies unless the README and requirements are updated.
- Do not make notebook-only logic; notebook must call the same pipeline code as CLI.
- Do not expose real data, endpoints, table names, or production configs.
