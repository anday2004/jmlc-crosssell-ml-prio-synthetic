---
name: jmlc-adaption-council
description: Use when adapting a real work ML project into a Junior ML Contest public artifact under NDA. Reviews article, README, notebook, code, diagrams, and claims for ML quality, causal validity, product clarity, reproducibility, personal contribution, and confidentiality.
---

# JMLC Adaptation Council

## Workflow

Review the artifact through five lenses, then produce a concise chairman summary.
Start by identifying the exact artifact version under review: article, README,
notebook, code, final metrics, or GitHub-ready folder.

1. ML Scientist: check target, features, baselines, model comparison, validation split, metrics, and whether predictive quality is confused with policy value.
2. Causal Reviewer: check treatment, outcome, randomization, propensities, overlap, leakage, IPS/SNIPS, and whether uplift claims are supported.
3. Product Reviewer: check business value, user story, constraints, why the model matters, and whether the contribution is legible to an admissions committee.
4. Engineering Reviewer: check reproducibility, repo structure, notebook/demo, tests, artifacts, dependency setup, and whether the public reconstruction can be run end to end.
5. NDA Reviewer: check that no real data, internal names, exact metrics, table schemas, partner names, screenshots, endpoints, or production details are exposed.

## Required Checks

- Public GitHub artifacts use the NDA-safe CrossSell wording; the separate contest submission may contain the more specific employer/project wording if the applicant intentionally keeps it outside the public repo.
- Article claims match what the public code actually demonstrates.
- Real closed-work claims are clearly separated from synthetic public evidence.
- All baselines are named consistently: random, NPV-only, response value, extra NPV, and synthetic oracle.
- The contribution section states what the applicant did without erasing team/product collaboration.
- The repo can be reviewed without access to employer systems.
- Any mention of AI use explains the applicant's workflow, not a vague "AI helped me".

## Red Flags

- "Production result" language attached to synthetic metrics.
- Causal claims without randomized logs, propensities, or overlap discussion.
- Transformer claims that do not explain why event order matters.
- Confusing response ranking `p(utilization) * NPV` with uplift ranking `NPV * (p1 - p0)`.
- NDA leaks through screenshots, internal names, exact values, hidden URLs, or real schemas.
- A public repo that only contains text but no runnable reconstruction.

## Output

Use this structure:

```markdown
Decision: ready / needs changes / blocked

Findings:
- P0: ...
- P1: ...
- P2: ...

Chairman Summary:
...

Next Steps:
1. ...
2. ...
3. ...
```

Prefer concrete file references and exact claims to fix. Do not suggest adding confidential details to make the project stronger.
