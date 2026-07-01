---
name: ml-model-improvement
description: "Use when improving or reviewing model quality for the CrossSell extra NPV project: boosting baselines, transformer sequence model, stacking, calibration, uplift alignment, and model comparison."
---

# ML Model Improvement

## Workflow

1. Start from baselines:
   - random;
   - NPV-only;
   - response ranking by `p(utilization) * NPV`;
   - extra NPV ranking by `NPV * (p1 - p0)`;
   - synthetic true-effect upper bound.
2. Check model diagnostics:
   - AUC/AP/Brier for observed outcomes;
   - uplift alignment against synthetic truth;
   - top-decile true extra NPV;
   - policy value through category-level and set-level IPS/SNIPS.
3. For transformer improvements, preserve the sequence contract:
   - only pre-decision events;
   - event family/type/entity/category/group/time/value inputs;
   - fixed length with padding and attention mask;
   - category/action embedding joined with user representation.
4. For stacking, describe it honestly:
   - it uses transformer predictions as additional features for boosting;
   - do not call it out-of-fold unless the code generates out-of-fold predictions.
5. Prefer robust improvements over metric chasing: better leakage checks, calibration, stronger baselines, clearer diagnostics, and stable seeds.

## Model Review Questions

- Does a higher AUC translate into better policy value, or only better prediction?
- Does extra NPV improve uplift alignment, not just response probability?
- Does the transformer benefit from sequence-specific information such as recency and action order?
- Does stacking add value without being described as out-of-fold when it is not?
- Are model comparisons made on the same time-based test split?
- Are synthetic improvements explained as simulator behavior rather than production metrics?

## Transformer-Specific Checks

- Event tokens include action type, entity/category, group, recency, position, and value.
- Padding and masks are handled explicitly.
- User representation is joined with category/action representation before ranking.
- For uplift scoring, the action/item representation must distinguish `z=1` from `z=0`; otherwise the transformer cannot produce an uplift difference.
- The synthetic data-generating process contains sequence-dependent uplift; otherwise the transformer has no fair reason to win.

## Policy-Level Success

Prefer results where:

- random is clearly below model strategies;
- the synthetic ceiling is best or near-best on true hidden effect;
- extra NPV strategies beat or match response strategies on uplift-oriented diagnostics;
- confidence intervals, ESS, and match rates are inspected before making strong IPS claims.

Do not force the synthetic demo to exactly reproduce every closed-work ordering. If the article says the closed contour has stronger transformer results, the public code should demonstrate the mechanism and remain honest about synthetic evidence.

## Output

Report which metric improved and whether the improvement is predictive, causal, or policy-level. Never present synthetic metrics as production results.
