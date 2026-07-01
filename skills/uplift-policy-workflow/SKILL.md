---
name: uplift-policy-workflow
description: "Use when designing, reviewing, or debugging uplift modeling and policy evaluation for the CrossSell extra NPV project: S-learner, randomized logs, category-level IPS, set-level IPS, propensities, overlap, and top-N policy constraints."
---

# Uplift Policy Workflow

## Checklist

1. Define the causal objects:
   - unit: user-period-category for scoring, user-period for a top-N set;
   - treatment: category shown vs not shown;
   - outcome: utilization or normalized value;
   - policy: top-N category selection under availability and `group_id` constraints.
2. Train S-learner:
   - one outcome model with `shown_feature`;
   - score each candidate twice with `shown_feature=1` and `shown_feature=0`;
   - compute `uplift = p1 - p0`;
   - compute `extra_npv = NPV * uplift`.
3. Evaluate policies:
   - category-level IPS/SNIPS for practical overlap;
   - set-level IPS/SNIPS for strict top-N policy evaluation;
   - ESS, match rate, bootstrap intervals, and DR-style diagnostics.
4. Check assumptions:
   - randomization respects the same product constraints as policy;
   - propensities are positive and reconstructable;
   - treatment is not predictable from pre-decision features;
   - no future events or target-window proxies enter features.
5. State limitations:
   - category-level IPS ignores some within-set interactions;
   - set-level IPS has fewer matches and higher variance;
   - DR-style diagnostics are not a full production DR estimator without cross-fitting.

## Evaluation Order

1. Confirm that the randomized assignment used the same eligibility and `group_id` constraints as the evaluated policy.
2. Build policy assignments for all strategies on the same candidate set.
3. Compute category-level IPS/SNIPS for overlap-rich diagnostics.
4. Compute set-level IPS/SNIPS as a stricter but noisier top-N check.
5. Compare against synthetic `true_extra_npv_value` only in the simulator.
6. Inspect ESS and match share before trusting a large IPS/SNIPS difference.

## Common Failure Modes

- Using response probability as uplift.
- Evaluating on categories that were not eligible for the user.
- Forgetting that `NPV * p1` and `NPV * (p1 - p0)` answer different business questions.
- Calling the synthetic ceiling a deployable strategy.
- Reporting category-level IPS as if it fully captures set-level interactions.
- Letting the policy re-rank unavailable categories after scoring all user-category pairs.

## Output

Summarize whether extra NPV beats response/random on synthetic truth, whether IPS evidence is stable, and which assumptions require more data or production validation.
