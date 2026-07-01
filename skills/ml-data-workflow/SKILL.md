---
name: ml-data-workflow
description: "Use when designing, reviewing, or debugging ML data for the CrossSell extra NPV project: event history, tabular features, time-based splits, randomized logs, propensities, availability/preselects, and leakage checks."
---

# ML Data Workflow

## Checklist

1. Identify the decision unit: `user_id, period_id` for a distribution and `user_id, period_id, category_id` for scoring.
2. Separate data surfaces:
   - tabular mart for boosting;
   - event history for transformer sequences;
   - randomized assignment log for uplift and policy evaluation.
3. Enforce the cutoff:
   - every feature and event must satisfy `event_time <= decision_time - data_lag`;
   - current show, current selection, current utilization, and future eligibility changes must be excluded.
4. Validate randomized logs:
   - fixed number of shown slots;
   - positive propensities;
   - no duplicate `group_id` inside one shown set;
   - treatment assignment is hard to predict from pre-decision features.
5. Verify public safety:
   - use synthetic IDs and buckets;
   - avoid real product names, exact windows, raw timestamps, internal table names, and exact values.

## Deep Checks

- For every `user_id, period_id`, there must be enough available `group_id` values to fill all slots.
- `show_propensity` must be positive for shown categories and consistent with the randomized policy.
- The same cutoff logic must be used for tabular features and transformer sequences.
- Synthetic `p0_true`, `p1_true`, and `extra_npv_true` may exist only because this is a simulator; never describe them as real production labels.
- Event fields should explain why the transformer can learn something beyond tabular aggregates: order, recency, action type, entity/category, and event value.
- Train/validation/test must be time-based. Random row splits are a leakage risk for user histories.
- Product availability and `group_id` diversity are part of the decision surface, not model labels. The model may score all user-category pairs, but the policy must select only from available categories.

## Debug Order

1. Check row counts and unique keys.
2. Check fixed slot count and unique `group_id` in shown sets.
3. Check cutoffs against `source_event_time`.
4. Check propensity distribution and effective overlap.
5. Check that synthetic ceiling has signal but does not leak into model features.

## Expected Artifacts

Produce or inspect:

- `events` sample with event fields and cutoff columns;
- `assignments` sample with `shown`, `show_propensity`, `p0/p1` in synthetic data;
- data checks table;
- short explanation of what each event family contributes to transformer and boosting.
