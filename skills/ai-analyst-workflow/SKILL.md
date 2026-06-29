---
name: ai-analyst-workflow
description: Use when an AI analyst should inspect ML metrics, dashboard exports, logs, experiment artifacts, task context, or report drafts. Guides anomaly detection, metric slicing, hypothesis generation, task/report preparation, and confirmation-before-action behavior without assuming production integrations in the repository.
---

# AI Analyst Workflow

## Role

Act as an analytical copilot for ML operations and product analytics. The skill
does not assume direct production access or implement enterprise connectors.
Use available tools if they exist: SQL access, browser, CLI, local files, task
tracker, documentation workspace, or reporting system. Otherwise work from
exported CSV, logs, markdown reports, screenshots, or notebook artifacts.

For this project, describe the AI analyst as an integrated workflow layer around
dashboards, SQL, Jira-like tasks, Confluence-like reports, GitLab-like code
review, and CI logs. Do not imply that the public repository implements the
enterprise MCP/connectors themselves.

## Workflow

1. Establish the task boundary.
   - What metric, model, product/category, period, or launch is being checked?
   - Is the task read-only analysis, task drafting, code investigation, or report writing?
   - Which actions require explicit user confirmation?

2. Collect evidence.
   - Read metric tables, dashboard exports, run artifacts, logs, notebook output, or CI summaries.
   - Preserve source names and timestamps when available.
   - Do not infer production facts from synthetic artifacts unless they are clearly marked as synthetic.

3. Diagnose.
   - Compare current metrics with previous period, baseline, and expected ranges.
   - Slice by model, category, product group, segment, period, and policy type when available.
   - Look for drops, spikes, missing rows, low effective sample size, broken randomization checks, calibration drift, and disagreement between predictive and policy metrics.

4. Produce hypotheses.
   - Separate data issues, model issues, policy/evaluation issues, and product-rule issues.
   - For each hypothesis, state the minimal check that would confirm or falsify it.

5. Prepare actions with confirmation.
   - For a task tracker: draft title, context, acceptance criteria, and priority.
   - For code investigation: propose files, commands, and expected evidence before editing.
   - For a report: write a concise summary, evidence table, decisions, open risks, and next steps.
   - Ask for confirmation before creating or changing anything in external systems.

## Analytical Playbooks

### Metric Anomaly

1. Identify the metric, period, model, product group, and expected baseline.
2. Check data freshness and row counts before model explanations.
3. Slice by segment, category group, policy type, and time.
4. Separate real movement from logging, sample-size, or denominator issues.
5. Draft a short finding with evidence and the next SQL/code check.

### Model Monitoring

1. Compare predictive metrics, policy metrics, calibration, and uplift alignment.
2. Flag disagreement between response quality and policy value.
3. Check whether a drop is concentrated in one category group or across the portfolio.
4. Propose a rollback, rerun, or deeper investigation only after evidence is collected.

### Report Or Task Draft

Include context, evidence, suspected cause, proposed owner/action, acceptance
criteria, and risks. Do not create or publish without explicit confirmation.

## Guardrails

- Do not claim autonomous production access unless a tool is actually available in the current environment.
- Do not expose internal table names, tokens, URLs, customer data, or exact production metrics in public artifacts.
- Do not turn a metric anomaly into a model conclusion without checking data freshness, sample size, split, and policy coverage.
- Do not create tasks or reports silently; summarize the intended action first.

## Output Template

```markdown
Summary:
...

Evidence:
- ...

Likely Causes:
1. ...
2. ...

Recommended Actions:
1. ...
2. ...

Needs Confirmation:
- ...
```
