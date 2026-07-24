# GAIA Evaluation Status

Last updated: 2026-07-24

## Executive summary

The current goal is to improve the default Agent Framework harness rather than
require every customer to tune GAIA-specific configuration.

The most important finding is that many earlier scores were corrupted by
deployment rate limits. Per-task HTTP 429 failures were caught by the GAIA
adapter and converted into empty, incorrect answers. Concurrent candidate and
baseline evaluations made this substantially worse.

After adding bounded transient retries, propagating exhausted transient errors,
and running one evaluation process at a time, the unchanged harness scored:

| Slice | Model | Result |
|---|---|---:|
| Seed 0, 20 tasks, clean run 1 | `gpt-5.2` | 13/20 |
| Seed 0, 20 tasks, clean run 2 | `gpt-5.2` | 13/20 |
| Seed 1, 20-task holdout | `gpt-5.2` | 15/20 |

These small slices suggest a clean baseline around 65-75%, but they are not a
replacement for a larger representative run. They are also not directly
comparable to published GPT-4o results or the earlier approximately 47.6%
number.

No evolved framework mutation has yet produced a confirmed general improvement.

## Current repository state

Branch base:

```text
6758e8f10 Add Darwin GAIA optimization pilot surface
```

The working tree currently contains an uncommitted evaluation-reliability
change in:

- `python/packages/eval-harness/agent_framework_eval_harness/benchmarks/_gaia.py`
- `python/packages/eval-harness/tests/test_gaia.py`

The change:

- Retries wrapped transient HTTP 408, 409, 429, and 5xx failures.
- Uses exponential backoff and honors numeric `Retry-After` headers.
- Traverses Agent Framework exception wrappers to find the underlying service
  exception.
- Holds the benchmark concurrency slot during backoff so queued tasks do not
  stampede an exhausted deployment.
- Propagates an exhausted transient failure, invalidating the benchmark instead
  of silently scoring the task as wrong.
- Uses a fresh agent session for every retry attempt.

The focused GAIA tests, Ruff, and Pyright pass with this change.

## Runtime configuration

The model actually used by the recent GAIA runs was:

```text
gpt-5.2 on the eastus2 Foundry deployment
```

This is controlled by `FOUNDRY_MODEL`. The comments in
`gaia_harness_eval.py` still discuss GPT-4o as the published comparison model,
so always record the runtime model with a result.

Darwin used Claude Sonnet 4.5 to generate code mutations. The mutation model and
the model evaluated on GAIA are separate.

## Relevant files

- `gaia_harness_eval.py`
  - Builds the harness agent and defines search, fetch, transcript, code, answer
    formatting, and optional reformulation behavior.
- `gaia_harness_candidate.py`
  - Holds the GAIA-specific instruction and settings surface used by the first
    Darwin pilot.
- `agent_framework_eval_harness/benchmarks/_gaia.py`
  - Loads deterministic task slices, runs tasks, scores answers, persists
    task-level results, and now handles transient service failures.
- `agent_framework/_harness/_agent.py`
  - Contains the default harness instructions and factory.
- `agent_framework/_harness/_loop.py`
  - Contains loop continuation, feedback, response aggregation, and iteration
    behavior.

## Evolution architecture

Darwin could not safely transport the full Agent Framework repository because
unrelated Git LFS assets repeatedly broke checkout, push, or post-commit hooks.
The working setup therefore uses small standalone candidate repositories.

For framework evolution:

1. Darwin mutates only copies of `_agent.py` and `_loop.py`.
2. The evaluator creates a temporary worktree at a pinned Agent Framework
   revision.
3. The candidate files are overlaid into that worktree.
4. Ruff, Pyright, targeted tests, and GAIA are run independently.
5. Compatibility diagnostics are recorded, but a runnable candidate still
   receives a GAIA score.

Workstation-specific locations:

```text
Darwin:                         C:\Users\bentho\src\Darwin
Framework candidate repository: C:\Users\bentho\src\gaia-darwin-framework-candidate
Configuration candidate repo:   C:\Users\bentho\src\gaia-darwin-candidate
```

Framework candidate baseline:

```text
5440d69173d0450a9b70a04fc57b265ba7ba1749
```

## Darwin search changes

The original idea sampler used weighted random selection with replacement.
Weights were probabilities rather than a priority queue, and the same idea
could be sampled repeatedly for the same parent.

A hybrid sampler was implemented in the Darwin checkout:

- Usually selects the highest-weight idea not yet tried on the selected parent.
- Reserves 20% of selections for weighted-random exploration.
- Tracks usage by `(idea, parent program)`.
- Allows an idea to be tried on a different descendant.
- Returns to fitness-adjusted reuse after a parent's idea pool is exhausted.

Experiment:

```text
Name:   optimize-agent-framework-gaia-core-v4
Run ID: 42822a96-b8a9-44f2-bbf9-9f58bb17308d
```

The sampler behaved as intended, but better idea coverage did not solve the
larger problems of noisy scoring and oversized speculative mutations.

## Results and interpretation

### Early framework evolution

Darwin produced several apparently strong five- and ten-task candidates,
including:

- Adaptive iteration limits.
- Stuck detection and adaptive feedback.
- Progressive thresholds.
- Tool success/failure tracking.
- A final-answer synthesis pass.

Some small slices showed large apparent gains, such as 1/5 to 4/5. Broader
promotion runs did not reproduce those gains.

Do not use the old rankings as evidence of framework improvement. Many runs
were executed concurrently against the same quota-limited deployment and logs
contained task-level 429 failures that were scored as wrong answers.

### Clean baseline

Two identical seed-0 runs completed all tasks and each scored 13/20. Each run
encountered five transient service failures, all recovered by the new retry
logic.

Task-level stability across those two runs:

| Outcome | Tasks |
|---|---:|
| Passed both runs | 11 |
| Passed one run | 4 |
| Failed both runs | 5 |

The five stable failures were:

1. Counting Mercedes Sosa studio albums in a date range.
2. Locating an exact deleted word in Cornell LII amendment notes.
3. Reading a blocked Bielefeld BASE result.
4. Locating an exact scikit-learn changelog symbol.
5. Resolving the adversarial Pineapple/Guava instruction trap.

### Failure-driven micro-mutations

Three small instruction hypotheses were evaluated independently:

1. Require the exact final fact to be grounded in fetched source text.
2. Try alternate sources or archives when a source is blocked.
3. For counts and date ranges, find the complete list or table rather than
   inferring from individual items or snippets.

The source-grounding and blocked-source hypotheses failed their diagnostic
tasks and were rejected.

The complete-list instruction showed a weak signal:

- Target counting task: candidate 2/5, unchanged baseline 0/5.
- Seed-0 full runs: candidate 15/20 and 13/20; baseline 13/20 and 13/20.
- Seed-1 holdout: candidate 15/20; baseline 15/20.
- Holdout task-level comparison: two gains and two losses.

This was not strong or causal enough to add to the default framework.

## What we learned

### Evaluation reliability

- Infrastructure failures must never become ordinary incorrect answers.
- Do not run baseline and candidate evaluations concurrently against this
  deployment. Serial execution is slower but produces usable data.
- Use `parallel=1` until quota behavior is understood. Internal agent loops can
  issue many model calls even for one benchmark task.
- A benchmark is invalid if transient retries are exhausted.
- Persist task-level responses, extracted answers, and errors for every run.

### Task selection and variance

- `seed` controls the shuffled task selection when `max_tasks` truncates the
  dataset. Changing the seed changes the slice; it is not a repeat.
- Repeat measurements must use the same seed, task count, and offset.
- Compare task-level paired outcomes, not only aggregate percentages.
- A five-task slice is useful only as a diagnostic canary, not as evidence of a
  general improvement.

### Evolution strategy

- More random Darwin rounds will not fix a weak fitness signal.
- Mutation ideas should be derived from reproduced failures, not generic
  brainstorming.
- Mutations should test one hypothesis and stay below roughly 50 changed lines.
- Large 140-400-line adaptive mechanisms were difficult to reason about and
  frequently failed compatibility checks.
- Ruff, Pyright, and tests should run once per unique candidate. They are
  compatibility diagnostics, not automatic score suppressors.
- Only benchmark setup or execution failure is a hard scoring blocker.
- Darwin is most useful after individual mutations are independently validated,
  when it can explore combinations and descendants.

## Recommended next steps when GAIA work resumes

1. Review and commit the transient retry/backoff change.
2. Ensure every Darwin candidate uses the revision containing that fix.
3. Set Darwin candidate evaluation to one process and `parallel=1`.
4. Run a larger clean baseline before evolving more code:
   - Prefer at least 50 tasks.
   - Use fixed seeds and offsets.
   - Repeat the same slice before drawing conclusions.
5. Classify reproduced failures using full responses and tool traces.
6. Create small single-hypothesis candidates for framework-addressable failures.
7. Use a staged funnel:
   - Repeated diagnostic tasks.
   - One fixed development slice.
   - A different-seed holdout.
   - Larger confirmation only for surviving candidates.
8. Do not promote a mutation unless it improves paired task outcomes without
   introducing comparable regressions.

## Commands

Run a clean fixed-slice baseline from the repository root:

```powershell
$env:GITHUB_TOKEN = gh auth token

uv run --project python --all-packages `
  --with "huggingface-hub>=0.20.0" `
  --with "pyarrow>=18.0.0" `
  --frozen python `
  python\samples\05-end-to-end\evaluation\gaia_harness_eval.py `
  --level 1 `
  --max-tasks 20 `
  --task-offset 0 `
  --parallel 1 `
  --timeout 300 `
  --seed 0 `
  --results-file .gaia-results.jsonl
```

Run one exact diagnostic task from the seeded slice:

```powershell
uv run --project python --all-packages `
  --with "huggingface-hub>=0.20.0" `
  --with "pyarrow>=18.0.0" `
  --frozen python `
  python\samples\05-end-to-end\evaluation\gaia_harness_eval.py `
  --level 1 `
  --max-tasks 1 `
  --task-offset <INDEX> `
  --parallel 1 `
  --timeout 300 `
  --seed 0 `
  --results-file .gaia-diagnostic.jsonl
```

Local detailed artifacts from this investigation are under:

```text
C:\Users\bentho\.copilot\session-state\48a8f153-5f8b-486c-8e80-2de40cfd089d\files\failure-driven-v1
C:\Users\bentho\.copilot\session-state\48a8f153-5f8b-486c-8e80-2de40cfd089d\files\gaia-promotion-v4
```

