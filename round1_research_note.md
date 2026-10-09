# Targeted clarification for sequential reciprocal matching

## Problem and hypothesis

The simulator makes daily introductions from an arriving pool with incomplete preferences, changing availability, and feedback that may be delayed or missing. A match must be mutually feasible under the kit's hard constraints; soft compatibility cannot make an infeasible pair eligible. The primary outcome is mutual second-meeting intention (MSMI) per 100 arrived members, averaged equally across the six scenario families.

**Hypothesis.** After resolving required hard constraints, asking a missing soft-preference question when a feasible candidate has already answered that field can modestly improve choices in information-rich scenarios. A local allocation repair may also reduce conflicts from selecting pairs independently. I expect less benefit when answers or reciprocal candidates are scarce. These are hypotheses about the synthetic simulator, not about real relationships.

## Feasibility and allocation

At each decision, I use only the current observed state. A pair is considered only when the public eligibility function reports it feasible in both directions, both members are available, and they have not previously been introduced. Missing required hard information blocks the pair; soft scores never override this gate.

The proposed policy ranks eligible pairs using observed agreement on four soft fields: relationship goal, relationship pace, lifestyle, and conversations. Each observed agreement adds 1, each observed disagreement subtracts 0.5, and an unknown value adds 0. These weights are heuristic design choices, not probabilities. The policy greedily selects non-overlapping pairs by this score, then tries a local repair that replaces one selected pair with two feasible pairs among those members and currently unmatched members when the combined score improves. This is not a globally optimal or multi-day allocation method.

I also evaluated a coverage-first alternative: maximum-cardinality matching on the day's feasible graph, followed by two-pair swaps that preserve cardinality and improve the same soft score. It is an experimental comparison, not the proposed policy.

## Clarification and incomplete data

The daily clarification budget is 12 units; a hard-constraint bundle costs 3 and a named soft-field question costs 1. The policy asks for missing required hard information first. With budget remaining, it ranks unanswered soft fields by how many currently feasible comparisons have an observed answer from the other member, then asks the most useful questions. It does not re-ask observed or declined fields.

A null or declined answer is unknown, not rejection or disagreement. Delayed or absent feedback is not treated as a negative outcome. The policy does not estimate probabilities or learn from selective feedback during this experiment. The simulator's 40 follow-up days are used for outcome maturation after the 60 decision days.

## Experiments and results

I evaluated public simulator seeds 101–120 on all six public variants: development, sparse, cold-start, delayed, shift, and drift. Each method therefore has 120 episodes (20 per family), using the same seed/variant worlds across methods. The public kit generates synthetic episodes with 200 members, 60 decision days, and 40 follow-up days. All 720 episodes across the six methods were valid. Results use the evaluator's equal-weighted mean of the six scenario-family MSMI scores. They are descriptive; public seeds do not establish performance on held-back assessment worlds.

| Method | MSMI per 100 arrived | Coverage | Clarification cost per episode |
|---|---:|---:|---:|
| Supplied greedy | 0.4333 | 0.3448 | 200.38 |
| Supplied no-clarification | 0.1833 | 0.1230 | 0.00 |
| Supplied random-feasible | 0.3667 | 0.3447 | 200.38 |
| Proposed targeted clarification + local repair | 0.4250 | 0.3458 | 256.93 |
| Ablation: no soft-field questions | 0.4125 | 0.3449 | 200.38 |
| Coverage-first allocation alternative | 0.3708 | 0.3444 | 255.92 |

| Scenario family | Greedy | Proposed | No soft questions | Coverage-first |
|---|---:|---:|---:|---:|
| Development | 0.675 | 0.725 | 0.650 | 0.500 |
| Sparse | 0.075 | 0.075 | 0.075 | 0.100 |
| Cold-start | 0.300 | 0.250 | 0.300 | 0.300 |
| Delayed | 0.375 | 0.400 | 0.350 | 0.475 |
| Shift | 0.550 | 0.475 | 0.500 | 0.425 |
| Drift | 0.625 | 0.625 | 0.600 | 0.425 |

The proposed policy did not exceed greedy on the primary score (0.4250 versus 0.4333). In paired episodes it had more, equal, or fewer MSMI outcomes in 17, 84, and 19 episodes, respectively, for two fewer outcomes overall. Coverage was similar. Removing soft-field questions reduced the score to 0.4125 and three fewer total outcomes than the proposed policy, while saving an average 56.56 clarification units per episode; this small descriptive difference does not establish a reliable value for soft questions. The coverage-first alternative scored 0.3708. The proposed policy's mean first-introduction wait among served members was 6.53 days; 130.84 arrived members per episode were unserved on average, and mean missing feedback was 39.16 records per episode.

## Limitations and failure cases

The strongest shortfall is sparse supply: the proposed policy scored 0.075 in sparse scenarios. It also fell below greedy in cold-start and shifted-outcome scenarios, despite higher scores in development and delayed scenarios. Reciprocal candidate scarcity, missing hard answers, and several members competing for the same candidate can defeat the daily local repair. The soft-question heuristic counts potentially useful comparisons but does not measure the true value of information. The policy does not optimize across future days or correct for selective feedback. Its extra questions also cost more than greedy clarification, and its results do not establish a robust advantage. All findings concern only the public synthetic simulator.

**Sources:** [Problem statement and public evaluation protocol](https://github.com/RomeoJulietLove/The-Sequential-Matching-Problem/blob/a8e26b35118cfa8e886a02f93984923e43ab64f6/PROBLEM_STATEMENT.md); [Round 1 and Round 2 submission requirements](https://github.com/RomeoJulietLove/The-Sequential-Matching-Problem/blob/a8e26b35118cfa8e886a02f93984923e43ab64f6/docs/SUBMISSION.md), release 1.0.0.
