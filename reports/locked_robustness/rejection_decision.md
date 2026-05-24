# Locked Robustness Rejection Decision

## Executive Summary
All locked candidates failed robustness validation and are rejected.

## Locked Candidate Results
- `A_H1_momentum50_h20` = **FAIL**
- `A_H1_momentum50_h40` = **FAIL**
- `A_H1_momentum50_h80` = **FAIL**
- `B_H4_mapos50_h80` = **FAIL**

## Assessment
- **Candidate A** was unstable across years and did not consistently beat random baselines.
- **Candidate B** showed strong descriptive expectancy in some periods, but failed robustness because it did not consistently beat random/opposite baselines, especially in OOS stability.
- The prior top-context strength is therefore more likely descriptive/overfit than robust enough for trading deployment.

## Final Decision
Reject these locked candidates.

Do **not** proceed to trade simulation.
Do **not** build an EA from these candidates.
