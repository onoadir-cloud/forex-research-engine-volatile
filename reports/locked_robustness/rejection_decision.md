# Locked Robustness Rejection Decision

All locked candidates failed the robustness gate and are rejected.

## Runtime Verdict
- `A_H1_momentum50_h20` = **FAIL**
- `A_H1_momentum50_h40` = **FAIL**
- `A_H1_momentum50_h80` = **FAIL**
- `B_H4_mapos50_h80` = **FAIL**

## Decision Rationale
1. Every locked candidate failed, so none qualify for progression.
2. Candidate A showed instability across years and did not consistently outperform random baselines.
3. Candidate B failed due to insufficient OOS/yearly stability, with weakness especially versus random baselines.
4. The previously strong top-context signal is likely descriptive/overfit rather than robust under locked testing.

## Final Recommendation
Reject these locked candidates.

Do **not** proceed to trade simulation, EA building, or further optimization for this set.
