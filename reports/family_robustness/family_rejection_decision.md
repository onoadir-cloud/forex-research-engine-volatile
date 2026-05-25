# Family-Level Directional-Run Robustness Rejection Decision

## Scope
This document is the final decision report for the family-level directional-run robustness study based on the generated outputs under `reports/family_robustness/`.

## Executive Conclusion
- **Do NOT build an EA from these directional-run direction-selection families.**
- **Do NOT proceed to trade simulation using these family outputs.**
- **Do NOT optimize these directional-run families further.**

## Decision Basis

### 1) Prior EURUSD locked robustness test outcome
The prior EURUSD locked robustness test failed all candidates. No candidate demonstrated a robust, directional selection edge that survives the robustness criteria.

### 2) Family-level multi-symbol robustness outcome
The family-level multi-symbol test also failed to demonstrate robust edge. Cross-symbol behavior does not provide stable support for deployable direction-selection logic.

### 3) Family-wide edge profile
Most method families show:
- negative `edge_vs_opposite`, and
- negative `edge_vs_random`.

This indicates the direction-selection signal is generally not better than opposite-side framing and not better than random direction choice.

### 4) Specific exception check: `momentum_50` on H4
`momentum_50` on H4 shows tiny positive `edge_vs_random` at some horizons. However, this does **not** pass acceptance because:
- `edge_vs_opposite` remains negative, and
- the positive effect is too small/inconsistent to qualify as robust.

Therefore, this exception does not overturn the rejection.

### 5) Interpretation of high `selected_mean_after_friction`
High `selected_mean_after_friction` only demonstrates that there is underlying movement in the market window. It does **not** prove that direction can be selected reliably **ex-ante** by these families.

## Final Decision
**Reject directional-run direction-selection families for EA development.**

Accordingly:
- Do not continue to trade simulation from these results.
- Do not proceed with further optimization on these families.
- Close this research branch as a negative finding for direction-selection viability.
