1. The family-level robustness test was executed successfully.
2. The tested method families were:
   - momentum_20
   - momentum_50
   - ma_position_50
   - ma_slope_20
   - recent_range_position
3. The result does not justify trade simulation or EA development.
4. Most method families did not beat opposite direction and random baseline robustly.
5. selected_mean_after_friction shows that movement exists, but not that direction can be selected ex-ante.
6. momentum_50 H4 showed tiny positive edge_vs_random in some rows, but it is not enough because edge_vs_opposite remains weak/negative and robustness is insufficient.
7. Final decision:
   Reject directional-run direction-selection families for EA development.
   Do not proceed to trade simulation.
   Do not build a robot from this research family.
