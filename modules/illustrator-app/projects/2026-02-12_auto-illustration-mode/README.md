Auto-illustration Mode — Frontend settings

Defaults: `minGapLines` = 2, `maxGapLines` = 8, `minScore` = unset. Mapping: 1 line ≈ 0.01 normalized page height.

When enabled, the `useAutoIllustration()` composable adds selected highlight ids into the shared `selectedHighlights` (Set<number>) using union semantics and preserves manual selections. Use the cogwheel dropdown to tune `minGapLines`, `maxGapLines`, and `minScore`.
