# Auto-illustration Mode — Implementation Tasks
Project folder: projects/2026-02-12_auto-illustration-mode/

Phases: Discovery → Backend (domain) → Frontend components → API routes & integration → Cleanup & tests

Notes: Keep changes small and focused. Where possible reuse existing flows (`open-editor`, `openEditorIds`, `ImageLayer`, `ImageEditor`, `scoreSegment()`).

## Discovery

- [ ] Review current `Highlight` WS payload format and list keys used (id, page, normalized coords, score).
- [ ] Confirm where `scoreSegment()` is invoked and what events are emitted (`highlight` events).
- [ ] Identify current `open-editor` / `openEditorIds` event handlers and parameters in `ImageLayer`/`ImageEditor`.

## Backend / Domain (minimal changes)

### Goal: Add batch-first support for image/enhance endpoints

- [ ] Add small batch helper function in `services/api/core/tools/text2image.py` to accept `texts: string[]` and return an array of results.
- [ ] Update `services/api/core/api.py` route handlers for `/gen-image-bytes` to accept `texts` (batch-only) and route to helper.
- [ ] Update `/enhance` handler to accept a list of texts (`texts: string[]`) and return batched enhancements.
- [ ] Add server-side validation for max batch size (configurable, default 50) and input sanitization.
- [ ] Add unit tests for new batch path and ordering of results.

## Frontend — Components & UI

### 1) Top bar toggle & cogwheel
 - [x] Add `AutoIllustrationToggle` UI element inside `services/frontend/app/pages/index.vue` top bar: place to the right of `ModelSelect.vue` and left of export controls.
 - [x] Create a small composable `useAutoIllustration()` to hold mode state, settings (`min_gap_lines`, `max_gap_lines`, `min_score`) and helper methods.
 - [x] Implement cogwheel dropdown using `ModelManager.vue` patterns (styles & show/hide) and include inputs for `min_gap`, `max_gap`, `min_score`, with validation: `max_gap > min_gap`.
  - [x] Add a `Clear` button inside the cogwheel dropdown that closes all open `ImageEditor` instances and removes auto-added ids from `selectedHighlights` (preserve manually-selected ids). Implement `useAutoIllustration().clearAutoSelections()` to perform this.

### 2) Selection UI patterns
- [ ] Do NOT modify `HeatmapViewer.vue` for any new indicators — leave the heatmap visuals unchanged.
 - [x] Add a button to run an on-demand selection pass (can be in cogwheel dropdown or top bar) wired to `useAutoIllustration().runPass()`.
 - [ ] Add a compact status bar component next to the cogwheel dropdown showing three fills for: selected (color A), enhanced (color B/pattern), and generated (color C/pattern). The bar should display percentages relative to the computed maximum possible selections (derived from gap rules and total segment count) and show a tooltip with a textual summary (counts and active state). Wire this component to `useAutoIllustration()` for live counts and active status.
 - [ ] Add animations to the `enhanced` and `generated` fills when those processes are active (e.g., subtle pulse or diagonal stripe animation). The `selected` fill remains static (instant). Ensure animations are CSS-driven and accessible (reduce-motion respects user preference).

- ### 3) Hook into scored highlights reactively
 - [x] Implement selection logic in `useAutoIllustration()` that watches the reactive `highlights` array (the scored segments array) and runs whenever the array changes (add/update/remove).
 - [x] Implement the in-order lookahead algorithm that processes the `highlights` array in its current order (document/physical order) and returns an array of selected ids.
 - [x] On algorithm output, add the returned ids into the shared `selectedHighlights` reactive Set/array (union), and emit `openEditorIds` (or call `open-editor` flow) for newly-added ids; ensure manual selections are preserved and the UI deduplicates already-open editors.
 - [x] Ensure the watcher is efficient (debounce or microtask) and tolerant to frequent updates (client-side workers or WS); do not rely on WS-only messages.
 - [ ] Track counts of selected, enhanced, and generated highlights in `useAutoIllustration()` and expose a derived `progress` object: `{ selectedCount, enhancedCount, generatedCount, maxPossible }` and `isActive` flag (true when enhancing/generating). The status bar component will consume this object to render fills and tooltip text.

### 4) Editor management
- [ ] Implement tracking of open editor IDs in `useAutoIllustration()` and manage opens without imposing a fixed upper limit.

### 5) Distance calculations
 - [x] Implement helper to compute normalized vertical delta → lines mapping: lines = delta / 0.01.
 - [ ] Use `pageAspectRatio` and page refs from `PdfViewer.vue` when computing y coordinates for accurate mapping.

## Frontend — Integration & minor changes

- [ ] Reuse `scoreSegment()` where appropriate to keep scoring pipeline consistent before selection.
 - [x] Update `scoreSegment()` (in `services/frontend/app/pages/index.vue`) so it only assigns `highlight.score` and does NOT mutate `selectedHighlights` (remove SCORE_THRESHOLD-based autoselect code).
 - [ ] Ensure HighlightNav still reads `selectedHighlights` for navigation; `useAutoIllustration()` will become the authoritive updater of `selectedHighlights` when enabled.
- [ ] In `ImageLayer`/`ImageEditor`, verify `openEditorIds` API signature and update call sites if needed to accept array input from auto-select code.

## API Routes & Tests

- [ ] Add route-level tests for `gen-image-bytes` and `enhance` batch behaviors.
- [ ] Add an integration test that calls the batch route with 3 entries and inspects results ordering.

## UX polish & validation

- [ ] Add validation messages in cogwheel dropdown for invalid `max_gap <= min_gap` and non-integer `min_score`.
 - [ ] Ensure the status bar tooltip includes: number selected, number enhanced, number of images generated, and whether the algorithm is currently enhancing/generating.

## Cleanup & docs

- [ ] Document new settings and default values in project `README` or a short front-end README (one paragraph).
- [ ] Add comments to `useAutoIllustration()` describing the selection algorithm and mapping rule (1 line = 0.01 normalized).
- [ ] Run lint/format on changed files and update any tests broken by refactors.

## Rollout checklist

- [ ] Feature flag the toggle behind a user preference or remote flag for gradual rollout.
- [ ] Manually smoke test: enable toggle, stream sample highlights, verify editors open and images generate.

---

Each task above should be implemented as a small commit. If you'd like, I can start by implementing the frontend composable `useAutoIllustration()` and the UI toggle + cogwheel dropdown.

```
