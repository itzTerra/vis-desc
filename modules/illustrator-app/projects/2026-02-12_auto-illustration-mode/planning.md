# Auto-illustration Mode — Project Plan

Date: 2026-02-12

## 1. Classification

- Project type: Feature enhancement (frontend-first, light backend additions)
- Scope: Client-driven rule-based selection algorithm + UI integration + backend batch endpoints compatibility
- Priority: Medium (UX improvement, iterative rollout)

## 2. Goals

- Provide an online rule-based algorithm that automatically selects high-score highlights and creates image editors for them.
- Make selection tunable via UI controls (toggle + cogwheel settings) and allow on-demand re-run.
- Support batching in backend image/enhance endpoints (batch-only API).
- Keep UI responsive; do not impose a hard limit on opened editors from auto-selection.

## 3. User stories & acceptance criteria

- User story 1: As a reader, I can enable "Auto-illustration mode" and have the app open image editors for selected highlights as the scored segments array changes.
  - Acceptance: When toggle enabled, the selection algorithm runs reactively whenever the scored segments array changes (most commonly when a scored segment is added). The algorithm returns only the ids of segments for which an `ImageEditor` should open; the UI opens editors for those ids.
  - Additional acceptance: When auto-illustration is enabled the algorithm will add its selected ids into the shared `selectedHighlights` set/array (i.e., union with manual selections) used by `HighlightNav` and other UI. Manual selection by clicking remains the primary way to control `selectedHighlights`. The existing SCORE_THRESHOLD-based auto-selection must be removed — scoring updates should not mutate selection. When auto-illustration is disabled, only manual selection mutates `selectedHighlights`.

- User story 2: As a user, I can tune selection behavior via a cogwheel dropdown: `min_gap`, `max_gap`, and optional `min_score`.
  - Acceptance: Settings visible in cogwheel dropdown; `min_gap` is at least the system default (user may increase); `max_gap` must be > `min_gap`; `min_score` may be unset.

- User story 3: As a power user, I can clear existing image editors and trigger an on-demand selection pass that processes the currently-known highlights and opens editors for chosen ones.
  - Acceptance: A UI affordance (button/menu item) for clearing image editors and running selection on demand and opens editors; useful after bulk import or after adjusting settings.

- User story 4: As a developer, backend endpoints accept batch requests for `enhance` and `gen-image-bytes`.
- Acceptance: Endpoints accept `texts: string[]` and return results with consistent batch formatting.

## 4. Functional requirements

 - Client-side selection algorithm:
  - Input: the reactive array of scored highlights (in physical document order) and current selection state. Each highlight contains normalized coordinates, page index, score, and id.
  - Output: an ordered array of selected highlight ids (only ids) that indicate which `ImageEditor` instances should open.
  - Behavior: The algorithm runs every time the scored highlights array changes. It processes the array in the same order it appears (document/physical order) and applies the min/max gap rules using a lookahead window to choose the best candidate within the window.
  - Reactivity behavior:
  - The algorithm must be invoked whenever the scored highlights array changes (Vue watcher or computed), not only on WebSocket messages. It must be agnostic to where scores come from (WS streaming, client-side worker inference, or server responses). The algorithm returns only ids. Additionally, when auto-illustration is enabled the algorithm will add its selected ids into the shared `selectedHighlights` reactive Set/array (without removing manually-selected ids). When disabled, only manual selection mutates `selectedHighlights`.

  - Score handling: `scoreSegment()` or any scoring path must only set `highlight.score` fields and must NOT add ids to `selectedHighlights`. Removing the existing SCORE_THRESHOLD-based autoselect is required.

- Gap rules and units:
  - Use normalized vertical coordinates (0..1) and the `pageAspectRatio` to compute vertical distance relevant to the current PDF viewport.
  - Default mapping: 1 line ≈ 0.01 of page height. Default `min_gap` = 2 lines → 0.02 normalized units. Allow tuning later.
  - `max_gap` defines lookahead window: when selecting a highlight, skip ahead by up to `max_gap` (in same units) to allow for higher rated highlights.

- Controls and settings:
  - Toggle in top bar placed to the right of `ModelSelect.vue` and left of Export controls in `index.vue` top bar.
  - Cogwheel dropdown styled using `ModelManager.vue` patterns; include inputs for `min_gap` (floating display in lines and normalized), `max_gap`, and optional `min_score` (0–100 int or unset).
  - Respect validation: `max_gap` must be > `min_gap`; UI shows validation state.

- Editor management:
  - Use existing flows: `open-editor`, `openEditorIds`, `editorStates`, and `ImageLayer` to create editors identical to manual create flow.
  - Provide a button to clear all ImageEditors.
  - Provide an explicit on-demand selection trigger to run a pass over current highlights.

 - Backend batch endpoints:
  - Add support to `/enhance` and `/gen-image-bytes` to accept `texts: string[]` as the primary (batch) API.
  - Return list of results in same order as inputs with clear success/failure per item.

## 5. Non-functional requirements

- Performance: selection algorithm runs in O(M log M) per batch where M = number of highlights in window; streaming (reactivity) friendly.
- UI responsiveness: avoid blocking main thread during selection; use requestAnimationFrame or web worker if necessary for large highlight volumes.
 - Backward compatibility: not required for these batch endpoints; the API will use batch-first semantics.
- Security / validation: validate batch sizes server-side (max batch 32 by default) to avoid resource exhaustion.

## 6. Current state analysis (key files)

- Frontend:
  - `services/frontend/app/pages/index.vue` — top-level page; contains top bar where toggle will be placed.
  - `services/frontend/app/components/ModelSelect.vue` — model dropdown pattern to reuse for toggle placement.
  - `services/frontend/app/components/ModelManager.vue` — styles and dropdown/cogwheel patterns to reuse for settings UI.
  - `services/frontend/app/components/PdfViewer.vue` — exposes pageAspectRatio and page refs needed for distance computations.
  - `services/frontend/app/components/HighlightLayer.vue` — emits highlight events and is the canonical source of highlights for selection.
  - `services/frontend/app/components/HeatmapViewer.vue` — unrelated to this feature; do NOT read highlights from or modify `HeatmapViewer.vue` in this project.
  - `services/frontend/app/components/ImageLayer.vue` & `ImageEditor.vue` — existing create/edit flows; must be reused to open editors.
  - `services/frontend/app/utils/models.ts` & `app/composables/useModelLoader.ts` — model management; follow pattern for cogwheel and dropdown.
  - `services/frontend/app/composables/useExport.ts` — export button positioning; toggle goes just left of Export.

- Backend:
  - `services/api/core/api.py` — route definitions for image endpoints; add batch support here.
  - `services/api/core/tasks.py` — background tasks for enhancement/generation; adapt to handle batch payloads or add thin batch wrapper.
  - `services/api/core/consumers.py` — websocket consumers that may forward generation tasks; leave unchanged unless needed.
  - `services/api/core/tools/text2image.py` — generation/enhancement logic; add batch helper functions.

## 7. Integration & patterns to follow

- Reuse `ModelSelect.vue` dropdown pattern for placement and consistency.

- Reuse styles and dropdown/cogwheel behavior from `ModelManager.vue`. Example pattern:

```vue
<!-- pattern: cogwheel dropdown -->
<div class="model-manager">
  <button class="cog" @click="toggle()">⚙️</button>
  <div v-if="open" class="dropdown">...settings...</div>
</div>
```

- Use existing `scoreSegment()` behavior and `highlight` events as input to the selection algorithm. Pseudocode:

```js
// on ws batch
const highlights = incomingBatch.highlights;
scoreSegment(highlights); // keep existing scoring pipeline
autoSelect.process(highlights);
```

- To open editors reuse `openEditorIds` or `open-editor` flow already used elsewhere:

```js
const toOpen = selectedIds.slice(0, maxOpen);
emit('openEditorIds', toOpen);
```

## 8. Algorithm design (client-side)

- Normalize inputs: for each highlight use {id, page, yCenterNorm, score} where yCenterNorm is normalized vertical center (0..1).

- Convert normalized vertical distance to "lines":
  - lines = yDelta / 0.01 // (1 line ≈ 0.01 of page height)
  - default `min_gap_lines` = 2 → normalized = 0.02

 - In-order lookahead selection (per-document order):
   1. Treat the highlights array as ordered by physical location in the PDF (this is the canonical input order).
   2. Iterate the array from start to end. At each position i consider the lookahead window of segments whose vertical center is within `max_gap` of segment i (and on the same page). From that window choose the segment with the highest score that also meets `min_score` (if set). If a candidate is chosen, emit its id into the output list and advance the scan to the first segment whose center lies beyond the chosen segment's center plus `min_gap` (so the min gap rule is enforced).
   3. Repeat until the end of the array. This guarantees deterministic, document-ordered selection while allowing the algorithm to pick higher-score segments within a local lookahead.

 - Cross-page considerations:
   - Run the same in-order algorithm across the whole array; segments on different pages will naturally be ordered and treated according to their absolute positions. If desired, the implementation may optionally process per-page slices, but the authoritative requirement is to follow the array order.

 - Reactivity behavior:
   - The algorithm must be invoked whenever the scored highlights array changes (Vue watcher or computed), not only on WebSocket messages. It must be agnostic to where scores come from (WS streaming, client-side worker inference, or server responses). The algorithm returns only ids; the UI layer is responsible for opening editors for returned ids and for deduplicating already-open editors.

## 9. Limits and config

 - Default mapping: 1 line = 0.01 normalized units.
  - Defaults: `min_gap_lines` = 2 (→ 0.02), `max_gap_lines` = 8 (→ 0.08), `min_score` = unset.

## 10. Decisions made now

- Client-side algorithm (no DB changes).
- Use normalized vertical distance mapping of 1 line = 0.01 page height; `min_gap` default 2 lines.
- Backend endpoints will accept `texts` (batch-only semantics preferred).
- No hard limit on simultaneous editors will be enforced by the auto-selection algorithm.

---

If any decision above needs to change (for example, mapping of lines or max editors), I can update this plan and regenerate tasks.
