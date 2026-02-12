Findings: scoring and `scoreSegment()`

- Definition:
  - `scoreSegment(segment: Segment)` is defined in `services/frontend/app/pages/index.vue` (function near the bottom of that file). It finds the matching highlight in the reactive `highlights` array and assigns `segmentHighlight.score = segment.score`.

- Invocations / call sites:
  - `services/frontend/app/pages/index.vue` — WebSocket `onMessage` handler: when `data.type` is `segment`, it calls `scoreSegment(data.content as Segment)`; when `batch`, it iterates the batch and calls `scoreSegment(segment)` for each.
  - `services/frontend/app/utils/models.ts` — some model `onSuccess` handlers call `scoreSegment(result)` for worker/batch evaluation flows (see the `onSuccess` hook entries for models which forward results to `scoreSegment`).
  - `services/frontend/app/pages/index.vue` also calls `scoreSegment()` in a few places where model callbacks are wired (see lines that call `modelInfo.onSuccess?.(data, modelInfo, socket, scoreSegment)` and direct `scoreSegment(...)` calls).

- Events emitted:
  - `scoreSegment()` itself does not emit DOM events or mutate `selectedHighlights` in the current source. It only assigns `score` onto the matching highlight object. Navigation/selection side-effects are handled elsewhere (hooks / custom hooks). The viewer exposes a `custom:goToHighlight` hook the app uses in other code paths.

- Does `scoreSegment()` mutate `selectedHighlights`?
  - No — in the current source `scoreSegment()` only sets `segmentHighlight.score = segment.score` and returns. Selection is handled by `useAutoIllustration()` (composable) or by other logic — older/dist code shows autoselect behavior by score threshold but the canonical source (`app/pages/index.vue`) does not modify `selectedHighlights` from `scoreSegment()`.
