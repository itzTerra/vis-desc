Findings: editor open handlers (`open-editor`, `openEditorIds`)

- Where `open-editor` is emitted:
  - `services/frontend/app/components/HighlightLayer.vue` emits `open-editor` with a single `highlightId` when the user clicks the "Illustrate" button in the highlight tooltip. (See `@click="$emit('open-editor', highlight.id)"` inside `HighlightLayer.vue`.)

- Where the event is handled:
  - `services/frontend/app/components/PdfViewer.vue` listens for `@open-editor="openImageEditor"` on the `HighlightLayer` and defines `function openImageEditor(highlightId: number)` which adds the id into `openEditorIds` (a `ref(new Set<number>())`).
  - `PdfViewer` also exposes `openEditors(ids: number[])` via `defineExpose` that can be called by other code (see `defineExpose({ openEditors: (ids: number[] = []) => { ... } })`). Example usage: `pdfViewer.value.openEditors([1,2,3])` (when `pdfViewer` is a component ref).

- ImageLayer / ImageEditor surface:
  - `services/frontend/app/components/ImageLayer.vue` accepts a prop `openEditorIds: Set<number>` and renders one `ImageEditor` per id with `v-for="editorId in openEditorIds"`. Each `ImageEditor` receives `:highlight-id="editorId"` and `:initial-text="getHighlightText(editorId)"`.
  - `services/frontend/app/components/ImageEditor.vue` props: `highlightId: number` and `initialText: string`. It emits events `delete`, `bringToFront`, `pointerDown`, and `state-change` (the latter emits an `EditorImageState` object including `highlightId` and `imageUrl`).

- Short usage examples:
  - Programmatic open (from parent using `pdfViewer` ref):
    - `const pv = pdfViewer.value as any; pv.openEditors([12, 34]);` — this will call `openImageEditor` for each id and add them to `openEditorIds`.
  - Emitting from highlight tooltip (user action): `this.$emit('open-editor', highlightId)` (handled by `PdfViewer.openImageEditor`).
