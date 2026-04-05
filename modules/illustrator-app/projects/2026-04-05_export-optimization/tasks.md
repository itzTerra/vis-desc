# Tasks

Tasks implement the recommended approach: single-file HTML with WebP conversion + lazy image parsing (Decisions D1-D4). If ZIP export is chosen instead, see notes at the bottom.

## WebP Conversion Utility

- [x] Add `async function convertToWebP(dataUri: string, quality = 0.80): Promise<string>` to `services/frontend/app/utils/export.ts` that loads a PNG data URI into an `Image`, draws to an offscreen canvas, and returns a WebP data URI via `canvas.toDataURL('image/webp', quality)`. Returns `Promise<string>`. Wrap `img.onload` in a `new Promise` -- set `img.src`, then `drawImage` + `toDataURL` inside `onload` callback. Note: if the browser does not support WebP encoding, `toDataURL` silently returns PNG; this is acceptable (best-effort size reduction).

## Export Snapshot Changes

- [x] Add a new `ExportSnapshot` field `imageData: Record<number, string>` to hold the WebP data URIs keyed by highlight ID, separate from the highlights array. Remove `imageUrl` from `ExportHighlight` (do not add `hasImage` -- it is redundant).
- [x] Update `createExportSnapshot()` to convert images inline: use `for...of` over `Object.entries(imageUrls)` with `await convertToWebP(url)` on each entry, collecting results into `snapshot.imageData`.

## HTML Template: Lazy Image Tags

- [x] In `generateExportHtml()`, emit one `<script type="application/json" data-image-id="[highlightId]">` tag per entry in `snapshot.imageData`, containing the WebP data URI as a JSON string. Place these tags after the modal overlay div and before the main `<script type="module">`.
- [x] Update the `hasImages` predicate in `generateExportHtml()` to check `Object.keys(snapshot.imageData).length > 0` instead of filtering `h.imageUrl`. Without this change, the modal overlay markup will never be emitted.
- [x] Update the highlights JS literal in the template to serialize highlights without `imageUrl` -- only `id`, `text`, and `polygons` (no `hasImage` field).
- [x] Add a helper function `getImageUrl(highlightId)` in the exported HTML's script that queries `document.querySelector('script[data-image-id="${id}"]')`, parses `.textContent` as JSON, and returns the data URI (or null if no tag exists). Cache the result so repeated calls for the same ID don't re-parse.
- [x] Update the `renderPage()` function in the template: for each highlight, call `const url = getImageUrl(highlight.id)` -- if null, skip button creation entirely. Otherwise, set `img.src = url` and `button.onclick = () => openImageModal(url)`. The URL is resolved once per button. Note: the existing `highlights.filter(h => h.imageUrl && h.polygons[...])` filter on line 412 of `export.ts` must be updated -- `h.imageUrl` will always be `undefined` after the change. Replace the whole filter with per-highlight `getImageUrl(h.id)` null check inside the loop body.

## Integration Verification

- [x] Verify the export button in `index.vue` and `useExport.ts` composable require no changes (they pass data through unchanged interfaces).
- [x] Verify the exported HTML renders correctly: PDF pages load lazily, image buttons appear on correct pages, image modal opens with WebP images. Test manually with a multi-page PDF containing several generated images.
- [x] Verify the exported HTML file size is meaningfully smaller than before for a document with multiple PNG images (expect 50-70% reduction on image data).

## Cleanup

- [x] Run lint from the repo root: `pnpm lint` (the root `package.json` has the lint script).
- [x] Run TypeScript/type checks via `docker compose run --rm frontend pnpm build` (the Nuxt build catches type errors).

---

## Notes: If ZIP Export Is Chosen Instead

If the decision changes from single-file HTML (D3) to ZIP export:

1. **`jszip` is already available**: It is listed in `services/frontend/package.json` -- no new dependency needed.
2. **Replace `downloadExport()`**: Instead of creating a single HTML Blob, create a ZIP containing `index.html` + `images/[id].webp` files using `jszip`.
3. **Replace script tags with relative paths**: Instead of `<script data-image-id>` tags, the HTML would reference `images/[id].webp` as relative `src` attributes. The `getImageUrl()` helper becomes unnecessary.
4. **Change download MIME type**: Use `application/zip` instead of `text/html`.
5. **Update `useExport.ts`**: The `confirmExport` function would produce a `.zip` filename instead of `.html`.
6. The WebP conversion tasks (first section) remain identical -- the format change only affects packaging.
