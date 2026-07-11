# PDF Export Redesign

## Problem

The current export produces a self-contained HTML file that renders the book PDF via PDF.js loaded from a CDN. This has four compounding problems:

- **No offline capability** — PDF.js is fetched from a CDN at load time, so the export is unusable without internet access.
- **RAM growth / freezing** — the hand-rolled lazy-loading (`IntersectionObserver` + manual scroll listener in `services/frontend/app/utils/export.ts`) renders pages as they scroll into view but never unloads pages that scroll out. After a few hundred pages, memory use climbs until the browser freezes.
- **Lost reading position** — when the page is unloaded (common on mobile after ~15 minutes backgrounded), both the PDF.js render state and scroll position reset on reload.
- **Awkward image interaction** — images appear as small thumbnail buttons beside their segment; clicking opens a full-size preview in a JS-driven modal dialog, all bespoke code duplicated between the live app and the export.

## Goals

- Export produces a literal PDF file, openable in any standard PDF viewer (Adobe, Apple Books, browser built-in viewer, etc.) — no custom renderer, no CDN dependency, fully offline.
- Reading position and smooth/low-RAM scrolling are handled by the user's own PDF viewer, not by app code.
- Generated illustration images remain reachable from their originating segment, with enough reference text to understand why the image was generated.
- Original book PDF content is preserved unmodified.

## Non-goals

- No changes to the live in-app viewer (`PdfViewer.vue` / `ImageLayer.vue`) — this redesign is scoped to the export feature only.
- No backend/API changes. The export pipeline stays entirely client-side, consistent with the current architecture (the Django API is stateless and holds no export-related logic today).
- No handling of heatmap/narrative-importance score overlays — the current export doesn't include them, so this redesign doesn't add them either.
- HTML export is removed, not kept as a fallback option.

## Approach

Generate the PDF entirely client-side using `pdf-lib`, replacing the current HTML-generation pipeline in `services/frontend/app/utils/export.ts` and the orchestration in `services/frontend/app/composables/useExport.ts`.

### Why client-side `pdf-lib` over alternatives

- **Server-side generation (Django + reportlab/pypdf)** was considered and rejected. The API is deliberately stateless today (no models, no persistence — see `core/api.py`). Adding a PDF-assembly endpoint would introduce that complexity for an operation a browser can already perform, and would add a network dependency to a step that currently works offline once the segments/images are in memory.
- **Background job via Dramatiq** was considered and rejected as overengineering — Dramatiq exists for the heavy NLP/image-generation work already in this app; PDF assembly completes in seconds and doesn't need a job queue.

### Why a real inline image (mid-paragraph reflow) isn't feasible

The uploaded book PDF is a fixed-layout document — segments are polygon overlays on existing pages, not a text-flow model the app owns. Inserting an image truly "into the flow" would require re-extracting and re-typesetting the entire book as new text, losing the original page's fonts, existing images, footnotes, and pagination. This was explicitly ruled out in favor of the appendix-link approach below.

## Design

### Pipeline

On export, using data already held in app state (original PDF bytes, `Highlight[]` with polygons, and each highlight's current generated image):

1. Load the original PDF bytes into a `pdf-lib` `PDFDocument`. Existing page content streams are never rewritten — all additions are new overlays or new pages appended after the original content. This guarantees the original PDF content is preserved byte-for-byte within its own content streams.
2. For each `Highlight` that has a current image (`imageUrl` is non-null in its `EditorImageState`):
   - Re-encode the image from WebP to JPEG via canvas (WebP is not a valid PDF embedded-image format; JPEG is used over PNG since these are photographic/illustrated content and JPEG keeps offline file size down).
   - Draw the JPEG as a thumbnail overlay on the highlight's original page, at the position derived from its polygon coordinates. Thumbnail size follows the current HTML export's convention: 72×72pt on desktop-scale pages, scaling down proportionally on smaller pages (matching the 48×48px mobile breakpoint precedent in `export.ts`).
   - Add an invisible PDF link annotation over the thumbnail's bounding box, pointing to that image's appendix page (to be created in step 3). This is a native PDF annotation — clickable in any standard viewer, no JavaScript required.
3. After all original pages, append one appendix page per image, ordered by the page number of the originating highlight:
   - Full-resolution JPEG image, centered on the page.
   - Below it, centered text:
     - If the original segment text (`Highlight.text`) differs from the generation prompt actually used (`EditorState.currentPrompt` / the relevant `EditorHistoryItem.text` for the selected image): two labeled blocks, "Original text" and "Generation prompt."
     - If they are identical: a single unlabeled text block (no redundant heading).
   - A "← back to page N" link annotation at the bottom, pointing back to the originating page.
4. Save the assembled PDF and trigger a browser download.

### Code removal

The entire custom PDF.js-from-CDN renderer in `export.ts` is deleted as part of this work: the `IntersectionObserver`-based `setupLazyLoading`, `renderPage`, the `pageQueue`/`MAX_PARALLEL_RENDERS` render throttle, and the modal/backdrop JS for image preview. None of it is needed once the export is a plain PDF file — the user's own PDF viewer handles lazy rendering, memory management, and reading-position persistence.

### Data reused (no new data flow needed)

- `Highlight.polygons` — page-relative positions for thumbnail placement (existing field, `services/frontend/app/types/common.d.ts:26`).
- `Highlight.text` — original segment text for the appendix.
- `EditorImageState.imageUrl` — the currently selected generated image per highlight (`common.d.ts:47-51`).
- `EditorState.currentPrompt` / matching `EditorHistoryItem.text` — the generation prompt text for the appendix.

## Error handling

- Highlights with no generated image (`imageUrl` is null) get no marker and no appendix entry — same as today's behavior where only illustrated segments show a thumbnail.
- If WebP-to-JPEG re-encoding fails for a given image, skip that image's marker and appendix entry rather than failing the whole export (matches the existing pattern of per-image tolerance rather than an all-or-nothing export).

## Risks / trade-offs

- Building a large PDF with many embedded full-resolution images client-side is momentarily RAM-heavy during export generation. This is a one-time cost paid when the user clicks "export," not a recurring cost paid every time they open the book to read — which is the actual problem being solved.
- Very image-heavy books will produce a correspondingly large appendix and file size; no cap is imposed by this design, matching current behavior where every illustrated segment gets an entry.
