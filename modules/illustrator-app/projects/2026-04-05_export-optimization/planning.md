---
type: improvement
size: M
---

# Export Optimization: WebP Conversion and Lazy Image Parsing

## Overview

The current HTML export embeds all generated images as PNG base64 data URIs directly inside a JavaScript literal (`JSON.stringify(snapshot.highlights)`). For books with many illustrations, this creates two problems: (1) PNG images at 512x512 are unnecessarily large when lossy compression is acceptable, and (2) V8 must parse the entire JSON string containing all image data at page load, even for images the user never scrolls to. This project converts images to WebP before export and restructures the exported HTML so image data is stored outside the main JS execution path and parsed lazily per page.

## Goals

- Reduce exported HTML file size by converting PNG images to WebP (lossy) before embedding.
- Improve exported HTML startup performance by deferring image data parsing until the page containing the image is rendered.
- Maintain the existing single-click export UX (no additional UI choices).

## User Stories

### As a user, I want my exported HTML files to be smaller so they download faster and take less disk space.

**Acceptance criteria:**

- [ ] Exported images are encoded as WebP instead of PNG.
- [ ] Image quality is visually acceptable at the chosen WebP quality level (target: 0.80).
- [ ] 512x512 resolution is preserved (no dimension reduction).

### As a user, I want the exported HTML to open quickly even when it contains many illustrations.

**Acceptance criteria:**

- [ ] The exported HTML does not parse all image data at startup.
- [ ] Image data for a given page is only parsed when that page is rendered.
- [ ] The first page renders without waiting for image data from other pages to be parsed.

### As a user, I want the export to work the same way as before (single button click, single file download).

**Acceptance criteria:**

- [ ] The export button behavior and placement are unchanged.
- [ ] The output is still a single `.html` file downloaded via Blob + anchor click.
- [ ] The exported HTML functions identically to the current version from a user perspective.

## Requirements

### Functional

- Convert each PNG data URI to WebP before embedding in the exported HTML.
- Use the browser Canvas API (`canvas.toDataURL('image/webp', quality)`) for conversion -- no new dependencies needed. If the browser does not support WebP encoding, `toDataURL` silently returns PNG; this is acceptable since size reduction is best-effort.
- Store image data in the exported HTML outside the main `<script type="module">` block, using per-highlight `<script type="application/json">` tags keyed by highlight ID.
- The exported HTML's page-render function should parse the relevant image data tags only when rendering a page that contains highlights with images.
- The highlights array in the main JS should contain metadata (id, text, polygons) but not `imageUrl` -- image URLs are resolved lazily from the embedded JSON tags.
- Provide a quality constant (default 0.80) that can be adjusted easily.

### Non-functional

- No new npm dependencies required (Canvas and Blob APIs are browser-native).
- Export time may increase slightly due to WebP conversion but should remain under 5 seconds for typical documents (10-30 images).
- WebP conversion is async (image loading via `img.onload`) and is performed sequentially via `for...of` with `await` inside `createExportSnapshot()`.
- The exported HTML must continue to work in all modern browsers (Chrome, Firefox, Safari, Edge).

## Scope

### Included

- WebP conversion of image data URIs during export snapshot creation.
- Restructuring the exported HTML template to use lazy image data tags.
- Updating the page-render function in the template to resolve images lazily.
- Manual verification of WebP conversion and lazy image loading in the exported HTML.

### Excluded

- Changing the export from single-file HTML to ZIP or multi-file format (tracked as a future option).
- Changing image resolution or adding quality settings to the UI.
- Server-side export or image processing.
- Any changes to the in-app image generation, editing, or display pipeline.

---

## Current State

The export flow works as follows:

1. User clicks "Export" button in the toolbar.
2. `handleExportConfirm()` in `index.vue` collects image URLs from `ImageLayer.getExportImages()`.
3. `useExport().confirmExport()` calls `createExportSnapshot()` which reads the PDF file into base64 and maps highlights with their image URLs (PNG data URIs like `data:image/png;base64,...`).
4. `generateExportHtml()` builds a self-contained HTML string with all data inlined:
   - PDF bytes as a base64 string in a JS template literal.
   - The entire highlights array (including image data URIs) serialized via `JSON.stringify()` into a JS `const`.
5. `downloadExport()` creates a Blob and triggers download via anchor click.

Images originate from the backend as base64 PNG strings (`data:image/png;base64,${image_b64}`) in `ImageEditor.applyGenerateResult()`.

The exported HTML already has lazy PDF page rendering via IntersectionObserver, but image data is eagerly parsed as part of the highlights JSON.

## Key Files

- `services/frontend/app/utils/export.ts` - Core export logic: `createExportSnapshot()`, `generateExportHtml()`, `downloadExport()`.
- `services/frontend/app/composables/useExport.ts` - Export composable used by the main page.
- `services/frontend/app/pages/index.vue` - Export button handler and image collection.
- `services/frontend/app/components/ImageEditor.vue` - Where PNG data URIs originate (`applyGenerateResult`).
- `services/frontend/app/components/ImageLayer.vue` - `getExportImages()` collects image URLs from editors.
- `services/frontend/app/types/common.d.ts` - `Highlight` and `EditorImageState` types.

## Existing Patterns

- The export template already uses IntersectionObserver for lazy page rendering with configurable thresholds.
- Image data flows as data URI strings (`data:image/png;base64,...`) through the component tree.
- The `arrayBufferToBase64Chunked()` utility in `export.ts` handles large binary-to-base64 conversion.
- Export types (`ExportHighlight`, `ExportSnapshot`) are defined locally in `export.ts`.

## Decisions

### D1: WebP conversion via Canvas API

**Decision**: Convert PNG data URIs to WebP using `canvas.drawImage()` + `canvas.toDataURL('image/webp', 0.80)` in the browser.

**Rationale**: The Canvas API is universally available in modern browsers and introduces no dependencies. WebP at quality 0.80 typically achieves 60-80% size reduction vs PNG for generated illustration-style images. The 512x512 images are small enough that canvas conversion is fast (< 50ms each). `convertToWebP` returns a `Promise<string>`. Wrap `img.onload` in a `new Promise`: set `img.src = dataUri`, then inside `onload` call `ctx.drawImage(img, 0, 0)` and resolve with `canvas.toDataURL('image/webp', quality)`. If the browser does not support WebP encoding, `toDataURL` silently returns PNG; this is acceptable since size reduction is best-effort.

**Implementation**: Add an `async function convertToWebP(dataUri: string, quality = 0.80): Promise<string>` utility in `export.ts` that loads the data URI into an Image, draws to an offscreen canvas, and returns a WebP data URI via `canvas.toDataURL('image/webp', quality)`. The function is called within `createExportSnapshot()` over the image entries -- no separate `convertAllImagesToWebP` function is needed.

### D2: Lazy image storage via per-highlight script tags

**Decision**: Store each image's data URI in a separate `<script type="application/json" data-image-id="[highlightId]">` tag in the exported HTML, rather than in the main JS highlights array.

**Rationale**: When images are inside `JSON.stringify(highlights)`, V8 must parse the entire multi-megabyte string at startup. By moving images to individual `<script type="application/json">` tags, the browser treats them as inert text nodes. The page-render function retrieves and parses only the relevant tags when a page scrolls into view, using `document.querySelector('script[data-image-id="X"]')` and reading `.textContent`. This avoids parsing image data for pages the user never visits.

**Implementation**: The highlights JS array retains `id`, `text`, and `polygons` -- no `hasImage` flag is needed. The `renderPage()` function, after rendering PDF content, calls `getImageUrl(highlight.id)` for each highlight and skips button creation if it returns null. The URL is resolved once per button: `const url = getImageUrl(highlight.id); img.src = url; button.onclick = () => openImageModal(url);`. The `hasImages` predicate in `generateExportHtml()` checks `Object.keys(snapshot.imageData).length > 0` (not `h.imageUrl`) to decide whether to include the modal overlay markup.

### D3: Single-file HTML remains the only export format

**Decision**: Keep the single-file HTML export as the sole option. Do not add ZIP export at this time.

**Rationale**: The WebP conversion and lazy parsing optimizations should bring file sizes to an acceptable range for typical use cases. ZIP export adds complexity (multi-file generation, different download mechanism) and can be added later if single-file sizes prove insufficient for very large books. Note that `jszip` is already available as a dependency, so the cost of adding ZIP export in the future is low.

**Future consideration**: If books with 100+ illustrations still produce impractically large files after WebP conversion, a ZIP export could be added as an alternative using `jszip` (already a dependency in `services/frontend/package.json`). This would extract images into separate files and reference them from the HTML. The lazy-parsing architecture from D2 would transfer naturally to this approach (replace script tag reads with fetch calls to relative paths).

### D4: Sequential WebP conversion via `for...of`

**Decision**: Convert all images to WebP sequentially using `for...of` with `await` over the image entries inside `createExportSnapshot()`.

**Rationale**: Image loading is async (`img.onload`). Conversions run sequentially via `for...of` with `await convertToWebP(...)` inside `createExportSnapshot`. Sequential is fine -- the bottleneck is image decode, not JS thread contention. Each conversion is fast (< 50ms for 512x512), so sequential execution is simple and sufficient. The results are collected into the `imageData` record.

### Note: jszip availability

`jszip` is already listed as a dependency in `services/frontend/package.json`. If ZIP export is needed in the future, the dependency is already available at no additional cost.

---

## Alternatives Considered

These were evaluated and excluded from this project's scope. They remain viable future options.

### A1: ZIP export (multi-file)

**How it works**: Use `jszip` (already a dependency) to package the PDF, images as separate WebP files, and a small `index.html` into a `.zip` download. Images are referenced by relative path. The user extracts and opens `index.html`.

**Benefits**: Dramatically smaller individual files. Images are separate files so the browser only decodes what is visible. No base64 overhead on the PDF (it can be stored as binary in the ZIP). For very large books (100+ illustrations, 200+ page PDFs), this is the most effective approach.

**Drawbacks**: Requires extraction before opening — less convenient. Two-step UX. The HTML references sibling files, so moving `index.html` out of the folder breaks it.

**When to choose over this project**: If books with 50+ illustrations still produce impractically large or slow-to-open files after WebP + lazy parsing, ZIP is the natural next step. `jszip` is already available so implementation cost is low.

### A2: Annotated PDF export via pdf-lib

**How it works**: Use `pdf-lib` (not currently a dependency) to take the original PDF and stamp AI-generated images alongside the highlighted pages, producing a new annotated PDF. No custom viewer — the user opens it in their native PDF reader.

**Benefits**: Very compact output (JPEG/WebP images embedded in PDF natively). No custom viewer code to maintain. Works fully offline. Native PDF readers are fast and memory-efficient.

**Drawbacks**: Requires `pdf-lib` (~500 KB). Loses the custom sidebar image layout and modal UI. Image positioning relative to highlight polygons is complex to replicate in PDF coordinates. No text layer customization. The user loses the interactive highlight overlay.

**When to choose**: If the primary goal shifts from "interactive viewer" to "archival document", this is the cleanest option.

### A3: Deflate/WASM compression of the PDF payload

**How it works**: Compress the PDF `ArrayBuffer` with `fflate` (pure JS, ~16 KB) before base64-encoding it. Embed the compressed bytes. In the exported viewer, decompress with `fflate` before passing to pdf.js.

**Benefits**: Text-heavy PDFs compress 50–80% with deflate. Could significantly reduce the largest component of the export file (the PDF itself).

**Drawbacks**: Requires adding `fflate` as a dependency (or bundling it inline into the exported HTML). Decompression runs on the browser's main thread when the export is opened, adding startup latency. pdf.js already supports streaming, but the decompressed bytes must be fully available first. Adds ~16 KB of inline JS to the exported file.

**When to choose**: If the PDF itself (not the images) is the primary size bottleneck — e.g., image-heavy scanned books where WebP image savings are already maximized. Best combined with A1 (ZIP) since ZIP uses deflate natively and avoids the inline WASM path entirely.

### A4: Server-side export

**How it works**: Send the PDF bytes and highlight data (image URLs, polygons) to a new Django API endpoint. The backend renders the HTML (or PDF) and streams it back as a download. Offloads memory pressure entirely from the browser.

**Benefits**: No browser RAM constraint. Can apply server-side image processing (resize, compress, format convert) more efficiently. Can produce formats like PDF or EPUB that are hard to generate client-side.

**Drawbacks**: Requires a new API endpoint, file upload (potentially large), and streaming response. Adds server-side storage/temp file handling. Breaks the fully client-side, stateless architecture of the current app. Increases latency (round-trip vs local generation).

**When to choose**: If browser crashes persist even after client-side optimizations, or if server-side formats (PDF annotation, EPUB) become a requirement.
