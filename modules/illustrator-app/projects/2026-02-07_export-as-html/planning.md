---
type: feature
size: M
---

# Export as HTML (PDF + Generated Images)

## Overview

Create an export flow that produces a standalone HTML file containing the current PDF rendered with `pdfjs-dist` and minimal controls for viewing generated images. The export focuses on portability and performance through embedded PDF bytes, inline worker setup, and lazy page rendering.

## Goals

- Export a single HTML file that renders the current PDF with `pdfjs-dist` using embedded bytes.
- Include only the PDF rendering surface plus buttons for generated images.
- Provide a simple export modal UX and an in-export image modal overlay at 512x512.

## User Stories

### As a user, I want to export my current PDF and generated images to a single HTML file so I can view and share it offline.

**Acceptance criteria:**

- [ ] Export produces a single HTML file that renders the PDF via embedded bytes and `pdfjs-dist`.
- [ ] Exported HTML includes only the PDF rendering area and buttons for generated images (no editor or scoring UI).
- [ ] Buttons appear only for highlights whose current editor history item has an `imageUrl`.
- [ ] Clicking a button opens a modal overlay showing the generated image at 512x512.

### As a user, I want large PDFs to stay responsive in the exported HTML so that page rendering does not stall my browser.

**Acceptance criteria:**

- [ ] Exported HTML lazy loads pages using IntersectionObserver.
- [ ] Visible pages render first; offscreen pages remain placeholders until scrolled into view.

### As a user, I want a clear export flow so I understand what file will be produced and can confirm it.

**Acceptance criteria:**

- [ ] Export action opens a file modal that lets me confirm the export (name and cancel).
- [ ] Successful export triggers a download of the generated HTML file.

## Requirements

### Functional

- Retain the uploaded PDF `File` in memory until export/reset to access bytes for embedding.
- Release retained PDF bytes after export completes or when the session is reset.
- Embed the PDF bytes directly in the exported HTML and render with `pdfjs-dist` on load.
- Inline the `pdfjs-dist` worker via a Blob URL to keep the export single-file and offline.
- Include generated image buttons only when the current editor history item has an `imageUrl`.
- Store generated image blobs at generation time for export portability.
- Convert image blobs to data URLs at export time for embedding in the exported HTML.
- Provide a modal overlay in the exported HTML that displays images at max 512x512 and scales down on small screens.
- Include an export file modal in the app UX for naming/confirming the HTML download.
- Use lazy loading for PDF pages in the exported HTML (IntersectionObserver, preload +/-3 pages, thresholds `[0, 0.1, 0.25, 0.5, 0.75, 1.0]` to match the viewer).
- Expose export image getters from `ImageEditor` and `ImageLayer` to collect current image data at export time.
- Export only the current history item image for open editors; closed editors are excluded.

### Non-functional

- Exported HTML should work offline with no external network dependencies.
- Configure `pdfjs-dist` to avoid external assets (no `cMapUrl` or remote fonts; prefer system fonts), accepting non-Latin fidelity tradeoffs.
- Export should avoid duplicate image data in memory by converting blobs to data URLs only at export time.
- PDF rendering should remain responsive on multi-page documents.
- Keep the exported UI minimal and focused on viewing.

## Scope

### Included

- Export flow and HTML template generation in the frontend.
- Minimal exported UI: PDF render surface plus generated image buttons and image modal.
- Lazy loading logic for pages in exported HTML.

### Excluded

- Editing, scoring, or model selection UI in exported HTML.
- Server-side HTML generation.
- PDF text extraction or highlight editing in the export.

---

## Current State

- The app renders PDFs in the viewer using `vue-pdf-embed`, with page visibility controlled by an IntersectionObserver-based lazy loader.
- Highlights are rendered in `HighlightLayer` and open per-highlight image editors that store prompt/image history locally via `useEditorHistory`.
- Generated images are attached to the current editor history item and displayed inside `ImageEditor` instances; there is no HTML export feature yet.
- The uploaded PDF bytes are not currently retained after creating a blob URL.

## Key Files

- `services/frontend/app/pages/index.vue` - Main page layout and PDF flow.
- `services/frontend/app/components/PdfViewer.vue` - PDF rendering and lazy page visibility logic.
- `services/frontend/app/components/HighlightLayer.vue` - Highlight overlays and editor entry point.
- `services/frontend/app/components/ImageLayer.vue` - Positions image editors relative to PDF pages.
- `services/frontend/app/components/ImageEditor.vue` - Editor history and generated image handling.
- `services/frontend/app/composables/useEditorHistory.ts` - Editor history and current item logic.
- `services/frontend/app/assets/css/main.css` - Global PDF and highlight styling.

## Existing Patterns

- IntersectionObserver-based page visibility with a preload window in `PdfViewer.vue`.
- Component-scoped state for editor history via `useEditorHistory`.
- DaisyUI modal and button styling used across dialogs and toolbars.

## Decisions

### C1: Embed PDF bytes and render via `pdfjs-dist` in exported HTML

**Decision**: Generate a standalone HTML file that embeds the PDF bytes and renders with `pdfjs-dist`, using lazy page loading in the export.

**Rationale**: Keeps exports self-contained and consistent with the in-app PDF experience while ensuring performance on large documents.

**Implementation**: Export template includes embedded data, inline worker setup, `pdfjs-dist` bundled as a direct frontend dependency and configured without external assets (system fonts, no `cMapUrl`), minimal styles, and a small script to render pages on demand using IntersectionObserver.

### Export Image Access

**Decision**: Collect exportable image data via `defineExpose` getters on `ImageEditor` and `ImageLayer`, scoped to currently open editors.

**Rationale**: Matches existing component-local editor history without introducing a new global store.

**Implementation**: `ImageEditor` stores the latest generated image `Blob` alongside its display URL and exposes `getExportImage()` via `defineExpose` returning `{ highlightId, imageBlob } | null` for the current history item. `ImageLayer` collects editor refs from the `v-for` (ref array) and exposes `getExportImages()` returning `{ highlightId -> imageBlob }` for currently open editors only; export converts these blobs to data URLs on demand.

### Export Button Placement

**Decision**: Place the “Export as HTML” button in the top bar immediately to the right of `ModelSelect`.

**Rationale**: Aligns with the requested location and keeps export in the primary workflow.

### Export Modal Pattern

**Decision**: Implement the export modal using the existing fixed overlay dialog pattern (same structure as `ModelDownloadDialog`).

**Rationale**: Reuses established UI and avoids introducing a new modal system.

### Future Options: A and B

**Decision**: Track these as future alternatives, not part of this implementation.

**Rationale**: Provide possible server-side or pre-rendered export paths if client-side rendering is insufficient.

**Implementation**:
- **A (pdf2htmlEX)**: Server-side conversion to static HTML/CSS.
- **B (PyMuPDF)**: Server-side rasterization to images with lightweight HTML layout.
