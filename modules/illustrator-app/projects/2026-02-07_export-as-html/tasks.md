# Tasks

## Domain

- [x] Retain the uploaded PDF bytes in app state for export (File or ArrayBuffer alongside `pdfUrl`).
- [ ] Extend editor history to store the latest generated image `Blob` for export.
- [ ] Expose `ImageEditor` getters via `defineExpose` and collect refs in `ImageLayer` to return `{ highlightId -> imageBlob }` for open editors.
- [ ] Define the export data snapshot (PDF bytes + collected image blob map, convert blobs to data URLs at export time).
- [x] Add `pdfjs-dist` as a frontend dependency and document the bundled version used for export.
- [ ] Create the HTML export template that embeds data, loads `pdfjs-dist`, configures offline-friendly assets (system fonts, no `cMapUrl`), wires inline worker setup, and lazy page rendering.
- [ ] Implement lazy loading in the exported HTML using IntersectionObserver.

## Components

- [ ] Add an export action entry point in the viewer/top bar that opens an OS file modal.
- [ ] Ensure export logic only includes image buttons for highlights whose current editor history item has an `imageUrl`.
- [ ] Add exported HTML UI elements: image buttons and a responsive modal overlay viewer (max 512x512).

## Routes

- [ ] Confirm export flow works within the existing index page without new routes.

## Cleanup

- [ ] Update help copy or guidance text to mention the HTML export.
- [ ] Manual verification: export with and without images, lazy loading behavior, modal image preview size, offline usage.
