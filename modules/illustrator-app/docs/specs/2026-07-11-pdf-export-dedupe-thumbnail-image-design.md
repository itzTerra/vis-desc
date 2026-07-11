# PDF Export: Dedupe Thumbnail/Appendix Image Embed

**Goal:** Fix a redundant embed in `buildExportPdf.ts` — each illustrated segment's image is currently embedded into the PDF twice (once for the in-page thumbnail marker, once for the appendix page), even though both draws use identical JPEG bytes. Embed once, reuse the `PDFImage` for both draws.

## Background

While investigating why a PDF inspector reported the full 512x512 illustration as the backing image for the small (~48-72pt) thumbnail marker, we found the real issue isn't the thumbnail's resolution — it's that `buildExportPdf` calls `doc.embedJpg()` twice for the same bytes:

- Pass 1 (thumbnail overlay): `jpegImage = await doc.embedJpg(jpegBytes)`, drawn small on the origin page.
- Pass 2 (appendix page): `jpegImage = await doc.embedJpg(entry.jpegBytes)`, drawn full-size on the appendix page.

`pdf-lib`'s `embedJpg` creates a new XObject on every call, regardless of whether identical bytes were already embedded — it does not deduplicate. So every illustrated segment currently stores two full copies of its image in the output PDF, roughly doubling the image-driven portion of the file size.

Downscaling the thumbnail to a separate smaller variant was considered and explicitly deferred (out of scope here) — the dedup fix alone removes the larger inefficiency with no quality trade-offs and no added complexity.

## Design

In `services/frontend/app/utils/pdfExport/buildExportPdf.ts`:

- Replace the `jpegBytes: Uint8Array` field on the `AppendixEntry` interface with `jpegImage: PDFImage`. (Confirmed `jpegBytes` on `AppendixEntry` is read nowhere else — only by the redundant Pass 2 `embedJpg` call being removed — so it's a straight replacement, not an addition.)
- In Pass 1, after `jpegImage = await doc.embedJpg(jpegBytes)`, store that `jpegImage` reference on the `appendixEntries` push instead of the raw bytes.
- In Pass 2 (the appendix loop), remove `const jpegImage = await doc.embedJpg(entry.jpegBytes);` and use `entry.jpegImage` directly.

No other files change. No new tests are needed — `pdf-lib`'s embed call count isn't something the existing `buildExportPdf.test.ts` suite asserts on (it checks page counts and annotations), and this change doesn't alter any observable output shape. Manual verification: re-run the existing test suite (must still pass unchanged) and, optionally, inspect a real export's `pdf-lib`-produced object count/file size before and after to confirm the reduction.

## Non-goals

- No change to thumbnail image resolution, quality, or generation pipeline.
- No new "downscaled thumbnail variant" — deferred, not part of this change.
