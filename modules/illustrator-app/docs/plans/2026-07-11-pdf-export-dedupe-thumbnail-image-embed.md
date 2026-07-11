# PDF Export: Dedupe Thumbnail/Appendix Image Embed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use olc-powers:subagent-driven-development (recommended) or olc-powers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop `buildExportPdf` from embedding each illustrated segment's JPEG twice — once for the in-page thumbnail, once for the appendix page — by reusing the single `PDFImage` returned from the first `embedJpg()` call for both draws.

**Architecture:** `buildExportPdf.ts` already embeds the JPEG once in Pass 1 (thumbnail overlay) and again in Pass 2 (appendix page) from the same raw bytes. Store the `PDFImage` object (not the raw bytes) on each `AppendixEntry` during Pass 1, and read it back in Pass 2 instead of calling `embedJpg` a second time.

**Tech Stack:** Nuxt 3 / TypeScript (existing), `pdf-lib` (existing dependency), Node's built-in `node:test` + `node --experimental-strip-types` for running the existing test suite (no new tests needed — see Task 1).

---

### Task 1: Reuse the embedded `PDFImage` instead of re-embedding in the appendix pass

**Files:**
- Modify: `services/frontend/app/utils/pdfExport/buildExportPdf.ts:17-24` (interface), `:70-104` (Pass 1), `:113-115` (Pass 2)

No new test is needed for this task. `buildExportPdf.test.ts` asserts on page counts and annotation presence, not on `embedJpg` call counts or embedded object counts, and this change doesn't alter any of those observable outputs — it only removes a redundant embed of already-identical bytes. The existing suite is the regression check: it must still pass unchanged after the edit (Step 3 below).

- [ ] **Step 1: Update the `AppendixEntry` interface to hold the embedded image instead of raw bytes**

In `services/frontend/app/utils/pdfExport/buildExportPdf.ts`, add `PDFImage` to the existing `pdf-lib` type import and swap the `jpegBytes` field for `jpegImage`:

```typescript
import { PDFDocument, StandardFonts } from "pdf-lib";
import type { PDFFont, PDFImage, PDFPage } from "pdf-lib";
```

```typescript
interface AppendixEntry {
  highlight: Highlight;
  prompt: string;
  jpegImage: PDFImage;
  originPage: PDFPage;
  originPageIndex: number;
  thumbnailRect: { x: number; y: number; width: number; height: number };
}
```

- [ ] **Step 2: Store the Pass-1 `PDFImage` on the appendix entry, and reuse it in Pass 2**

Still in `services/frontend/app/utils/pdfExport/buildExportPdf.ts`, in the Pass 1 loop, change the `appendixEntries.push` call to store `jpegImage` (the value already produced by `await doc.embedJpg(jpegBytes)` two lines above it) instead of `jpegBytes`:

```typescript
    appendixEntries.push({
      highlight,
      prompt: imageData.prompt,
      jpegImage,
      originPage,
      originPageIndex: placement.page,
      thumbnailRect: rect,
    });
```

Then, in the Pass 2 (appendix) loop, delete the redundant re-embed line and read the image off the entry instead:

```typescript
  for (const entry of appendixEntries) {
    const appendixPage = doc.addPage([appendixWidth, appendixHeight]);
    const jpegImage = entry.jpegImage;
```

(This replaces the old `const jpegImage = await doc.embedJpg(entry.jpegBytes);` line. Everything below it — `maxImageWidth`, `scale`, `drawImage`, etc. — is unchanged since it already reads from the local `jpegImage` variable.)

- [ ] **Step 3: Run the existing test suite to confirm no regression**

Run:
```bash
docker compose run --rm frontend node --experimental-strip-types --test app/utils/pdfExport/buildExportPdf.test.ts
```
Expected: all 4 existing tests pass unchanged (page-count assertions, annotation-count assertions) — same as before the edit.

- [ ] **Step 4: Typecheck**

Run (requires the `api` container running first):
```bash
docker compose up -d api
docker compose run --rm frontend npx nuxi typecheck
```
Expected: no new type errors. `entry.jpegImage` must resolve to `PDFImage` and satisfy `drawImage`'s expected argument type, and the removed `jpegBytes` field must have no remaining references (confirmed in the Background section of the design — it's read nowhere else).

- [ ] **Step 5: Commit**

```bash
git add services/frontend/app/utils/pdfExport/buildExportPdf.ts
git commit -m "fix(frontend): embed each PDF export image only once, reuse for thumbnail and appendix"
```

---

## Self-Review Notes

- **Spec coverage:** The spec's three bullet points (interface field swap, Pass 1 stores the `PDFImage`, Pass 2 drops its `embedJpg` call and reuses the entry's image) are each covered by Step 1 and Step 2 above. The spec's "No other files change" and "No new tests are needed" constraints are honored — only `buildExportPdf.ts` is touched, and the plan explains why no test file changes.
- **Placeholder scan:** No TBD/TODO markers; every step shows the exact code to write and the exact command to run with its expected result.
- **Type consistency:** `jpegImage: PDFImage` in the interface (Step 1) matches the value pushed in Step 2 (`jpegImage` from `await doc.embedJpg(jpegBytes)`, already declared as `PDFImage` by `pdf-lib`'s types) and the read-back (`entry.jpegImage`) in Pass 2.
