# PDF Export Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use olc-powers:subagent-driven-development (recommended) or olc-powers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current HTML export (a self-contained HTML file that renders the PDF via CDN-loaded PDF.js) with a literal, offline-openable PDF file built client-side with `pdf-lib`, per `docs/specs/2026-07-11-pdf-export-design.md`. As part of this, fix the thumbnail-duplication bug: when a segment (`Highlight`) has polygon data on more than one page (e.g. because it visually spans a page boundary in the source PDF), only the **later** page gets a thumbnail marker — not both.

**Architecture:** New `services/frontend/app/utils/pdfExport/` directory holds small, independently-testable pure/near-pure modules (thumbnail placement geometry, text wrapping, link-annotation helper, image re-encoding) plus an orchestrator (`buildExportPdf.ts`) that assembles the final PDF with `pdf-lib`: original pages are left untouched except for a thumbnail image overlay + link annotation per illustrated segment, followed by one appended appendix page per image (full-res image, original text / prompt, back-link). The existing HTML-generation pipeline in `utils/export.ts` is deleted; `useExport.ts` and `index.vue` are updated to call the new pipeline.

**Tech Stack:** Nuxt 3 / Vue 3 / TypeScript (existing), `pdf-lib` (new dependency), Node's built-in `node:test` + `node --experimental-strip-types` for unit tests on the pure modules (matches the existing `heatmapUtils.test.ts` convention already in the repo — no test framework is currently installed).

---

## Before you start

- All frontend commands run inside the Docker container per project convention: `docker compose run --rm frontend <cmd>` (from the `illustrator-app` module root, i.e. one level above `services/frontend`).
- New test files in `services/frontend/app/utils/pdfExport/` must use **relative imports with explicit `.ts` extensions** for any runtime (non-type-only) import of a sibling module in that directory. This is required so the tests are runnable directly with `node --experimental-strip-types --test <file>`, which does not understand the Nuxt `~` alias. Type-only imports (`import type { X } from "~/types/common"`) are fine to keep using the `~` alias — Node's type-stripping erases them entirely before module resolution runs, so they never need to resolve at runtime. (Verified: a `~`-aliased *type-only* import does not break `node --experimental-strip-types --test`; a `~`-aliased *runtime* import does.)
- Run each `pdfExport/*.test.ts` file with:
  ```bash
  docker compose run --rm frontend node --experimental-strip-types --test app/utils/pdfExport/<name>.test.ts
  ```
  (from `services/frontend` inside the container's working directory — the compose service's working dir is already `services/frontend` in this project; if a task's run command 404s on the path, `cd services/frontend` first.)

---

### Task 1: Add the `pdf-lib` dependency

**Files:**
- Modify: `services/frontend/package.json`

- [ ] **Step 1: Add the dependency**

Edit `services/frontend/package.json`, adding `pdf-lib` to `dependencies` (alphabetical position, between `onnxruntime-web` and `protobufjs`):

```json
    "onnxruntime-web": "^1.24.1",
    "pdf-lib": "^1.17.1",
    "protobufjs": "^8.0.0",
```

- [ ] **Step 2: Install**

```bash
docker compose run --rm frontend pnpm install
```

Expected: install completes, `pdf-lib` appears in `services/frontend/node_modules/pdf-lib`, and `pnpm-lock.yaml` is updated.

- [ ] **Step 3: Commit**

```bash
git add services/frontend/package.json pnpm-lock.yaml
git commit -m "chore(frontend): add pdf-lib dependency for PDF export redesign"
```

(If `pnpm-lock.yaml` lives elsewhere, e.g. `services/frontend/pnpm-lock.yaml`, adjust the path — check with `git status` before committing.)

---

### Task 2: Thumbnail placement geometry (includes the split-segment bug fix)

**Files:**
- Create: `services/frontend/app/utils/pdfExport/thumbnailPlacement.ts`
- Test: `services/frontend/app/utils/pdfExport/thumbnailPlacement.test.ts`

This is the module that decides **which page** gets a segment's thumbnail and **where** on that page. The bug fix is `selectThumbnailPage`: today's HTML export (`services/frontend/app/utils/export.ts:453`) puts a thumbnail on *every* page key present in `Highlight.polygons`, so a segment whose polygons span two pages gets two thumbnails. The fix: only the highest (later) page key gets one.

- [ ] **Step 1: Write the failing tests**

```typescript
// services/frontend/app/utils/pdfExport/thumbnailPlacement.test.ts
import assert from "node:assert/strict";
import test from "node:test";

import {
  selectThumbnailPage,
  computeMinY,
  computeThumbnailPlacement,
  computeThumbnailSize,
  computeThumbnailRect,
} from "./thumbnailPlacement.ts";

test("selectThumbnailPage picks the only page when a segment doesn't span pages", () => {
  assert.equal(selectThumbnailPage({ 3: [[0, 0.1, 0.2, 0.1, 0.2, 0.2, 0, 0.2]] }), 3);
});

test("selectThumbnailPage picks the later page when a segment spans a page break", () => {
  // Regression test for the export bug: a segment split across pages 4 and 5
  // must resolve to a single thumbnail page, and it must be the later one.
  const polygons = {
    4: [[0, 0.9, 0.2, 0.9, 0.2, 1.0, 0, 1.0]],
    5: [[0, 0.0, 0.2, 0.0, 0.2, 0.1, 0, 0.1]],
  };
  assert.equal(selectThumbnailPage(polygons), 5);
});

test("selectThumbnailPage returns null for a segment with no page data", () => {
  assert.equal(selectThumbnailPage({}), null);
});

test("computeMinY finds the smallest normalized y across all polygon points", () => {
  const polygons = [
    [0, 0.5, 0.2, 0.5, 0.2, 0.6, 0, 0.6],
    [0, 0.3, 0.2, 0.3, 0.2, 0.4, 0, 0.4],
  ];
  assert.equal(computeMinY(polygons), 0.3);
});

test("computeThumbnailPlacement uses only the selected page's polygons for minY", () => {
  const polygons = {
    4: [[0, 0.9, 0.2, 0.9, 0.2, 1.0, 0, 1.0]], // minY 0.9 on page 4 — must be ignored
    5: [[0, 0.05, 0.2, 0.05, 0.2, 0.1, 0, 0.1]], // minY 0.05 on page 5 — this one wins
  };
  const placement = computeThumbnailPlacement(polygons);
  assert.deepEqual(placement, { page: 5, minY: 0.05 });
});

test("computeThumbnailPlacement returns null when polygons is empty", () => {
  assert.equal(computeThumbnailPlacement({}), null);
});

test("computeThumbnailSize returns the base size at the reference page width", () => {
  assert.equal(computeThumbnailSize(612), 72);
});

test("computeThumbnailSize scales down for narrower pages but clamps to minSize", () => {
  assert.equal(computeThumbnailSize(300), 48); // 72 * 300/612 ≈ 35.3 -> clamped to 48
});

test("computeThumbnailSize never exceeds baseSize for wider pages", () => {
  assert.equal(computeThumbnailSize(1200), 72);
});

test("computeThumbnailRect anchors to the top-right near minY, converted to PDF's bottom-up y axis", () => {
  // pageHeight 800, size 72, minY 0 (top of page) -> box should hug the top edge
  const rect = computeThumbnailRect(0, 600, 800, 72);
  assert.equal(rect.width, 72);
  assert.equal(rect.height, 72);
  assert.equal(rect.x, 600 - 72 - 8); // right-aligned with an 8pt margin
  assert.equal(rect.y, 800 - 72); // clamped so the box stays on the page
});

test("computeThumbnailRect clamps the bottom edge so the box never runs off the page", () => {
  const rect = computeThumbnailRect(1, 600, 800, 72); // minY 1 = bottom of page
  assert.equal(rect.y, 0);
});
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
docker compose run --rm frontend node --experimental-strip-types --test app/utils/pdfExport/thumbnailPlacement.test.ts
```

Expected: FAIL — `Cannot find module './thumbnailPlacement.ts'` (file doesn't exist yet).

- [ ] **Step 3: Implement**

```typescript
// services/frontend/app/utils/pdfExport/thumbnailPlacement.ts
export interface ThumbnailPlacement {
  page: number;
  minY: number;
}

export interface ThumbnailRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

/**
 * A Highlight's polygons can have entries on more than one page when the
 * segment's source text visually spans a page break. Only the later page
 * gets a thumbnail marker, so a split segment never shows two thumbnails.
 */
export function selectThumbnailPage(polygons: Record<number, number[][]>): number | null {
  const pages = Object.keys(polygons).map(Number);
  if (pages.length === 0) return null;
  return Math.max(...pages);
}

export function computeMinY(polygons: number[][]): number {
  let minY = Infinity;
  for (const polygon of polygons) {
    for (let i = 1; i < polygon.length; i += 2) {
      minY = Math.min(minY, polygon[i]);
    }
  }
  return minY;
}

export function computeThumbnailPlacement(
  polygons: Record<number, number[][]>,
): ThumbnailPlacement | null {
  const page = selectThumbnailPage(polygons);
  if (page === null) return null;

  const pagePolygons = polygons[page];
  if (!pagePolygons || pagePolygons.length === 0) return null;

  const minY = computeMinY(pagePolygons);
  if (!Number.isFinite(minY) || minY < 0) return null;

  return { page, minY };
}

/**
 * Thumbnails are 72x72pt at the reference page width (US Letter, 612pt),
 * scaling down proportionally for narrower pages, clamped to a 48pt floor —
 * matches the 72px desktop / 48px mobile-breakpoint convention from the
 * previous HTML export (services/frontend/app/utils/export.ts).
 */
export function computeThumbnailSize(
  pageWidth: number,
  baseSize = 72,
  basePageWidth = 612,
  minSize = 48,
): number {
  const scaled = baseSize * (pageWidth / basePageWidth);
  return clamp(scaled, minSize, baseSize);
}

/**
 * Positions the thumbnail box in the page's top-right area, vertically
 * centered on the segment's starting point (minY, normalized 0-1, top-down —
 * same convention as Highlight.polygons). PDF page coordinates are
 * bottom-up, so this flips the y axis.
 */
export function computeThumbnailRect(
  minY: number,
  pageWidth: number,
  pageHeight: number,
  size: number,
  margin = 8,
): ThumbnailRect {
  const x = clamp(pageWidth - size - margin, 0, Math.max(0, pageWidth - size));
  const y = clamp(pageHeight * (1 - minY) - size / 2, 0, Math.max(0, pageHeight - size));
  return { x, y, width: size, height: size };
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
docker compose run --rm frontend node --experimental-strip-types --test app/utils/pdfExport/thumbnailPlacement.test.ts
```

Expected: PASS, all 11 tests green.

- [ ] **Step 5: Commit**

```bash
git add services/frontend/app/utils/pdfExport/thumbnailPlacement.ts services/frontend/app/utils/pdfExport/thumbnailPlacement.test.ts
git commit -m "feat(frontend): add thumbnail placement geometry, fixing duplicate thumbnails on split segments"
```

---

### Task 3: Text wrapping utility

**Files:**
- Create: `services/frontend/app/utils/pdfExport/textWrap.ts`
- Test: `services/frontend/app/utils/pdfExport/textWrap.test.ts`

Needed for the appendix pages, which draw the original segment text and generation prompt as wrapped paragraphs using a `pdf-lib` font's width measurement. Kept generic (`measureWidth` is injected) so it doesn't need `pdf-lib` or a real font to test.

- [ ] **Step 1: Write the failing tests**

```typescript
// services/frontend/app/utils/pdfExport/textWrap.test.ts
import assert from "node:assert/strict";
import test from "node:test";

import { wrapText } from "./textWrap.ts";

// Fixed-width fake measurer: 10 units per character, for predictable tests.
const measureWidth = (s: string) => s.length * 10;

test("wrapText keeps a short string on one line", () => {
  assert.deepEqual(wrapText("hello world", 200, measureWidth), ["hello world"]);
});

test("wrapText breaks onto a new line once the max width would be exceeded", () => {
  // "hello world" is 110 units; "hello" is 50, "hello world" is 110 > 100
  assert.deepEqual(wrapText("hello world", 100, measureWidth), ["hello", "world"]);
});

test("wrapText wraps multiple words across several lines", () => {
  const result = wrapText("one two three four five", 60, measureWidth);
  assert.deepEqual(result, ["one two", "three", "four", "five"]);
});

test("wrapText keeps an over-long single word on its own line rather than dropping it", () => {
  const result = wrapText("supercalifragilisticexpialidocious short", 100, measureWidth);
  assert.deepEqual(result, ["supercalifragilisticexpialidocious", "short"]);
});

test("wrapText collapses repeated whitespace", () => {
  assert.deepEqual(wrapText("hello    world", 200, measureWidth), ["hello world"]);
});

test("wrapText returns an empty array for empty input", () => {
  assert.deepEqual(wrapText("", 200, measureWidth), []);
});
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
docker compose run --rm frontend node --experimental-strip-types --test app/utils/pdfExport/textWrap.test.ts
```

Expected: FAIL — `Cannot find module './textWrap.ts'`.

- [ ] **Step 3: Implement**

```typescript
// services/frontend/app/utils/pdfExport/textWrap.ts
/**
 * Greedy word-wrap: appends words to the current line until measureWidth
 * would exceed maxWidth, then starts a new line. An unbreakable overlong
 * word is placed on its own line rather than truncated.
 */
export function wrapText(
  text: string,
  maxWidth: number,
  measureWidth: (s: string) => number,
): string[] {
  const words = text.split(/\s+/).filter(Boolean);
  const lines: string[] = [];
  let current = "";

  for (const word of words) {
    const candidate = current ? `${current} ${word}` : word;
    if (!current || measureWidth(candidate) <= maxWidth) {
      current = candidate;
    } else {
      lines.push(current);
      current = word;
    }
  }

  if (current) lines.push(current);
  return lines;
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
docker compose run --rm frontend node --experimental-strip-types --test app/utils/pdfExport/textWrap.test.ts
```

Expected: PASS, all 6 tests green.

- [ ] **Step 5: Commit**

```bash
git add services/frontend/app/utils/pdfExport/textWrap.ts services/frontend/app/utils/pdfExport/textWrap.test.ts
git commit -m "feat(frontend): add word-wrap utility for PDF appendix text"
```

---

### Task 4: Image data URI byte conversion + browser JPEG re-encode

**Files:**
- Create: `services/frontend/app/utils/pdfExport/imageReencode.ts`
- Test: `services/frontend/app/utils/pdfExport/imageReencode.test.ts`

Generated illustration images are held as `data:image/png;base64,...` URIs (see `applyGenerateResult` in `services/frontend/app/components/ImageEditor.vue:222-226`). `pdf-lib` can embed PNG directly, but per the spec we standardize on JPEG for file-size reasons, so every image is re-encoded via canvas regardless of its source format. `dataUriToBytes` is pure (base64 decode) and unit-tested; `reencodeToJpeg` needs a real `Image`/`canvas`, which isn't available under `node:test` — it's covered by the manual QA pass in Task 12, same as the previous `convertToWebP` helper it replaces (`services/frontend/app/utils/export.ts:15-32`) was never unit tested either.

- [ ] **Step 1: Write the failing test**

```typescript
// services/frontend/app/utils/pdfExport/imageReencode.test.ts
import assert from "node:assert/strict";
import test from "node:test";

import { dataUriToBytes } from "./imageReencode.ts";

test("dataUriToBytes decodes the base64 payload back to the original bytes", () => {
  const original = new Uint8Array([1, 2, 3, 250, 0, 255]);
  const base64 = Buffer.from(original).toString("base64");
  const dataUri = `data:application/octet-stream;base64,${base64}`;

  assert.deepEqual(Array.from(dataUriToBytes(dataUri)), Array.from(original));
});

test("dataUriToBytes handles a real-shaped image/jpeg data URI prefix", () => {
  const original = new Uint8Array([0xff, 0xd8, 0xff, 0xe0]); // JPEG magic bytes
  const base64 = Buffer.from(original).toString("base64");
  const dataUri = `data:image/jpeg;base64,${base64}`;

  assert.deepEqual(Array.from(dataUriToBytes(dataUri)), Array.from(original));
});
```

- [ ] **Step 2: Run test to verify it fails**

```bash
docker compose run --rm frontend node --experimental-strip-types --test app/utils/pdfExport/imageReencode.test.ts
```

Expected: FAIL — `Cannot find module './imageReencode.ts'`.

- [ ] **Step 3: Implement**

```typescript
// services/frontend/app/utils/pdfExport/imageReencode.ts
/** Decodes the base64 payload of a data URI into raw bytes. */
export function dataUriToBytes(dataUri: string): Uint8Array {
  const commaIndex = dataUri.indexOf(",");
  const base64 = commaIndex >= 0 ? dataUri.slice(commaIndex + 1) : dataUri;
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

/**
 * Re-encodes any browser-decodable image data URI (PNG today, from
 * ImageEditor's applyGenerateResult) to JPEG bytes suitable for
 * PDFDocument#embedJpg. Requires a browser environment (Image + canvas);
 * not covered by node:test — see Task 12's manual QA pass.
 */
export async function reencodeToJpeg(dataUri: string, quality = 0.85): Promise<Uint8Array> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        reject(new Error("Canvas 2D context unavailable"));
        return;
      }
      ctx.drawImage(img, 0, 0);
      resolve(dataUriToBytes(canvas.toDataURL("image/jpeg", quality)));
    };
    img.onerror = () => reject(new Error("Failed to load image for re-encoding"));
    img.src = dataUri;
  });
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
docker compose run --rm frontend node --experimental-strip-types --test app/utils/pdfExport/imageReencode.test.ts
```

Expected: PASS, both tests green.

- [ ] **Step 5: Commit**

```bash
git add services/frontend/app/utils/pdfExport/imageReencode.ts services/frontend/app/utils/pdfExport/imageReencode.test.ts
git commit -m "feat(frontend): add data URI byte decoding and JPEG re-encode for PDF export"
```

---

### Task 5: Internal link annotation helper

**Files:**
- Create: `services/frontend/app/utils/pdfExport/linkAnnotations.ts`
- Test: `services/frontend/app/utils/pdfExport/linkAnnotations.test.ts`

`pdf-lib` has no high-level "add a clickable internal link" API; this wraps the documented low-level recipe (a `Link` annotation dict with a `Dest` pointing at a page reference) used for both the thumbnail-to-appendix links and the appendix's "back to page N" links. Because `pdf-lib` runs fine in plain Node (no DOM needed for document assembly), this is tested against the real library, not a mock.

- [ ] **Step 1: Write the failing test**

```typescript
// services/frontend/app/utils/pdfExport/linkAnnotations.test.ts
import assert from "node:assert/strict";
import test from "node:test";
import { PDFDocument, PDFArray, PDFDict, PDFName } from "pdf-lib";

import { addInternalLink } from "./linkAnnotations.ts";

test("addInternalLink attaches a Link annotation pointing at the target page", async () => {
  const doc = await PDFDocument.create();
  const fromPage = doc.addPage([600, 800]);
  const toPage = doc.addPage([600, 800]);

  addInternalLink(doc, fromPage, { x: 10, y: 20, width: 72, height: 72 }, toPage);

  const annots = fromPage.node.Annots();
  assert.ok(annots instanceof PDFArray, "Annots array should exist on the page");
  assert.equal(annots.size(), 1);

  const annotDict = doc.context.lookup(annots.get(0)) as PDFDict;
  assert.equal(annotDict.lookup(PDFName.of("Subtype"), PDFName)?.toString(), "/Link");

  const rect = annotDict.lookup(PDFName.of("Rect"), PDFArray);
  assert.deepEqual(
    rect.asArray().map((n) => Number(n.toString())),
    [10, 20, 82, 92],
  );

  const dest = annotDict.lookup(PDFName.of("Dest"), PDFArray);
  assert.equal(dest.get(0).toString(), toPage.ref.toString());
});

test("addInternalLink appends to an existing Annots array instead of replacing it", async () => {
  const doc = await PDFDocument.create();
  const fromPage = doc.addPage([600, 800]);
  const toPageA = doc.addPage([600, 800]);
  const toPageB = doc.addPage([600, 800]);

  addInternalLink(doc, fromPage, { x: 0, y: 0, width: 10, height: 10 }, toPageA);
  addInternalLink(doc, fromPage, { x: 20, y: 20, width: 10, height: 10 }, toPageB);

  const annots = fromPage.node.Annots();
  assert.equal(annots.size(), 2);
});
```

- [ ] **Step 2: Run test to verify it fails**

```bash
docker compose run --rm frontend node --experimental-strip-types --test app/utils/pdfExport/linkAnnotations.test.ts
```

Expected: FAIL — `Cannot find module './linkAnnotations.ts'`.

- [ ] **Step 3: Implement**

```typescript
// services/frontend/app/utils/pdfExport/linkAnnotations.ts
import { PDFArray, PDFDocument, PDFName, PDFPage } from "pdf-lib";

export interface LinkRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * Adds an invisible, clickable internal link annotation to fromPage that
 * navigates the viewer to toPage when the given rect is clicked. Uses
 * pdf-lib's low-level object API since pdf-lib has no built-in helper for
 * internal (same-document) links.
 */
export function addInternalLink(
  doc: PDFDocument,
  fromPage: PDFPage,
  rect: LinkRect,
  toPage: PDFPage,
): void {
  const linkDict = doc.context.obj({
    Type: "Annot",
    Subtype: "Link",
    Rect: [rect.x, rect.y, rect.x + rect.width, rect.y + rect.height],
    Border: [0, 0, 0],
    Dest: [toPage.ref, "Fit"],
  });
  const linkRef = doc.context.register(linkDict);

  const existingAnnots = fromPage.node.Annots();
  if (existingAnnots) {
    existingAnnots.push(linkRef);
  } else {
    fromPage.node.set(PDFName.of("Annots"), doc.context.obj([linkRef]));
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
docker compose run --rm frontend node --experimental-strip-types --test app/utils/pdfExport/linkAnnotations.test.ts
```

Expected: PASS, both tests green. If `pdf-lib`'s actual low-level API shape differs slightly from what's written above (e.g. `PDFDict#lookup` argument order, or `Annots()` returning `undefined` vs throwing), the test failure output will show the exact mismatch — adjust the implementation to match `pdf-lib`'s real behavior and re-run; the test *expectations* (one Link annotation, correct Rect, Dest pointing at the target page ref, appending not replacing) should not need to change.

- [ ] **Step 5: Commit**

```bash
git add services/frontend/app/utils/pdfExport/linkAnnotations.ts services/frontend/app/utils/pdfExport/linkAnnotations.test.ts
git commit -m "feat(frontend): add internal PDF link annotation helper"
```

---

### Task 6: Shared `ExportImageData` type + pass prompt text out of the editor

**Files:**
- Modify: `services/frontend/app/types/common.d.ts`
- Modify: `services/frontend/app/components/ImageEditor.vue:204-211,232-240`
- Modify: `services/frontend/app/components/ImageLayer.vue:466-480`

The appendix page needs both the generated image and the prompt text that produced it (`EditorHistoryItem.text` for the selected history entry — see `services/frontend/app/composables/useEditorHistory.ts:8-10`). Today `ImageEditor.getExportImage()` only returns `{ highlightId, imageUrl }`. This task adds `prompt` to that return value and threads it through `ImageLayer.getExportImages()`.

- [ ] **Step 1: Add the shared type**

In `services/frontend/app/types/common.d.ts`, add after the `EditorImageState` type (line 51):

```typescript
export type ExportImageData = {
  imageUrl: string;
  prompt: string;
};
```

- [ ] **Step 2: Update `ImageEditor.vue` to return the prompt**

In `services/frontend/app/components/ImageEditor.vue`, replace the `getExportImage` function (lines 204-211):

```typescript
function getExportImage() {
  const item = currentHistoryItem.value;
  if (!item?.imageUrl) return null;
  return {
    highlightId: props.highlightId,
    imageUrl: item.imageUrl,
    prompt: item.text,
  };
}
```

(No other changes needed in this file — `defineExpose` at lines 232-240 already exposes `getExportImage` by reference, so the new return shape flows through automatically.)

- [ ] **Step 3: Update `ImageLayer.vue`'s `getExportImages` to the new shape**

In `services/frontend/app/components/ImageLayer.vue`, replace the `getExportImages` function (lines 466-477):

```typescript
function getExportImages(): Record<number, ExportImageData> {
  const result: Record<number, ExportImageData> = {};

  for (const editor of editorRefs.value) {
    if (!editor) continue;
    const exportData = editor.getExportImage();
    if (exportData) {
      result[exportData.highlightId] = { imageUrl: exportData.imageUrl, prompt: exportData.prompt };
    }
  }
  return result;
}
```

And update the import at the top of the file (line 31) to include the new type:

```typescript
import type { ActionState, EditorImageState, ExportImageData, Highlight } from "~/types/common";
```

- [ ] **Step 4: Type-check**

```bash
docker compose run --rm frontend npx nuxi typecheck
```

Expected: no new type errors from these three files. (`index.vue`'s call site is updated in Task 10 — until then, TypeScript will flag the mismatched `imageUrls: Record<number, string>` argument at the `confirmExport` call in `index.vue`; that's expected and resolved in Task 10.)

- [ ] **Step 5: Commit**

```bash
git add services/frontend/app/types/common.d.ts services/frontend/app/components/ImageEditor.vue services/frontend/app/components/ImageLayer.vue
git commit -m "feat(frontend): thread generation prompt text through export image data"
```

---

### Task 7: PDF export orchestrator (`buildExportPdf`)

**Files:**
- Create: `services/frontend/app/utils/pdfExport/buildExportPdf.ts`
- Test: `services/frontend/app/utils/pdfExport/buildExportPdf.test.ts`

This assembles the final PDF: draws a thumbnail + link on each illustrated segment's page (using Task 2's placement, skipping duplicate pages per the bug fix), then appends one appendix page per image (full-res image, text, back-link), using Task 3's wrapping and Task 5's link helper. The `reencodeToJpeg` dependency (Task 4, browser-only) is injectable so this can be tested with a fake that returns a static, valid JPEG fixture — `pdf-lib` itself runs fine in plain Node.

- [ ] **Step 1: Write the failing tests**

```typescript
// services/frontend/app/utils/pdfExport/buildExportPdf.test.ts
import assert from "node:assert/strict";
import test from "node:test";
import { PDFDocument } from "pdf-lib";

import { buildExportPdf } from "./buildExportPdf.ts";
import { dataUriToBytes } from "./imageReencode.ts";

// A minimal valid 1x1 JPEG, used so PDFDocument#embedJpg has real bytes to parse.
const TINY_JPEG_DATA_URI =
  "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAj/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k=";

async function buildBlankPdf(pageCount: number, size: [number, number] = [612, 792]) {
  const doc = await PDFDocument.create();
  for (let i = 0; i < pageCount; i++) doc.addPage(size);
  return doc.save();
}

const fakeReencode = async () => dataUriToBytes(TINY_JPEG_DATA_URI);

test("buildExportPdf preserves the original page count plus one appendix page per illustrated segment", async () => {
  const pdfBytes = await buildBlankPdf(3);
  const highlights = [
    { id: 1, text: "First segment", polygons: { 0: [[0, 0.1, 0.2, 0.1, 0.2, 0.2, 0, 0.2]] } },
    { id: 2, text: "Second segment", polygons: { 1: [[0, 0.3, 0.2, 0.3, 0.2, 0.4, 0, 0.4]] } },
  ];
  const images = {
    1: { imageUrl: TINY_JPEG_DATA_URI, prompt: "First segment" },
    2: { imageUrl: TINY_JPEG_DATA_URI, prompt: "a different prompt" },
  };

  const resultBytes = await buildExportPdf({
    pdfBytes,
    highlights,
    images,
    reencodeToJpeg: fakeReencode,
  });

  const resultDoc = await PDFDocument.load(resultBytes);
  assert.equal(resultDoc.getPageCount(), 3 + 2);
});

test("buildExportPdf skips segments with no generated image", async () => {
  const pdfBytes = await buildBlankPdf(1);
  const highlights = [
    { id: 1, text: "Illustrated", polygons: { 0: [[0, 0.1, 0.2, 0.1, 0.2, 0.2, 0, 0.2]] } },
    { id: 2, text: "Not illustrated", polygons: { 0: [[0, 0.5, 0.2, 0.5, 0.2, 0.6, 0, 0.6]] } },
  ];
  const images = {
    1: { imageUrl: TINY_JPEG_DATA_URI, prompt: "Illustrated" },
  };

  const resultBytes = await buildExportPdf({ pdfBytes, highlights, images, reencodeToJpeg: fakeReencode });
  const resultDoc = await PDFDocument.load(resultBytes);
  assert.equal(resultDoc.getPageCount(), 1 + 1); // one original + one appendix page
});

test("buildExportPdf places only one thumbnail link when a segment spans two pages", async () => {
  const pdfBytes = await buildBlankPdf(2);
  const highlights = [
    {
      id: 1,
      text: "Spans a page break",
      polygons: {
        0: [[0, 0.9, 0.2, 0.9, 0.2, 1.0, 0, 1.0]],
        1: [[0, 0.0, 0.2, 0.0, 0.2, 0.1, 0, 0.1]],
      },
    },
  ];
  const images = { 1: { imageUrl: TINY_JPEG_DATA_URI, prompt: "Spans a page break" } };

  const resultBytes = await buildExportPdf({ pdfBytes, highlights, images, reencodeToJpeg: fakeReencode });
  const resultDoc = await PDFDocument.load(resultBytes);
  const pages = resultDoc.getPages();

  const page0Annots = pages[0].node.Annots();
  const page1Annots = pages[1].node.Annots();

  assert.equal(page0Annots, undefined, "the earlier page must not get a thumbnail link");
  assert.ok(page1Annots && page1Annots.size() === 1, "the later page must get exactly one thumbnail link");
});

test("buildExportPdf continues past an image that fails to re-encode", async () => {
  const pdfBytes = await buildBlankPdf(1);
  const highlights = [
    { id: 1, text: "Bad image", polygons: { 0: [[0, 0.1, 0.2, 0.1, 0.2, 0.2, 0, 0.2]] } },
    { id: 2, text: "Good image", polygons: { 0: [[0, 0.5, 0.2, 0.5, 0.2, 0.6, 0, 0.6]] } },
  ];
  const images = {
    1: { imageUrl: "data:image/png;base64,broken", prompt: "Bad image" },
    2: { imageUrl: TINY_JPEG_DATA_URI, prompt: "Good image" },
  };
  const failingReencode = async (dataUri: string) => {
    if (dataUri.includes("broken")) throw new Error("re-encode failed");
    return dataUriToBytes(TINY_JPEG_DATA_URI);
  };

  const resultBytes = await buildExportPdf({ pdfBytes, highlights, images, reencodeToJpeg: failingReencode });
  const resultDoc = await PDFDocument.load(resultBytes);
  assert.equal(resultDoc.getPageCount(), 1 + 1); // only the good image gets an appendix page
});
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
docker compose run --rm frontend node --experimental-strip-types --test app/utils/pdfExport/buildExportPdf.test.ts
```

Expected: FAIL — `Cannot find module './buildExportPdf.ts'`.

- [ ] **Step 3: Implement**

```typescript
// services/frontend/app/utils/pdfExport/buildExportPdf.ts
import { PDFDocument, PDFPage, StandardFonts } from "pdf-lib";
import type { PDFFont } from "pdf-lib";
import type { Highlight, ExportImageData } from "~/types/common";

import { computeThumbnailPlacement, computeThumbnailRect, computeThumbnailSize } from "./thumbnailPlacement.ts";
import { addInternalLink } from "./linkAnnotations.ts";
import { wrapText } from "./textWrap.ts";
import { reencodeToJpeg as defaultReencodeToJpeg } from "./imageReencode.ts";

export interface BuildExportPdfOptions {
  pdfBytes: ArrayBuffer | Uint8Array;
  highlights: Highlight[];
  images: Record<number, ExportImageData>;
  reencodeToJpeg?: (dataUri: string) => Promise<Uint8Array>;
}

interface AppendixEntry {
  highlight: Highlight;
  prompt: string;
  jpegBytes: Uint8Array;
  originPage: PDFPage;
  originPageIndex: number;
  thumbnailRect: { x: number; y: number; width: number; height: number };
}

const MARGIN = 48;
const BODY_SIZE = 11;
const LABEL_SIZE = 12;
const LINE_HEIGHT = 14;
const BACK_LINK_SIZE = 10;

export async function buildExportPdf(options: BuildExportPdfOptions): Promise<Uint8Array> {
  const { pdfBytes, highlights, images } = options;
  const reencodeToJpeg = options.reencodeToJpeg ?? defaultReencodeToJpeg;

  const doc = await PDFDocument.load(pdfBytes);
  const originalPages = doc.getPages();

  // Pass 1: draw a thumbnail overlay on each illustrated segment's chosen
  // page, deferring appendix creation (and the forward link into it) until
  // every appendix page's index is known.
  const appendixEntries: AppendixEntry[] = [];

  for (const highlight of highlights) {
    const imageData = images[highlight.id];
    if (!imageData) continue;

    let jpegBytes: Uint8Array;
    try {
      jpegBytes = await reencodeToJpeg(imageData.imageUrl);
    } catch {
      continue; // per spec: skip this image's marker/appendix rather than failing the export
    }

    const placement = computeThumbnailPlacement(highlight.polygons);
    if (!placement) continue;

    const originPage = originalPages[placement.page];
    if (!originPage) continue;

    let jpegImage;
    try {
      jpegImage = await doc.embedJpg(jpegBytes);
    } catch {
      continue;
    }

    const { width: pageWidth, height: pageHeight } = originPage.getSize();
    const size = computeThumbnailSize(pageWidth);
    const rect = computeThumbnailRect(placement.minY, pageWidth, pageHeight, size);

    originPage.drawImage(jpegImage, { x: rect.x, y: rect.y, width: rect.width, height: rect.height });

    appendixEntries.push({
      highlight,
      prompt: imageData.prompt,
      jpegBytes,
      originPage,
      originPageIndex: placement.page,
      thumbnailRect: rect,
    });
  }

  // Appendix pages are ordered by the page number of the originating highlight.
  appendixEntries.sort((a, b) => a.originPageIndex - b.originPageIndex);

  const font = await doc.embedFont(StandardFonts.Helvetica);
  const boldFont = await doc.embedFont(StandardFonts.HelveticaBold);
  const [appendixWidth, appendixHeight] = originalPages.length > 0
    ? [originalPages[0].getSize().width, originalPages[0].getSize().height]
    : [612, 792];

  for (const entry of appendixEntries) {
    const appendixPage = doc.addPage([appendixWidth, appendixHeight]);
    const jpegImage = await doc.embedJpg(entry.jpegBytes);

    const maxImageWidth = appendixWidth - MARGIN * 2;
    const maxImageHeight = appendixHeight * 0.55;
    const imageDims = jpegImage.scaleToFit(maxImageWidth, maxImageHeight);
    const imageX = (appendixWidth - imageDims.width) / 2;
    const imageY = appendixHeight - MARGIN - imageDims.height;

    appendixPage.drawImage(jpegImage, {
      x: imageX,
      y: imageY,
      width: imageDims.width,
      height: imageDims.height,
    });

    let cursorY = imageY - LINE_HEIGHT * 2;
    cursorY = drawAppendixText(appendixPage, font, boldFont, entry.highlight.text, entry.prompt, cursorY);

    const backLinkText = `← back to page ${entry.originPageIndex + 1}`;
    const backLinkY = MARGIN;
    appendixPage.drawText(backLinkText, { x: MARGIN, y: backLinkY, size: BACK_LINK_SIZE, font });
    const backLinkWidth = font.widthOfTextAtSize(backLinkText, BACK_LINK_SIZE);
    addInternalLink(
      doc,
      appendixPage,
      { x: MARGIN, y: backLinkY - 2, width: backLinkWidth, height: BACK_LINK_SIZE + 4 },
      entry.originPage,
    );

    // Forward link: clicking the thumbnail on the origin page opens this appendix page.
    addInternalLink(doc, entry.originPage, entry.thumbnailRect, appendixPage);
  }

  return doc.save();
}

/** Draws original-text / prompt blocks below the image, returning the new cursor y. */
function drawAppendixText(
  page: PDFPage,
  font: PDFFont,
  boldFont: PDFFont,
  originalText: string,
  prompt: string,
  startY: number,
): number {
  const { width: pageWidth } = page.getSize();
  const maxWidth = pageWidth - MARGIN * 2;
  const measure = (s: string) => font.widthOfTextAtSize(s, BODY_SIZE);

  let y = startY;
  const sameText = originalText.trim() === prompt.trim();

  const drawBlock = (label: string | null, text: string) => {
    if (label) {
      page.drawText(label, { x: MARGIN, y, size: LABEL_SIZE, font: boldFont });
      y -= LINE_HEIGHT;
    }
    for (const line of wrapText(text, maxWidth, measure)) {
      page.drawText(line, { x: MARGIN, y, size: BODY_SIZE, font });
      y -= LINE_HEIGHT;
    }
    y -= LINE_HEIGHT / 2;
  };

  if (sameText) {
    drawBlock(null, originalText);
  } else {
    drawBlock("Original text", originalText);
    drawBlock("Generation prompt", prompt);
  }

  return y;
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
docker compose run --rm frontend node --experimental-strip-types --test app/utils/pdfExport/buildExportPdf.test.ts
```

Expected: PASS, all 4 tests green. If `doc.embedJpg` rejects the hardcoded `TINY_JPEG_DATA_URI` fixture as invalid, replace it with the byte content of any small real `.jpg` file (e.g. `Buffer.from(fs.readFileSync(path)).toString("base64")` from a throwaway 1x1 JPEG) — the test *assertions* don't depend on the fixture's specific bytes, only on it being a JPEG `embedJpg` accepts.

- [ ] **Step 5: Commit**

```bash
git add services/frontend/app/utils/pdfExport/buildExportPdf.ts services/frontend/app/utils/pdfExport/buildExportPdf.test.ts
git commit -m "feat(frontend): add pdf-lib PDF export orchestrator with appendix pages and internal links"
```

---

### Task 8: Download helper

**Files:**
- Create: `services/frontend/app/utils/pdfExport/download.ts`

Replaces `downloadExport` from `services/frontend/app/utils/export.ts:580-593`, adapted for binary PDF bytes instead of an HTML string.

- [ ] **Step 1: Implement**

```typescript
// services/frontend/app/utils/pdfExport/download.ts
export async function downloadPdf(pdfBytes: Uint8Array, filename: string): Promise<void> {
  const blob = new Blob([pdfBytes], { type: "application/pdf" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
```

There's no meaningful unit test here (it's pure DOM side-effecting glue, same as the code it replaces) — it's covered by Task 12's manual QA pass.

- [ ] **Step 2: Commit**

```bash
git add services/frontend/app/utils/pdfExport/download.ts
git commit -m "feat(frontend): add PDF blob download helper"
```

---

### Task 9: Wire `useExport.ts` to the new pipeline

**Files:**
- Modify: `services/frontend/app/composables/useExport.ts`

- [ ] **Step 1: Replace the composable's body**

Replace the full contents of `services/frontend/app/composables/useExport.ts`:

```typescript
import { ref } from "vue";
import type { ExportImageData, Highlight } from "~/types/common";
import { buildExportPdf } from "~/utils/pdfExport/buildExportPdf";
import { downloadPdf } from "~/utils/pdfExport/download";

interface ExportResult {
  pdfBytes: Uint8Array;
}

export function useExport() {
  const showExportDialog = ref(false);
  const isExporting = ref(false);

  async function exportPdf(
    pdfFile: File | null,
    highlights: Highlight[],
    images: Record<number, ExportImageData>
  ): Promise<ExportResult> {
    if (!pdfFile) {
      throw new Error("PDF file is required for export");
    }

    isExporting.value = true;

    try {
      const pdfBytes = await pdfFile.arrayBuffer();
      const resultBytes = await buildExportPdf({ pdfBytes, highlights, images });
      return { pdfBytes: resultBytes };
    } finally {
      isExporting.value = false;
    }
  }

  async function confirmExport(
    pdfFile: File | null,
    highlights: Highlight[],
    images: Record<number, ExportImageData>,
    filename: string
  ): Promise<void> {
    const result = await exportPdf(pdfFile, highlights, images);
    const finalFilename = filename.toLowerCase().endsWith(".pdf") ? filename : `${filename}.pdf`;
    await downloadPdf(result.pdfBytes, finalFilename);
  }

  return {
    showExportDialog,
    isExporting,
    exportPdf,
    confirmExport,
  };
}
```

- [ ] **Step 2: Type-check**

```bash
docker compose run --rm frontend npx nuxi typecheck
```

Expected: the only remaining error should be in `index.vue`'s call site (fixed in Task 10) — everything inside `useExport.ts` and `pdfExport/` should type-check cleanly.

- [ ] **Step 3: Commit**

```bash
git add services/frontend/app/composables/useExport.ts
git commit -m "feat(frontend): switch useExport to the pdf-lib export pipeline"
```

---

### Task 10: Update the export button handler in `index.vue`

**Files:**
- Modify: `services/frontend/app/pages/index.vue:525-555`

- [ ] **Step 1: Update `handleExportConfirm`**

Replace `handleExportConfirm` (lines 525-555 of `services/frontend/app/pages/index.vue`):

```typescript
async function handleExportConfirm() {
  try {
    if (!pdfFile.value) {
      useNotifier().error("PDF file is required for export");
      return;
    }

    if (!pdfViewer.value) {
      useNotifier().error("PDF viewer is not ready");
      return;
    }

    const imageLayer = pdfViewer.value.$refs.imageLayer as ImageLayer | undefined;
    if (!imageLayer || typeof (imageLayer as any).getExportImages !== "function") {
      useNotifier().error("Image layer is not ready");
      return;
    }

    const images = imageLayer.getExportImages();

    await confirmExport(
      pdfFile.value,
      highlights,
      images,
      `${pdfFile.value.name.replace(/\.pdf$/i, "")}-export`
    );
  } catch (error) {
    console.error("Export failed:", error);
    useNotifier().error("Failed to export PDF");
  }
}
```

(Only the local variable rename `imageUrls` → `images` changes here — `getExportImages()`'s return type already changed to `Record<number, ExportImageData>` in Task 6, and `confirmExport` already expects that shape after Task 9, so this is purely a rename for clarity; TypeScript will catch it if anything's mismatched.)

- [ ] **Step 2: Type-check**

```bash
docker compose run --rm frontend npx nuxi typecheck
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add services/frontend/app/pages/index.vue
git commit -m "feat(frontend): update export handler for the new PDF export pipeline"
```

---

### Task 11: Remove the old HTML export pipeline

**Files:**
- Delete: `services/frontend/app/utils/export.ts`

Per the spec's "Code removal" section (`docs/specs/2026-07-11-pdf-export-design.md:58-60`): the CDN-loaded PDF.js renderer, lazy-loading/`IntersectionObserver` logic, and modal JS are all deleted — none of it is referenced anymore after Tasks 9-10.

- [ ] **Step 1: Confirm nothing else references the old module**

```bash
grep -rn "utils/export\"\|utils/export'" services/frontend/app --include="*.ts" --include="*.vue"
```

Expected: no output (only `useExport.ts`/`pdfExport/*` should be referenced anywhere now, and `useExport.ts` no longer imports from `utils/export`).

- [ ] **Step 2: Delete the file**

```bash
git rm services/frontend/app/utils/export.ts
```

- [ ] **Step 3: Type-check and lint**

```bash
docker compose run --rm frontend npx nuxi typecheck
docker compose run --rm frontend pnpm dlx eslint --ext .ts,.js,.vue app --no-fix
```

(Or, from the module root: `pnpm lint:ci`.) Expected: no errors.

- [ ] **Step 4: Run the full `pdfExport` test suite once more**

```bash
docker compose run --rm frontend node --experimental-strip-types --test app/utils/pdfExport/*.test.ts
```

Expected: all tests across all five `pdfExport/*.test.ts` files pass.

- [ ] **Step 5: Commit**

```bash
git commit -m "refactor(frontend): remove the CDN-based HTML export pipeline"
```

---

### Task 12: Manual QA pass

**Files:** none (verification only)

Automated tests cover the geometry, wrapping, link-annotation, and orchestration logic, but not the browser-only canvas re-encoding (Task 4) or the actual PDF viewer experience. Run this pass before considering the feature done.

- [ ] **Step 1: Start the dev server**

```bash
docker compose run --rm frontend pnpm dev
```

- [ ] **Step 2: Upload a multi-page book PDF, generate at least two illustrations**

Include at least one segment whose highlighted polygon visually spans two pages (scroll to find or construct one) — this is the case the bug fix targets.

- [ ] **Step 3: Export and open the resulting PDF file**

Confirm:
- The downloaded file is named `<original-name>-export.pdf` and opens in a standard PDF viewer (not just the browser).
- The original pages render unchanged (no missing text/images from the source book).
- Each illustrated segment shows exactly one small thumbnail image — including the segment that spans a page break, which must show its thumbnail on the **later** page only, not both.
- Clicking a thumbnail's area jumps to its appendix page.
- Each appendix page shows the full-resolution image, the segment text (and a separate "Generation prompt" block only when it differs from the original text), and a "← back to page N" link that jumps back correctly.
- A segment with no generated image gets no thumbnail and no appendix page.

- [ ] **Step 4: Report results**

If any check fails, file it as a follow-up fix before merging — do not silently patch without re-running the affected automated tests first (Tasks 2-7) to confirm the regression isn't already covered by an existing test that needs strengthening.

---

## Self-Review Notes

- **Spec coverage:** Pipeline steps 1-4 of `docs/specs/2026-07-11-pdf-export-design.md` are covered by Tasks 2 (placement/sizing), 4 (re-encode), 5 (links), 7 (orchestration: overlay + appendix + back-link), 8-9 (save/download). "Code removal" is Task 11. "Data reused" fields (`Highlight.polygons`, `Highlight.text`, `EditorImageState.imageUrl`/`EditorState.currentPrompt`) are covered via Task 6's `ExportImageData` threading. "Error handling" (no image → no marker/entry; re-encode failure → skip that image only) is covered by Task 7's `try/catch` and its dedicated test. The requested bug fix (thumbnail only on the later page for a split segment) is Task 2's `selectThumbnailPage` plus Task 7's dedicated regression test.
- **Non-goals respected:** no changes to `PdfViewer.vue`/`ImageLayer.vue`'s live rendering (only its `getExportImages` export-time data shape, which is in scope), no backend changes, no heatmap overlay handling added.
- **Type consistency:** `ExportImageData` is defined once (Task 6, `common.d.ts`) and reused verbatim by `ImageEditor.vue`, `ImageLayer.vue`, `useExport.ts`, and `buildExportPdf.ts` — no duplicate/renamed shapes across tasks.
