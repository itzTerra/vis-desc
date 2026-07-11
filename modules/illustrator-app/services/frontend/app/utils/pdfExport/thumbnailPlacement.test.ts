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
