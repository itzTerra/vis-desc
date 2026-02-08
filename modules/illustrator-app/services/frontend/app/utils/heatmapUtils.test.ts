import assert from "node:assert/strict";
import test from "node:test";

import type { Highlight } from "~/types/common";

import {
  createSegmentArray,
  getViewportPercentage,
  heatmapPixelToNormalized,
  normalizedToHeatmapPixel,
  renderHeatmapCanvas,
  scoreToBrightness,
  scoreToOpacity,
} from "~/utils/heatmapUtils";

test("normalizedToHeatmapPixel maps to heatmap space", () => {
  const result = normalizedToHeatmapPixel(0.5, 0.25, 1, 1.5, 100, 300, 3);
  assert.deepEqual(result, { x: 50, y: 187.5 });
});

test("getViewportPercentage clamps output", () => {
  const result = getViewportPercentage(100, 1000, 200);
  assert.equal(result.topPercent, 0.1);
  assert.equal(result.heightPercent, 0.2);
  assert.equal(result.topPixels, 100);
  assert.equal(result.heightPixels, 200);
});

test("heatmapPixelToNormalized returns normalized coordinates", () => {
  const result = heatmapPixelToNormalized(50, 150, 1, 200, 2);
  assert.deepEqual(result, { page: 1, normalizedX: 0.5, normalizedY: 0.5 });
});

test("score helpers clamp values", () => {
  assert.equal(scoreToOpacity(1), 0.9);
  assert.equal(scoreToOpacity(0), 0.15);
  assert.equal(scoreToBrightness(1), 40);
  assert.equal(scoreToBrightness(0), 220);
});

test("createSegmentArray filters and sorts segments", () => {
  const highlights: Highlight[] = [
    {
      id: 2,
      text: "Two",
      polygons: { 1: [[0.2, 0.3]] },
      score: 0.2,
    },
    {
      id: 1,
      text: "One",
      polygons: { 0: [[0.1, 0.2]] },
      score: 0.9,
    },
    {
      id: 3,
      text: "No score",
      polygons: { 0: [[0.4, 0.5]] },
    },
  ];

  const segments = createSegmentArray(highlights);
  assert.equal(segments.length, 2);
  assert.deepEqual(segments[0], {
    highlightId: 1,
    pageNum: 0,
    polygonPoints: [[0.1, 0.2]],
    score: 0.9,
  });
  assert.deepEqual(segments[1], {
    highlightId: 2,
    pageNum: 1,
    polygonPoints: [[0.2, 0.3]],
    score: 0.2,
  });
});

test("renderHeatmapCanvas returns a sized canvas", () => {
  const originalDocument = globalThis.document;
  const fillCalls: Array<[number, number, number, number]> = [];
  const ctxStub = {
    fillStyle: "",
    globalAlpha: 1,
    fillRect: (...args: [number, number, number, number]) => {
      fillCalls.push(args);
    },
  } as unknown as CanvasRenderingContext2D;

  const canvasStub = {
    width: 0,
    height: 0,
    getContext: () => ctxStub,
  } as unknown as HTMLCanvasElement;

  globalThis.document = {
    createElement: () => canvasStub,
  } as unknown as Document;

  try {
    const canvas = renderHeatmapCanvas(
      [{ highlightId: 1, pageNum: 0, polygonPoints: [[0.1, 0.2]], score: 0.5 }],
      100,
      1,
      1
    );
    assert.equal(canvas.width, 100);
    assert.equal(canvas.height, 100);
    assert.ok(fillCalls.length >= 1);
  } finally {
    globalThis.document = originalDocument;
  }
});
