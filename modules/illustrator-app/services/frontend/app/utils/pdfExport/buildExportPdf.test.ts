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
