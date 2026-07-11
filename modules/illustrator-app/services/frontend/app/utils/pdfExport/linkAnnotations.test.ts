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
  assert.ok(annots, "Annots array should exist on the page");
  assert.equal(annots.size(), 2);
});
