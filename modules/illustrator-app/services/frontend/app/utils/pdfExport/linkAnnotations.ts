import { PDFDocument, PDFName, PDFPage } from "pdf-lib";

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
