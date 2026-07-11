import { PDFDocument, StandardFonts } from "pdf-lib";
import type { PDFFont, PDFPage } from "pdf-lib";
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
const BACK_LINK_SIZE = 18;
const THUMBNAIL_MARGIN = 8;

export async function buildExportPdf(options: BuildExportPdfOptions): Promise<Uint8Array> {
  const { pdfBytes, highlights, images } = options;
  const reencodeToJpeg = options.reencodeToJpeg ?? defaultReencodeToJpeg;

  const doc = await PDFDocument.load(pdfBytes);
  const originalPages = doc.getPages();
  // Captured before any page is widened for a thumbnail, so appendix pages
  // keep the original (unwidened) page dimensions.
  const [appendixWidth, appendixHeight] = originalPages.length > 0
    ? [originalPages[0].getSize().width, originalPages[0].getSize().height]
    : [612, 792];

  // Pass 1: draw a thumbnail overlay on each illustrated segment's chosen
  // page, deferring appendix creation (and the forward link into it) until
  // every appendix page's index is known.
  const appendixEntries: AppendixEntry[] = [];

  // Pages that get a thumbnail are widened once, so the thumbnail sits in
  // newly added blank space to the right of the original content instead of
  // overlapping it. Widening keeps existing content anchored at the same
  // (bottom-left-origin) coordinates, so it's applied at most once per page.
  const widenedPages = new Map<number, { pageWidth: number; size: number }>();

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

    let widened = widenedPages.get(placement.page);
    if (!widened) {
      const { width: origWidth, height: origHeight } = originPage.getSize();
      const size = computeThumbnailSize(origWidth);
      const pageWidth = origWidth + size + THUMBNAIL_MARGIN * 2;
      originPage.setSize(pageWidth, origHeight);
      widened = { pageWidth, size };
      widenedPages.set(placement.page, widened);
    }
    const { pageWidth, size } = widened;
    const { height: pageHeight } = originPage.getSize();
    const rect = computeThumbnailRect(placement.minY, pageWidth, pageHeight, size, THUMBNAIL_MARGIN);

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

  for (const entry of appendixEntries) {
    const appendixPage = doc.addPage([appendixWidth, appendixHeight]);
    const jpegImage = await doc.embedJpg(entry.jpegBytes);

    const maxImageWidth = appendixWidth - MARGIN * 2;
    const maxImageHeight = appendixHeight * 0.55;
    // Never upscale past the image's native resolution — only shrink to fit.
    const scale = Math.min(maxImageWidth / jpegImage.width, maxImageHeight / jpegImage.height, 1);
    const imageDims = { width: jpegImage.width * scale, height: jpegImage.height * scale };
    const imageX = (appendixWidth - imageDims.width) / 2;
    const imageY = appendixHeight - MARGIN - imageDims.height;

    appendixPage.drawImage(jpegImage, {
      x: imageX,
      y: imageY,
      width: imageDims.width,
      height: imageDims.height,
    });

    // Back link sits centered directly below the image, sized to be an
    // obvious, easy-to-hit tap target.
    const backLinkText = `<- back to page ${entry.originPageIndex + 1}`;
    const backLinkWidth = font.widthOfTextAtSize(backLinkText, BACK_LINK_SIZE);
    const backLinkX = (appendixWidth - backLinkWidth) / 2;
    const backLinkY = imageY - LINE_HEIGHT * 1.5;
    appendixPage.drawText(backLinkText, { x: backLinkX, y: backLinkY, size: BACK_LINK_SIZE, font });
    addInternalLink(
      doc,
      appendixPage,
      { x: backLinkX - 6, y: backLinkY - 6, width: backLinkWidth + 12, height: BACK_LINK_SIZE + 12 },
      entry.originPage,
    );

    let cursorY = backLinkY - LINE_HEIGHT * 2.5;
    cursorY = drawAppendixText(appendixPage, font, boldFont, entry.highlight.text, entry.prompt, cursorY);

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
