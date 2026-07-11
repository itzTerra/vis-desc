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

    const backLinkText = `<- back to page ${entry.originPageIndex + 1}`;
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
