export interface ThumbnailPlacement {
  page: number;
  centerY: number;
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

export function computeMaxY(polygons: number[][]): number {
  let maxY = -Infinity;
  for (const polygon of polygons) {
    for (let i = 1; i < polygon.length; i += 2) {
      maxY = Math.max(maxY, polygon[i]);
    }
  }
  return maxY;
}

export function computeThumbnailPlacement(
  polygons: Record<number, number[][]>,
): ThumbnailPlacement | null {
  const page = selectThumbnailPage(polygons);
  if (page === null) return null;

  const pagePolygons = polygons[page];
  if (!pagePolygons || pagePolygons.length === 0) return null;

  const minY = computeMinY(pagePolygons);
  const maxY = computeMaxY(pagePolygons);
  if (!Number.isFinite(minY) || minY < 0 || !Number.isFinite(maxY)) return null;

  return { page, centerY: (minY + maxY) / 2 };
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
 * centered on the segment's vertical midpoint (centerY, normalized 0-1,
 * top-down — same convention as Highlight.polygons). PDF page coordinates
 * are bottom-up, so this flips the y axis.
 */
export function computeThumbnailRect(
  centerY: number,
  pageWidth: number,
  pageHeight: number,
  size: number,
  margin = 8,
): ThumbnailRect {
  const x = clamp(pageWidth - size - margin, 0, Math.max(0, pageWidth - size));
  const y = clamp(pageHeight * (1 - centerY) - size / 2, 0, Math.max(0, pageHeight - size));
  return { x, y, width: size, height: size };
}
