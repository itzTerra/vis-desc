import type { Highlight } from "~/types/common";

import { clamp } from "~/utils/utils";

export type HeatmapPixel = {
  x: number;
  y: number;
};

export type HeatmapNormalizedPoint = {
  page: number;
  normalizedX: number;
  normalizedY: number;
};

export type ViewportPercentage = {
  topPercent: number;
  heightPercent: number;
  topPixels: number;
  heightPixels: number;
};

export type HeatmapSegment = {
  highlightId: number;
  pageNum: number;
  polygonPoints: number[][];
  score: number;
};

/**
 * Converts normalized PDF coordinates into heatmap pixel space.
 */
export function normalizedToHeatmapPixel(
  normalizedX: number,
  normalizedY: number,
  pageNum: number,
  pageAspectRatio: number,
  heatmapWidth: number,
  heatmapHeight: number,
  totalPageCount: number
): HeatmapPixel | undefined {
  if (pageNum >= totalPageCount) {
    return undefined;
  }

  const pageHeight = pageAspectRatio * heatmapWidth;
  const cumulativeOffset = pageNum * pageHeight;
  const withinPageOffset = normalizedY * pageHeight;
  const x = clamp(normalizedX * heatmapWidth, 0, heatmapWidth);
  const y = clamp(cumulativeOffset + withinPageOffset, 0, heatmapHeight);

  return { x, y };
}

/**
 * Calculates viewport percentages relative to the total scrollable height.
 */
export function getViewportPercentage(
  scrollY: number,
  totalHeight: number,
  viewportHeight: number
): ViewportPercentage {
  if (totalHeight <= 0) {
    return { topPercent: 0, heightPercent: 0, topPixels: 0, heightPixels: 0 };
  }

  const topPercent = clamp(scrollY / totalHeight, 0, 1);
  const heightPercent = clamp(viewportHeight / totalHeight, 0, 1);

  return {
    topPercent,
    heightPercent,
    topPixels: topPercent * totalHeight,
    heightPixels: heightPercent * totalHeight,
  };
}

/**
 * Converts heatmap pixel coordinates back to normalized PDF coordinates.
 */
export function heatmapPixelToNormalized(
  pixelX: number,
  pixelY: number,
  pageAspectRatio: number,
  heatmapHeight: number,
  totalPageCount: number
): HeatmapNormalizedPoint {
  if (totalPageCount <= 0 || pageAspectRatio <= 0 || heatmapHeight <= 0) {
    return { page: 0, normalizedX: 0, normalizedY: 0 };
  }

  const pageHeight = heatmapHeight / totalPageCount;
  const heatmapWidth = pageHeight / pageAspectRatio;
  const clampedX = clamp(pixelX, 0, heatmapWidth);
  const clampedY = clamp(pixelY, 0, heatmapHeight);
  const page = clamp(Math.floor(clampedY / pageHeight), 0, totalPageCount - 1);
  const normalizedX = clamp(clampedX / heatmapWidth, 0, 1);
  const normalizedY = clamp((clampedY - page * pageHeight) / pageHeight, 0, 1);

  return { page, normalizedX, normalizedY };
}

/**
 * Maps a segment score to a canvas opacity value.
 */
export function scoreToOpacity(score: number): number {
  const opacity = 0.15 + score * 0.75;
  return clamp(opacity, 0.15, 0.9);
}

/**
 * Maps a segment score to a grayscale brightness value.
 */
export function scoreToBrightness(score: number): number {
  const brightness = 220 - score * 180;
  return Math.round(clamp(brightness, 40, 220));
}

/**
 * Renders heatmap segments to a cached canvas for fast reuse.
 */
export function renderHeatmapCanvas(
  segments: HeatmapSegment[],
  heatmapWidth: number,
  pageAspectRatio: number,
  totalPageCount: number
): HTMLCanvasElement {
  const heatmapHeight = Math.ceil(totalPageCount * pageAspectRatio * heatmapWidth);
  if (typeof document === "undefined") {
    return { width: heatmapWidth, height: heatmapHeight } as HTMLCanvasElement;
  }

  const canvas = document.createElement("canvas");
  canvas.width = heatmapWidth;
  canvas.height = heatmapHeight;

  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return canvas;
  }

  try {
    ctx.fillStyle = "#E8E8E8";
    ctx.fillRect(0, 0, heatmapWidth, heatmapHeight);

    for (const segment of segments) {
      const opacity = scoreToOpacity(segment.score);
      const brightness = scoreToBrightness(segment.score);
      ctx.globalAlpha = opacity;
      ctx.fillStyle = `rgb(${brightness}, ${brightness}, ${brightness})`;

      for (const [x, y] of segment.polygonPoints) {
        const point = normalizedToHeatmapPixel(
          x,
          y,
          segment.pageNum,
          pageAspectRatio,
          heatmapWidth,
          heatmapHeight,
          totalPageCount
        );
        if (!point) {
          continue;
        }
        ctx.fillRect(Math.round(point.x), Math.round(point.y), 2, 2);
      }
    }
  } catch (error) {
    console.warn("Heatmap render failed", error);
  } finally {
    ctx.globalAlpha = 1;
  }

  return canvas;
}

/**
 * Builds a renderable heatmap segment list from highlight data.
 */
export function createSegmentArray(highlights: Highlight[]): HeatmapSegment[] {
  const segments: HeatmapSegment[] = [];

  for (const highlight of highlights) {
    if (highlight.score === undefined) {
      continue;
    }

    for (const [pageStr, polygonPoints] of Object.entries(highlight.polygons)) {
      if (!polygonPoints.length) {
        continue;
      }
      segments.push({
        highlightId: highlight.id,
        pageNum: Number(pageStr),
        polygonPoints,
        score: highlight.score,
      });
    }
  }

  segments.sort((a, b) => {
    if (a.pageNum !== b.pageNum) {
      return a.pageNum - b.pageNum;
    }
    if (a.highlightId !== b.highlightId) {
      return a.highlightId - b.highlightId;
    }
    return a.score - b.score;
  });

  return segments;
}
