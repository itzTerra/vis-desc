<template>
  <div
    ref="containerElement"
    class="heatmap-container fixed left-0 top-0 h-full transition-all duration-200 border-r bg-base-300/50 border-base-300"
    :style="{
      width: isExpanded ? `${HEATMAP_WIDTH}px` : '0',
    }"
  >
    <div class="relative h-full w-full">
      <button
        type="button"
        class="btn btn-sm heatmap-toggle-button"
        :aria-pressed="isExpanded"
        aria-label="Toggle heatmap"
        @click="isExpanded = !isExpanded"
      >
        <Icon
          name="lucide:chevron-left"
          size="14"
          class="heatmap-toggle-icon"
          :class="{ 'heatmap-toggle-icon--collapsed': !isExpanded }"
          aria-hidden="true"
        />
      </button>
      <Transition name="heatmap-expand">
        <div v-show="isExpanded" class="h-full w-full overflow-hidden" @pointerdown.left="handleHeatmapClick">
          <canvas
            ref="canvasElement"
            :width="HEATMAP_WIDTH - IMAGE_INDICATOR_STRIP_WIDTH"
            :height="heatmapHeight"
            class="h-full cursor-pointer"
            :style="{
              width: `${HEATMAP_WIDTH - IMAGE_INDICATOR_STRIP_WIDTH}px`,
            }"
          />
          <svg
            :viewBox="`0 0 ${HEATMAP_WIDTH} ${heatmapHeight}`"
            class="absolute inset-0 h-full w-full"
            preserveAspectRatio="none"
            role="img"
            aria-label="Heatmap image indicators"
          >
            <rect
              :x="0"
              :y="viewportY"
              :width="HEATMAP_WIDTH"
              :height="viewportRectHeight"
              class="viewport-rect"
              :class="{ 'viewport-rect--dragging': isDragging }"
            />
            <template v-for="dot in segmentDots" :key="dot.highlightId">
              <ellipse
                v-if="dot.hasImage"
                :cx="dot.x"
                :cy="dot.y"
                rx="5"
                :ry="5 * radiusScale"
                class="image-dot image-dot--has pointer-events-none"
              />
              <ellipse
                v-else
                :cx="dot.x"
                :cy="dot.y"
                rx="4"
                :ry="4 * radiusScale"
                class="image-dot image-dot--none pointer-events-none"
              />
            </template>
          </svg>
        </div>
      </Transition>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { EditorImageState, Highlight } from "~/types/common";
import {
  createSegmentArray,
  getFirstPolygonPoints,
  heatmapPixelToNormalized,
  normalizedToHeatmapPixel,
  renderHeatmapCanvas,
  type PagePolygons
} from "~/utils/heatmapUtils";
import { clamp } from "~/utils/utils";

const HEATMAP_WIDTH = 56;
const IMAGE_INDICATOR_STRIP_WIDTH = 16; // Justified on the right of heatmap container

const props = defineProps<{
  highlights: Highlight[];
  currentPage: number;
  pageAspectRatio: number;
  pageRefs: Element[];
  editorStates: EditorImageState[];
}>();

const isExpanded = ref(true);
const containerElement = ref<HTMLElement | null>(null);
const canvasElement = ref<HTMLCanvasElement | null>(null);
const cachedHeatmapCanvas = ref<HTMLCanvasElement | null>(null);
const scrollOffset = ref(0);
const totalHeight = ref(0);
const viewportHeight = ref(0);
const pageTop = ref(0);
const isDragging = ref(false);
const dragStartY = ref(0);
const dragStartScrollOffset = ref(0);
const dragTarget = ref<SVGRectElement | null>(null);
let scrollRafId: number | null = null;

const stickyHeaderOffset = ref(0);
const bottomOffset = ref(0);
const documentScrollHeight = ref(0);

const radiusScale = computed(() => {
  if (!containerElement.value || heatmapHeight.value <= 0) {
    return 1;
  }

  const containerHeight = containerElement.value.clientHeight;
  if (containerHeight <= 0) {
    return 1;
  }

  return heatmapHeight.value / containerHeight;
});

const totalPages = computed(() => {
  const maxPageNum = Math.max(
    -1,
    ...props.highlights.flatMap(h => Object.keys(h.polygons).map(Number))
  );
  return Math.max(props.pageRefs.length, maxPageNum + 1) || 1;
});

// Use the actual drawing width (excluding the image indicator strip)
const drawingWidth = HEATMAP_WIDTH - IMAGE_INDICATOR_STRIP_WIDTH;

const heatmapHeight = computed(() => {
  return Math.ceil(totalPages.value * props.pageAspectRatio * drawingWidth);
});

// Maps viewport size to heatmap canvas space for drag-to-scroll UI
const viewportRectHeight = computed(() => {
  if (totalHeight.value <= 0) {
    return 20;
  }

  // Use actual visible PDF height (accounting for sticky header and bottom controls)
  const visiblePdfHeight = Math.max(0, viewportHeight.value - stickyHeaderOffset.value - bottomOffset.value);
  const rectHeight = (visiblePdfHeight / totalHeight.value) * heatmapHeight.value;
  return Math.max(rectHeight, 20);
});

const trackHeight = computed(() => {
  return Math.max(0, heatmapHeight.value - viewportRectHeight.value);
});

const actualContentHeight = computed(() => {
  // Calculate real-time content height from first to last page
  const lastPageElement = props.pageRefs[props.pageRefs.length - 1];
  if (!lastPageElement) {
    return totalHeight.value;
  }

  // Depend on scrollOffset to trigger re-evaluation on scroll
  const _ = scrollOffset.value;

  const lastPageRect = lastPageElement.getBoundingClientRect();
  const lastPageAbsoluteBottom = lastPageRect.bottom + window.scrollY;
  return lastPageAbsoluteBottom - pageTop.value;
});

const scrollableHeight = computed(() => {
  return Math.max(0, actualContentHeight.value - viewportHeight.value);
});

// Map document scroll position to heatmap viewport rectangle Y coordinate
// Linear mapping: scroll ratio (0-1) → vertical position in heatmap track
const viewportY = computed(() => {
  if (scrollableHeight.value <= 0 || trackHeight.value <= 0) {
    return 0;
  }

  const result = clamp(
    (scrollOffset.value / scrollableHeight.value) * trackHeight.value,
    0,
    trackHeight.value
  );

  // Debug logging
  // const lastPageRect = props.pageRefs[props.pageRefs.length - 1]?.getBoundingClientRect();
  // const lastPageBottom = lastPageRect ? lastPageRect.bottom + window.scrollY : 0;
  // const lastPageIndex = props.pageRefs.length - 1;

  // console.log("[HeatmapViewer]", {
  //   scrollOffset: scrollOffset.value,
  //   scrollableHeight: scrollableHeight.value,
  //   trackHeight: trackHeight.value,
  //   viewportRectHeight: viewportRectHeight.value,
  //   heatmapHeight: heatmapHeight.value,
  //   totalHeight: totalHeight.value,
  //   viewportHeight: viewportHeight.value,
  //   viewportY: result,
  //   bottomGap: heatmapHeight.value - (result + viewportRectHeight.value),
  //   windowScrollY: window.scrollY,
  //   pageTop: pageTop.value,
  //   stickyHeaderOffset: stickyHeaderOffset.value,
  //   bottomOffset: bottomOffset.value,
  //   documentScrollHeight: documentScrollHeight.value,
  //   lastPageBottom,
  //   viewportBottom: window.scrollY + viewportHeight.value,
  //   numPageRefs: props.pageRefs.length,
  //   lastPageIndex,
  //   totalPages: totalPages.value,
  // });

  return result;
});

const segmentDots = computed<SegmentDot[]>(() => {
  if (props.highlights.length === 0) {
    return [];
  }

  const editorStateById = new Map(
    props.editorStates.map(state => [state.highlightId, state])
  );
  const dots: SegmentDot[] = [];

  // Converts normalized coordinates [0,1] from PDF space to heatmap pixel space
  for (const highlight of props.highlights) {
    if (!editorStateById.has(highlight.id)) {
      continue;
    }
    const pageNums = Object.keys(highlight.polygons).map(Number).sort((a, b) => a - b);
    if (pageNums.length === 0) {
      continue;
    }

    let pageNum: number | null = null;
    let pagePoints: number[][] | null = null;
    for (const candidatePage of pageNums) {
      const pagePolygons = highlight.polygons[candidatePage] as PagePolygons | undefined;
      if (pagePolygons === undefined || pagePolygons === null) {
        continue;
      }

      const candidatePoints = getFirstPolygonPoints(pagePolygons);
      if (candidatePoints !== null && candidatePoints !== undefined) {
        pageNum = candidatePage;
        pagePoints = candidatePoints;
        break;
      }
    }

    if (pageNum === null || pagePoints === null) {
      continue;
    }

    // Points are assumed to be in normalized coordinates [0,1] within the page
    let sumX = 0;
    let sumY = 0;
    for (const [x, y] of pagePoints) {
      sumX += x;
      sumY += y;
    }

    const normalizedX = clamp(sumX / pagePoints.length, 0, 1);
    const normalizedY = clamp(sumY / pagePoints.length, 0, 1);

    // Accounts for page stacking (cumulative Y offset) and aspect ratio
    const pixel = normalizedToHeatmapPixel(
      normalizedX,
      normalizedY,
      pageNum,
      props.pageAspectRatio,
      drawingWidth,
      heatmapHeight.value,
      totalPages.value
    );

    if (pixel === undefined) {
      continue;
    }

    const hasImage = editorStateById.get(highlight.id)?.hasImage ?? false;
    const centerX = HEATMAP_WIDTH - IMAGE_INDICATOR_STRIP_WIDTH / 2;
    dots.push({
      highlightId: highlight.id,
      pageNum,
      normalizedX,
      normalizedY,
      x: centerX,
      y: pixel.y,
      hasImage,
    });
  }

  return dots;
});

function renderHeatmap() {
  if (!canvasElement.value) {
    return;
  }

  const segments = createSegmentArray(props.highlights);
  const renderedCanvas = renderHeatmapCanvas(
    segments,
    drawingWidth,
    props.pageAspectRatio,
    totalPages.value
  );

  cachedHeatmapCanvas.value = renderedCanvas;

  const ctx = canvasElement.value.getContext("2d");
  if (!ctx) {
    return;
  }

  ctx.clearRect(0, 0, canvasElement.value.width, canvasElement.value.height);
  ctx.drawImage(renderedCanvas, 0, 0);
}

function handleHeatmapClick(event: PointerEvent) {
  const canvas = canvasElement.value;
  if (canvas === null) {
    return;
  }

  const rect = canvas.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) {
    return;
  }

  // Account for canvas scaling (logical pixels vs display size)
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const clickX = (event.clientX - rect.left) * scaleX;
  const clickY = (event.clientY - rect.top) * scaleY;

  // Reverse transformation: heatmap pixels → normalized PDF coordinates
  const normalized = heatmapPixelToNormalized(
    clickX,
    clickY,
    drawingWidth,
    heatmapHeight.value,
  );

  window.scrollTo({
    top: pageTop.value + normalized.normalizedY * actualContentHeight.value
  });

  startDrag(event);
}

function measureLayout() {
  const pageElements = props.pageRefs;
  const firstPageRect = pageElements[0]?.getBoundingClientRect();
  const lastPageRect = pageElements[pageElements.length - 1]?.getBoundingClientRect();
  const viewportHeightValue = window.innerHeight;

  // Calculate absolute positions: firstPageRect.top is relative to viewport, add scrollY for absolute
  const pageTopValue = firstPageRect ? firstPageRect.top + window.scrollY : 0;
  const pageBottomValue = lastPageRect ? lastPageRect.bottom + window.scrollY : pageTopValue;

  // Total height is the distance from first page top to last page bottom in absolute coordinates
  const totalHeightValue = pageBottomValue - pageTopValue;

  // console.log("[measureLayout]", {
  //   numPages: pageElements.length,
  //   firstPageRect: firstPageRect ? { top: firstPageRect.top, bottom: firstPageRect.bottom } : null,
  //   lastPageRect: lastPageRect ? { top: lastPageRect.top, bottom: lastPageRect.bottom } : null,
  //   windowScrollY: window.scrollY,
  //   pageTopValue,
  //   pageBottomValue,
  //   totalHeightValue,
  // });

  stickyHeaderOffset.value = Math.max(0, firstPageRect?.top ?? 0);

  // Measure bottom offset: space between last page bottom and viewport bottom
  // This accounts for PDF controls and other UI elements at the bottom
  if (lastPageRect) {
    bottomOffset.value = Math.max(0, viewportHeightValue - lastPageRect.bottom);
  } else {
    bottomOffset.value = 0;
  }

  pageTop.value = pageTopValue;
  totalHeight.value = totalHeightValue;
  viewportHeight.value = viewportHeightValue;
  documentScrollHeight.value = document.documentElement.scrollHeight;
  scrollOffset.value = window.scrollY - pageTopValue;
}

function handleScroll() {
  if (scrollRafId !== null) {
    return;
  }

  scrollRafId = window.requestAnimationFrame(() => {
    scrollRafId = null;
    scrollOffset.value = window.scrollY - pageTop.value;
  });
}

function startDrag(event: PointerEvent) {
  if (isDragging.value) {
    return;
  }

  dragTarget.value = event.currentTarget as SVGRectElement | null;
  dragTarget.value?.setPointerCapture(event.pointerId);
  isDragging.value = true;
  dragStartY.value = event.clientY;
  dragStartScrollOffset.value = window.scrollY - pageTop.value;
  window.addEventListener("pointermove", onDrag);
  window.addEventListener("pointerup", endDrag);
  window.addEventListener("pointercancel", endDrag);
}

function onDrag(event: PointerEvent) {
  if (!isDragging.value) {
    return;
  }

  if (heatmapHeight.value <= 0 || totalHeight.value <= 0) {
    return;
  }

  const canvas = canvasElement.value;
  if (!canvas) {
    return;
  }

  const rect = canvas.getBoundingClientRect();
  if (rect.height <= 0) {
    return;
  }

  // Convert screen pixel delta to heatmap canvas pixel delta
  const screenDeltaY = event.clientY - dragStartY.value;
  const scaleY = canvas.height / rect.height;
  const heatmapDeltaY = screenDeltaY * scaleY;

  // Map heatmap pixel movement directly to scroll distance
  const scrollDelta = (heatmapDeltaY / heatmapHeight.value) * totalHeight.value;
  const nextScrollOffset = clamp(dragStartScrollOffset.value + scrollDelta, 0, scrollableHeight.value);

  window.scrollTo({ top: pageTop.value + nextScrollOffset });
}

function endDrag(event: PointerEvent) {
  if (!isDragging.value) {
    return;
  }

  dragTarget.value?.releasePointerCapture(event.pointerId);
  dragTarget.value = null;
  isDragging.value = false;
  window.removeEventListener("pointermove", onDrag);
  window.removeEventListener("pointerup", endDrag);
  window.removeEventListener("pointercancel", endDrag);
}

watchDebounced(
  () => props.highlights,
  () => {
    cachedHeatmapCanvas.value = null;
    if (isExpanded.value) {
      renderHeatmap();
    }
  },
  { debounce: 200, deep: true }
);

watch(
  () => actualContentHeight.value,
  () => {
    if (isExpanded.value) {
      renderHeatmap();
    }
  }
);

watchEffect(() => {
  if (isExpanded.value && canvasElement.value) {
    renderHeatmap();
  }
});

onMounted(() => {
  measureLayout();
  window.addEventListener("scroll", handleScroll, { passive: true });
  window.addEventListener("resize", measureLayout);
  if (isExpanded.value) {
    renderHeatmap();
  }
});

onBeforeUnmount(() => {
  window.removeEventListener("scroll", handleScroll);
  window.removeEventListener("resize", measureLayout);
  window.removeEventListener("pointermove", onDrag);
  window.removeEventListener("pointerup", endDrag);
  window.removeEventListener("pointercancel", endDrag);
  if (scrollRafId !== null) {
    window.cancelAnimationFrame(scrollRafId);
    scrollRafId = null;
  }
});

watch(
  () => props.pageRefs.length,
  () => {
    measureLayout();
  }
);

export interface SegmentDot {
  highlightId: number;
  pageNum: number;
  normalizedX: number;
  normalizedY: number;
  x: number;
  y: number;
  hasImage: boolean;
}
</script>

<style scoped>
.heatmap-container {
  padding-bottom: 34px; /* PDF control panel height */
}

.heatmap-toggle-icon {
  transition: transform 200ms ease;
}

.heatmap-toggle-icon--collapsed {
  transform: rotate(180deg);
}

.heatmap-toggle-button {
  position: absolute;
  right: 0;
  top: 50%;
  transform: translate(100%, -50%);
  height: 60px;
  width: 15px;
  opacity: 0.7;
  padding-left: 0;
  padding-right: 0;
  border-start-start-radius: 0;
  border-end-start-radius: 0;
}

.heatmap-expand-enter-active,
.heatmap-expand-leave-active {
  transition: opacity 200ms ease;
}

.heatmap-expand-enter-from,
.heatmap-expand-leave-to {
  opacity: 0;
}

.image-dot {
  opacity: 0.85;
}

.image-dot--has {
  fill: var(--color-primary);
  stroke: var(--color-primary);
  stroke-width: 2px;
}

.image-dot--none {
  fill: transparent;
  stroke: var(--color-primary);
  stroke-width: 2px;
}

.viewport-rect {
  fill: var(--color-info);
  fill-opacity: 0.3;
  stroke: var(--color-info);
  stroke-width: 2px;
  pointer-events: all;
  cursor: grab;
  transition: y 120ms linear, height 120ms linear;
}

.viewport-rect--dragging {
  cursor: grabbing;
  transition: none;
}
</style>
