<template>
  <div
    ref="containerElement"
    class="heatmap-container fixed left-0 top-0 h-full transition-all duration-200 border-r"
    :class="isExpanded ? `w-[${HEATMAP_WIDTH}px]` : 'w-0'"
    :style="{
      backgroundColor: isExpanded ? 'transparent' : 'rgba(245, 245, 245, 0.7)',
      borderColor: '#D0D0D0'
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
        <div v-show="isExpanded" class="h-full w-full overflow-hidden" @click="handleHeatmapClick">
          <canvas
            ref="canvasElement"
            :width="HEATMAP_WIDTH"
            :height="canvasHeight"
            class="heatmap-canvas cursor-pointer"
          />
          <svg
            :viewBox="`0 0 ${HEATMAP_WIDTH} ${canvasHeight}`"
            class="absolute inset-0 h-full w-full"
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
              @pointerdown="startDrag"
            />
            <template v-for="dot in segmentDots" :key="dot.highlightId">
              <circle
                v-if="dot.hasImage"
                :cx="dot.x"
                :cy="dot.y"
                r="4.5"
                class="image-dot image-dot--has pointer-events-none"
                role="img"
                :aria-label="`Segment ${dot.highlightId} has image`"
              />
              <circle
                v-else
                :cx="dot.x"
                :cy="dot.y"
                r="4"
                class="image-dot image-dot--none pointer-events-none"
                role="img"
                :aria-label="`Segment ${dot.highlightId} has no image`"
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

const HEATMAP_WIDTH = 72;

const props = defineProps<{
  highlights: Highlight[];
  currentPage: number;
  pageAspectRatio: number;
  pageRefs: Element[];
  editorStates: EditorImageState[];
}>();

const emit = defineEmits<{
  navigate: [page: number, normalizedY: number];
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

const totalPages = computed(() => {
  const maxPageNum = Math.max(
    -1,
    ...props.highlights.flatMap(h => Object.keys(h.polygons).map(Number))
  );
  return Math.max(props.pageRefs.length, maxPageNum + 1) || 1;
});

const canvasHeight = computed(() => {
  return Math.ceil(totalPages.value * props.pageAspectRatio * HEATMAP_WIDTH);
});

// Maps viewport size to heatmap canvas space for drag-to-scroll UI
const viewportRectHeight = computed(() => {
  if (totalHeight.value <= 0) {
    return 20;
  }

  const rectHeight = (viewportHeight.value / totalHeight.value) * canvasHeight.value;
  return Math.max(rectHeight, 20);
});

const trackHeight = computed(() => {
  return Math.max(0, canvasHeight.value - viewportRectHeight.value);
});

const scrollableHeight = computed(() => {
  return Math.max(0, totalHeight.value - viewportHeight.value);
});

// Map document scroll position to heatmap viewport rectangle Y coordinate
// Linear mapping: scroll ratio (0-1) → vertical position in heatmap track
const viewportY = computed(() => {
  if (scrollableHeight.value <= 0 || trackHeight.value <= 0) {
    return 0;
  }

  return clamp(
    (scrollOffset.value / scrollableHeight.value) * trackHeight.value,
    0,
    trackHeight.value
  );
});

const segmentDots = computed<SegmentDot[]>(() => {
  if (props.highlights.length === 0) {
    return [];
  }

  const editorStateById = new Map(
    props.editorStates.map(state => [state.highlightId, state])
  );
  const dots: SegmentDot[] = [];
  const heatmapHeight = canvasHeight.value;

  // Converts normalized coordinates [0,1] from PDF space to heatmap pixel space
  for (const highlight of props.highlights) {
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
      HEATMAP_WIDTH,
      heatmapHeight,
      totalPages.value
    );

    if (pixel === undefined) {
      continue;
    }

    const hasImage = editorStateById.get(highlight.id)?.hasImage ?? false;
    dots.push({
      highlightId: highlight.id,
      pageNum,
      normalizedX,
      normalizedY,
      x: pixel.x,
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
    HEATMAP_WIDTH,
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

function handleHeatmapClick(event: MouseEvent) {
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
    props.pageAspectRatio,
    canvasHeight.value,
    totalPages.value
  );

  emit("navigate", normalized.page, normalized.normalizedY);
}

function measureLayout() {
  const pageElements = props.pageRefs;
  const firstPageRect = pageElements[0]?.getBoundingClientRect();
  const lastPageRect = pageElements[pageElements.length - 1]?.getBoundingClientRect();
  const pageTopValue = firstPageRect ? firstPageRect.top + window.scrollY : 0;
  const totalHeightValue = firstPageRect && lastPageRect
    ? lastPageRect.bottom - firstPageRect.top
    : 0;
  const viewportHeightValue = containerElement.value?.getBoundingClientRect().height ?? window.innerHeight;

  pageTop.value = pageTopValue;
  totalHeight.value = totalHeightValue;
  viewportHeight.value = viewportHeightValue;
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
  dragStartScrollOffset.value = scrollOffset.value;
  window.addEventListener("pointermove", onDrag);
  window.addEventListener("pointerup", endDrag);
  window.addEventListener("pointercancel", endDrag);
}

function onDrag(event: PointerEvent) {
  if (!isDragging.value) {
    return;
  }

  if (trackHeight.value <= 0 || scrollableHeight.value <= 0) {
    return;
  }

  // Reverse mapping: heatmap pixel delta → PDF scroll distance
  const deltaY = event.clientY - dragStartY.value;
  const scrollDelta = (deltaY / trackHeight.value) * scrollableHeight.value;
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
.heatmap-canvas {
  width: 100%;
  height: 100%;
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
  width: 20px;
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
  fill: #f5c542;
  stroke: #6b4d00;
  stroke-width: 1.5px;
}

.image-dot--none {
  fill: transparent;
  stroke: #9ca3af;
  stroke-width: 1.5px;
}

.viewport-rect {
  fill: rgba(74, 144, 226, 0.15);
  stroke: #4a90e2;
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
