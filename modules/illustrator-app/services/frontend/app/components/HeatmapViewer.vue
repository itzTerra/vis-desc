<template>
  <div
    class="heatmap-container fixed left-0 top-0 h-full transition-all duration-200 border-r"
    :class="isExpanded ? 'w-24' : 'w-0'"
    :style="{
      backgroundColor: isExpanded ? 'transparent' : 'rgba(245, 245, 245, 0.7)',
      borderColor: '#D0D0D0'
    }"
  >
    <button
      type="button"
      class="btn btn-sm btn-circle absolute"
      style="left: -26px; top: 8px;"
      :aria-pressed="isExpanded"
      aria-label="Toggle heatmap"
      @click="isExpanded = !isExpanded"
    >
      <Icon :name="isExpanded ? 'lucide:chevron-left' : 'lucide:chevron-right'" size="18" aria-hidden="true" />
    </button>
    <div class="relative h-full w-full overflow-hidden">
      <canvas
        ref="canvasElement"
        :width="HEATMAP_WIDTH"
        :height="canvasHeight"
        class="heatmap-canvas cursor-pointer"
        :class="isExpanded ? 'block' : 'hidden'"
      />
      <svg
        v-if="isExpanded"
        :viewBox="`0 0 ${HEATMAP_WIDTH} ${canvasHeight}`"
        class="absolute inset-0 h-full w-full pointer-events-none"
        role="img"
        aria-label="Heatmap image indicators"
      >
        <template v-for="dot in segmentDots" :key="dot.highlightId">
          <circle
            v-if="dot.hasImage"
            :cx="dot.x"
            :cy="dot.y"
            r="4.5"
            class="image-dot image-dot--has"
            role="img"
            :aria-label="`Segment ${dot.highlightId} has image`"
          />
          <circle
            v-else
            :cx="dot.x"
            :cy="dot.y"
            r="4"
            class="image-dot image-dot--none"
            role="img"
            :aria-label="`Segment ${dot.highlightId} has no image`"
          />
        </template>
      </svg>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watchEffect } from "vue";
import { watchDebounced } from "@vueuse/core";

import type { EditorState, Highlight } from "~/types/common";
import { createSegmentArray, normalizedToHeatmapPixel, renderHeatmapCanvas } from "~/utils/heatmapUtils";
import { clamp } from "~/utils/utils";

const HEATMAP_WIDTH = 96;

const props = defineProps<{
  highlights: Highlight[];
  currentPage: number;
  pageAspectRatio: number;
  pageRefs: Element[];
  editorStates: EditorState[];
}>();

defineEmits<{
  navigate: [page: number, normalizedY: number];
}>();

const isExpanded = ref(true);
const canvasElement = ref<HTMLCanvasElement | null>(null);
const cachedHeatmapCanvas = ref<HTMLCanvasElement | null>(null);

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

const segmentDots = computed<SegmentDot[]>(() => {
  if (!props.highlights.length) {
    return [];
  }

  const editorStateById = new Map(
    props.editorStates.map(state => [state.highlightId, state])
  );
  const dots: SegmentDot[] = [];
  const heatmapHeight = canvasHeight.value;

  for (const highlight of props.highlights) {
    const pageNums = Object.keys(highlight.polygons).map(Number);
    if (!pageNums.length) {
      continue;
    }

    const pageNum = Math.min(...pageNums);
    const pagePoints = highlight.polygons[pageNum];
    if (!pagePoints.length) {
      continue;
    }

    let sumX = 0;
    let sumY = 0;
    for (const [x, y] of pagePoints) {
      sumX += x;
      sumY += y;
    }

    const normalizedX = clamp(sumX / pagePoints.length, 0, 1);
    const normalizedY = clamp(sumY / pagePoints.length, 0, 1);
    const pixel = normalizedToHeatmapPixel(
      normalizedX,
      normalizedY,
      pageNum,
      props.pageAspectRatio,
      HEATMAP_WIDTH,
      heatmapHeight,
      totalPages.value
    );

    if (!pixel) {
      continue;
    }

    const hasImage = Boolean(editorStateById.get(highlight.id)?.imageUrl);
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
  if (isExpanded.value) {
    renderHeatmap();
  }
});

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
</style>
