<template>
  <div
    class="fixed left-0 top-0 h-full transition-all duration-200"
    :class="isExpanded ? 'w-24' : 'w-0'"
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
      <canvas ref="heatmapCanvas" class="block h-full w-full" />
      <svg class="absolute inset-0 h-full w-full" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from "vue";

import type { EditorState, Highlight } from "~/types/common";

defineProps<{
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
const heatmapCanvas = ref<HTMLCanvasElement | null>(null);

export interface HeatmapSegment {
  highlightId: number;
  pageNum: number;
  polygonPoints: number[][];
  score: number;
  hasImage: boolean;
}

export interface SegmentDot {
  highlightId: number;
  pageNum: number;
  normalizedX: number;
  normalizedY: number;
  hasImage: boolean;
}
</script>
