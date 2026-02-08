<template>
  <div class="fixed left-0 top-0 h-full w-24">
    <div class="relative h-full w-full">
      <canvas ref="heatmapCanvas" class="block h-full w-full" />
      <svg class="absolute inset-0 h-full w-full" />
      <button
        type="button"
        class="btn btn-sm btn-circle absolute"
        style="left: -26px; top: 8px;"
        :aria-pressed="isExpanded"
        aria-label="Toggle heatmap"
      >
        <Icon :name="isExpanded ? 'lucide:chevron-left' : 'lucide:chevron-right'" size="18" aria-hidden="true" />
      </button>
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
