<template>
  <div
    ref="imagesContainer"
    class="w-[512px] border-l border-base-300 select-none"
  >
    <div class="relative">
      <ImageEditor
        v-for="editorId in openEditorIds"
        :key="editorId"
        :highlight-id="editorId"
        :initial-text="getHighlightText(editorId)"
        :style="getEditorPositionStyle(editorId)"
        class="absolute pointer-events-auto"
        @delete="closeImageEditor(editorId)"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import type { Highlight } from "~/types/common";

const props = defineProps<{
  highlights: Highlight[];
  openEditorIds: Set<number>;
  pageAspectRatio: number;
  pageHeight: number;
  currentPage: number;
  pageRefs?: Element[];
}>();

const emit = defineEmits<{
  "close-editor": [highlightId: number];
}>();

const imagesContainer = ref<HTMLElement | null>(null);
const EDITOR_WIDTH = 512;
const PDF_COORDINATE_HEIGHT = 850; // Typical PDF coordinate height from polygon data.

function getHighlightText(highlightId: number): string {
  const highlight = props.highlights.find(h => h.id === highlightId);
  return highlight?.text || "";
}

function getEditorPositionStyle(highlightId: number) {
  const highlight = props.highlights.find(h => h.id === highlightId);
  if (!highlight) return {};

  const polygons = highlight.polygons[props.currentPage];
  if (!polygons || !polygons[0]) return {};

  // Vertical position: align with highlight's Y coordinate (minimum Y of all polygon points)
  const minY = Math.min(...polygons.map(p => p[1]));
  const topOffset = (minY / PDF_COORDINATE_HEIGHT) * props.pageHeight;

  return {
    top: `${topOffset}px`,
    left: "0",
    width: `${EDITOR_WIDTH}px`,
    position: "absolute" as const,
    touchAction: "none" as const,
  };
}

function closeImageEditor(highlightId: number) {
  emit("close-editor", highlightId);
}
</script>

