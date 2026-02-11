<template>
  <div
    ref="imagesContainer"
    class="w-[512px] select-none"
  >
    <div class="relative">
      <ImageEditor
        v-for="editorId in openEditorIds"
        :key="editorId"
        ref="editorRefs"
        :highlight-id="editorId"
        :initial-text="getHighlightText(editorId)"
        :style="getEditorPositionStyle(editorId)"
        class="absolute pointer-events-auto"
        @delete="closeImageEditor(editorId)"
        @bring-to-front="bringImageToFront(editorId)"
        @pointer-down="onEditorPointerDown($event, editorId)"
        @state-change="handleEditorStateChange"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import type { ImageEditor } from "#components";
import type { EditorImageState, Highlight } from "~/types/common";

const props = defineProps<{
  pdfEmbedWrapper: HTMLElement | null;
  highlights: Highlight[];
  openEditorIds: Set<number>;
  pageAspectRatio: number;
  pageHeight: number;
  currentPage: number;
  pageRefs?: Element[];
}>();

const emit = defineEmits<{
  "close-editor": [highlightId: number];
  "editor-state-change": [state: EditorImageState];
}>();

const EDITOR_NO_IMAGE_HEIGHT = 156; // controls height when no image
const EDITOR_MAX_POSSIBLE_HEIGHT = 512 + EDITOR_NO_IMAGE_HEIGHT; // Image + controls

const imagesContainer = ref<HTMLElement | null>(null);

const highlightMap = computed(() => {
  const map: Record<number, Highlight> = {};
  for (const h of props.highlights) {
    map[h.id] = h;
  }
  return map;
});

function getHighlightText(highlightId: number): string {
  const highlight = highlightMap.value[highlightId];
  return highlight?.text || "";
}

// function getEditorPositionStyle(highlightId: number) {
//   const highlight = highlightMap.value[highlightId];
//   if (!highlight) return {};

//   const polygons = highlight.polygons[props.currentPage - 1];
//   if (!polygons || !polygons[0]) return {};

//   // Vertical position: align with highlight's Y coordinate (minimum Y of all polygon points)
//   const minY = Math.min(...polygons.map(p => p[1]));
//   const topOffset = minY * props.pageHeight;

//   return {
//     top: `${topOffset}px`,
//     left: "0",
//     width: `${EDITOR_WIDTH}px`,
//     position: "absolute" as const,
//     touchAction: "none" as const,
//   };
// }

function getEditorTop(highlightId: number) {
  // Try user-overridden drag positions
  const existingTop = imagePositions[highlightId]?.top;
  if (existingTop !== null && existingTop !== undefined) return existingTop;

  if (!props.pdfEmbedWrapper) return 0;
  const pdfEmbedBounding = props.pdfEmbedWrapper.getBoundingClientRect();
  const containerBounding = imagesContainer.value?.getBoundingClientRect();
  const highlight = highlightMap.value[highlightId];
  const highlightBounding = getHighlightBoundingNormalized(highlight);
  if (!highlightBounding) return 0;

  // Determine page index: polygon keys are 0-based (see usage in index.vue where +1 is applied)
  const firstPageKey = Object.keys(highlight.polygons || {})[0];
  const pageIndex = Number(firstPageKey) || 0; // 0-based
  // Attempt to use actual page element metrics for more accurate placement
  const pageEl = props.pageRefs?.[pageIndex] as HTMLElement | undefined;
  let globalTop: number;
  if (pageEl) {
    const pageRect = pageEl.getBoundingClientRect();
    const actualPageHeight = pageRect.height || (() => {
      const ratio = props.pageAspectRatio;
      return pdfEmbedBounding.width * ratio; // fallback
    })();
    globalTop = pageRect.top + highlightBounding.y * actualPageHeight + window.scrollY;
  } else {
    // Fallback to estimated cumulative height method
    const ratio = props.pageAspectRatio;
    const width = pdfEmbedBounding.width;
    const pageHeight = width * ratio;
    const pageOffset = pageIndex * pageHeight; // pageIndex is 0-based
    globalTop = pdfEmbedBounding.top + pageOffset + highlightBounding.y * pageHeight + window.scrollY;
  }
  const containerTopGlobal = (containerBounding?.top || 0) + window.scrollY;
  let top = globalTop - containerTopGlobal;

  if (highlightBounding.height < EDITOR_NO_IMAGE_HEIGHT) {
    // align center of image to center of highlight
    top = top - EDITOR_NO_IMAGE_HEIGHT / 2 + highlightBounding.height / 2;
  }

  return top;
}

function getEditorPositionStyle(highlightId: number) {
  const top = getEditorTop(highlightId);
  return {
    top: `${top}px`,
    left: "0px",
    width: "100%",
    position: "absolute",
    touchAction: "none",
    zIndex: imageZIndices[highlightId] || 1
  };
}

type ImagePosition = { top: number; left: number };
const imagePositions = reactive<Record<number, ImagePosition>>({});
const imageZIndices = reactive<Record<number, number>>({});
let zCounter = 1;
const draggingId = ref<number | null>(null);
let draggingEl: HTMLElement | null = null;
// (Deprecated) dragStartPointer replaced by page-based coordinates
let dragStartPos: ImagePosition | null = null;
// Track page-based starting coordinates (includes scroll offset) so we can adjust during scroll
let dragStartPage: { x: number; y: number } | null = null;
// Last known client (viewport) pointer position; used to recompute deltas on scroll events
let lastPointerClient: { x: number; y: number } = { x: 0, y: 0 };

function getHighlightBoundingNormalized(highlight: Highlight) {
  // Need to convert normalized first polygon to bounding box
  if (!highlight.polygons) return null;
  const firstPoly = highlight.polygons[Object.keys(highlight.polygons)[0] as any];
  // get lowest and highest of y and the other extremes for x
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const [x, y] of firstPoly) {
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
  }
  return { x: minX, y: minY, width: maxX - minX, height: maxY - minY };
}

function ensureImagePosition(highlightId: number) {
  if (!imagePositions[highlightId]) {
    // Initialize from computed style
    const style = getEditorPositionStyle(highlightId) as any;
    const parsePx = (v: string | undefined) => v ? Number(v.toString().replace(/px$/, "")) : 0;
    imagePositions[highlightId] = {
      top: parsePx(style.top),
      left: parsePx(style.left)
    };
  }
  return imagePositions[highlightId];
}

function onEditorPointerDown(e: PointerEvent, highlightId: number) {
  if (!(e.target instanceof HTMLElement)) return;
  // Ignore drag start when clicking overlay controls
  if ((e.target as HTMLElement).closest(".image-controls")) return;
  // Only allow primary (usually left) mouse button / primary pointer to initiate drag
  // Avoid starting drag on right-click (button === 2) or middle-click (button === 1)
  if (e.button !== 0) return;
  draggingId.value = highlightId;
  draggingEl = e.currentTarget as HTMLElement;
  const pos = ensureImagePosition(highlightId);
  dragStartPage = { x: e.pageX, y: e.pageY };
  lastPointerClient = { x: e.clientX, y: e.clientY };
  dragStartPos = { ...pos };
  (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
  window.addEventListener("pointermove", onEditorPointerMove);
  window.addEventListener("pointerup", onEditorPointerUp, { once: true });
}

function onEditorPointerMove(e: PointerEvent) {
  if (draggingId.value == null || !dragStartPage || !dragStartPos) return;
  lastPointerClient = { x: e.clientX, y: e.clientY };
  const deltaX = e.pageX - dragStartPage.x;
  const deltaY = e.pageY - dragStartPage.y;
  updateDraggedPosition(deltaX, deltaY);
}

function onEditorPointerUp() {
  draggingId.value = null;
  draggingEl = null;
  dragStartPos = null;
  dragStartPage = null;
  window.removeEventListener("pointermove", onEditorPointerMove);
}

function updateDraggedPosition(deltaX: number, deltaY: number) {
  if (draggingId.value == null || !dragStartPos) return;
  const pos = imagePositions[draggingId.value];
  if (!pos) return;
  const container = imagesContainer.value;
  if (container) {
    const ch = container.clientHeight;
    // const cw = container.clientWidth;
    const eh = draggingEl?.offsetHeight || EDITOR_MAX_POSSIBLE_HEIGHT;
    // const ew = draggingEl?.offsetWidth || cw;
    pos.top = clamp(dragStartPos.top + deltaY, 0, Math.max(0, ch - eh));
    // pos.left = clamp(dragStartPos.left + deltaX, 0, Math.max(0, cw - ew));
  } else {
    pos.top = dragStartPos.top + deltaY;
    // pos.left = dragStartPos.left + deltaX;
  }
}

function onScrollWhileDragging() {
  // When the user scrolls during a drag without moving the pointer (e.g. wheel / two-finger scroll),
  // recompute position using page deltas so the element appears anchored to the pointer.
  if (draggingId.value == null || !dragStartPage) return;
  const currentPageX = lastPointerClient.x + window.scrollX;
  const currentPageY = lastPointerClient.y + window.scrollY;
  const deltaX = currentPageX - dragStartPage.x;
  const deltaY = currentPageY - dragStartPage.y;
  updateDraggedPosition(deltaX, deltaY);
}

onMounted(() => {
  window.addEventListener("scroll", onScrollWhileDragging, { passive: true });
});

onBeforeUnmount(() => {
  window.removeEventListener("pointermove", onEditorPointerMove);
  window.removeEventListener("scroll", onScrollWhileDragging);
});

function bringImageToFront(highlightId: number) {
  zCounter += 1;
  imageZIndices[highlightId] = zCounter;
}

function closeImageEditor(highlightId: number) {
  emit("close-editor", highlightId);
}

function handleEditorStateChange(state: EditorImageState) {
  emit("editor-state-change", state);
}

const editorRefs = ref<Array<InstanceType<typeof ImageEditor> | null>>([]);

function getExportImages(): Record<number, Blob> {
  const result: Record<number, Blob> = {};

  for (const editor of editorRefs.value) {
    if (!editor) continue;
    const exportData = editor.getExportImage();
    if (exportData) {
      result[exportData.highlightId] = exportData.imageBlob;
    }
  }

  return result;
}

defineExpose({
  getExportImages
});
</script>

