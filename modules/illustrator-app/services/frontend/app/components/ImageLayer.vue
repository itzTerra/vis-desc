<template>
  <div
    ref="imagesContainer"
    class="w-[512px] border-l border-base-300 select-none"
  >
    <div class="relative">
      <figure
        v-for="(highlight, index) in highlights.filter(h => h.imageUrl || h.imageLoading)"
        :key="index"
        :style="(getHighlightImageStyle(highlight) as StyleValue)"
        :title="highlight.text"
        class="group shadow-sm rounded bg-base-200/30 backdrop-blur-sm hover:shadow cursor-grab active:cursor-grabbing transition"
        @pointerdown="(e) => onImagePointerDown(e, highlight)"
      >
        <div v-if="highlight.imageUrl" class="absolute top-1 right-1 flex gap-1 opacity-0 group-hover:opacity-100 transition pointer-events-auto image-controls z-50">
          <button class="btn btn-xs btn-circle btn-primary" title="Bring to front" @click.stop="bringImageToFront(highlight)">
            <Icon name="lucide:arrow-up" class="w-3 h-3" />
          </button>
          <button class="btn btn-xs btn-circle btn-error" title="Delete image" @click.stop="deleteImage(highlight)">
            <Icon name="lucide:x" class="w-3 h-3" />
          </button>
        </div>
        <Transition>
          <div v-if="highlight.imageLoading" class="flex justify-center items-center w-[512px] h-[512px] pointer-events-none select-none">
            <div class="loading loading-spinner loading-md" />
          </div>
          <img v-else :src="highlight.imageUrl" alt="ai-illustration" draggable="false" class="pointer-events-none select-none">
        </Transition>
      </figure>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { StyleValue } from "vue";
import type { Highlight } from "~/types/common";

const props = defineProps<{
  pdfEmbedWrapper: HTMLElement | null;
  highlights: Highlight[];
  pageAspectRatio: number;
  pageHeight: number;
  pageRefs?: Element[]; // optional list of page wrapper elements (index = pageNum - 1)
}>();

const imagesContainer = ref<HTMLElement | null>(null);

const IMAGE_HEIGHT = 512;

function getHighlightImageStyle(highlight: Highlight) {
  if (!props.pdfEmbedWrapper) return {};
  const pdfEmbedBounding = props.pdfEmbedWrapper.getBoundingClientRect();
  const containerBounding = imagesContainer.value?.getBoundingClientRect();
  const highlightBounding = getHighlightBoundingNormalized(highlight);
  if (!highlightBounding) return {};
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
    globalTop = pageRect.top + window.scrollY + highlightBounding.y * actualPageHeight;
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

  if (highlightBounding.height < IMAGE_HEIGHT) {
    // align center of image to center of highlight
    top = top - IMAGE_HEIGHT / 2 + highlightBounding.height / 2;
  }
  // Allow user-overridden drag positions
  const existing = imagePositions[highlight.id];
  if (existing) {
    top = existing.top;
  }
  const left = imagePositions[highlight.id]?.left || 0;
  return {
    top: `${top}px`,
    left: `${left}px`,
    width: "100%",
    maxHeight: `${IMAGE_HEIGHT}px`,
    position: "absolute",
    touchAction: "none",
    zIndex: imageZIndices[highlight.id] || 1
  };
}

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

function ensureImagePosition(highlight: Highlight) {
  if (!imagePositions[highlight.id]) {
    // Initialize from computed style
    const style = getHighlightImageStyle(highlight) as any;
    const parsePx = (v: string | undefined) => v ? Number(v.toString().replace(/px$/, "")) : 0;
    imagePositions[highlight.id] = {
      top: parsePx(style.top),
      left: parsePx(style.left)
    };
  }
  return imagePositions[highlight.id];
}

function onImagePointerDown(e: PointerEvent, highlight: Highlight) {
  if (!(e.target instanceof HTMLElement)) return;
  // Ignore drag start when clicking overlay controls
  if ((e.target as HTMLElement).closest(".image-controls")) return;
  // Only allow primary (usually left) mouse button / primary pointer to initiate drag
  // Avoid starting drag on right-click (button === 2) or middle-click (button === 1)
  if (e.button !== 0) return;
  draggingId.value = highlight.id;
  draggingEl = e.currentTarget as HTMLElement;
  const pos = ensureImagePosition(highlight);
  dragStartPage = { x: e.pageX, y: e.pageY };
  lastPointerClient = { x: e.clientX, y: e.clientY };
  dragStartPos = { ...pos };
  (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
  window.addEventListener("pointermove", onImagePointerMove);
  window.addEventListener("pointerup", onImagePointerUp, { once: true });
}

function onImagePointerMove(e: PointerEvent) {
  if (draggingId.value == null || !dragStartPage || !dragStartPos) return;
  lastPointerClient = { x: e.clientX, y: e.clientY };
  const deltaX = e.pageX - dragStartPage.x;
  const deltaY = e.pageY - dragStartPage.y;
  updateDraggedPosition(deltaX, deltaY);
}

function onImagePointerUp() {
  draggingId.value = null;
  draggingEl = null;
  dragStartPos = null;
  dragStartPage = null;
  window.removeEventListener("pointermove", onImagePointerMove);
}

function updateDraggedPosition(deltaX: number, deltaY: number) {
  if (draggingId.value == null || !dragStartPos) return;
  const pos = imagePositions[draggingId.value];
  if (!pos) return;
  const container = imagesContainer.value;
  if (container) {
    const ch = container.clientHeight;
    const cw = container.clientWidth;
    const eh = draggingEl?.offsetHeight || IMAGE_HEIGHT;
    const ew = draggingEl?.offsetWidth || cw;
    pos.top = clamp(dragStartPos.top + deltaY, 0, Math.max(0, ch - eh));
    pos.left = clamp(dragStartPos.left + deltaX, 0, Math.max(0, cw - ew));
  } else {
    pos.top = dragStartPos.top + deltaY;
    pos.left = dragStartPos.left + deltaX;
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
  window.removeEventListener("pointermove", onImagePointerMove);
  window.removeEventListener("scroll", onScrollWhileDragging);
});

function clamp(v: number, min: number, max: number) {
  return v < min ? min : (v > max ? max : v);
}

function bringImageToFront(highlight: Highlight) {
  zCounter += 1;
  imageZIndices[highlight.id] = zCounter;
}

function deleteImage(highlight: Highlight) {
  window.removeEventListener("pointermove", onImagePointerMove);
  if (draggingId.value === highlight.id) {
    draggingId.value = null;
    draggingEl = null;
    dragStartPos = null;
  }
  highlight.imageUrl = undefined as any;
  highlight.imageLoading = false;
  delete imagePositions[highlight.id];
  delete imageZIndices[highlight.id];
}

function reset() {
  for (const key in imagePositions) {
    delete imagePositions[key];
  }
  for (const key in imageZIndices) {
    delete imageZIndices[key];
  }
  zCounter = 1;
}

defineExpose({
  bringImageToFront,
  deleteImage,
  reset
});
</script>

