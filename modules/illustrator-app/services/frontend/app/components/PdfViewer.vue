<template>
  <div ref="pdfViewer" class="flex bg-base-100 max-w-360 mx-auto">
    <ClientOnly>
      <div ref="pdfEmbedWrapper" class="grow pdf-wrapper bg-base-200" :style="{ width: `${pdfWidth}px`}" data-help-target="viewer">
        <div
          v-for="pageNum in pageNums"
          :key="pageNum"
          ref="pageRefs"
          class="page-wrapper"
        >
          <VuePdfEmbed
            v-if="pageVisibility[pageNum]"
            annotation-layer
            :text-layer="textLayerEnabled"
            :source="doc"
            :page="pageNum"
            :width="pdfWidthScaled"
            class="pdf-embed"
            :style="{
              width: `${pdfWidthScaled}px`,
              height: `${pdfPageHeight}px`,
            }"
            @rendered="onRendered"
          />
          <div
            v-else
            class="skeleton"
            :data-page="pageNum"
            :style="{
              width: `${pdfWidthScaled}px`,
              height: `${pdfPageHeight}px`,
              margin: '0 auto',
            }"
          />
        </div>
        <HighlightLayer
          v-if="isPdfRendered && highlights.length"
          ref="highlightLayer"
          :key="highlightLayerKey"
          :highlights="highlights"
          :selected-highlights="selectedHighlights"
          :page-refs="pageRefs"
          :page-visibility="pageVisibility"
          :default-page-size="{ width: pdfWidthScaled, height: pdfPageHeight }"
          @select="toggleHighlightSelection"
          @open-editor="(id) => openImageEditors([id])"
        />
      </div>
    </ClientOnly>
    <ImageLayer
      ref="imageLayer"
      :pdf-embed-wrapper="pdfEmbedWrapper"
      :highlights="highlights"
      :open-editor-ids="openEditorIds"
      :page-aspect-ratio="pageAspectRatio"
      :page-refs="pageRefs"
      :page-height="pdfPageHeight"
      :current-page="currentPage"
      :auto-enabled="props.autoEnabled"
      :style="{ width: IMAGES_WIDTH + 'px' }"
      data-help-target="images"
      @close-editor="(id) => closeImageEditors([id])"
      @editor-state-change="updateEditorState"
    />
    <HeatmapViewer
      v-if="isPdfRendered"
      class="z-40"
      :style="heatmapStyle"
      :highlights="highlights"
      :current-page="currentPage"
      :page-aspect-ratio="pageAspectRatio"
      :page-refs="pageRefs"
      :editor-states="editorStates"
    />
    <div class="fixed bottom-0 left-0 flex justify-center z-120">
      <div class="join flex items-center bg-base-100 space-x-2 border border-base-content/25 rounded-tr">
        <button class="join-item btn btn-sm px-2 me-0" :disabled="currentPage <= 1" title="Previous Page" @click="setCurrentPage(currentPage - 1)">
          <Icon name="lucide:chevron-up" size="16" />
        </button>
        <input v-model="currentPageModel" type="number" class="input input-sm join-item px-2 text-base w-12 text-end" min="1" :max="pageNums.length" @keydown.enter="setCurrentPage(currentPageModel)" @blur="setCurrentPage(currentPageModel)">
        <div class="join-item">
          /&nbsp;{{ pageNums.length }}
        </div>
        <button class="join-item btn btn-sm px-2 me-1" :disabled="currentPage >= pageNums.length" title="Next Page" @click="setCurrentPage(currentPage + 1)">
          <Icon name="lucide:chevron-down" size="16" />
        </button>
        <div class="join-item">
          <button class="btn btn-sm px-2" :disabled="pdfScale <= 0.1" title="Zoom Out" @click="setPdfScale(pdfScale - 0.1)">
            <Icon name="lucide:zoom-out" size="16" />
          </button>
          <label class="join-item input input-sm px-2 text-base w-16 gap-1">
            <input v-model="pdfScaleModel" type="number" min="10" max="300" step="10" class="text-end" @keydown.enter="setPdfScale(pdfScaleModel / 100)" @blur="setPdfScale(pdfScaleModel / 100)">
            %
          </label>
          <button class="btn btn-sm px-2" :disabled="pdfScale >= 3" title="Zoom In" @click="setPdfScale(pdfScale + 0.1)">
            <Icon name="lucide:zoom-in" size="16" />
          </button>
        </div>
        <label class="join-item pe-2 flex items-center gap-1">
          <input v-model="textLayerEnabled" type="checkbox" class="checkbox checkbox-sm">
          Render text (slow)
        </label>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { debounce } from "lodash-es";
import { DEFAULT_PAGE_ASPECT_RATIO, scrollIntoView } from "~/utils/utils";
import "vue-pdf-embed/dist/styles/annotationLayer.css";
import "vue-pdf-embed/dist/styles/textLayer.css";
import VuePdfEmbed, { useVuePdfEmbed } from "vue-pdf-embed";

import type { EditorImageState, Highlight } from "~/types/common";

import HeatmapViewer from "~/components/HeatmapViewer.vue";
// import { GlobalWorkerOptions } from "vue-pdf-embed/dist/index.essential.mjs";
// import PdfWorker from "pdfjs-dist/build/pdf.worker.mjs?url";
// GlobalWorkerOptions.workerSrc = PdfWorker;

const IMAGES_WIDTH = 512;

const emit = defineEmits<{
  "pdf:rendered": [];
}>();
const PRELOAD_PAGES = 3;

const props = defineProps<{
  pdfUrl: string;
  autoEnabled?: boolean;
}>();

watch(() => props.pdfUrl, () => {
  reset();
});

const highlights = defineModel<Highlight[]>("highlights", { required: true });
const selectedHighlights = defineModel<Set<number>>("selectedHighlights", { required: true });

const pdfUrlComputed = computed(() => props.pdfUrl);

const nuxtApp = useNuxtApp();
const highlightLayer = ref<InstanceType<typeof import("~/components/HighlightLayer.vue").default> | null>(null);
const imageLayer = ref<InstanceType<typeof import("~/components/ImageLayer.vue").default> | null>(null);
const pdfViewer = ref<HTMLElement | null>(null);
const pdfEmbedWrapper = ref<HTMLElement | null>(null);
const { doc } = useVuePdfEmbed({ source: pdfUrlComputed });
const highlightLayerKey = ref(0);
const pdfScaleModel = ref(100);
const pdfScale = ref(1);
const pdfWidth = ref<number>(800);
const pdfWidthScaled = computed(() => pdfWidth.value * pdfScale.value);
const textLayerEnabled = ref(false);

const openEditorIds = ref(new Set<number>());
const editorStates = ref<EditorImageState[]>([]);
const heatmapOffset = ref({ top: 0, height: 0 });
const heatmapStyle = computed(() => ({
  top: `${heatmapOffset.value.top}px`,
  height: `${heatmapOffset.value.height}px`
}));

function openImageEditors(highlightIds: number[]) {
  openEditorIds.value = new Set([...openEditorIds.value, ...highlightIds]);
  // Initialize state immediately so dots appear right away.
  // Batch all new entries into a single reactive write to avoid queuing
  // PdfViewer/HeatmapViewer for update once per highlight.
  const existingIds = new Set(editorStates.value.map(s => s.highlightId));
  const newEntries = highlightIds
    .filter(id => !existingIds.has(id))
    .map(id => ({ highlightId: id, imageUrl: null as string | null, hasImage: false }));
  if (newEntries.length > 0) {
    editorStates.value = [...editorStates.value, ...newEntries];
  }
}

function closeImageEditors(highlightIds: number[]) {
  const newSet = new Set(openEditorIds.value);
  for (const highlightId of highlightIds) {
    newSet.delete(highlightId);
  }
  openEditorIds.value = newSet;
  // Remove editor state so dots vanish
  editorStates.value = editorStates.value.filter(state => !highlightIds.includes(state.highlightId));
}

function updateEditorState(nextState: EditorImageState) {
  const index = editorStates.value.findIndex(state => state.highlightId === nextState.highlightId);
  if (index === -1) {
    editorStates.value.push(nextState);
    return;
  }

  editorStates.value[index] = nextState;
}

async function setPdfScale(scale: number) {
  if (isNaN(scale)) return;
  pdfScale.value = Math.min(3, Math.max(0.1, scale));
  pdfScaleModel.value = Math.round(pdfScale.value * 100);
  await nextTick();
  highlightLayerKey.value++; // force re-render highlight layer
}

// 0-indexed
const pageRefs = ref<Element[]>([]);
// 1-indexed
const pageVisibility = ref<Record<number, boolean>>({});
let pageIntersectionObserver: IntersectionObserver | undefined;
let virtualScrollDebounce: ReturnType<typeof setTimeout> | undefined;
const currentPageModel = ref(1);
const currentPage = ref(1);
watch(currentPage, (newPage) => {
  currentPageModel.value = newPage;
});

function setCurrentPage(pageNum: number) {
  if (isNaN(pageNum) || pageNum < 1 || pageNum > pageNums.value.length) {
    currentPageModel.value = currentPage.value;
    return;
  }
  currentPage.value = Math.round(pageNum);
  const pageElement = pageRefs.value[currentPage.value - 1] as HTMLElement | undefined;
  if (pageElement) {
    pageElement.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

// 1-indexed
const pageNums = computed(() =>
  doc.value ? [...Array(doc.value.numPages + 1).keys()].slice(1) : []
);
// 1-indexed
const pageAspectRatio = ref<number>(DEFAULT_PAGE_ASPECT_RATIO);

// Pre-fetch first page ratio to improve initial placeholder accuracy
watch(doc, async (d) => {
  if (d) {
    const page = await d.getPage(1);
    const vp = page.getViewport({ scale: 1 });
    pageAspectRatio.value = vp.height / vp.width;
  }
}, { immediate: true });

const pdfPageHeight = computed(() => {
  return pdfWidthScaled.value * pageAspectRatio.value;
});

const pageIntersectionRatios = new Map<number, number>();

const resetPageIntersectionObserver = () => {
  pageIntersectionObserver?.disconnect();
  // Clear any previously tracked ratios to avoid stale values
  pageIntersectionRatios.clear();
  pageIntersectionObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      const index = pageRefs.value.indexOf(entry.target);
      if (index === -1) return;
      const pageNum = pageNums.value[index];
      pageIntersectionRatios.set(pageNum, entry.intersectionRatio);
    });

    if (virtualScrollDebounce) {
      clearTimeout(virtualScrollDebounce);
    }
    virtualScrollDebounce = setTimeout(() => {
      // page with the largest intersection ratio
      currentPage.value = pageIntersectionRatios.size > 0
        ? Array.from(pageIntersectionRatios.entries()).sort((a, b) => a[1] - b[1]).pop()![0]
        : 1;
      const loadRangeStart = Math.max(1, currentPage.value - PRELOAD_PAGES);
      const loadRangeEnd = Math.min(pageNums.value.length, currentPage.value + PRELOAD_PAGES);

      for (let i = 1; i <= pageNums.value.length; i++) {
        pageVisibility.value[i] = i >= loadRangeStart && i <= loadRangeEnd;
      }
    }, 150); // 150ms debounce
  }, {
    root: null,
    threshold: [0, 0.1, 0.25, 0.5, 0.75, 1.0],
  });

  // Observe only valid mounted elements
  pageRefs.value.forEach((element) => {
    if (element && element instanceof Element) {
      pageIntersectionObserver!.observe(element);
    }
  });
};

watch(pageNums, (newPageNums) => {
  if (newPageNums.length > 0) {
    // Initialize with the first few pages for better UX
    const initialLoadCount = Math.min(PRELOAD_PAGES + 1, newPageNums.length);
    const initialVisibility: Record<number, boolean> = {};
    for (let i = 1; i <= initialLoadCount; i++) {
      initialVisibility[i] = true;
    }
    pageVisibility.value = initialVisibility;
  }
  nextTick(resetPageIntersectionObserver);
});

function getFirstPolygonPageIndex(polygons: any): number | null {
  if (!polygons) return null;
  // Array-style polygons (simple list) -> assume page index 0
  if (Array.isArray(polygons)) return 0;
  const keys = Object.keys(polygons || {});
  if (keys.length === 0) return null;
  const k = Number(keys[0]);
  return Number.isFinite(k) ? k : null;
}

async function goToHighlight(highlight: Highlight) {
  // Determine a valid 0-based page index for the first polygon (null = unknown)
  const firstPolyPage = getFirstPolygonPageIndex(highlight.polygons);
  if (firstPolyPage === null) return;
  const targetPageNum = firstPolyPage + 1; // convert 0-based -> 1-based
  if (pageVisibility.value[targetPageNum] !== true) {
    const start = Math.max(1, targetPageNum - PRELOAD_PAGES);
    const end = Math.min(pageNums.value.length, targetPageNum + PRELOAD_PAGES);
    for (let i = start; i <= end; i++) {
      if (pageVisibility.value[i] !== true) {
        pageVisibility.value[i] = true;
      }
    }
    await nextTick();
  }
  // Scroll to the highlight
  await scrollIntoView(() => `[data-segment-id="${highlight.id}"]`);
  // Spawn a marker animation to the left
  highlightLayer.value?.spawnMarker(highlight.id);
}
nuxtApp.hook("custom:goToHighlight", goToHighlight);

function toggleHighlightSelection(id: number) {
  if (selectedHighlights.value.has(id)) {
    selectedHighlights.value.delete(id);
  } else {
    selectedHighlights.value.add(id);
  }
}

function reset() {
  openEditorIds.value.clear();
  editorStates.value = [];
  pageVisibility.value = {};
  pageIntersectionRatios.clear();
  currentPage.value = 1;
  if (virtualScrollDebounce) {
    clearTimeout(virtualScrollDebounce);
    virtualScrollDebounce = undefined;
  }
  highlightLayerKey.value++;
  // console.log("resetting");
}

nuxtApp.$debugPanel.track("pageVisibility", pageVisibility);
nuxtApp.$debugPanel.track("pageIntersectionRatios", pageIntersectionRatios);

async function pageResizeHandler() {
  updateHeatmapOffset();
  if (pdfEmbedWrapper.value) {
    const containerWidth = pdfViewer.value ? pdfViewer.value.clientWidth : window.innerWidth;
    pdfWidth.value = containerWidth - IMAGES_WIDTH;
    // console.log(`Setting PDF width to ${pdfWidth.value}px`);
    pdfEmbedWrapper.value.style.width = `${pdfWidth.value}px`;
    await nextTick();
    highlightLayerKey.value++; // force re-render highlight layer
  } else {
    console.warn("pdfEmbedWrapper is null");
  }
};
const pageResizeHandlerDebounced = debounce(pageResizeHandler, 200);

function updateHeatmapOffset() {
  if (!pdfViewer.value) return;
  const rect = pdfViewer.value.getBoundingClientRect();
  const topBarHeight = document.querySelector(".top-bar")?.getBoundingClientRect().height ?? 0;
  const top = Math.max(rect.top, topBarHeight);
  heatmapOffset.value = {
    top,
    height: Math.max(0, window.innerHeight - top)
  };
}

const isPdfRendered = ref(false);

function onRendered() {
  if (!isPdfRendered.value) {
    isPdfRendered.value = true;
    pageResizeHandler();
  }
  emit("pdf:rendered");
}

onMounted(() => {
  pageResizeHandlerDebounced();
  updateHeatmapOffset();
  window.addEventListener("resize", pageResizeHandlerDebounced);
});

onBeforeUnmount(() => {
  window.removeEventListener("resize", pageResizeHandlerDebounced);
  pageIntersectionObserver?.disconnect();
  if (virtualScrollDebounce) {
    clearTimeout(virtualScrollDebounce);
  }
});

defineExpose({
  getPageCount: () => pageNums.value.length,
  imageLayer,
  openImageEditors,
  closeImageEditors,
  // expose sizing helpers for parent components
  pageAspectRatio,
  pageRefs,
});
</script>
