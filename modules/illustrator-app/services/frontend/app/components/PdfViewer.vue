<template>
  <div class="flex bg-base-100">
    <ClientOnly>
      <div ref="pdfEmbedWrapper" class="flex-grow pdf-wrapper bg-base-200" :style="{ width: `${pdfWidth}px`}" data-help-target="viewer">
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
            @rendered="$emit('pdf:rendered')"
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
          v-if="highlights.length"
          ref="highlightLayer"
          :key="highlightLayerKey"
          :highlights="highlights"
          :selected-highlights="selectedHighlights"
          :page-refs="pageRefs"
          :page-visibility="pageVisibility"
          :default-page-size="{ width: pdfWidthScaled, height: pdfPageHeight }"
          @select="toggleHighlightSelection"
          @open-editor="openImageEditor"
        />
      </div>
    </ClientOnly>
    <ImageLayer
      ref="imageLayer"
      :highlights="highlights"
      :open-editor-ids="openEditorIds"
      :page-aspect-ratio="pageAspectRatio"
      :page-refs="pageRefs"
      :page-height="pdfPageHeight"
      :current-page="currentPage"
      :style="{ width: IMAGES_WIDTH + 'px' }"
      data-help-target="images"
      @close-editor="closeImageEditor"
    />
    <div class="fixed bottom-0 left-0 flex justify-center z-200">
      <div class="join flex items-center bg-base-100 space-x-2 border border-base-content/25 rounded-tr">
        <button class="join-item btn btn-sm px-3 me-0" :disabled="currentPage <= 1" title="Previous Page" @click="setCurrentPage(currentPage - 1)">
          <Icon name="lucide:chevron-up" size="16" />
        </button>
        <input v-model="currentPageModel" type="number" class="input input-sm join-item px-2 text-base w-12 text-end" min="1" :max="pageNums.length" @keydown.enter="setCurrentPage(currentPageModel)" @blur="setCurrentPage(currentPageModel)">
        <div class="join-item">
          /&nbsp;{{ pageNums.length }}
        </div>
        <button class="join-item btn btn-sm px-3" :disabled="currentPage >= pageNums.length" title="Next Page" @click="setCurrentPage(currentPage + 1)">
          <Icon name="lucide:chevron-down" size="16" />
        </button>
        <div class="join-item ms-4 me-0">
          <button class="btn btn-sm px-2" :disabled="pdfScale <= 0.1" title="Zoom Out" @click="setPdfScale(pdfScale - 0.1)">
            <Icon name="lucide:zoom-out" size="16" />
          </button>
          <button class="btn btn-sm px-2" :disabled="pdfScale >= 3" title="Zoom In" @click="setPdfScale(pdfScale + 0.1)">
            <Icon name="lucide:zoom-in" size="16" />
          </button>
        </div>
        <label class="join-item input input-sm px-2 text-base w-16 gap-1">
          <input v-model="pdfScaleModel" type="number" min="10" max="300" step="10" class="text-end" @keydown.enter="setPdfScale(pdfScaleModel / 100)" @blur="setPdfScale(pdfScaleModel / 100)">
          %
        </label>
        <label class="join-item px-2 flex items-center gap-1">
          <input v-model="textLayerEnabled" type="checkbox" class="checkbox checkbox-sm">
          Render text layer (slow)
        </label>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { Highlight } from "~/types/common";
import { debounce } from "lodash-es";
import "vue-pdf-embed/dist/styles/annotationLayer.css";
import "vue-pdf-embed/dist/styles/textLayer.css";
import VuePdfEmbed, { useVuePdfEmbed } from "vue-pdf-embed";
// import { GlobalWorkerOptions } from "vue-pdf-embed/dist/index.essential.mjs";
// import PdfWorker from "pdfjs-dist/build/pdf.worker.mjs?url";
// GlobalWorkerOptions.workerSrc = PdfWorker;

const IMAGES_WIDTH = 512;
const PRELOAD_PAGES = 3;

const props = defineProps<{
  pdfUrl: string;
}>();

watch(() => props.pdfUrl, () => {
  reset();
});

const highlights = defineModel<Highlight[]>("highlights", { required: true });
const selectedHighlights = defineModel<Set<number>>("selectedHighlights", { required: true });

defineEmits<{
  (e: "pdf:rendered"): void;
}>();

const pdfUrlComputed = computed(() => props.pdfUrl);

const nuxtApp = useNuxtApp();
const highlightLayer = ref<InstanceType<typeof import("~/components/HighlightLayer.vue").default> | null>(null);
const imageLayer = ref<InstanceType<typeof import("~/components/ImageLayer.vue").default> | null>(null);
const pdfEmbedWrapper = ref<HTMLElement | null>(null);
const { doc } = useVuePdfEmbed({ source: pdfUrlComputed });
const highlightLayerKey = ref(0);
const pdfScaleModel = ref(100);
const pdfScale = ref(1);
const pdfWidth = ref<number>(800);
const pdfWidthScaled = computed(() => pdfWidth.value * pdfScale.value);
const textLayerEnabled = ref(false);

const openEditorIds = ref(new Set<number>());

function openImageEditor(highlightId: number) {
  openEditorIds.value.add(highlightId);
}

function closeImageEditor(highlightId: number) {
  openEditorIds.value.delete(highlightId);
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

  pageRefs.value.forEach((element) => {
    pageIntersectionObserver!.observe(element);
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

async function goToHighlight(highlight: Highlight) {
  // Make page of the first polygon visible
  const firstPolyPage = highlight.polygons ? Number(Object.keys(highlight.polygons)[0]) : null;
  if (firstPolyPage && !pageVisibility.value[firstPolyPage + 1]) {
    const targetPageNum = firstPolyPage + 1;
    for (let i = Math.max(1, targetPageNum - PRELOAD_PAGES); i <= Math.min(pageNums.value.length, targetPageNum + PRELOAD_PAGES); i++) {
      if (!pageVisibility.value[i]) {
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
  pageVisibility.value = {};
  pageIntersectionRatios.clear();
  currentPage.value = 1;
  if (virtualScrollDebounce) {
    clearTimeout(virtualScrollDebounce);
    virtualScrollDebounce = undefined;
  }
  highlightLayerKey.value++;
  console.log("resetting");
}

nuxtApp.$debugPanel.track("pageVisibility", pageVisibility);
nuxtApp.$debugPanel.track("pageIntersectionRatios", pageIntersectionRatios);

async function pageResizeHandler() {
  if (pdfEmbedWrapper.value) {
    const windowWidth = window.innerWidth;
    pdfWidth.value = windowWidth - IMAGES_WIDTH;
    console.log(`Setting PDF width to ${pdfWidth.value}px`);
    pdfEmbedWrapper.value.style.width = `${pdfWidth.value}px`;
    await nextTick();
    highlightLayerKey.value++; // force re-render highlight layer
  } else {
    console.warn("pdfEmbedWrapper is null");
  }
};
const pageResizeHandlerDebounced = debounce(pageResizeHandler, 200);

onMounted(() => {
  pageResizeHandlerDebounced();
  window.addEventListener("resize", pageResizeHandlerDebounced);
});

onBeforeUnmount(() => {
  window.removeEventListener("resize", pageResizeHandlerDebounced);
  pageIntersectionObserver?.disconnect();
  if (virtualScrollDebounce) {
    clearTimeout(virtualScrollDebounce);
  }
});
</script>
