<template>
  <div>
    <div class="top-bar w-full px-4 py-2 bg-base-200 flex flex-col sm:flex-row gap-4 justify-between sm:items-center sticky top-0 z-10 h-[114px] sm:h-[58px]">
      <div class="flex items-center flex-grow-1 w-100">
        <div class="flex items-end gap-4 flex-grow-1">
          <input type="file" accept="application/pdf" class="file-input" @change="handleFileUpload">
          <div class="flex items-center join bg-base-100 ps-2 space-x-2 border border-base-content/25 rounded">
            <Icon name="lucide:component" class="join-item" />
            <select v-model="modelSelect" class="join-item select select-ghost select-sm w-auto text-nowrap ps-1">
              <option value="all_minilm_l6_v2" disabled>
                all-MiniLM-L6-v2
              </option>
              <option value="random">
                Random [debug]
              </option>
              <option value="deberta_mnli">
                DeBERTa-MNLI
              </option>
            </select>
          </div>
        </div>
        <div v-if="isLoading || isCancelled" class="flex items-center ms-4">
          <div class="flex flex-col">
            <progress class="progress progress-info w-52" :value="scoredSegments.length" :max="segmentCount" />
            <div class="flex justify-between mt-0.5 text-sm opacity-60">
              <template v-if="!isCancelled && segmentCount">
                <span>{{ scoredSegments.length }}/{{ segmentCount }}</span>
                <span>~{{ formatEta(etaMs) }}&nbsp;remaining</span>
              </template>
              <template v-else-if="!isCancelled">
                <span>Loading...</span>
              </template>
              <template v-else>
                <span>{{ scoredSegments.length }}/{{ segmentCount }}</span>
                <span>Loading canceled</span>
              </template>
            </div>
          </div>
          <button v-if="!isCancelled" class="btn btn-error btn-sm mx-2" @click="cancelSegmentLoading">
            Stop
          </button>
        </div>
      </div>
      <TextSearch
        v-model:index="currentSearchIndex"
        v-model:current-set-index="currentSearchSetIndex"
        :input="searchInput"
        :found="currentSearchSet.length"
        :searched-set-count="2"
        @prev="onSearchPrevNext"
        @next="onSearchPrevNext"
        @cycle="changeSearchSet"
      />
    </div>
    <div v-if="pdfUrl" class="flex bg-base-100">
      <div class="flex-grow">
        <div class="relative border border-gray-300 rounded">
          <ClientOnly>
            <div ref="pdfEmbedWrapper" class="pdf-wrapper">
              <div v-for="pageNum in pageNums" :key="pageNum" ref="pageRefs" :style="(pagePlaceholderStyles[pageNum - 1] as StyleValue)">
                <VuePdfEmbed
                  v-if="pageVisibility[pageNum]"
                  annotation-layer
                  text-layer
                  :source="doc"
                  :page="pageNum"
                  class="pdf-embed"
                />
                <div
                  v-else
                  class="skeleton h-full w-full"
                  :data-page="pageNum"
                  style="position:absolute; inset:0;"
                />
              </div>
              <HighlightLayer
                v-if="highlights.length"
                ref="highlightLayer"
                :highlights="highlights"
                :selected-highlights="selectedHighlights"
                :page-refs="pageRefs"
                :page-visibility="pageVisibility"
                @select="(index) => toggleHighlightSelection(index)"
              />
            </div>
          </ClientOnly>
        </div>
      </div>
      <div class="w-[512px] h-[90dvh]">
        <figure
          v-for="(highlight, index) in highlights.filter(h => h.imageUrl || h.imageLoading)"
          :key="index"
          :style="(getHighlightImageStyle(highlight) as StyleValue)"
          :title="highlight.text"
        >
          <Transition>
            <div v-if="highlight.imageLoading" class="flex justify-center items-center">
              <div class="loading loading-spinner loading-md" />
            </div>
            <img v-else :src="highlight.imageUrl" alt="ai-illustration">
          </Transition>
        </figure>
      </div>
    </div>
    <div v-else class="hero bg-custom">
      <div class="hero-content flex-col lg:flex-row-reverse max-w-[64rem]">
        <img src="/vis-desc-image.png" alt="hero-image" width="400" height="400">
        <div>
          <h1 class="text-5xl font-bold">
            Upload a file in PDF format to get started
          </h1>
          <p class="py-6">
            The tool is is designed to evaluate literature in English language, specially fiction, travel and history genres. Results for other types of content may vary.
          </p>
          <div class="max-w-xl">
            <label
              class="flex justify-center w-full h-24 px-4 transition border-2 border-dashed rounded-md appearance-none cursor-pointer text-base-content/50 border-base-content/50 hover:border-base-content/30 focus:outline-none hover:bg-primary/10"
            >
              <span class="flex items-center space-x-2 ">
                <Icon name="lucide:upload" />
                <span class="font-medium">
                  Drop file here or click in this area
                </span>
              </span>
              <input type="file" accept="application/pdf" name="file_upload" class="hidden" @change="handleFileUpload">
            </label>
          </div>
        </div>
      </div>
    </div>
    <BottomBar />
    <BackToTop />
    <Alert />
  </div>
</template>

<script setup lang="ts">
import type { Highlight } from "~/types/common";
import VuePdfEmbed, { useVuePdfEmbed } from "vue-pdf-embed";
import type { StyleValue } from "vue";
import "vue-pdf-embed/dist/styles/annotationLayer.css";
import "vue-pdf-embed/dist/styles/textLayer.css";

type Segment = { text: string, score: number };
type SocketMessage = { content: unknown, type: "segment" | "batch" | "info" | "error" | "success" };

const SCORE_THRESHOLD = 0.95;

const nuxtApp = useNuxtApp();
const call = nuxtApp.$api;

const pdfUrl = ref<string | null>(null);
const modelSelect = ref<string>("random");
const isLoading = ref(false);
const isCancelled = ref(false);
const segmentCount = ref(0);
const highlights = reactive<Highlight[]>([]);
const selectedHighlights = reactive<Set<number>>(new Set());
const searchInput = ref("");
const currentSearchIndex = ref(0);
enum SearchSet {
  AllHighlights = 0,
  SelectedHighlights = 1
}
const currentSearchSetIndex = ref<SearchSet>(SearchSet.AllHighlights);

const highlightLayer = ref<InstanceType<typeof import("~/components/HighlightLayer.vue").default> | null>(null);
const pdfEmbedWrapper = ref<HTMLElement | null>(null);
const { doc } = useVuePdfEmbed({ source: pdfUrl });
const pdfRenderKey = ref(0);
const pageRefs = ref<Element[]>([]);
const pageVisibility = ref<Record<number, boolean>>({});
let pageIntersectionObserver: IntersectionObserver | undefined;

const pageNums = computed(() =>
  doc.value ? [...Array(doc.value.numPages + 1).keys()].slice(1) : []
);
const PRELOAD_PAGES = 4;
const DEFAULT_PAGE_ASPECT_RATIO = 1.4142; // A4 fallback (height / width)
const pageAspectRatios = reactive<Record<number, number>>({});
const containerWidth = ref(0);
const resizeObserver: ResizeObserver | null = new ResizeObserver(entries => {
  for (const entry of entries) {
    containerWidth.value = entry.contentRect.width;
  }
});

// Pre-fetch first page ratio to improve initial placeholder accuracy
watch(doc, async (d) => {
  if (d) {
    const page = await d.getPage(1);
    const vp = page.getViewport({ scale: 1 });
    pageAspectRatios[1] = vp.height / vp.width;
  }
}, { immediate: true });

watch(pdfEmbedWrapper, (newVal, oldVal) => {
  if (oldVal && resizeObserver) {
    resizeObserver.unobserve(oldVal);
    return;
  }
  if (newVal && resizeObserver) {
    resizeObserver.observe(newVal);
    containerWidth.value = newVal.getBoundingClientRect().width;
  }
});

const pageHeight = computed(() => {
  const ratio = pageAspectRatios[1] || DEFAULT_PAGE_ASPECT_RATIO;
  return containerWidth.value ? containerWidth.value * ratio : 1000; // fallback height
});

const pagePlaceholderStyles = computed(() => {
  return pageNums.value.map(() => ({
    minHeight: `${pageHeight.value}px`,
    position: "relative",
  }));
});

const resetPageIntersectionObserver = () => {
  pageIntersectionObserver?.disconnect();
  pageIntersectionObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        const index = pageRefs.value.indexOf(entry.target);
        if (index === -1) return;
        const pageNum = pageNums.value[index];
        if (!pageVisibility.value[pageNum]) {
          pageVisibility.value[pageNum] = true;

          // (Optional) refine aspect ratio once page is requested
          doc.value?.getPage(pageNum)
            .then((p) => {
              if (p) {
                const vp = p.getViewport({ scale: 1 });
                pageAspectRatios[pageNum] = vp.height / vp.width;
              }
            })
            .catch((e) => {
              console.warn("Failed to fetch page for ratio", pageNum, e);
            });
        }
      }
    });
  }, {
    root: null,
    rootMargin: `${PRELOAD_PAGES * pageHeight.value}px 0px`, // pre-load ahead
    threshold: 0.05, // require some actual area visible
  });
  pageRefs.value.forEach((element) => {
    pageIntersectionObserver!.observe(element);
  });
};

watch(pageNums, (newPageNums) => {
  pageVisibility.value = { [newPageNums[0]]: true };
  nextTick(resetPageIntersectionObserver);
});

onBeforeUnmount(() => {
  resizeObserver?.disconnect();
  pageIntersectionObserver?.disconnect();
});

const scoredSegments = computed(() =>
  highlights.filter(h => typeof h.score === "number" && h.score_received_at) as (Highlight & {
    score: number;
    score_received_at: number;
  })[]
);

/** Heuristic time remaining based on the time taken to get already scored segments */
const etaMs = computed(() => {
  if (scoredSegments.value.length <= 1) return 0;
  const timeTaken = scoredSegments.value[scoredSegments.value.length - 1].score_received_at - scoredSegments.value[0].score_received_at;
  return (timeTaken / (scoredSegments.value.length - 1)) * (segmentCount.value - scoredSegments.value.length);
});

const socket = useWebSocket("ws://localhost:8000/ws/progress/", {
  immediate: false,
  onConnected: () => {
    console.log("socket connected");
  },
  onDisconnected: onDisconnected,
  onMessage: (ws, e) => {
    const data = JSON.parse(e.data) as SocketMessage;
    // console.log("socket received:", data);
    switch (data.type) {
    case "batch":
    {
      const batch = data.content as Segment[];
      for (const segment of batch) {
        const foundSegment = highlights.find((seg) => seg.text === segment.text);
        if (foundSegment) {
          foundSegment.score = segment.score;
          foundSegment.score_received_at = Date.now();
        }
        highlightSegment(segment);
      }
      break;
    }
    case "segment":
    {
      const foundSegment = highlights.find((seg) => seg.text === (data.content as Segment).text);
      if (foundSegment) {
        foundSegment.score = (data.content as Segment).score;
        foundSegment.score_received_at = Date.now();
      }
      highlightSegment(data.content as Segment);
      break;
    }
    case "info":
      console.log("WS[INFO]:", data.content);
      break;
    case "error":
      console.error("WS[ERROR]:", data.content);
      break;
    case "success":
      console.log("WS[SUCCESS]:", data.content);
      ws.close();
      useNotifier().success("Document processed successfully");
      break;
    }
  },
});

function fullReset() {
  socket.close();
  onDisconnected();
  isCancelled.value = false;
  pdfUrl.value = null;
  pdfRenderKey.value++;
  selectedHighlights.clear();
  searchInput.value = "";
}

const handleFileUpload = async (event: any) => {
  const file = event.target?.files?.[0];
  if (!file) {
    return;
  }
  event.target.value = "";
  fullReset();

  isLoading.value = true;
  pdfUrl.value = URL.createObjectURL(file);
  const formData = new FormData();
  formData.append("pdf", file, file.name);
  formData.append("model", modelSelect.value);
  call("/api/process/pdf", {
    method: "POST",
    body: formData as any
  }).then((data) => {
    segmentCount.value = data.segment_count;
    // collect initial aligned segments with polygons if provided
    if (Array.isArray(data.segments)) {
      highlights.splice(0, highlights.length, ...data.segments);
    }
    socket.open();
    socket.send(JSON.stringify({ ws_key: data.ws_key }));
  }).catch((error) => {
    console.error("Failed to process PDF:", error);
    isLoading.value = false;
    useNotifier().error("Failed to process document. Please try again.");
  });
};

function onDisconnected() {
  console.log("socket disconnected");
  isLoading.value = false;
}

function cancelSegmentLoading() {
  if (!window.confirm("Are you sure you want to cancel the segment scoring?")) {
    return;
  }
  isCancelled.value = true;
  socket.close();
  onDisconnected();
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

const IMAGE_HEIGHT = 512;

function getHighlightImageStyle(highlight: Highlight) {
  if (!pdfEmbedWrapper.value) return {};
  const pdfEmbedBounding = pdfEmbedWrapper.value.getBoundingClientRect();
  const highlightBounding = getHighlightBoundingNormalized(highlight);
  if (!highlightBounding) return {};
  // convert to real coordinates using page aspect ratio
  const ratio = pageAspectRatios[1] || DEFAULT_PAGE_ASPECT_RATIO;
  const width = pdfEmbedBounding.width;
  const height = width * ratio;
  let top = pdfEmbedBounding.top + highlightBounding.y * height + window.scrollY;

  if (highlightBounding.height < IMAGE_HEIGHT) {
    // align center of image to center of highlight
    top = top - IMAGE_HEIGHT / 2 + highlightBounding.height / 2;
  }
  console.log("getHighlightImageStyle", { top, highlightBounding, pdfEmbedBounding });
  return {
    top: `${top}px`,
    position: "relative"
  };
}

function highlightSegment(segment: Segment) {
  for (const highlight of highlights) {
    if (highlight.text === segment.text) {
      if ((highlight?.score ?? 0) >= SCORE_THRESHOLD && !selectedHighlights.has(highlight.id)) {
        selectedHighlights.add(highlight.id);
      }
    }
  }
}

function toggleHighlightSelection(index: number) {
  const highlight = highlights[index];
  assertIsDefined(highlight);
  if (selectedHighlights.has(highlight.id)) {
    selectedHighlights.delete(highlight.id);
  } else {
    selectedHighlights.add(highlight.id);
  }
}

async function onSearchPrevNext(index: number) {
  const highlight = currentSearchSet.value[index];
  console.log("onSearchPrevNext called with index:", index);
  console.log("highlight:", highlight);
  if (highlight) {
    searchInput.value = highlight.text || "";
    const highlightsIndex = currentSearchSet.value === highlights ? index : highlights.findIndex(h => h.text === highlight.text);
    console.log("highlightsIndex:", highlightsIndex);
    // Make page of the first polygon visible
    const firstPolyPage = highlight.polygons ? Number(Object.keys(highlight.polygons)[0]) : null;
    if (firstPolyPage && !pageVisibility.value[firstPolyPage + 1]) {
      pageVisibility.value[firstPolyPage + 1] = true;
      // Wait 1 second for the page to render
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    // Scroll to the highlight
    highlightLayer.value?.highlightRefs[highlightsIndex]?.scrollIntoView({
      block: "center"
    });
  }
}

const currentSearchSet = computed(() => {
  if (currentSearchSetIndex.value === SearchSet.AllHighlights) {
    return highlights || [];
  } else if (currentSearchSetIndex.value === SearchSet.SelectedHighlights) {
    return highlights.filter(h => selectedHighlights.has(h.id)) || [];
  }
  return [];
});

watch([highlights, selectedHighlights], () => {
  if (currentSearchSet.value.length === 1) {
    onSearchPrevNext(0);
  }
});

watch(currentSearchIndex, (newIndex) => {
  console.log("newIndex:", newIndex);
});

async function changeSearchSet() {
  await nextTick();
  const index = currentSearchSet.value.findIndex(h => h.text === searchInput.value);
  if (index !== -1) {
    currentSearchIndex.value = index;
  } else {
    currentSearchIndex.value = 0;
  }
  onSearchPrevNext(currentSearchIndex.value);
}

/** Round to full minutes if > 90 sec, else round to full seconds */
function formatEta(etaMs: number) {
  if (etaMs > 90000) {
    return `${Math.round(etaMs / 60 / 1000)} min`;
  }
  return `${Math.round(etaMs / 1000)} sec`;
}

nuxtApp.$debugPanel.track("containerWidth", containerWidth);
nuxtApp.$debugPanel.track("pageVisibility", pageVisibility);
</script>
