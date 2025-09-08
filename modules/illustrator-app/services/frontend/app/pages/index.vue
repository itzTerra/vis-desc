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
              <option value="deberta_mnli" disabled>
                DeBERTa-MNLI
              </option>
            </select>
          </div>
        </div>
        <EvalProgress
          :is-loading="isLoading"
          :is-cancelled="isCancelled"
          :highlights="highlights"
          @cancel="cancelSegmentLoading"
        />
      </div>
      <HighlightNav
        ref="highlightNav"
        :highlights="highlights"
        :selected-highlights="selectedHighlights"
        @update="goToHighlight"
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
                  style="position: absolute; inset:0;"
                />
              </div>
              <HighlightLayer
                v-if="highlights.length"
                ref="highlightLayer"
                :highlights="highlights"
                :selected-highlights="selectedHighlights"
                :page-refs="pageRefs"
                :page-visibility="pageVisibility"
                @select="toggleHighlightSelection"
                @gen-image="genImage"
              />
            </div>
          </ClientOnly>
        </div>
      </div>
      <ImageLayer
        ref="imageLayer"
        :highlights="highlights"
        :page-aspect-ratios="pageAspectRatios"
        :pdf-embed-wrapper="pdfEmbedWrapper"
        :page-refs="pageRefs"
      />
    </div>
    <Hero v-else @file-selected="handleFileUpload" />
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
// Holds start timestamp (ms) when loading begins; null when idle
const isLoading = ref<number | null>(null);
const isCancelled = ref(false);
const highlights = reactive<Highlight[]>([]);
const selectedHighlights = reactive<Set<number>>(new Set());

const highlightLayer = ref<InstanceType<typeof import("~/components/HighlightLayer.vue").default> | null>(null);
const highlightNav = ref<InstanceType<typeof import("~/components/HighlightNav.vue").default> | null>(null);
const imageLayer = ref<InstanceType<typeof import("~/components/ImageLayer.vue").default> | null>(null);
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

const handleFileUpload = async (event: any) => {
  const file = event.target?.files?.[0];
  if (!file) {
    return;
  }
  event.target.value = "";
  fullReset();

  isLoading.value = Date.now();
  pdfUrl.value = URL.createObjectURL(file);
  const formData = new FormData();
  formData.append("pdf", file, file.name);
  formData.append("model", modelSelect.value);
  call("/api/process/pdf", {
    method: "POST",
    body: formData as any
  }).then((data) => {
    // collect initial aligned segments with polygons if provided
    if (Array.isArray(data.segments)) {
      highlights.splice(0, highlights.length, ...data.segments);
    }
    socket.open();
    socket.send(JSON.stringify({ ws_key: data.ws_key }));
  }).catch((error) => {
    console.error("Failed to process PDF:", error);
    isLoading.value = null;
    useNotifier().error("Failed to process document. Please try again.");
  });
};

const socket = useWebSocket("ws://localhost:8000/ws/progress/", {
  immediate: false,
  onConnected: () => {
    console.log("WS: connected");
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
        scoreSegment(segment);
      }
      break;
    }
    case "segment":
    {
      scoreSegment(data.content as Segment);
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

function onDisconnected() {
  console.log("WS: disconnected");
  isLoading.value = null;
}

function cancelSegmentLoading() {
  if (!window.confirm("Are you sure you want to cancel the segment scoring?")) {
    return;
  }
  isCancelled.value = true;
  socket.close();
  onDisconnected();
}

function scoreSegment(segment: Segment) {
  const segmentHighlight = highlights.find((seg) => seg.text === segment.text);
  if (!segmentHighlight) {
    return;
  }
  segmentHighlight.score = segment.score;
  if ((segmentHighlight?.score ?? 0) >= SCORE_THRESHOLD && !selectedHighlights.has(segmentHighlight.id)) {
    // Autoselect high-score highlights
    selectedHighlights.add(segmentHighlight.id);
    if (selectedHighlights.size === 1) {
      // If this is the first selected highlight, navigate to it
      goToHighlight(segmentHighlight);
    }
  }
}

async function goToHighlight(highlight: Highlight) {
  // Make page of the first polygon visible
  const firstPolyPage = highlight.polygons ? Number(Object.keys(highlight.polygons)[0]) : null;
  if (firstPolyPage && !pageVisibility.value[firstPolyPage + 1]) {
    pageVisibility.value[firstPolyPage + 1] = true;
    await nextTick();
  }
  // Scroll to the highlight
  await scrollIntoView(() => `[data-segment-id="${highlight.id}"]`);
  // Spawn a marker animation to the left
  highlightLayer.value?.spawnMarker(highlight.id);
}


function toggleHighlightSelection(id: number) {
  if (selectedHighlights.has(id)) {
    selectedHighlights.delete(id);
  } else {
    selectedHighlights.add(id);
  }
}

async function genImage(highlightId: number) {
  const realHighlight = highlights.find(h => h.id === highlightId);
  if (!realHighlight) return;

  realHighlight.imageLoading = true;
  const res = await call("/api/gen-image-bytes", {
    method: "POST",
    body: { text: realHighlight.text }
  });
  const blob = new Blob([res as any], { type: "image/png" });
  const url = URL.createObjectURL(blob);
  realHighlight.imageUrl = url;
  realHighlight.imageLoading = false;
  imageLayer.value?.bringImageToFront(realHighlight);
  return { blob, url };
}

function fullReset() {
  isCancelled.value = false;
  socket.close();
  onDisconnected();
  highlightNav.value?.reset();
  imageLayer.value?.reset();
  selectedHighlights.clear();
  highlights.length = 0;
  pdfUrl.value = null;
  pdfRenderKey.value++;
}

nuxtApp.$debugPanel.track("containerWidth", containerWidth);
nuxtApp.$debugPanel.track("pageVisibility", pageVisibility);

onBeforeUnmount(() => {
  resizeObserver?.disconnect();
  pageIntersectionObserver?.disconnect();
});
</script>
