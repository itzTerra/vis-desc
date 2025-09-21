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
      />
    </div>
    <PdfViewer
      v-if="pdfUrl"
      v-model:highlights="highlights"
      v-model:selected-highlights="selectedHighlights"
      :pdf-url="pdfUrl"
    />
    <Hero v-else @file-selected="handleFileUpload" />
    <BottomBar />
    <BackToTop />
    <Alert />
  </div>
</template>

<script setup lang="ts">
import type { Highlight } from "~/types/common";

type Segment = { text: string, score: number };
type SocketMessage = { content: unknown, type: "segment" | "batch" | "info" | "error" | "success" };

const SCORE_THRESHOLD = 0.95;

const { $api: call, callHook } = useNuxtApp();
const runtimeConfig = useRuntimeConfig();

const pdfUrl = ref<string | null>(null);
const modelSelect = ref<string>("random");
// Holds start timestamp (ms) when loading begins; null when idle
const isLoading = ref<number | null>(null);
const isCancelled = ref(false);
const highlights = reactive<Highlight[]>([]);
const selectedHighlights = reactive<Set<number>>(new Set());
const highlightNav = ref<InstanceType<typeof import("~/components/HighlightNav.vue").default> | null>(null);

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

const socket = useWebSocket(`${runtimeConfig.public.wsBaseUrl}/ws/progress/`, {
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
      callHook("custom:goToHighlight", segmentHighlight);
    }
  }
}

function fullReset() {
  isCancelled.value = false;
  socket.close();
  onDisconnected();
  highlightNav.value?.reset();
  selectedHighlights.clear();
  highlights.length = 0;
  pdfUrl.value = null;
}
</script>
