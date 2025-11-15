<template>
  <div>
    <div
      class="top-bar w-full px-4 py-2 bg-base-200 flex flex-col sm:flex-row gap-3 justify-between sm:items-center sticky top-0 z-10 h-[114px] sm:h-[58px]"
    >
      <div class="flex items-center flex-grow-1 w-100">
        <div class="flex items-end gap-4 flex-grow-1">
          <input
            type="file"
            accept="application/pdf"
            class="file-input"
            data-help-target="file"
            @change="handleFileUpload"
          >
          <ModelSelect v-model="modelSelect" data-help-target="model" />
        </div>
        <EvalProgress
          data-help-target="progress"
          :is-loading="isLoading"
          :is-cancelled="isCancelled"
          :highlights="highlights"
          @cancel="cancelSegmentLoading"
        />
      </div>
      <HighlightNav
        ref="highlightNav"
        data-help-target="nav"
        :highlights="highlights"
        :selected-highlights="selectedHighlights"
      />
      <ThemeToggle data-help-target="theme" />
    </div>
    <PdfViewer
      v-if="pdfUrl || showHelp"
      v-model:highlights="highlights"
      v-model:selected-highlights="selectedHighlights"
      :pdf-url="pdfUrl || helpPdfUrl"
    />
    <Hero v-else @file-selected="handleFileUpload" />
    <div class="bottom-bar">
      <div v-if="!seenHelpOnce" class="tooltip tooltip-open tooltip-info tooltip-left">
        <div class="tooltip-content pointer-events-auto p-0">
          <div class="relative flex items-end px-2 pb-2 h-[40px]">
            Start a 1-minute guided tour now!
            <button class="absolute right-1 top-0 cursor-pointer text-xs" @click="seenHelpOnce = true">
              ✕
            </button>
          </div>
        </div>
        <button class="btn btn-circle btn-sm btn-info text-lg animate-bounce" title="Help" @click="toggleHelp(true)">
          ?
        </button>
      </div>
      <button v-else class="btn btn-circle btn-sm btn-info text-lg" title="Help" @click="toggleHelp(true)">
        ?
      </button>
      <NuxtLink
        :href="$config.public.githubUrl"
        target="_blank"
        class="flex items-end space-x-1 bg-base-200 rounded-2xl shadow hover:opacity-70"
      >
        <Icon name="uil:github" size="34px" />
        <div class="pe-2 text-xs/3 flex flex-col grow-0 font-light font-mono">
          <span class="text-base-content/70">v{{ $config.public.appVersion }}</span>
          <small class="truncate text-base-content/50 max-w-[60px]">#{{ $config.public.commitHash }}</small>
        </div>
      </NuxtLink>
    </div>
    <HelpOverlay v-if="showHelp" :help-steps="helpSteps" @cancel="toggleHelp(false)" />
    <BackToTop />
    <Alert />
  </div>
</template>

<script setup lang="ts">
import type { Highlight } from "~/types/common";
import HelpOverlay, { type Step } from "~/components/HelpOverlay.vue";

type Segment = { text: string, score: number };
type SocketMessage = { content: unknown, type: "segment" | "batch" | "info" | "error" | "success" };

const SCORE_THRESHOLD = 0.95;

const { $api: call, callHook, $config } = useNuxtApp();
const runtimeConfig = useRuntimeConfig();

const pdfUrl = ref<string | null>(null);
const modelSelect = ref<ModelValue>(MODELS.filter(model => !model.disabled)[0].value);
// Holds start timestamp (ms) when loading begins; null when idle
const isLoading = ref<number | null>(null);
const isCancelled = ref(false);
const highlights = reactive<Highlight[]>([]);
const selectedHighlights = reactive<Set<number>>(new Set());
const highlightNav = ref<InstanceType<typeof import("~/components/HighlightNav.vue").default> | null>(null);

const showHelp = ref(false);
const seenHelpOnce = useLocalStorage("seenHelpOnce", false);
const highlightsHelpBackup: Highlight[] = [];

function toggleHelp(toValue?: boolean) {
  showHelp.value = toValue !== undefined ? toValue : !showHelp.value;
  if (showHelp.value) {
    highlightsHelpBackup.splice(0, highlightsHelpBackup.length, ...highlights);
    highlights.splice(0, highlights.length,
      { id: 1, text: "This is a scored segment.", score: 0.98, score_received_at: Date.now(), polygons: [ [ [0.1, 0.20], [0.3, 0.20], [0.3, 0.25], [0.1, 0.25] ] ] },
      { id: 2, text: "This is an unscored segment.", score: undefined, score_received_at: Date.now(), polygons: [ [ [0.1, 0.30], [0.35, 0.30], [0.35, 0.35], [0.1, 0.35] ] ] }
    );
    if (!seenHelpOnce.value) {
      seenHelpOnce.value = true;
    }
  } else {
    highlights.splice(0, highlights.length, ...highlightsHelpBackup);
  }
}

const helpPdfUrl = `${runtimeConfig.app.baseURL}/sample.pdf`;
const helpImageUrl = `${runtimeConfig.app.baseURL}/sample.jpg`;
const helpSteps: Step[] = [
  // > Open model menu
  {
    selector: "[data-help-target=\"model\"]",
    title: "Model Selection",
    message: "Select the AI model to use for scoring text segments.",
    position: "bottom",
    onEnter: () => {
      const modelSelectEl = document.querySelector("[data-help-target=\"model\"] [role=\"button\"]") as HTMLElement;
      modelSelectEl.focus();
    },
    onLeave: () => {
      (document.activeElement as HTMLElement)?.blur();
    }
  },
  {
    selector: "[data-help-target=\"file\"]",
    title: "File Upload",
    message: "Upload your PDF document here by clicking or dragging.",
    position: "bottom"
  },
  // > Simulate processing
  {
    selector: "[data-help-target=\"progress\"]",
    title: "Scoring Progress",
    message: "Monitor the progress of text segment scoring here.",
    position: "bottom",
    onEnter: () => {
      isLoading.value = Date.now();
    },
    onLeave: () => {
      isLoading.value = null;
    }
  },
  // > Open the nav panel
  {
    selector: "[data-help-target=\"nav\"]",
    title: "Segment Navigation",
    message: "Navigate through scored segments using this panel. Filter selected segments only or set a limit to a minimum and/or a maximum score. Sort segments by score or index. Click on a segment to jump to its location in the document.",
    position: "bottom",
    onEnter: () => {
      const navEl = document.querySelector("[data-help-target=\"nav\"] summary") as HTMLElement;
      navEl.click();
    },
    onLeave: () => {
      const navEl = document.querySelector("[data-help-target=\"nav\"] summary") as HTMLElement;
      (navEl.parentElement as HTMLDetailsElement).open = false;
    }
  },
  // > Show a single static pdf page with static highlights
  {
    selector: "[data-help-target=\"viewer\"]",
    title: "Document Viewer",
    message: "View your PDF with scored segments. Select promising segments to filter them later using left-click. Generate an image from a segment by hovering over it and clicking the 'Generate image' button in the opened tooltip.",
    position: "right",
    onEnter: () => {
      // Hover first highlight to show tooltip
      const firstHighlightEl = document.querySelector(".pdf-highlight") as HTMLElement;
      firstHighlightEl?.dispatchEvent(new MouseEvent("mouseenter", { bubbles: true }));
    }
  },
  // > Show a static image
  {
    selector: "[data-help-target=\"images\"]",
    title: "Image Viewer",
    message: "View images generated from segments. Move generated images by holding left-click and dragging. Delete images by clicking on the x icon.",
    position: "left",
    onEnter: () => {
      highlights[1].imageLoading = true;
      setTimeout(() => {
        highlights[1].imageUrl = helpImageUrl;
        highlights[1].imageLoading = false;
      }, 1000);
    },
  },
  {
    selector: "[data-help-target=\"theme\"]",
    title: "Theme Switch",
    message: "Switch between light and dark mode.",
    position: "bottom"
  }
];

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
<style scoped>
.bottom-bar {
  position: fixed;
  bottom: 0;
  width: 100%;
  padding: 0.5rem 0.5rem;
  display: flex;
  align-items: center;
  justify-content: end;
  gap: 0.5rem;
}
</style>
