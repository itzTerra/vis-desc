<template>
  <div>
    <div
      class="top-bar w-full px-4 py-2 bg-base-200 flex flex-col sm:flex-row gap-3 justify-between sm:items-center sticky top-0 z-50 h-[114px] sm:h-[58px]"
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
          <ModelSelect v-model="modelSelect" data-help-target="model" @request-model-download="handleModelDownloadRequest" />
          <button
            v-if="pdfUrl"
            class="btn btn-sm btn-outline"
            title="Export as HTML"
            data-help-target="export"
            :disabled="isLoading !== null || highlights.length === 0"
            @click="showExportDialog = true"
          >
            <Icon name="lucide:download" size="18" />
            Export
          </button>
        </div>
        <EvalProgress
          data-help-target="progress"
          :is-loading="isLoading"
          :is-cancelled="isCancelled"
          :highlights="highlights"
          @cancel="cancelSegmentLoading"
        />
      </div>
      <div class="flex items-center gap-3 self-end">
        <HighlightNav
          ref="highlightNav"
          data-help-target="nav"
          :highlights="highlights"
          :selected-highlights="selectedHighlights"
        />
        <ThemeToggle data-help-target="theme" />
      </div>
    </div>
    <PdfViewer
      v-if="pdfUrl || showHelp"
      ref="pdfViewer"
      v-model:highlights="highlights"
      v-model:selected-highlights="selectedHighlights"
      :pdf-url="pdfUrl || helpPdfUrl"
      @pdf:rendered="onPdfRendered"
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
      <button class="btn btn-circle btn-sm btn-neutral" title="Debug" @click="$debugPanel.toggle()">
        <Icon name="uil:bug" size="20px" />
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
    <ModelManager
      ref="modelManagerRef"
      v-model:pending-model="pendingModel"
      v-model:is-expanded="modelManagerExpanded"
      @model-ready="onModelReady"
    />
    <ModelDownloadDialog
      v-if="pendingModel && pendingModel.transformersConfig"
      :is-open="showModelDownloadDialog"
      :model-info="pendingModel"
      @confirm="confirmModelDownload"
      @cancel="cancelModelDownload"
    />
    <ExportDialog
      :is-open="showExportDialog"
      :image-count="exportedImageCount"
      @confirm="handleExportConfirm"
      @cancel="showExportDialog = false"
    />
    <BackToTop />
    <Alert />
  </div>
</template>

<script setup lang="ts">
import type { Highlight, Segment } from "~/types/common";
import HelpOverlay, { type Step } from "~/components/HelpOverlay.vue";
import type { ModelManager } from "#components";

type SocketMessage = { content: unknown, type: "segment" | "batch" | "info" | "error" | "success" };

const SCORE_THRESHOLD = 0.95;

const { $api: call, callHook, $config, $debugPanel } = useNuxtApp();
const runtimeConfig = useRuntimeConfig();

const pdfUrl = ref<string | null>(null);
const pdfFile = ref<File | null>(null);
const pdfRenderedQueue = [] as (() => void)[];
const modelSelect = ref<ModelId>(MODELS.filter(model => !model.disabled)[0].id);
const pendingModel = ref<TModelInfo | null>(null);
// Holds start timestamp (ms) when loading begins; null when idle
const isLoading = ref<number | null>(null);
const isCancelled = ref(false);
const highlights = reactive<Highlight[]>([]);
const selectedHighlights = reactive<Set<number>>(new Set());
const highlightNav = ref<InstanceType<typeof import("~/components/HighlightNav.vue").default> | null>(null);

const showHelp = ref(false);
const seenHelpOnce = useLocalStorage("seenHelpOnce", false);
const highlightsHelpBackup: Highlight[] = [];
const showModelDownloadDialog = ref(false);
const modelManagerRef = ref<InstanceType<typeof ModelManager> | null>(null);
const modelManagerExpanded = ref(false);
const pdfViewer = ref<InstanceType<typeof import("~/components/PdfViewer.vue").default> | null>(null);
const showExportDialog = ref(false);
const { confirmExport } = useExport();

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

const helpPdfUrl = `${runtimeConfig.app.baseURL}sample.pdf`;
// const helpImageUrl = `${runtimeConfig.app.baseURL}sample.jpg`;
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
  // TODO update
  // > Show a static image
  // {
  //   selector: "[data-help-target=\"images\"]",
  //   title: "Image Viewer",
  //   message: "View images generated from segments. Move generated images by holding left-click and dragging. Delete images by clicking on the x icon.",
  //   position: "left",
  //   onEnter: () => {
  //     highlights[1].imageLoading = true;
  //     setTimeout(() => {
  //       highlights[1].imageUrl = helpImageUrl;
  //       highlights[1].imageLoading = false;
  //     }, 1000);
  //   },
  // },
  {
    selector: "[data-help-target=\"export\"]",
    title: "Export as HTML",
    message: "Export your PDF and generated images as a single standalone HTML file. The exported file includes your PDF with all pages lazy-loaded for performance, and a gallery of generated images that can be viewed in a modal. The HTML export works offline without external dependencies.",
    position: "bottom"
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
  pdfFile.value = file;
  pdfUrl.value = URL.createObjectURL(file);
  const formData = new FormData();
  formData.append("pdf", file, file.name);
  formData.append("model", modelSelect.value);
  const modelInfo = getModelById(modelSelect.value);

  if (!modelInfo || !modelInfo.apiUrl || modelInfo.disabled) {
    useNotifier().error("Selected model is not available.");
    isLoading.value = null;
    return;
  }

  const canContinue = handleModelDownloadRequest(modelSelect.value);
  if (!canContinue) {
    isLoading.value = null;
    return;
  }

  call(modelInfo.apiUrl, {
    method: "POST",
    body: formData as any
  }).then((data: any) => {
    // collect initial aligned segments with polygons if provided
    if (data?.segments && Array.isArray(data.segments)) {
      highlights.splice(0, highlights.length, ...data.segments);
    }

    modelInfo.onSuccess?.(data, modelInfo, socket, scoreSegment);
    // socket.open();
    // socket.send(JSON.stringify({ ws_key: data.ws_key }));
  }).catch((error) => {
    isLoading.value = null;
    console.error("Failed to process PDF:", error);
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
  isLoading.value = null;

  const modelInfo = getModelById(modelSelect.value);
  modelInfo?.onCancel?.(modelInfo);
  // socket.close();
  // onDisconnected();
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
  if (pdfUrl.value) {
    URL.revokeObjectURL(pdfUrl.value);
  }
  pdfUrl.value = null;
  pdfFile.value = null;
}

const exportedImageCount = computed(() => {
  if (!pdfViewer.value) return 0;
  const imageLayer = pdfViewer.value.$refs.imageLayer;
  if (!imageLayer || typeof imageLayer.getExportImages !== "function") return 0;
  const images = imageLayer.getExportImages();
  return Object.keys(images).length;
});

async function handleExportConfirm(filename: string) {
  try {
    if (!pdfFile.value) {
      useNotifier().error("PDF file is required for export");
      return;
    }

    if (!pdfViewer.value) {
      useNotifier().error("PDF viewer is not ready");
      return;
    }

    const imageLayer = pdfViewer.value.$refs.imageLayer;
    if (!imageLayer || typeof imageLayer.getExportImages !== "function") {
      useNotifier().error("Image layer is not ready");
      return;
    }

    const imageBlobs = imageLayer.getExportImages();

    await confirmExport(
      pdfFile.value,
      highlights,
      imageBlobs,
      filename
    );

    showExportDialog.value = false;
    useNotifier().success("PDF exported successfully");
  } catch (error) {
    console.error("Export failed:", error);
    useNotifier().error("Failed to export PDF");
  }
}

function handleModelDownloadRequest(modelId: ModelId): boolean {
  const modelInfo = getModelById(modelId);
  if (!modelInfo || !("transformersConfig" in modelInfo) || !modelInfo.transformersConfig) {
    return true;
  }

  const { getModelCacheStatus } = useModelLoader();
  const status = getModelCacheStatus(modelInfo as TModelInfo);

  if (status === "cached") {
    return true;
  }

  if (status === "downloading") {
    useNotifier().info(`Model "${modelInfo.label}" is already downloading.`);
    modelManagerExpanded.value = true;
    return false;
  }

  pendingModel.value = modelInfo as TModelInfo;
  showModelDownloadDialog.value = true;
  return false;
}

function onModelReady(modelId: ModelId) {
  useNotifier().success(`${getModelById(modelId).label} is now ready to use`);
}

function confirmModelDownload() {
  if (!pendingModel.value) return;

  modelManagerRef.value?.queueDownload(pendingModel.value);
  showModelDownloadDialog.value = false;
  pendingModel.value = null;
}

function cancelModelDownload() {
  showModelDownloadDialog.value = false;
  pendingModel.value = null;
}

function onPdfRendered() {
  while (pdfRenderedQueue.length > 0) {
    const fn = pdfRenderedQueue.shift();
    fn?.();
  }
}

onMounted(() => {
  $debugPanel.addAction(
    "📋 Print Highlights",
    () => {
      console.log("Current highlights:", highlights);
      console.table(highlights.map(h => ({
        id: h.id,
        score: h.score,
        selected: selectedHighlights.has(h.id),
        text: h.text.slice(0, 50) + (h.text.length > 50 ? "..." : "")
      })));
    },
    "Print current highlights to console"
  );

  $debugPanel.addAction(
    "📄 Load Example PDF",
    async () => {
      try {
        const response = await fetch(`${runtimeConfig.app.baseURL}example.pdf`);
        if (!response.ok) {
          throw new Error(`Failed to fetch example PDF: ${response.statusText}`);
        }
        const blob = await response.blob();
        const file = new File([blob], "example.pdf", { type: "application/pdf" });
        pdfFile.value = file;
        pdfUrl.value = URL.createObjectURL(file);

        // Load example segments JSON after PDF is rendered
        pdfRenderedQueue.push(async () => {
          const segments = (await import("~/assets/data/example-segments.json").then(m => m.default || m)) as Highlight[];
          highlights.splice(0, highlights.length, ...segments);

          useNotifier().success("Example PDF and segments loaded successfully");
          console.log("Loaded example with", highlights.length, "segments");
        });
      } catch (error) {
        console.error("Failed to load example data:", error);
        useNotifier().error("Failed to load example data");
      }
    },
    "Load example PDF and segments"
  );
});
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
  z-index: 100;
}
</style>
