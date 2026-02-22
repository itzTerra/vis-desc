<template>
  <div>
    <div
      class="top-bar w-full px-3 py-2 bg-base-200 flex flex-col lg:flex-row gap-3 justify-between lg:items-center sticky top-0 z-50 lg:h-14.5"
    >
      <div class="flex items-center gap-2 min-w-44">
        <input
          type="file"
          accept="application/pdf,.txt,text/plain"
          class="file-input file-input-primary min-w-12"
          data-help-target="file"
          :disabled="!isAppReady"
          @change="handleFileUpload"
        >
        <ModelSelect v-model="selectedModel" data-help-target="model" @request-model-download="(scorerId) => checkScorer(SCORERS.find(s => s.id === scorerId)!)" />
      </div>
      <div class="flex items-center gap-2">
        <EvalProgress
          v-model="isLoading"
          data-help-target="progress"
          :is-cancelled="isCancelled"
          :highlights="highlights"
          :current-stage="currentStage"
          :scorer="selectedScorer"
          @cancel="cancelSegmentLoading"
        />
        <AutoIllustration
          v-model:enabled="autoIllustration.enabled.value"
          v-model:min-gap-pages="autoIllustration.minGapPages.value"
          v-model:max-gap-pages="autoIllustration.maxGapPages.value"
          v-model:min-score="autoIllustration.minScore.value"
          v-model:enable-enhance="autoIllustration.enableEnhance.value"
          v-model:enable-generate="autoIllustration.enableGenerate.value"
          :run-pass="autoIllustration.runPass"
          :clear-auto-selections="autoIllustration.clearAutoSelections"
          :progress="autoIllustration.progress.value"
          data-help-target="auto-illustration"
        />
        <button
          v-if="pdfUrl || showHelp"
          class="btn btn-primary btn-soft"
          title="Export as HTML"
          data-help-target="export"
          :disabled="(!pdfFile || !pdfViewer) && !showHelp"
          @click="handleExportConfirm"
        >
          <Icon name="lucide:download" size="18" />
          Export
        </button>
      </div>
      <div class="flex items-center gap-2 ms-auto min-w-58">
        <HighlightNav
          ref="highlightNav"
          data-help-target="nav"
          :highlights="highlights"
          :selected-highlights="selectedHighlights"
          class="min-w-48"
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
      :auto-enabled="autoIllustration.enabled.value"
      @pdf:rendered="onPdfRendered"
    />
    <Hero v-else :disabled="!isAppReady" @file-selected="handleFileUpload" />
    <div class="bottom-bar flex-col items-end md:flex-row md:items-center">
      <div v-if="!seenHelpOnce" class="tooltip tooltip-open tooltip-info tooltip-left">
        <div class="tooltip-content pointer-events-auto p-0">
          <div class="relative flex items-end px-2 py-2">
            Start a 1-minute guided tour now!
            <button class="ms-2 cursor-pointer text-xs" @click="seenHelpOnce = true">
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
      <CacheManager
        ref="cacheManagerRef"
        v-model:is-expanded="cacheManagerExpanded"
        data-help-target="cache"
      />
      <button class="btn btn-circle btn-sm btn-warning" title="Debug" @click="$debugPanel.toggle()">
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
          <small class="truncate text-base-content/50 max-w-15">#{{ $config.public.commitHash }}</small>
        </div>
      </NuxtLink>
    </div>
    <HelpOverlay v-if="showHelp" :help-steps="helpSteps" @cancel="toggleHelp(false)" />
    <DownloadConfirmDialog
      v-if="groupInConfirmation"
      :model-info="groupInConfirmation"
      @confirm="confirmModelDownload"
      @cancel="cancelModelDownload"
    />
    <BackToTop />
    <Alert />
  </div>
</template>

<script setup lang="ts">
import type { Highlight, Segment } from "~/types/common";
import type { CacheManager } from "#components";
import { SCORERS, MODEL_GROUPS, type Scorer } from "~/utils/models";
import type { ModelGroup } from "~/types/cache";

type ImageLayer = InstanceType<typeof import("~/components/ImageLayer.vue").default>;

const { $api: call, $config, $debugPanel, $appReadyState, callHook, hook } = useNuxtApp();
const runtimeConfig = useRuntimeConfig();

const isAppReady = computed(() => $appReadyState?.apiReady && $appReadyState?.scorerWorkerReady && !$appReadyState?.apiError);

const pdfUrl = ref<string | null>(null);
const pdfFile = ref<File | null>(null);
const pdfRenderedQueue = [] as (() => void)[];
const { selectedModel } = useAppStorage();
const selectedScorer = computed(() =>
  SCORERS.find(s => s.id === selectedModel.value)
);
const currentStage = ref<string | undefined>(undefined);
const isLoading = ref<number | null>(null);
const isCancelled = ref(false);
const highlights = reactive<Highlight[]>([]);
const selectedHighlights = reactive<Set<number>>(new Set());
const highlightNav = ref<InstanceType<typeof import("~/components/HighlightNav.vue").default> | null>(null);

const groupInConfirmation = ref<ModelGroup | null>(null);
const cacheManagerRef = ref<InstanceType<typeof CacheManager> | null>(null);
const cacheManagerExpanded = ref(false);
const pdfViewer = ref<InstanceType<typeof import("~/components/PdfViewer.vue").default> | null>(null);
const { confirmExport } = useExport();

const { showHelp, seenHelpOnce, helpSteps, toggleHelp, helpPdfUrl } = useHelpSteps({
  isLoading,
  currentStage,
  highlights,
  pdfViewer,
  cacheManagerRef,
});

const handleFileUpload = async (event: any) => {
  const file = event.target?.files?.[0];
  if (!file) {
    return;
  }
  event.target.value = "";
  fullReset();

  const isTxt = file.name.endsWith(".txt") || file.type === "text/plain";

  isLoading.value = Date.now();
  currentStage.value = isTxt ? "Processing TXT..." : "Processing PDF...";

  if (!isTxt) {
    pdfFile.value = file;
  }

  const scorer = selectedScorer.value;
  if (!scorer) {
    useNotifier().error("No scorer selected.");
    isLoading.value = null;
    currentStage.value = undefined;
    return;
  }

  if (!checkScorer(scorer)) {
    isLoading.value = null;
    currentStage.value = undefined;
    return;
  }

  if (!isTxt) {
    pdfUrl.value = URL.createObjectURL(file);
  }

  const formData = new FormData();
  formData.append(isTxt ? "txt" : "pdf", file, file.name);
  formData.append("model", selectedModel.value);

  const endpoint = isTxt
    ? (scorer.socketBased ? "/api/process/txt" : "/api/segment/txt")
    : (scorer.socketBased ? "/api/process/pdf" : "/api/segment/pdf");

  try {
    const data = await call(endpoint, {
      method: "POST",
      body: formData as any
    });

    if (isTxt && (data as any)?.pdf_base64) {
      const pdfBytes = Uint8Array.from(atob((data as any).pdf_base64), c => c.charCodeAt(0));
      pdfFile.value = new File([pdfBytes], file.name.replace(/\.txt$/i, ".pdf"), { type: "application/pdf" });
      pdfUrl.value = URL.createObjectURL(pdfFile.value);
    }

    if (data?.segments && Array.isArray(data.segments)) {
      highlights.splice(0, highlights.length, ...data.segments);
    }

    if (scorer.socketBased) {
      socket.open();
      // Wait for socket to open
      await new Promise<void>((resolve, reject) => {
        const timeout = setTimeout(() => {
          stopWatcher();
          reject(new Error("Socket connection timeout"));
        }, 10000);

        const stopWatcher = watch(
          () => socket.status.value,
          (status) => {
            if (status === "OPEN") {
              clearTimeout(timeout);
              stopWatcher();
              resolve();
            } else if (status === "CLOSED") {
              clearTimeout(timeout);
              stopWatcher();
              reject(new Error("Socket connection closed"));
            }
          },
          { immediate: true }
        );
      });
      socket.send(JSON.stringify({ ws_key: (data as any).ws_key }));

      currentStage.value = "Scoring...";
      isLoading.value = Date.now();
    } else {
      await scorer.score(
        data,
        (progress) => {
          currentStage.value = progress.stage;
          isLoading.value = progress.startedAt;
          for (const segment of progress.results || []) {
            scoreSegment(segment);
          }
        },
        scorer.socketBased ? socket : undefined
      );
      isLoading.value = null;
      currentStage.value = undefined;
    }
  } catch (error) {
    isLoading.value = null;
    currentStage.value = undefined;
    console.error("Failed to process PDF:", error);
    useNotifier().error("Failed to process document. Please try again.");
  }
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
      isLoading.value = null;
      currentStage.value = undefined;
      break;
    }
  },
});

function onDisconnected() {
  console.log("WS: disconnected");
  isLoading.value = null;
  currentStage.value = undefined;
}

function cancelSegmentLoading() {
  if (!window.confirm("Are you sure you want to cancel the segment scoring?")) {
    return;
  }
  isCancelled.value = true;
  isLoading.value = null;
  currentStage.value = undefined;

  const scorer = selectedScorer.value;
  if (scorer) {
    scorer.dispose();
  }

  socket.close();
}

function scoreSegment(segment: Segment) {
  const segmentHighlight = highlights.find((seg) => (typeof (segment as any).id === "number" && seg.id === (segment as any).id) || seg.text === segment.text);
  if (!segmentHighlight) return;
  segmentHighlight.score = segment.score;
}

// expose PdfViewer sizing to the composable for accurate cross-page distance math
const pageAspectRatioRef = computed(() => {
  const pv = pdfViewer.value as any;
  if (!pv) return 1;
  const par = pv.pageAspectRatio;
  if (par && typeof par === "object" && "value" in par) return par.value as number;
  return typeof par === "number" ? par : 1;
});

// instantiate auto-illustration composable
// when composable requests editors to be opened, forward to PdfViewer's exposed method
const onNewPendingOpenIds = async (ids: number[], actionOpts: { enhance: boolean; generate: boolean }) => {
  if (!Array.isArray(ids) || ids.length === 0) return;
  pdfViewer.value?.openImageEditors?.(ids);
  // wait for editors to mount
  await nextTick();
  const layer = (pdfViewer.value as any)?.imageLayer as ImageLayer | undefined;
  if (layer && typeof layer.runAutoActions === "function") {
    try {
      layer.runAutoActions(ids, actionOpts);
    } catch (e) {
      console.error("auto actions failed", e);
    }
  }
};
const onCatchupActions = (ids: number[], actionOpts: { enhance: boolean; generate: boolean }) => {
  if (!Array.isArray(ids) || ids.length === 0) return;
  const layer = (pdfViewer.value as any)?.imageLayer as ImageLayer | undefined;
  if (layer && typeof layer.runAutoActions === "function") {
    try {
      layer.runAutoActions(ids, actionOpts);
    } catch (e) {
      console.error("catchup actions failed", e);
    }
  }
};
const autoIllustration = useAutoIllustration({ highlights, selectedHighlights, pageAspectRatio: pageAspectRatioRef, onNewPendingOpenIds, onCatchupActions });

function fullReset() {
  isCancelled.value = false;
  currentStage.value = undefined;
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

async function handleExportConfirm() {
  try {
    if (!pdfFile.value) {
      useNotifier().error("PDF file is required for export");
      return;
    }

    if (!pdfViewer.value) {
      useNotifier().error("PDF viewer is not ready");
      return;
    }

    const imageLayer = pdfViewer.value.$refs.imageLayer as ImageLayer | undefined;
    if (!imageLayer || typeof (imageLayer as any).getExportImages !== "function") {
      useNotifier().error("Image layer is not ready");
      return;
    }

    const imageUrls = imageLayer.getExportImages();

    await confirmExport(
      pdfFile.value,
      highlights,
      imageUrls,
      `${pdfFile.value.name.replace(/\.pdf$/i, "")}-export`
    );
  } catch (error) {
    console.error("Export failed:", error);
    useNotifier().error("Failed to export PDF");
  }
}

function checkScorer(scorer: Scorer) {
  if (scorer.disabled) {
    useNotifier().error("Selected scorer is not available.");
    return false;
  }
  const modelGroup = MODEL_GROUPS.find(g => g.id === scorer.id);
  if (!modelGroup) {
    return true;
  }

  const groupStatus = cacheManagerRef.value?.getGroupDownloadStatus(modelGroup);
  if (groupStatus === "downloading") {
    cacheManagerExpanded.value = true;
    return false;
  }
  if (groupStatus !== "cached") {
    callHook("custom:downloadNeeded", modelGroup.id);
    return false;
  }
  return true;
}

function onPdfRendered() {
  while (pdfRenderedQueue.length > 0) {
    const fn = pdfRenderedQueue.shift();
    fn?.();
  }
}

function cancelModelDownload() {
  groupInConfirmation.value = null;
}

function confirmModelDownload() {
  cacheManagerRef.value?.queueDownload(groupInConfirmation.value!);
  groupInConfirmation.value = null;
}

onMounted(() => {
  hook("custom:downloadNeeded", (groupId: string) => {
    const group = MODEL_GROUPS.find(g => g.id === groupId);
    if (group) {
      groupInConfirmation.value = group;
    }
  });

  $debugPanel.addAction(
    "🔬 Text Scorer",
    () => {
      navigateTo("/scorer");
    },
    "Open the text scorer page"
  );

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
  right: 0;
  padding: 0.5rem 0.5rem;
  display: flex;
  justify-content: end;
  gap: 0.5rem;
  z-index: 130;
}
</style>
