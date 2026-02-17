<template>
  <div>
    <div
      class="top-bar w-full px-4 py-2 bg-base-200 flex flex-col lg:flex-row gap-3 justify-between lg:items-center sticky top-0 z-50 lg:h-14.5"
    >
      <div class="flex items-center gap-3">
        <input
          type="file"
          accept="application/pdf"
          class="file-input file-input-primary"
          data-help-target="file"
          @change="handleFileUpload"
        >
        <ModelSelect v-model="selectedModel" data-help-target="model" @request-model-download="(scorerId) => checkScorer(SCORERS.find(s => s.id === scorerId)!)" />
      </div>
      <div class="flex items-center gap-3">
        <EvalProgress
          v-model="isLoading"
          data-help-target="progress"
          :is-cancelled="isCancelled"
          :highlights="highlights"
          :current-stage="currentStage"
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
          :is-active="(autoIllustration.isActive?.value) ?? (((autoIllustration.progress?.value?.enhancedCount ?? 0) > 0) || ((autoIllustration.progress?.value?.generatedCount ?? 0) > 0))"
          data-help-target="auto-illustration"
        />
        <button
          v-if="pdfUrl"
          class="btn btn-primary btn-outline"
          title="Export as HTML"
          data-help-target="export"
          :disabled="isLoading !== null || highlights.length === 0"
          @click="handleExportConfirm"
        >
          <Icon name="lucide:download" size="18" />
          Export
        </button>
      </div>
      <div class="flex items-center gap-3 ms-auto">
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
      @editor-state-change="onEditorStateChange"
      @editor-enhanced="onEditorEnhanced"
    />
    <Hero v-else @file-selected="handleFileUpload" />
    <div class="bottom-bar">
      <div v-if="!seenHelpOnce" class="tooltip tooltip-open tooltip-info tooltip-left">
        <div class="tooltip-content pointer-events-auto p-0">
          <div class="relative flex items-end px-2 pb-2 h-10">
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
      <CacheManager
        ref="cacheManagerRef"
        v-model:is-expanded="cacheManagerExpanded"
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
import HelpOverlay, { type Step } from "~/components/HelpOverlay.vue";
import type { CacheManager } from "#components";
import { SCORERS, MODEL_GROUPS, type Scorer } from "~/utils/models";
import type { ModelGroup } from "~/types/cache";

type SocketMessage = { content: unknown, type: "segment" | "batch" | "info" | "error" | "success" };

const { $api: call, $config, $debugPanel, callHook, hook } = useNuxtApp();
const runtimeConfig = useRuntimeConfig();

const pdfUrl = ref<string | null>(null);
const pdfFile = ref<File | null>(null);
const pdfRenderedQueue = [] as (() => void)[];
const firstScorer = SCORERS.find(scorer => !scorer.disabled);
const selectedModel = ref<string>(firstScorer ? firstScorer.id : (SCORERS[0]?.id ?? "random"));
const selectedScorer = computed(() =>
  SCORERS.find(s => s.id === selectedModel.value)
);
const currentStage = ref<string | undefined>(undefined);
const isLoading = ref<number | null>(null);
const isCancelled = ref(false);
const highlights = reactive<Highlight[]>([]);
const selectedHighlights = reactive<Set<number>>(new Set());
const highlightNav = ref<InstanceType<typeof import("~/components/HighlightNav.vue").default> | null>(null);

const showHelp = ref(false);
const seenHelpOnce = useLocalStorage("seenHelpOnce", false);
const highlightsHelpBackup: Highlight[] = [];
const groupInConfirmation = ref<ModelGroup | null>(null);
const cacheManagerRef = ref<InstanceType<typeof CacheManager> | null>(null);
const cacheManagerExpanded = ref(false);
const pdfViewer = ref<InstanceType<typeof import("~/components/PdfViewer.vue").default> | null>(null);
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
  currentStage.value = "Processing PDF...";
  pdfFile.value = file;

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

  pdfUrl.value = URL.createObjectURL(file);

  const needsSocket = scorer.id === "random";

  const formData = new FormData();
  formData.append("pdf", file, file.name);
  formData.append("model", selectedModel.value);

  try {
    const data = await call(needsSocket ? "/api/process/pdf" : "/api/process/seg/pdf", {
      method: "POST",
      body: formData as any
    });

    if (data?.segments && Array.isArray(data.segments)) {
      highlights.splice(0, highlights.length, ...data.segments);
    }

    if (needsSocket) {
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
        needsSocket ? socket : undefined
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
const autoIllustration = useAutoIllustration({ highlights, selectedHighlights, pageAspectRatio: pageAspectRatioRef });

// when composable requests editors to be opened, forward to PdfViewer's exposed method
watch(autoIllustration.pendingOpenIds, async () => {
  const list = autoIllustration.consumePendingOpenIds();
  if (!Array.isArray(list) || list.length === 0) return;
  pdfViewer.value?.openEditors?.(list);
  // wait for editors to mount
  await nextTick();
  const enhance = (autoIllustration.enableEnhance?.value ?? true);
  const generate = (autoIllustration.enableGenerate?.value ?? true);
  const layer = (pdfViewer.value as any)?.imageLayer as any | undefined;
  if (layer && typeof layer.runAutoActions === "function") {
    // run auto actions on the newly opened editors and update counts based on results
    try {
      const res = await layer.runAutoActions(list, { enhance, generate });
      if (res && Array.isArray(res.enhancedIds) && res.enhancedIds.length) {
        autoIllustration.markEnhanced?.(res.enhancedIds || []);
      }
      if (res && Array.isArray(res.generatedIds) && res.generatedIds.length) {
        autoIllustration.markGenerated?.(res.generatedIds || []);
      }
    } catch (e) {
      console.error("auto actions failed", e);
    }
  }
});

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

    const imageLayer = pdfViewer.value.$refs.imageLayer;
    if (!imageLayer || typeof (imageLayer as any).getExportImages !== "function") {
      useNotifier().error("Image layer is not ready");
      return;
    }

    const imageBlobs = (imageLayer as any).getExportImages();

    await confirmExport(
      pdfFile.value,
      highlights,
      imageBlobs,
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
  initApp();

  hook("custom:downloadNeeded", (groupId: string) => {
    const group = MODEL_GROUPS.find(g => g.id === groupId);
    if (group) {
      groupInConfirmation.value = group;
    }
  });

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

function onEditorStateChange(nextState: any) {
  if (!nextState || typeof nextState.highlightId !== "number") return;
  if (nextState.hasImage) {
    autoIllustration.markGenerated?.([nextState.highlightId]);
  }
}

function onEditorEnhanced(highlightId: number) {
  if (typeof highlightId !== "number") return;
  autoIllustration.markEnhanced?.([highlightId]);
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
  z-index: 90;
}
</style>
