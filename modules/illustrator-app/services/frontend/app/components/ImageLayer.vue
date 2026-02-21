<template>
  <div
    ref="imagesContainer"
    class="w-lg select-none"
  >
    <div class="relative">
      <ImageEditor
        v-for="editorId in openEditorIds"
        :key="editorId"
        ref="editorRefs"
        :highlight-id="editorId"
        :initial-text="getHighlightText(editorId)"
        :score="highlightMap[editorId]?.score"
        :enhance-state="getEnhanceState(editorId)"
        :generate-state="getGenerateState(editorId)"
        :style="getEditorPositionStyle(editorId)"
        class="absolute pointer-events-auto"
        @delete="closeImageEditor(editorId)"
        @bring-to-front="bringImageToFront(editorId)"
        @pointer-down="onEditorPointerDown($event, editorId)"
        @state-change="handleEditorStateChange"
        @enhance="onEnhance"
        @generate="onGenerate"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import type { ImageEditor } from "#components";
import type { ActionState, EditorImageState, Highlight } from "~/types/common";

const props = defineProps<{
  pdfEmbedWrapper: HTMLElement | null;
  highlights: Highlight[];
  openEditorIds: Set<number>;
  pageAspectRatio: number;
  pageHeight: number;
  currentPage: number;
  pageRefs?: Element[];
  autoEnabled?: boolean;
}>();

const emit = defineEmits<{
  "close-editor": [highlightId: number];
  "editor-state-change": [state: EditorImageState];
}>();


const { $api, $config, hook } = useNuxtApp();

const EDITOR_NO_IMAGE_HEIGHT = 156; // controls height when no image
const EDITOR_MAX_POSSIBLE_HEIGHT = 512 + EDITOR_NO_IMAGE_HEIGHT; // Image + controls

const imagesContainer = ref<HTMLElement | null>(null);

interface EnhanceQueueRecord {
  highlightId: number;
  prompt: string;
  auto?: boolean;
}
interface GenerateQueueRecord {
  highlightId: number;
  prompt: string;
  auto?: boolean;
}

const enhanceQueue = reactive<EnhanceQueueRecord[]>([]);
const generateQueue = reactive<GenerateQueueRecord[]>([]);

const enhanceStates = reactive<Record<number, ActionState>>({});
const generateStates = reactive<Record<number, ActionState>>({});

function getEnhanceState(id: number): ActionState {
  return enhanceStates[id] ?? "idle";
}

function getGenerateState(id: number): ActionState {
  return generateStates[id] ?? "idle";
}

// --- Throughput/Batch/Interval Calculation ---
const ENHANCE_THROUGHPUT = $config.public.enhanceThroughput;
const GENERATE_THROUGHPUT = $config.public.generateThroughput;
const BATCH_LIMIT = 8;
function calcBatchAndInterval(throughput: number) {
  // throughput: things/sec
  // Want to maximize batch size up to BATCH_LIMIT, but not exceed throughput per second
  // Choose interval so that batchSize/interval = throughput/sec
  // If throughput > BATCH_LIMIT: batchSize = BATCH_LIMIT, interval = BATCH_LIMIT/throughput sec
  // If 1 <= throughput <= BATCH_LIMIT: batchSize = round(throughput), interval = 1 sec
  // If throughput < 1: batchSize = 1, interval = 1/throughput sec
  if (throughput >= BATCH_LIMIT) {
    return { batchSize: BATCH_LIMIT, intervalMs: Math.round(1000 * (BATCH_LIMIT / throughput)) };
  } else if (throughput >= 1) {
    return { batchSize: Math.round(throughput), intervalMs: 1000 };
  } else {
    return { batchSize: 1, intervalMs: Math.round(1000 / throughput) };
  }
}

const enhanceConfig = calcBatchAndInterval(ENHANCE_THROUGHPUT);
const generateConfig = calcBatchAndInterval(GENERATE_THROUGHPUT);
// --- Event Handlers for Editor ---
function onEnhance(highlightId: number, prompt: string, auto: boolean = false) {
  const existingIdx = enhanceQueue.findIndex(r => r.highlightId === highlightId && r.prompt === prompt);
  if (existingIdx !== -1) {
    enhanceQueue.splice(existingIdx, 1);
    enhanceStates[highlightId] = "idle";
  } else {
    enhanceQueue.push({ highlightId, prompt, auto });
    enhanceStates[highlightId] = "queued";
  }
}

function onGenerate(highlightId: number, prompt: string, auto: boolean = false) {
  const existingIdx = generateQueue.findIndex(r => r.highlightId === highlightId && r.prompt === prompt);
  if (existingIdx !== -1) {
    generateQueue.splice(existingIdx, 1);
    generateStates[highlightId] = "idle";
  } else {
    generateQueue.push({ highlightId, prompt, auto });
    generateStates[highlightId] = "queued";
  }
}

const highlightMap = computed(() => {
  const map: Record<number, Highlight> = {};
  for (const h of props.highlights) {
    map[h.id] = h;
  }
  return map;
});

function getHighlightText(highlightId: number): string {
  const highlight = highlightMap.value[highlightId];
  return highlight?.text || "";
}

function getEditorTop(highlightId: number) {
  // Try user-overridden drag positions
  const existingTop = imagePositions[highlightId]?.top;
  if (existingTop !== null && existingTop !== undefined) return existingTop;

  if (!props.pdfEmbedWrapper) return 0;
  const pdfEmbedBounding = props.pdfEmbedWrapper.getBoundingClientRect();
  const containerBounding = imagesContainer.value?.getBoundingClientRect();
  const highlight = highlightMap.value[highlightId];
  const highlightBounding = getHighlightBoundingNormalized(highlight);
  if (!highlightBounding) return 0;

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
    globalTop = pageRect.top + highlightBounding.y * actualPageHeight + window.scrollY;
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

  if (highlightBounding.height < EDITOR_NO_IMAGE_HEIGHT) {
    // align center of image to center of highlight
    top = top - EDITOR_NO_IMAGE_HEIGHT / 2 + highlightBounding.height / 2;
  }

  return top;
}

function getEditorPositionStyle(highlightId: number) {
  const top = getEditorTop(highlightId);
  return {
    top: `${top}px`,
    left: "0px",
    width: "100%",
    position: "absolute",
    touchAction: "none",
    zIndex: imageZIndices[highlightId] || 1
  };
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

function ensureImagePosition(highlightId: number) {
  if (!imagePositions[highlightId]) {
    // Initialize from computed style
    const style = getEditorPositionStyle(highlightId) as any;
    const parsePx = (v: string | undefined) => v ? Number(v.toString().replace(/px$/, "")) : 0;
    imagePositions[highlightId] = {
      top: parsePx(style.top),
      left: parsePx(style.left)
    };
  }
  return imagePositions[highlightId];
}

function onEditorPointerDown(e: PointerEvent, highlightId: number) {
  if (!(e.target instanceof HTMLElement)) return;
  // Ignore drag start when clicking overlay controls
  if ((e.target as HTMLElement).closest(".image-controls")) return;
  // Only allow primary (usually left) mouse button / primary pointer to initiate drag
  // Avoid starting drag on right-click (button === 2) or middle-click (button === 1)
  if (e.button !== 0) return;
  draggingId.value = highlightId;
  draggingEl = e.currentTarget as HTMLElement;
  const pos = ensureImagePosition(highlightId);
  dragStartPage = { x: e.pageX, y: e.pageY };
  lastPointerClient = { x: e.clientX, y: e.clientY };
  dragStartPos = { ...pos };
  (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
  window.addEventListener("pointermove", onEditorPointerMove);
  window.addEventListener("pointerup", onEditorPointerUp, { once: true });
}

function onEditorPointerMove(e: PointerEvent) {
  if (draggingId.value == null || !dragStartPage || !dragStartPos) return;
  lastPointerClient = { x: e.clientX, y: e.clientY };
  const deltaX = e.pageX - dragStartPage.x;
  const deltaY = e.pageY - dragStartPage.y;
  updateDraggedPosition(deltaX, deltaY);
}

function onEditorPointerUp() {
  draggingId.value = null;
  draggingEl = null;
  dragStartPos = null;
  dragStartPage = null;
  window.removeEventListener("pointermove", onEditorPointerMove);
}

function updateDraggedPosition(deltaX: number, deltaY: number) {
  if (draggingId.value == null || !dragStartPos) return;
  const pos = imagePositions[draggingId.value];
  if (!pos) return;
  const container = imagesContainer.value;
  if (container) {
    const ch = container.clientHeight;
    // const cw = container.clientWidth;
    const eh = draggingEl?.offsetHeight || EDITOR_MAX_POSSIBLE_HEIGHT;
    // const ew = draggingEl?.offsetWidth || cw;
    pos.top = clamp(dragStartPos.top + deltaY, 0, Math.max(0, ch - eh));
    // pos.left = clamp(dragStartPos.left + deltaX, 0, Math.max(0, cw - ew));
  } else {
    pos.top = dragStartPos.top + deltaY;
    // pos.left = dragStartPos.left + deltaX;
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

  hook("custom:clearAutoImageEditors", () => {
    // Clear queues, reset states, and close all editors
    enhanceQueue.splice(0, enhanceQueue.length);
    generateQueue.splice(0, generateQueue.length);
    for (const id in enhanceStates) enhanceStates[Number(id)] = "idle";
    for (const id in generateStates) generateStates[Number(id)] = "idle";
    for (const id of props.openEditorIds) {
      emit("close-editor", id);
    }
  });
});

onBeforeUnmount(() => {
  window.removeEventListener("pointermove", onEditorPointerMove);
  window.removeEventListener("scroll", onScrollWhileDragging);
});

function bringImageToFront(highlightId: number) {
  zCounter += 1;
  imageZIndices[highlightId] = zCounter;
}

function closeImageEditor(highlightId: number) {
  emit("close-editor", highlightId);
}

function handleEditorStateChange(state: EditorImageState) {
  emit("editor-state-change", state);
}


// Global reactive map from highlightId to editor instance
const idToEditor = reactive(new Map<number, InstanceType<typeof ImageEditor>>());

const editorRefs = ref<Array<InstanceType<typeof ImageEditor> | null>>([]);

// Track enhanced texts for each highlightId
const enhancedTexts = new Map<number, string>();

async function processEnhanceQueue() {
  if (enhanceQueue.length === 0) return;
  const skipAuto = !props.autoEnabled;
  // Build batch without removing auto records when the toggle is off
  const batch: EnhanceQueueRecord[] = [];
  for (let i = 0; i < enhanceQueue.length && batch.length < enhanceConfig.batchSize; i++) {
    if (skipAuto && enhanceQueue[i].auto) continue;
    batch.push(enhanceQueue[i]);
  }
  for (const rec of batch) {
    const idx = enhanceQueue.indexOf(rec);
    if (idx !== -1) enhanceQueue.splice(idx, 1);
    enhanceStates[rec.highlightId] = "processing";
  }
  if (batch.length === 0) return;
  const ids = batch.map(r => r.highlightId);
  const texts = batch.map(r => r.prompt);
  if (texts.length === 0) return;
  const res = await $api("/api/enhance", { method: "POST", body: { texts } });
  const failedIds: number[] = [];
  for (let i = 0; i < (res?.length || 0); i++) {
    const r = res[i];
    const id = ids[i];
    const ed = idToEditor.get(id);
    if (r && r.ok && r.text !== undefined && r.text !== null && ed && typeof ed.applyEnhanceResult === "function") {
      ed?.applyEnhanceResult(r.text);
      // Track enhanced text for this highlightId
      enhancedTexts.set(id, r.text);
      if (!batch[i].auto) {
        useNotifier().success("Prompt enhanced");
      }
    } else {
      console.error("/api/enhance error", r?.error);
      failedIds.push(id);
      // If there is a corresponding record waiting in the generate queue, remove it since enhance failed and generate likely won't work
      const genIdx = generateQueue.findIndex(r => r.highlightId === id);
      if (genIdx !== -1) {
        generateQueue.splice(genIdx, 1);
        generateStates[id] = "idle";
      }
    }
    enhanceStates[id] = "idle";
    if (failedIds.length > 0) {
      useNotifier().error(`Enhance failed for ${failedIds.length}/${ids.length} item(s).`);
    }
  }
}

async function processGenerateQueue() {
  if (generateQueue.length === 0) return;
  const skipAuto = !props.autoEnabled;
  // Only process up to generateConfig.batchSize at a time, skipping auto records when disabled
  // and those with requireEnhanced=true if not yet enhanced
  const readyBatch: GenerateQueueRecord[] = [];
  for (const rec of generateQueue) {
    if (readyBatch.length >= generateConfig.batchSize) break;
    if (skipAuto && rec.auto) continue;
    if (rec.auto) {
      // Check if enhanced: look up enhancedTexts for this highlightId
      const ed = idToEditor.get(rec.highlightId);
      const currentPrompt = ed && typeof ed.getCurrentPrompt === "function" ? ed.getCurrentPrompt() : getHighlightText(rec.highlightId);
      const enhancedText = enhancedTexts.get(rec.highlightId);
      if (!enhancedText || currentPrompt !== enhancedText) continue;
    }
    readyBatch.push(rec);
  }
  if (readyBatch.length === 0) return;
  // Remove processed from queue and mark as processing
  for (const rec of readyBatch) {
    const idx = generateQueue.findIndex(r => r.highlightId === rec.highlightId);
    if (idx !== -1) generateQueue.splice(idx, 1);
    generateStates[rec.highlightId] = "processing";
  }
  const ids = readyBatch.map(r => r.highlightId);
  const texts = ids.map(id => {
    const ed = idToEditor.get(id);
    if (ed && typeof ed.getCurrentPrompt === "function") return ed.getCurrentPrompt();
    return getHighlightText(id);
  });
  if (texts.length === 0) return;
  const res = await $api("/api/gen-image-bytes", { method: "POST", body: { texts } });
  const failedIds: number[] = [];
  for (let i = 0; i < (res?.length || 0); i++) {
    const r = res[i];
    const id = ids[i];
    const ed = idToEditor.get(id);
    if (r && r.ok && r.image_b64 && ed && typeof ed.applyGenerateResult === "function") {
      ed.applyGenerateResult(r.image_b64);
      if (!readyBatch[i].auto) {
        useNotifier().success("Image generated");
      }
    } else {
      console.error("/api/gen-image-bytes error", r?.error);
      failedIds.push(id);
    }
    generateStates[id] = "idle";
    if (failedIds.length > 0) {
      useNotifier().error(`Image generation failed for ${failedIds.length}/${ids.length} item(s).`);
    }
  }
}

// Polling for queue processing
let enhanceInterval: ReturnType<typeof setInterval> | null = null;
let generateInterval: ReturnType<typeof setInterval> | null = null;

onMounted(() => {
  window.addEventListener("scroll", onScrollWhileDragging, { passive: true });
  enhanceInterval = setInterval(processEnhanceQueue, enhanceConfig.intervalMs);
  generateInterval = setInterval(processGenerateQueue, generateConfig.intervalMs);
});

onBeforeUnmount(() => {
  window.removeEventListener("pointermove", onEditorPointerMove);
  window.removeEventListener("scroll", onScrollWhileDragging);
  if (enhanceInterval) clearInterval(enhanceInterval);
  if (generateInterval) clearInterval(generateInterval);
});

function getExportImages(): Record<number, string> {
  const result: Record<number, string> = {};

  for (const editor of editorRefs.value) {
    if (!editor) continue;
    const exportData = editor.getExportImage();
    if (exportData) {
      result[exportData.highlightId] = exportData.imageUrl;
    }
  }
  return result;
}

defineExpose({
  getExportImages,
  runAutoActions: (ids: number[] = [], opts: { enhance?: boolean; generate?: boolean } = {}) => {
    if (!Array.isArray(ids) || ids.length === 0) return { enhancedIds: [], generatedIds: [] };
    const enhance = !!opts.enhance;
    const generate = !!opts.generate;
    // Add to queues instead of direct API calls
    if (enhance) {
      for (const id of ids) {
        // Trigger enhance in the editor
        const ed = idToEditor.get(id);
        if (!ed || typeof ed.triggerEnhance !== "function") {
          console.error("Editor not found in runAutoActions for enhance", id);
          continue;
        }
        ed!.triggerEnhance(true);
      }
    }
    if (generate) {
      for (const id of ids) {
        // Trigger generate in the editor, with requireEnhanced=true so it waits for enhance if needed
        const ed = idToEditor.get(id);
        if (!ed || typeof ed.triggerGenerate !== "function") {
          console.error("Editor not found in runAutoActions for generate", id);
          continue;
        }
        ed!.triggerGenerate(true);
      }
    }
  }
});
// Keep idToEditor map in sync with editorRefs
watch(
  () => editorRefs.value,
  (editors) => {
    for (const editor of editors) {
      if (!editor) continue;
      const id = editor.getHighlightId?.();
      if (id != null) {
        idToEditor.set(id, editor);
      }
    }
  },
  { immediate: true, deep: true }
);
</script>

