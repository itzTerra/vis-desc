<template>
  <div class="image-editor relative bg-base-100 border-l border-base-300 flex flex-col" style="width: 100%;">
    <!-- Collapse Button (protrudes LEFT into PDF viewer area) -->
    <button
      class="btn btn-sm btn-circle btn-primary absolute top-2 z-15"
      style="left: -26px;"
      :title="isOpen ? 'Collapse editor' : 'Expand editor'"
      :aria-label="isOpen ? 'Collapse editor' : 'Expand editor'"
      :aria-expanded="isOpen"
      @click="isOpen = !isOpen"
    >
      <Icon :name="isOpen ? 'lucide:chevron-right' : 'lucide:chevron-left'" size="18" />
    </button>

    <!-- Editor Content -->
    <Transition name="slide">
      <div v-show="isOpen" class="flex-1 overflow-y-auto shadow border-2 rounded border-base-300">
        <!-- Top Bar -->
        <div class="flex items-center bg-base-200 overflow-hidden">
          <div class="grow w-full min-w-0 cursor-grab active:cursor-grabbing select-none flex items-center gap-1.5 px-1.5 overflow-hidden" @pointerdown="(e) => emit('pointerDown', e)">
            <span v-if="props.score !== undefined" class="badge badge-sm badge-secondary badge-soft shrink-0 font-mono">{{ (props.score * 100).toFixed(0) }}</span>
            <span class="text-xs text-base-content/40 truncate">{{ props.initialText }}</span>
          </div>
          <div class="join join-horizontal ms-auto">
            <button
              class="btn btn-xs btn-neutral join-item"
              :aria-label="isExpanded ? 'Collapse editor' : 'Expand editor'"
              :title="isExpanded ? 'Collapse editor' : 'Expand editor'"
              @click="isExpanded = !isExpanded"
            >
              <Icon :name="isExpanded ? 'lucide:chevrons-down-up' : 'lucide:chevrons-up-down'" size="14" />
            </button>
            <button class="btn btn-xs btn-neutral join-item" title="Bring to front" @click="emit('bringToFront')">
              <Icon name="lucide:bring-to-front" size="14" />
            </button>
            <button class="btn btn-xs btn-error join-item" title="Delete image" @click="handleDelete">
              <Icon name="lucide:x" size="14" />
            </button>
          </div>
        </div>
        <!-- Prompt Area (visible when expanded) -->
        <div v-if="isExpanded">
          <div class="form-control">
            <label for="prompt-textarea" class="label hidden">
              <span class="label-text">Image Prompt</span>
            </label>
            <textarea
              id="prompt-textarea"
              v-model="currentPrompt"
              class="textarea textarea-bordered w-full h-24 resize-none"
              placeholder="Enter or paste image prompt..."
              :disabled="enhanceLoading || generateLoading"
            />
          </div>
          <div class="flex justify-between gap-2 my-1">
            <button
              class="btn btn-sm btn-secondary flex-1"
              :disabled="enhanceLoading || (enhanceState !== 'queued' && currentPrompt === '')"
              :aria-busy="enhanceLoading"
              :title="enhanceState === 'queued' ? 'Cancel enhance' : 'Enhance prompt'"
              @click="() => handleEnhance()"
            >
              <span v-if="enhanceLoading" class="loading loading-spinner loading-sm" />
              <Icon v-else-if="enhanceState === 'queued'" name="lucide:x" size="18" />
              <Icon v-else name="lucide:wand" size="18" />
              {{ enhanceState === 'queued' ? 'Cancel' : 'Enhance' }}
            </button>
            <!-- History Display -->
            <div v-if="history.length > 1" class="flex gap-2 items-center">
              <button
                class="btn btn-xs"
                :disabled="isAtStart"
                aria-label="Previous history entry"
                title="Previous"
                @click="navigatePrevious"
              >
                <Icon name="lucide:chevron-left" aria-hidden="true" size="14" />
              </button>
              <div class="text-xs text-base-content/70 flex-1 text-center" role="status">
                {{ historyIndex + 1 }} / {{ history.length }}
              </div>
              <button
                class="btn btn-xs"
                :disabled="isAtEnd"
                aria-label="Next history entry"
                title="Next"
                @click="navigateNext"
              >
                <Icon name="lucide:chevron-right" aria-hidden="true" size="14" />
              </button>
            </div>
            <button
              class="btn btn-sm btn-primary flex-1"
              :disabled="generateLoading || (generateState !== 'queued' && currentPrompt === '')"
              :aria-busy="generateLoading"
              :title="generateState === 'queued' ? 'Cancel generation' : 'Generate image'"
              @click="() => handleGenerate()"
            >
              <span v-if="generateLoading" class="loading loading-spinner loading-sm" />
              <Icon v-else-if="generateState === 'queued'" name="lucide:x" size="18" />
              <Icon v-else name="lucide:book-image" size="18" />
              {{ generateState === 'queued' ? 'Cancel' : 'Generate' }}
            </button>
          </div>
        </div>

        <!-- Generated Image Display (INSIDE editor) -->
        <div v-if="hasImage || generateLoading" class="rounded bg-base-200/30 backdrop-blur-sm">
          <img v-if="imageUrl" :src="imageUrl" class="w-full h-auto rounded" alt="AI generated image from prompt">
          <div v-if="generateLoading" class="flex justify-center items-center h-32" role="status">
            <span class="loading loading-dots" />
            <span class="sr-only">Generating image, please wait...</span>
          </div>
        </div>
      </div>
    </Transition>
  </div>
</template>

<script setup lang="ts">
import type { ActionState, EditorImageState } from "~/types/common";

import { useEditorHistory } from "~/composables/useEditorHistory";

const props = defineProps<{
  highlightId: number;
  initialText: string;
  score?: number;
  enhanceState?: ActionState;
  generateState?: ActionState;
}>();

const emit = defineEmits<{
  delete: [];
  bringToFront: [];
  pointerDown: [event: PointerEvent];
  "state-change": [state: EditorImageState];
  enhance: [highlightId: number, prompt: string, auto: boolean];
  generate: [highlightId: number, prompt: string, auto: boolean];
}>();

const isOpen = ref(true);
const isExpanded = ref(false);

const enhanceLoading = computed(() => props.enhanceState === "processing");
const generateLoading = computed(() => props.generateState === "processing");

const currentPrompt = ref(props.initialText);

const {
  history,
  historyIndex,
  currentHistoryItem,
  isAtStart,
  isAtEnd,
  addToHistory,
  navigatePrevious,
  navigateNext,
} = useEditorHistory();

addToHistory(props.initialText);

const imageUrl = computed(() => currentHistoryItem.value?.imageUrl ?? "");
const hasImage = computed(() => imageUrl.value !== "");

watchImmediate(isOpen, (open) => {
  if (open) {
    isExpanded.value = !hasImage.value;
  }
});

watch(currentHistoryItem, (item) => {
  if (item) {
    currentPrompt.value = item.text;
  }

});

watch(() => currentHistoryItem.value?.imageUrl, () => {
  emit("state-change", {
    highlightId: props.highlightId,
    imageUrl: imageUrl.value,
    hasImage: hasImage.value,
  });
});

function handleEnhance(auto: boolean = false) {
  if (enhanceLoading.value) return;
  const prompt = currentPrompt.value.trim();
  if (prompt === "" && props.enhanceState !== "queued") return;
  emit("enhance", props.highlightId, prompt, auto);
}

function handleGenerate(auto: boolean = false) {
  if (generateLoading.value) return;
  const prompt = currentPrompt.value.trim();
  if (prompt === "" && props.generateState !== "queued") return;
  emit("generate", props.highlightId, prompt, auto);
}

function handleDelete() {
  emit("delete");
}

function getExportImage() {
  const item = currentHistoryItem.value;
  if (!item?.imageBlob) return null;
  return {
    highlightId: props.highlightId,
    imageBlob: item.imageBlob
  };
}

// Helpers to apply batched API results from parent
function applyEnhanceResult(text: string) {
  currentPrompt.value = text;
  addToHistory(currentPrompt.value);
}

function applyGenerateResult(image_b64: string) {
  const url = `data:image/png;base64,${image_b64}`;
  addToHistory(currentPrompt.value.trim(), url, undefined);
}

function getCurrentPrompt() {
  return currentPrompt.value;
}

defineExpose({
  getExportImage,
  getHighlightId: () => props.highlightId,
  triggerEnhance: handleEnhance,
  triggerGenerate: handleGenerate,
  applyEnhanceResult,
  applyGenerateResult,
  getCurrentPrompt,
});

onBeforeUnmount(() => {
  if (imageUrl.value?.startsWith("blob:")) {
    URL.revokeObjectURL(imageUrl.value);
  }
});
</script>

<style scoped>
.image-editor {
  min-height: 0;
}

.image-editor .overflow-y-auto {
  scrollbar-width: thin;
  scrollbar-color: hsl(var(--bc) / 0.2) transparent;
}

.image-editor .overflow-y-auto::-webkit-scrollbar {
  width: 6px;
}

.image-editor .overflow-y-auto::-webkit-scrollbar-thumb {
  background-color: hsl(var(--bc) / 0.2);
  border-radius: 3px;
}

.image-editor .overflow-y-auto::-webkit-scrollbar-thumb:hover {
  background-color: hsl(var(--bc) / 0.3);
}
</style>
