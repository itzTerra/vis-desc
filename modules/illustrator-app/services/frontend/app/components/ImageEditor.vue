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
          <div class="flex-grow w-full cursor-grab active:cursor-grabbing select-none" @pointerdown="(e) => emit('pointerDown', e)">
            &nbsp;
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
              @keydown.ctrl.enter="handleGenerate"
            />
          </div>
          <div class="flex justify-between gap-2 my-1">
            <button
              class="btn btn-sm btn-secondary flex-1"
              :disabled="enhanceLoading || generateLoading || !currentPrompt"
              :aria-busy="enhanceLoading"
              @click="handleEnhance"
            >
              <span v-if="enhanceLoading" class="loading loading-spinner loading-sm" />
              <Icon v-else name="lucide:wand" size="18" />
              Enhance
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
              :disabled="enhanceLoading || generateLoading || !currentPrompt"
              :aria-busy="generateLoading"
              title="Generate image"
              @click="handleGenerate"
            >
              <span v-if="generateLoading" class="loading loading-spinner loading-sm" />
              <Icon v-else name="lucide:book-image" size="18" />
              Generate
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
import { useEditorHistory } from "~/composables/useEditorHistory";

const props = defineProps<{
  highlightId: number;
  initialText: string;
}>();

const emit = defineEmits<{
  delete: [];
  bringToFront: [];
  pointerDown: [event: PointerEvent];
}>();

const { $api } = useNuxtApp();

const isOpen = ref(true);
const isExpanded = ref(false);

const enhanceLoading = ref(false);
const generateLoading = ref(false);

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

const imageUrl = computed(() => currentHistoryItem.value?.imageUrl || "");
const hasImage = computed(() => !!imageUrl.value);

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

async function handleEnhance() {
  const prompt = currentPrompt.value.trim();
  if (!prompt) return;

  enhanceLoading.value = true;

  try {
    const res = await $api("/api/enhance", {
      method: "POST",
      body: { text: prompt },
    });

    if (!res || !res.text) {
      useNotifier().error("Enhance failed");
      return;
    }
    // const res = await promptEnhancer.enhance(prompt, PromptBridgeMode.Compress);

    currentPrompt.value = res.text;
    addToHistory(currentPrompt.value);

    useNotifier().success("Prompt enhanced");
  } catch (error) {
    useNotifier().error("Enhance request failed");
    console.error(error);
  } finally {
    enhanceLoading.value = false;
  }
}

async function handleGenerate() {
  const prompt = currentPrompt.value.trim();
  if (!prompt) return;

  generateLoading.value = true;

  try {
    const res = await $api("/api/gen-image-bytes", {
      method: "POST",
      body: { text: prompt },
    });

    if (!res) {
      useNotifier().error("Image generation failed");
      return;
    }

    const blob = new Blob([res as any], { type: "image/png" });
    const url = URL.createObjectURL(blob);

    addToHistory(prompt, url, blob);

    useNotifier().success("Image generated");
  } catch (error) {
    useNotifier().error("Image generation failed");
    console.error(error);
  } finally {
    generateLoading.value = false;
  }
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

defineExpose({
  getExportImage
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
