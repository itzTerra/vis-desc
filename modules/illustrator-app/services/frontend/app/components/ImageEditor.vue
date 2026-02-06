<template>
  <div class="image-editor relative bg-base-100 border-l border-base-300 flex flex-col" style="width: 100%;">
    <!-- Collapse Button (protrudes LEFT into PDF viewer area) -->
    <button
      class="btn btn-sm btn-circle absolute top-2 z-10"
      style="left: -20px;"
      :title="isExpanded ? 'Collapse editor' : 'Expand editor'"
      :aria-label="isExpanded ? 'Collapse editor' : 'Expand editor'"
      :aria-expanded="isExpanded"
      @click="isExpanded = !isExpanded"
    >
      <Icon :name="isExpanded ? 'lucide:chevron-right' : 'lucide:chevron-left'" />
    </button>

    <!-- Editor Content (visible when expanded) -->
    <Transition name="slide">
      <div v-show="isExpanded" class="flex-1 overflow-y-auto p-4 space-y-4">
        <!-- Top Bar -->
        <div class="flex justify-between items-center">
          <h2 class="text-sm font-semibold">
            Image Editor
          </h2>
        </div>

        <!-- Prompt Area (hidden if image exists) -->
        <template v-if="!hasImage">
          <div class="form-control">
            <label for="prompt-textarea" class="label">
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
          <div class="flex gap-2">
            <button
              class="btn btn-sm btn-primary flex-1"
              :disabled="enhanceLoading || generateLoading || !currentPrompt"
              :aria-busy="enhanceLoading"
              @click="handleEnhance"
            >
              <Icon v-if="enhanceLoading" name="lucide:loader" class="animate-spin" aria-hidden="true" />
              Enhance
            </button>
            <button
              class="btn btn-sm btn-secondary flex-1"
              :disabled="enhanceLoading || generateLoading || !currentPrompt"
              :aria-busy="generateLoading"
              title="Generate image (Ctrl+Enter)"
              @click="handleGenerate"
            >
              <Icon v-if="generateLoading" name="lucide:loader" class="animate-spin" aria-hidden="true" />
              Generate
            </button>
          </div>
        </template>

        <!-- Generated Image Display (INSIDE editor) -->
        <div v-if="hasImage || generateLoading" class="border rounded p-2">
          <img v-if="imageUrl" :src="imageUrl" class="w-full h-auto" alt="AI generated image from prompt">
          <div v-if="generateLoading" class="flex justify-center items-center h-32" role="status">
            <Icon name="lucide:loader" class="animate-spin" size="32" aria-hidden="true" />
            <span class="sr-only">Generating image, please wait...</span>
          </div>
        </div>

        <!-- History Display -->
        <div v-if="history.length > 0" class="border-t pt-4">
          <h3 class="text-sm font-semibold mb-2">
            History
          </h3>
          <div class="flex gap-2 items-center mb-2">
            <button
              class="btn btn-xs"
              :disabled="isAtStart"
              aria-label="Previous history entry"
              title="Previous"
              @click="navigatePrevious"
            >
              <Icon name="lucide:chevron-left" aria-hidden="true" />
            </button>
            <div class="text-xs text-base-content/70 flex-1 text-center" role="status">
              {{ currentHistoryItem ? `${historyIndex + 1} of ${history.length}` : 'Editing' }}
            </div>
            <button
              class="btn btn-xs"
              :disabled="isAtEnd"
              aria-label="Next history entry"
              title="Next"
              @click="navigateNext"
            >
              <Icon name="lucide:chevron-right" aria-hidden="true" />
            </button>
          </div>
          <div v-if="currentHistoryItem" class="bg-base-200 p-2 rounded text-xs mb-2 max-h-20 overflow-auto">
            {{ currentHistoryItem.text }}
          </div>
        </div>

        <!-- Delete Button -->
        <button
          class="btn btn-sm btn-error w-full"
          @click="handleDelete"
        >
          <Icon name="lucide:trash-2" aria-hidden="true" />
          Delete Editor
        </button>
      </div>
    </Transition>

    <!-- Collapsed State: Show image if exists -->
    <div v-if="!isExpanded && hasImage" class="p-2">
      <img v-if="imageUrl" :src="imageUrl" class="w-full h-auto" alt="AI generated image, click expand to view editor">
    </div>
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
}>();

const { $api } = useNuxtApp();

const isExpanded = ref(false);
const currentPrompt = ref(props.initialText);

// Editor owns image state
const imageUrl = ref<string | null>(null);
const enhanceLoading = ref(false);
const generateLoading = ref(false);

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

const hasImage = computed(() => !!imageUrl.value);

watch(isExpanded, (expanded) => {
  if (expanded) {
    nextTick(() => {
      document.getElementById("prompt-textarea")?.focus();
    });
  }
});

watch(currentHistoryItem, (item) => {
  if (item) {
    currentPrompt.value = item.text;
  }
});

watch(imageUrl, (newUrl, oldUrl) => {
  if (oldUrl?.startsWith("blob:")) {
    URL.revokeObjectURL(oldUrl);
  }
});

onBeforeUnmount(() => {
  if (imageUrl.value?.startsWith("blob:")) {
    URL.revokeObjectURL(imageUrl.value);
  }
});

async function handleEnhance() {
  if (!currentPrompt.value.trim()) return;

  enhanceLoading.value = true;

  try {
    const res = await $api("/api/enhance", {
      method: "POST",
      body: { text: currentPrompt.value },
    });

    if (!res || !res.text) {
      useNotifier().error("Enhance failed");
      return;
    }

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
  if (!currentPrompt.value.trim()) return;

  generateLoading.value = true;

  try {
    const res = await $api("/api/gen-image-bytes", {
      method: "POST",
      body: { text: currentPrompt.value },
    });

    if (!res) {
      useNotifier().error("Image generation failed");
      return;
    }

    const blob = new Blob([res as any], { type: "image/png" });
    const url = URL.createObjectURL(blob);

    imageUrl.value = url;
    addToHistory(currentPrompt.value, url);

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
</script>

<style scoped>
.slide-enter-active,
.slide-leave-active {
  transition: opacity 0.3s ease, transform 0.3s ease;
}

.slide-enter-from,
.slide-leave-to {
  opacity: 0;
  transform: translateX(10px);
}

.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border-width: 0;
}

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
