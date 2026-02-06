<template>
  <div class="image-editor relative bg-base-100 border-l border-base-300" style="width: 100%;">
    <!-- Collapse Button (protrudes LEFT into PDF viewer area) -->
    <button
      class="btn btn-sm btn-circle absolute top-2"
      style="left: -20px;"
      :title="isExpanded ? 'Collapse' : 'Expand'"
      @click="isExpanded = !isExpanded"
    >
      <Icon :name="isExpanded ? 'lucide:chevron-right' : 'lucide:chevron-left'" />
    </button>

    <!-- Editor Content (visible when expanded) -->
    <Transition name="slide">
      <div v-show="isExpanded" class="p-4 space-y-4">
        <!-- Top Bar -->
        <div class="flex justify-between items-center">
          <span class="text-sm font-semibold">Image Editor</span>
        </div>

        <!-- Prompt Area (hidden if image exists) -->
        <template v-if="!hasImage">
          <textarea
            v-model="currentPrompt"
            class="textarea textarea-bordered w-full h-24"
            placeholder="Enter or paste image prompt..."
            :disabled="enhanceLoading || generateLoading"
          />
          <div class="flex gap-2">
            <button
              class="btn btn-sm btn-primary flex-1"
              :disabled="enhanceLoading || generateLoading || !currentPrompt"
              @click="handleEnhance"
            >
              <Icon v-if="enhanceLoading" name="lucide:loader" class="animate-spin" />
              Enhance
            </button>
            <button
              class="btn btn-sm btn-secondary flex-1"
              :disabled="enhanceLoading || generateLoading || !currentPrompt"
              @click="handleGenerate"
            >
              <Icon v-if="generateLoading" name="lucide:loader" class="animate-spin" />
              Generate
            </button>
          </div>
        </template>

        <!-- Generated Image Display (INSIDE editor) -->
        <div v-if="hasImage || generateLoading" class="border rounded p-2">
          <img v-if="imageUrl" :src="imageUrl" class="w-full h-auto" alt="Generated image">
          <div v-if="generateLoading" class="flex justify-center items-center h-32">
            <Icon name="lucide:loader" class="animate-spin" size="32" />
          </div>
        </div>

        <!-- History Display -->
        <div v-if="history.length > 0" class="border-t pt-4">
          <div class="text-sm font-semibold mb-2">
            History
          </div>
          <div class="flex gap-2 items-center mb-2">
            <button
              class="btn btn-xs"
              :disabled="isAtStart"
              @click="navigatePrevious"
            >
              <Icon name="lucide:chevron-left" />
            </button>
            <div class="text-xs text-base-content/70 flex-1 text-center">
              {{ currentHistoryItem ? `${historyIndex + 1}/${history.length}` : 'Editing' }}
            </div>
            <button
              class="btn btn-xs"
              :disabled="isAtEnd"
              @click="navigateNext"
            >
              <Icon name="lucide:chevron-right" />
            </button>
          </div>
          <div v-if="currentHistoryItem" class="bg-base-200 p-2 rounded text-xs mb-2 max-h-20 overflow-auto">
            {{ currentHistoryItem.text }}
          </div>
          <button
            class="btn btn-xs btn-outline w-full"
            @click="clearHistory"
          >
            Clear History
          </button>
        </div>

        <!-- Delete Button -->
        <button
          class="btn btn-sm btn-error w-full"
          @click="handleDelete"
        >
          <Icon name="lucide:trash-2" />
          Delete Editor
        </button>
      </div>
    </Transition>

    <!-- Collapsed State: Show image if exists -->
    <div v-if="!isExpanded && hasImage" class="p-2">
      <img v-if="imageUrl" :src="imageUrl" class="w-full h-auto" alt="Generated image">
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
  clearHistory,
} = useEditorHistory();

const hasImage = computed(() => !!imageUrl.value);

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
    const { $api: call } = useNuxtApp();
    
    const res = await call("/api/enhance", {
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
    const { $api: call } = useNuxtApp();

    const res = await call("/api/gen-image-bytes", {
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
  transition: opacity 0.3s;
}

.slide-enter-from,
.slide-leave-to {
  opacity: 0;
}
</style>
