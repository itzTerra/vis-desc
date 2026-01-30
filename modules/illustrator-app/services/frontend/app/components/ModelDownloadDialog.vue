<template>
  <div v-if="isOpen" class="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
    <div class="bg-base-100 rounded-lg shadow-lg p-6 max-w-md w-full mx-4">
      <h2 class="text-lg font-bold mb-4">
        Download {{ modelInfo.label }}
      </h2>
      <div class="space-y-3 mb-6 text-sm">
        <div>
          <p class="font-semibold text-base-content/80">
            Model Size
          </p>
          <p class="text-base-content/60">
            {{ modelInfo.transformersConfig.sizeMb }} MB
          </p>
        </div>
        <div>
          <p class="font-semibold text-base-content/80">
            Caching
          </p>
          <p class="text-base-content/60">
            The model will be cached in your browser for offline use after the first download.
            You can clear browser cache to remove it.
          </p>
        </div>
        <div>
          <p class="font-semibold text-base-content/80">
            Privacy
          </p>
          <p class="text-base-content/60">
            The model runs entirely in your browser. No data is sent to external servers during inference.
            Download is from Hugging Face CDN.
          </p>
        </div>
      </div>

      <div class="flex gap-2 justify-end">
        <button
          class="btn btn-ghost btn-sm"
          @click="handleCancel"
        >
          Cancel
        </button>
        <button
          class="btn btn-primary btn-sm"
          @click="handleConfirm"
        >
          Download
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
withDefaults(defineProps<{
  isOpen: boolean;
  modelInfo: ModelInfo & { transformersConfig: TransformersModelConfig };
}>(), {});

const emit = defineEmits<{
  confirm: [];
  cancel: [];
}>();

const handleConfirm = () => {
  emit("confirm");
};

const handleCancel = () => {
  emit("cancel");
};
</script>
