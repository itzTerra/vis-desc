<template>
  <div v-if="isOpen" class="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
    <div class="bg-base-100 rounded-lg shadow-lg p-6 max-w-md w-full mx-4">
      <h2 class="text-lg font-bold mb-4">
        Export as HTML
      </h2>
      <div class="space-y-4 mb-6">
        <div>
          <label for="export-filename" class="label">
            <span class="label-text">Filename</span>
          </label>
          <input
            id="export-filename"
            v-model="filename"
            type="text"
            class="input input-bordered w-full"
            placeholder="export.html"
            @keydown.enter="handleConfirm"
          >
        </div>
        <div class="text-sm text-base-content/70">
          <p v-if="imageCount > 0" class="mb-2">
            ✓ PDF with {{ imageCount }} generated image{{ imageCount === 1 ? '' : 's' }}
          </p>
          <p v-else class="mb-2">
            ✓ PDF only (no images generated)
          </p>
          <p>The exported file will be fully offline-compatible.</p>
        </div>
      </div>

      <div class="flex gap-2 justify-end">
        <button
          class="btn btn-ghost btn-sm"
          :disabled="isExporting"
          @click="handleCancel"
        >
          Cancel
        </button>
        <button
          class="btn btn-primary btn-sm"
          :disabled="isExporting || !filename.trim()"
          :aria-busy="isExporting"
          @click="handleConfirm"
        >
          <span v-if="isExporting" class="loading loading-spinner loading-sm" />
          <span v-else>Export</span>
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
const props = withDefaults(defineProps<{
  isOpen: boolean;
  imageCount?: number;
}>(), {
  imageCount: 0,
});

const emit = defineEmits<{
  confirm: [filename: string];
  cancel: [];
}>();

const filename = ref("export.html");
const isExporting = ref(false);

const handleConfirm = async () => {
  if (!filename.value.trim()) return;
  isExporting.value = true;
  try {
    emit("confirm", filename.value.trim());
  } finally {
    isExporting.value = false;
  }
};

const handleCancel = () => {
  emit("cancel");
};

watch(
  () => props.isOpen,
  (isOpen) => {
    if (isOpen) {
      filename.value = "export.html";
      nextTick(() => {
        const input = document.getElementById("export-filename") as HTMLInputElement;
        input?.select();
      });
    }
  }
);
</script>
