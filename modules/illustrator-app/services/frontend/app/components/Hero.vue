<template>
  <div class="hero bg-custom">
    <div class="hero-content flex-col lg:flex-row-reverse max-w-5xl">
      <img src="/vis-desc-image.png" alt="hero-image" width="400" height="400">
      <div>
        <h1 class="text-5xl font-bold">
          Upload a PDF or TXT file to get started
        </h1>
        <p class="py-6">
          The tool is is designed to evaluate literature in English language, specially fiction, travel and history genres. Results for other types of content may vary.
        </p>
        <div class="max-w-xl">
          <label
            :class="[
              'flex justify-center w-full h-24 px-4 border-2 border-dashed rounded-md appearance-none',
              disabled
                ? 'cursor-not-allowed text-primary/40 border-primary/40 bg-primary/5'
                : [
                  'cursor-pointer transition text-primary/80 border-primary/80 bg-primary/5',
                  (isDragOver ? 'hover:text-primary hover:border-primary focus:outline-none hover:bg-primary/15 text-primary border-primary bg-primary/15' : 'hover:text-primary hover:border-primary focus:outline-none hover:bg-primary/15')
                ]
            ]"
            @dragover.prevent="onDragOver"
            @dragenter.prevent="onDragEnter"
            @dragleave.prevent="onDragLeave"
            @drop="onDrop"
          >
            <span class="flex items-center space-x-2 ">
              <Icon name="lucide:upload" />
              <span class="font-medium">
                Drop a PDF or TXT file here or click in this area
              </span>
            </span>
            <input type="file" accept="application/pdf,.txt,text/plain" name="file_upload" class="hidden" :disabled="disabled" @change="$emit('fileSelected', $event)">
          </label>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from "vue";
const props = defineProps<{
  disabled?: boolean;
}>();
const emit = defineEmits<{
  fileSelected: [event: any];
}>();

const isDragOver = ref(false);

function onDragOver() {
  if (props.disabled) return;
  isDragOver.value = true;
}
function onDragEnter() {
  if (props.disabled) return;
  isDragOver.value = true;
}
function onDragLeave() {
  if (props.disabled) return;
  isDragOver.value = false;
}
function onDrop(e: DragEvent) {
  if (props.disabled) return;
  e.preventDefault();
  isDragOver.value = false;
  const files = e.dataTransfer?.files;
  if (files && files.length > 0) {
    // Wrap in a synthetic event to match input @change
    emit("fileSelected", { target: { files } });
  }
}
</script>
