<template>
  <div class="hero bg-custom">
    <div class="hero-content flex-col lg:flex-row-reverse max-w-[64rem]">
      <img src="/vis-desc-image.png" alt="hero-image" width="400" height="400">
      <div>
        <h1 class="text-5xl font-bold">
          Upload a file in PDF format to get started
        </h1>
        <p class="py-6">
          The tool is is designed to evaluate literature in English language, specially fiction, travel and history genres. Results for other types of content may vary.
        </p>
        <div class="max-w-xl">
          <label
            class="flex justify-center w-full h-24 px-4 transition border-2 border-dashed rounded-md appearance-none cursor-pointer text-base-content/50 border-base-content/50 hover:border-base-content/30 focus:outline-none hover:bg-primary/10"
          >
            <span class="flex items-center space-x-2 ">
              <Icon name="lucide:upload" />
              <span class="font-medium">
                Drop file here or click in this area
              </span>
            </span>
            <input type="file" accept="application/pdf" name="file_upload" class="hidden" @change="$emit('fileSelected', $event)">
          </label>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
defineEmits<{
  fileSelected: [event: any];
}>();
</script>
