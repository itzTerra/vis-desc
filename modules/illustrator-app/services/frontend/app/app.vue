<template>
  <NuxtLoadingIndicator />
  <NuxtPage />
  <div v-if="!appFullyReady" class="fixed bottom-0 left-1/2 transform -translate-x-1/2 z-500">
    <div v-if="appReady?.apiError" class="text-center badge badge-lg rounded-b-none badge-error">
      <Icon name="lucide:alert-triangle" size="20" class="inline" />
      <p>{{ statusMessage }}</p>
      <button class="btn btn-sm btn-link" @click="() => reloadNuxtApp()">
        Retry
      </button>
    </div>
    <div v-else class="text-center badge badge-lg rounded-b-none badge-warning">
      <span class="loading loading-dots" />
      <p>{{ statusMessage }}</p>
    </div>
  </div>
</template>
<script setup lang="ts">
import type { AppReadyState } from '~/types/common';

let appReady = ref<AppReadyState | null>(null);

onNuxtReady(() => {
  appReady.value = useNuxtApp().$appReadyState;
})

const statusMessage = computed(() => {
  if (appReady.value?.apiError) {
    return `Error: ${appReady.value.apiError}`;
  } else if (appReady.value && !appReady.value.apiReady) {
    return "Waking up API";
  } else if (appReady.value && !appReady.value.scorerWorkerReady) {
    return "Loading scorer worker";
  } else {
    return "Loading app";
  }
});

const appFullyReady = computed(() => appReady.value?.apiReady && appReady.value?.scorerWorkerReady && !appReady.value?.apiError);
</script>
