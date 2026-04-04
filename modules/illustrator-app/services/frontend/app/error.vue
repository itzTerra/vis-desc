<template>
  <div>
    <h2>{{ error?.status }}: {{ error?.message }}</h2>
  </div>
</template>

<script setup lang="ts">
import type { NuxtError } from "#app";

const props = defineProps({
  // eslint-disable-next-line vue/require-default-prop
  error: Object as () => NuxtError,
});

if (import.meta.client && props.error?.status === 404) {
  // Force a full page reload so GitHub Pages 404 redirect mechanism runs
  // and the login guard is not bypassed by in-SPA navigation.
  // eslint-disable-next-line no-self-assign
  window.location.href = window.location.href;
}
</script>
