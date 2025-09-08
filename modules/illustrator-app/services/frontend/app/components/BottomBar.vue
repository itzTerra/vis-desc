<template>
  <div class="bottom-bar">
    <NuxtLink :href="$config.public.githubUrl" target="_blank" class="flex items-end space-x-1 bg-base-200 rounded-xl shadow hover:opacity-70">
      <Icon name="uil:github" size="28px" />
      <div class="pe-2 text-xs/3 flex flex-col grow-0 font-light font-mono">
        <span class="text-base-content/70">v{{ $config.public.appVersion }}</span>
      </div>
    </NuxtLink>
    <ThemeToggle />
  </div>
</template>

<script setup lang="ts">
const { $config } = useNuxtApp();
</script>
<style scoped>
.bottom-bar {
  position: fixed;
  bottom: 0;
  width: 100%;
  padding: 0.5rem 0.5rem;
  display: flex;
  align-items: center;
  justify-content: end;
  gap: 0.5rem;
}
</style>
