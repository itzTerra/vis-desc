<template>
  <label class="btn btn-sm items-center" :class="$slots.default ? 'gap-2 px-3 w-full justify-start' : 'btn-circle'">
    <div class="swap swap-rotate" :title="colorMode === 'dark' ? 'Switch to Light' : 'Switch to Dark'">
      <input :value="colorMode" type="checkbox" class="theme-controller" checked @click="onToggle">
      <Icon name="lucide:moon" size="20" class="swap-on" />
      <Icon name="lucide:sun" size="20" class="swap-off" />
    </div>
    <slot />
  </label>
</template>

<script setup lang="ts">
const colorMode = useColorMode({
  attribute: "data-theme"
});

function onToggle() {
  colorMode.value = colorMode.value === "dark" ? "light" : "dark";
}
</script>
