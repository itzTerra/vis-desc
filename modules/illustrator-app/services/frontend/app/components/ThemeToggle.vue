<template>
  <label class="btn btn-sm btn-circle items-center">
    <div class="swap swap-rotate">
      <input :value="colorMode" type="checkbox" class="theme-controller" checked @click="onToggle">
      <Icon name="lucide:moon" size="16" class="swap-on" />
      <Icon name="lucide:sun" size="16" class="swap-off" />
    </div>
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
