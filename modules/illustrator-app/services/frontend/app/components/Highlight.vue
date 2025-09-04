<template>
  <div
    class="dropdown dropdown-bottom dropdown-end dropdown-hover highlight-dropdown"
    @mouseenter="onHighlightHover(index)"
    @mouseleave="onHighlightLeave"
  >
    <div class="highlight-dropdown-content" :class="dropdownContentClasses[index]" @click="$emit('select', index)">
        &nbsp;
    </div>
    <div
      class="dropdown-content bg-base-100 hover:bg-base-200 rounded-box z-1 w-64 p-2 shadow-sm"
    >
      <div class="hidden">
        {{ highlight.text }}
      </div>
      <div v-if="highlight.score" class="flex">
        <div class="stat pa-2">
          <div class="stat-value">
            {{ (highlight.score * 100).toFixed(0) }}
          </div>
          <div class="stat-figure text-secondary">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              class="inline-block h-8 w-8 stroke-current"
            >
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                stroke-width="2"
                d="M13 10V3L4 14h7v7l9-11h-7z"
              />
            </svg>
          </div>
          <button
            class="btn btn-sm btn-primary ms-auto"
            :disabled="!highlight.text"
            @click="genImage(highlight)"
          >
            Generate image <Icon name="lucide:chevron-right" size="24" />
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { Highlight } from "~/types/common";

defineEmits<{
  select: [index: number]
}>();

const { $api } = useNuxtApp();

const props = defineProps<{
  highlight: Highlight;
  index: number;
  highlights: Highlight[];
  selectedHighlights: Highlight[];
}>();

const dropdownContentClasses = computed(() =>
  props.highlights.map((highlight, index) => ({
    "highlight-dropdown-content__hovered": hoveredHighlightIndex.value === index,
    "highlight-selected": props.selectedHighlights.includes(highlight),
  }))
);

const hoveredHighlightIndex = ref<number | null>(null);
function onHighlightHover(index: number) {
  hoveredHighlightIndex.value = index;
  // console.log(`Hovered over highlight at index: ${index}`);
}
function onHighlightLeave() {
  hoveredHighlightIndex.value = null;
  // console.log("Mouse left highlight");
}

async function genImage(highlight: Highlight) {
  if (!highlight.text) return;

  highlight.imageLoading = true;
  const res = await $api("/api/gen-image-bytes", {
    method: "POST",
    body: { text: highlight.text }
  });
  console.log(res);
  const blob = new Blob([res as any], { type: "image/png" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.target = "_blank";
  link.click();
  highlight.imageUrl = url;
  highlight.imageLoading = false;
  return { blob, url };
}
</script>

