<template>
  <div v-if="isLoading || isCancelled" class="flex items-center ms-4">
    <div class="flex flex-col">
      <progress class="progress progress-info w-52" :value="scoredSegmentCount" :max="highlights.length" />
      <div class="flex justify-between mt-0.5 text-sm opacity-60">
        <template v-if="!isCancelled && highlights.length">
          <span>{{ scoredSegmentCount }}/{{ highlights.length }}</span>
          <span>~{{ formatEta(etaMs) }}&nbsp;remaining</span>
        </template>
        <template v-else-if="!isCancelled">
          <span>Loading...</span>
        </template>
        <template v-else>
          <span>{{ scoredSegmentCount }}/{{ highlights.length }}</span>
          <span>Loading canceled</span>
        </template>
      </div>
    </div>
    <button v-if="!isCancelled" class="btn btn-error btn-sm ms-2" @click="$emit('cancel')">
      Stop
    </button>
  </div>
</template>

<script setup lang="ts">
const props = defineProps<{
  isLoading: number | null;
  isCancelled: boolean;
  highlights: { score?: number }[];
}>();

defineEmits<{
  cancel: [];
}>();

const scoredSegmentCount = computed(() =>
  props.highlights.filter(h => typeof h.score === "number").length
);

const etaMs = computed(() => {
  if (!props.isLoading) return 0;
  const scored = scoredSegmentCount.value;
  const total = props.highlights.length;
  const remaining = total - scored;
  if (scored <= 0 || remaining <= 0) return 0;
  const elapsed = Date.now() - props.isLoading;
  if (elapsed <= 0) return 0;
  return (elapsed / scored) * remaining;
});

/** Round to full minutes if > 90 sec, else round to full seconds */
function formatEta(etaMs: number) {
  if (etaMs > 90000) {
    return `${Math.round(etaMs / 60 / 1000)} min`;
  }
  return `${Math.round(etaMs / 1000)} sec`;
}
</script>
