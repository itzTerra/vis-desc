<template>
  <div v-if="isLoading || isCancelled" class="flex items-center">
    <div class="flex flex-col">
      <template v-if="currentStage === 'Scoring...'">
        <progress class="progress progress-secondary w-52" :value="scoredSegmentCount" :max="highlights.length" />
        <div class="flex justify-between mt-0.5 text-sm opacity-60 gap-1">
          <template v-if="isCancelled || isPaused">
            <span>Scoring {{ isCancelled ? "canceled" : "paused" }}</span>
            <span>{{ scoredSegmentCount }}/{{ highlights.length }}</span>
          </template>
          <template v-else-if="highlights.length">
            <span>{{ currentStage }}</span>
            <span>{{ scoredSegmentCount }}/{{ highlights.length }}</span>
            <span>ETA&nbsp;{{ formatEta(etaMs) }}</span>
          </template>
          <template v-else>
            <span>Loading...</span>
          </template>
        </div>
      </template>
      <template v-else>
        <span>{{ currentStage || "Initializing..." }}</span>
      </template>
    </div>
    <!-- Socket-based: show Stop button; Worker-based: show Pause/Continue button -->
    <template v-if="!isCancelled">
      <button
        v-if="scorer && scorer.socketBased"
        class="btn btn-error btn-sm ms-2"
        title="Stop scoring"
        @click="$emit('cancel')"
      >
        <Icon name="lucide:stop-square" size="18px" />
      </button>
      <button
        v-else-if="scorer && !scorer.socketBased"
        class="btn btn-warning btn-sm ms-2"
        :title="isPaused ? 'Resume scoring' : 'Pause scoring'"
        @click="togglePause()"
      >
        <Icon v-if="!isPaused" name="lucide:pause" size="18px" />
        <Icon v-else name="lucide:play" size="18px" />
      </button>
    </template>
  </div>
</template>

<script setup lang="ts">
const props = defineProps<{
  isCancelled: boolean;
  highlights: { score?: number }[];
  currentStage?: string;
  scorer?: { socketBased: boolean } | null;
}>();

defineEmits<{
  cancel: [];
}>();

const isLoading = defineModel<number | null>();

const scoredSegmentCount = computed(() =>
  props.highlights.filter(h => typeof h.score === "number").length
);

const isPaused = ref(false);

function togglePause() {
  isPaused.value = !isPaused.value;
  const scorerWorker = getScorerWorker();
  scorerWorker.postMessage({ type: isPaused.value ? "pause" : "continue" });
}

watchOnce(
  () => props.highlights.length > 0 && scoredSegmentCount.value === props.highlights.length,
  isComplete => {
    if (isComplete) {
      useNotifier().success("Document processed successfully");
      isLoading.value = null;
    }
  }
);

const etaMs = computed(() => {
  if (!isLoading.value) return 0;
  const scored = scoredSegmentCount.value;
  const total = props.highlights.length;
  const remaining = total - scored;
  if (scored <= 0 || remaining <= 0) return 0;
  const elapsed = Date.now() - isLoading.value;
  if (elapsed <= 0) return 0;
  return (elapsed / scored) * remaining;
});

/** Round to full minutes if > 90 sec, else round to full seconds */
function formatEta(etaMs: number) {
  if (etaMs === 0) {
    return "??";
  }
  if (etaMs > 90000) {
    return `${Math.round(etaMs / 60 / 1000)} min`;
  }
  return `${Math.round(etaMs / 1000)} sec`;
}
</script>
