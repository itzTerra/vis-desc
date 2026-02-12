<template>
  <div class="auto-illustration inline-flex items-center gap-2">
    <label class="flex items-center gap-2">
      <input v-model="enabledComputed" type="checkbox" class="toggle toggle-sm">
      <span class="text-sm">Auto-illustration</span>
    </label>
    <div class="relative">
      <button class="btn btn-ghost btn-square btn-sm" aria-label="Settings" @click="open = !open" :aria-expanded="open ? 'true' : 'false'" :aria-controls="'auto-illustration-settings-dropdown'">
        <Icon name="lucide:settings" size="16" />
      </button>
      <div v-if="open" id="auto-illustration-settings-dropdown" role="region" class="dropdown-content card p-3 mt-2 bg-base-100 shadow-lg w-64 z-50">
        <div class="mb-2">
          <label class="label">
            <span class="label-text">Min gap (lines)</span>
          </label>
          <input v-model.number="minGapLinesComputed" type="number" class="input input-sm w-full" min="0">
        </div>
        <div class="mb-2">
          <label class="label">
            <span class="label-text">Max gap (lines)</span>
          </label>
            <input v-model.number="maxGapLinesComputed" type="number" class="input input-sm w-full" :min="(Number.isFinite(minGapLinesComputed) ? minGapLinesComputed : 0) + 0.1" :aria-invalid="maxInvalid ? 'true' : 'false'" :aria-describedby="maxInvalid ? 'auto-max-gap-error' : null">
            <p v-if="maxInvalid" id="auto-max-gap-error" class="text-xs text-warning mt-1" role="alert">Max gap must be greater than min gap.</p>
        </div>
        <div class="mb-2">
          <label class="label">
            <span class="label-text">Min score (optional)</span>
          </label>
            <input v-model.number="minScoreComputed" type="number" class="input input-sm w-full" min="0" max="100" step="1" placeholder="unset (leave empty)" :aria-invalid="minScoreInvalid ? 'true' : 'false'" :aria-describedby="minScoreInvalid ? 'auto-min-score-error' : null">
            <p v-if="minScoreInvalid" id="auto-min-score-error" class="text-xs text-warning mt-1" role="alert">Min score must be an integer percentage (enter values like 75 for 75%).</p>
        </div>
        <div class="flex justify-between mt-2">
          <button class="btn btn-sm btn-warning" @click="clearAutoSelections">
            Clear
          </button>
          <div class="space-x-2">
            <button class="btn btn-sm" @click="runPass">
              Run Now
            </button>
            <button class="btn btn-sm btn-outline" @click="open = false">
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, isRef, computed } from "vue";
import type { Ref } from "vue";

const props = defineProps<{
  enabled: Ref<boolean> | boolean;
  minGapLines: Ref<number> | number;
  maxGapLines: Ref<number> | number;
  minScore: Ref<number | null> | number | null;
  runPass: () => void;
  clearAutoSelections: () => void;
}>();

const emit = defineEmits<["update:enabled", "update:minGapLines", "update:maxGapLines", "update:minScore"]>();

const open = ref(false);

const enabledComputed = computed<boolean>({
  get: () => (isRef(props.enabled) ? props.enabled.value : (props.enabled as boolean)),
  set: (v: boolean) => {
    if (isRef(props.enabled)) (props.enabled as Ref<boolean>).value = v;
    else emit("update:enabled", v);
  }
});

const minGapLinesComputed = computed<number>({
  get: () => (isRef(props.minGapLines) ? (props.minGapLines as Ref<number>).value : (props.minGapLines as number)),
  set: (v: number) => {
    if (isRef(props.minGapLines)) (props.minGapLines as Ref<number>).value = v;
    else emit("update:minGapLines", v);
  }
});

const maxGapLinesComputed = computed<number>({
  get: () => (isRef(props.maxGapLines) ? (props.maxGapLines as Ref<number>).value : (props.maxGapLines as number)),
  set: (v: number) => {
    if (isRef(props.maxGapLines)) (props.maxGapLines as Ref<number>).value = v;
    else emit("update:maxGapLines", v);
  }
});

const minScoreComputed = computed<number | null>({
  // Expose the value in the UI as percent (0..100) for easier integer input
  get: () => {
    const raw = isRef(props.minScore) ? (props.minScore as Ref<number | null>).value : (props.minScore as number | null);
    if (raw === null || raw === undefined) return null;
    if (typeof raw === "number" && Number.isFinite(raw)) return Math.round(raw * 100);
    return null;
  },
  set: (v: number | null) => {
    // Accept a percent (0..100) from the UI and normalize to 0..1 for storage
    // Normalize to an integer percent first to keep validation and storage consistent.
    let vNorm: number | null = null;
    if (typeof v === "number" && Number.isFinite(v)) {
      const vi = Math.round(v);
      if (vi >= 0 && vi <= 100) vNorm = vi / 100;
    }
    if (isRef(props.minScore)) (props.minScore as Ref<number | null>).value = vNorm;
    else emit("update:minScore", vNorm);
  }
});

const runPass = props.runPass;
const clearAutoSelections = props.clearAutoSelections;

const maxInvalid = computed(() => {
  const min = minGapLinesComputed.value ?? 0;
  const max = maxGapLinesComputed.value ?? 0;
  return Number.isFinite(min) && Number.isFinite(max) && max <= min;
});

const minScoreInvalid = computed(() => {
  const v = minScoreComputed.value;
  if (v === null || v === undefined) return false;
  // displayed as percent (0..100) — ensure integer percent in range
  if (!Number.isFinite(v)) return true;
  if (v < 0 || v > 100) return true;
  return !Number.isInteger(v);
});
</script>

<style scoped>
.dropdown-content { right: 0; }
</style>
