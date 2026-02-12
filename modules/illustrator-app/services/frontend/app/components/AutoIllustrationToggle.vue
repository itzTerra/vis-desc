<template>
  <div class="auto-illustration inline-flex items-center gap-2">
    <label class="flex items-center gap-2">
      <input v-model="enabledComputed" type="checkbox" class="toggle toggle-sm">
      <span class="text-sm">Auto-illustration</span>
    </label>
    <div class="relative">
      <button class="btn btn-ghost btn-square btn-sm" aria-label="Settings" @click="open = !open">
        <Icon name="lucide:settings" size="16" />
      </button>
      <div v-if="open" class="dropdown-content card p-3 mt-2 bg-base-100 shadow-lg w-64 z-50">
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
          <input v-model.number="maxGapLinesComputed" type="number" class="input input-sm w-full" :min="minGapLinesComputed + 0.1">
        </div>
        <div class="mb-2">
          <label class="label">
            <span class="label-text">Min score (optional)</span>
          </label>
          <input v-model.number="minScoreComputed" type="number" class="input input-sm w-full" min="0" max="1" step="0.01" placeholder="unset (leave empty)">
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
import { ref, watch, isRef, computed } from "vue";
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
  get: () => (isRef(props.minScore) ? (props.minScore as Ref<number | null>).value : (props.minScore as number | null)),
  set: (v: number | null) => {
    const vNorm = (typeof v === "number" && !Number.isNaN(v)) ? v : null;
    if (isRef(props.minScore)) (props.minScore as Ref<number | null>).value = vNorm;
    else emit("update:minScore", vNorm);
  }
});

const runPass = props.runPass;
const clearAutoSelections = props.clearAutoSelections;

watch([minGapLinesComputed, maxGapLinesComputed], ([min, max]) => {
  if (max <= min) {
    maxGapLinesComputed.value = min + 1;
  }
});
</script>

<style scoped>
.dropdown-content { right: 0; }
</style>
