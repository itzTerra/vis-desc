<template>
  <div class="auto-illustration control inline-flex flex-col">
    <div class="inline-flex items-center gap-2">
      <label class="flex items-center gap-2">
        <input v-model="enabled" type="checkbox" class="toggle toggle-sm toggle-secondary">
        <span class="text-sm whitespace-nowrap">Auto-illustrate</span>
      </label>
      <div class="dropdown">
        <div
          tabindex="0"
          role="button"
          class="btn btn-ghost btn-square btn-sm"
          aria-label="Settings"
        >
          <Icon name="lucide:settings" size="16" />
        </div>
        <div
          tabindex="-1"
          class="dropdown-content card p-3 mt-2 bg-base-300 shadow-lg w-56 z-50"
        >
          <div class="mb-2">
            <label class="label">
              <span class="label-text">Min gap (pages)</span>
            </label>
            <input
              v-model.number="minGapPages"
              type="number"
              step="0.1"
              class="input input-sm w-full"
              :min="MIN_GAP_PAGES_TOTAL"
            >
          </div>
          <div class="mb-2">
            <label class="label">
              <span class="label-text">Max gap (pages)</span>
            </label>
            <input
              v-model.number="maxGapPages"
              type="number"
              step="0.1"
              class="input input-sm w-full"
              :min="MIN_GAP_PAGES_TOTAL"
              :aria-invalid="maxInvalid ? 'true' : 'false'"
              :aria-describedby="maxInvalid ? 'auto-max-gap-error' : undefined"
            >
            <p v-if="maxInvalid" id="auto-max-gap-error" class="text-xs text-warning mt-1" role="alert">
              Max gap must be greater than min gap.
            </p>
          </div>
          <div class="mb-2">
            <label class="label">
              <span class="label-text">Min score (optional)</span>
            </label>
            <input
              v-model.number="minScorePercent"
              type="number"
              class="input input-sm w-full"
              min="0"
              max="100"
              step="1"
              placeholder="unset (leave empty)"
              :aria-invalid="minScoreInvalid ? 'true' : 'false'"
              :aria-describedby="minScoreInvalid ? 'auto-min-score-error' : undefined"
            >
            <p v-if="minScoreInvalid" id="auto-min-score-error" class="text-xs text-warning mt-1" role="alert">
              Min score must be an integer percentage (enter values like 75 for 75%).
            </p>
          </div>
          <div class="flex flex-col gap-2 mb-2">
            <label class="flex items-center gap-2">
              <input v-model="enableEnhance" type="checkbox" class="toggle toggle-sm toggle-secondary">
              <span class="text-sm">Enhance</span>
            </label>
            <label class="flex items-center gap-2">
              <input v-model="enableGenerate" type="checkbox" class="toggle toggle-sm toggle-secondary">
              <span class="text-sm">Generate</span>
            </label>
          </div>
          <div class="flex justify-between mt-2">
            <button class="btn btn-sm btn-error" @click="clearAutoSelections">
              <Icon name="lucide:trash" size="16" /> Clear
            </button>
            <div class="space-x-2">
              <button class="btn btn-sm btn-primary" @click="runPass">
                <Icon name="lucide:play" size="16" /> Run
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Status bar below the toggle -->
    <div class="auto-illustration-status" :title="tooltipText" :aria-label="tooltipText" role="group">
      <div class="bar" aria-hidden="true">
        <div class="segment selected" :style="{ width: normalized.selected + '%' }" />
        <div
          class="segment enhanced"
          :class="{ active: isActive && progress?.enhanceRequestedCount }"
          :style="{ width: normalized.enhanced + '%' }"
        />
        <div
          class="segment generated"
          :class="{ active: isActive && progress?.generateRequestedCount }"
          :style="{ width: normalized.generated + '%' }"
        />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { MIN_GAP_PAGES_TOTAL } from "~/composables/useAutoIllustration";

const props = defineProps<{
  progress?: Progress | null;
  runPass?: () => void;
  clearAutoSelections?: () => void;
}>();

const enabled = defineModel<boolean>("enabled", { required: true });
const minGapPages = defineModel<number>("minGapPages", { required: true });
const maxGapPages = defineModel<number>("maxGapPages", { required: true });
const minScore = defineModel<number | null>("minScore", { required: true });
const enableEnhance = defineModel<boolean>("enableEnhance", { required: true });
const enableGenerate = defineModel<boolean>("enableGenerate", { required: true });

const isActive = computed(() => props.progress?.enhanceRequestedCount || props.progress?.generateRequestedCount);

const selectedCount = computed(() => props.progress?.selectedCount ?? 0);
const enhancedCount = computed(() => props.progress?.enhancedCount ?? 0);
const generatedCount = computed(() => props.progress?.generatedCount ?? 0);
const maxPossible = computed(() => Math.max(1, props.progress?.maxPossible ?? 1));

// Normalize percentages so the three bars never overflow the container
const normalized = computed(() => {
  const raw = {
    selected: ((selectedCount.value - Math.max(0, enhancedCount.value, generatedCount.value)) / maxPossible.value) * 100,
    enhanced: ((enhancedCount.value - Math.max(0, generatedCount.value)) / maxPossible.value) * 100,
    generated: (generatedCount.value / maxPossible.value) * 100,
  };
  const total = raw.selected + raw.enhanced + raw.generated;
  const scale = total > 100 ? 100 / total : 1;
  const sel = Math.round(raw.selected * scale);
  const enh = Math.round(raw.enhanced * scale);
  const gen = Math.round(raw.generated * scale);
  return { selected: sel, enhanced: enh, generated: gen };
});

const tooltipText = computed(() => {
  return `Selected: ${selectedCount.value}/${maxPossible.value} | Enhanced: ${enhancedCount.value} | Generated: ${generatedCount.value} | Active: ${isActive.value ? "✓" : "✕"}`;
});

// Present `minScore` as an integer percent (0..100) in the UI while storing 0..1
const minScorePercent = computed<number | null>({
  get: () => {
    const raw = unref(minScore);
    if (raw === null || raw === undefined) return null;
    if (typeof raw === "number" && Number.isFinite(raw)) return Math.round(raw * 100);
    return null;
  },
  set: (v: number | null) => {
    let vNorm: number | null = null;
    if (typeof v === "number" && Number.isFinite(v)) {
      const vi = Math.round(v);
      if (vi >= 0 && vi <= 100) vNorm = vi / 100;
    }
    minScore.value = vNorm;
  }
});

const runPass = props.runPass ?? (() => {});
const clearAutoSelections = props.clearAutoSelections ?? (() => {});

const maxInvalid = computed(() => {
  const min = minGapPages.value ?? 0;
  const max = maxGapPages.value ?? 0;
  return Number.isFinite(min) && Number.isFinite(max) && max <= min;
});

const minScoreInvalid = computed(() => {
  const v = minScorePercent.value;
  if (v === null || v === undefined) return false;
  if (!Number.isFinite(v)) return true;
  if (v < 0 || v > 100) return true;
  return !Number.isInteger(v);
});
</script>

<style scoped>
.auto-illustration {
  display: inline-flex;
}

.auto-illustration .relative {
  display: inline-block;
}

.dropdown-content {
  right: 0;
}

.auto-illustration-status {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.bar {
  display: flex;
  width: 100%;
  height: 8px;
  background: var(--color-base-300);
  border-radius: 6px;
  overflow: hidden;
}

.segment {
  height: 100%;
  position: relative;
}
.segment:first-child {
  border-top-left-radius: inherit;
  border-bottom-left-radius: inherit;
}
.segment:last-child {
  border-top-right-radius: inherit;
  border-bottom-right-radius: inherit;
}

.segment.selected {
  background: var(--color-primary);
}

.segment.enhanced {
  background: var(--color-secondary);
}

.segment.generated {
  background: var(--color-success);
}

.segment.active {
  box-shadow: 0 0 6px rgba(0, 0, 0, 0.08) inset;
}

.segment::after {
  content: '';
  position: absolute;
  pointer-events: none;
  border-radius: inherit;
  opacity: 0.9;
  will-change: transform, background-position;
}

.segment.enhanced.active::after {
  top: 0;
  left: 0;
  bottom: 0;
  width: 40%;
  background: linear-gradient(90deg, rgba(255, 255, 255, 0) 0%, rgba(255, 255, 255, 0.35) 50%, rgba(255, 255, 255, 0) 100%);
  transform: translateX(-120%);
  animation: shimmer-translate 2.4s ease-in-out infinite;
}

.segment.generated.active::after {
  inset: 0;
  background-image: repeating-linear-gradient(135deg, rgba(255, 255, 255, 0.06) 0 6px, transparent 6px 12px);
  background-size: 24px 24px;
  animation: stripes-move 3s linear infinite;
}

@keyframes shimmer-translate {
  0% {
    transform: translateX(-120%);
  }

  50% {
    transform: translateX(20%);
  }

  100% {
    transform: translateX(120%);
  }
}

@keyframes stripes-move {
  from {
    background-position: 0 0;
  }

  to {
    background-position: 24px 24px;
  }
}

@media (prefers-reduced-motion: reduce) {

  .segment.enhanced.active::after,
  .segment.generated.active::after {
    animation: none;
    transform: none;
    background-position: initial;
    opacity: 0.35;
  }
}
</style>
