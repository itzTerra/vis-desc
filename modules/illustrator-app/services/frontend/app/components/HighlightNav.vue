<template>
  <div class="flex items-center">
    <details class="dropdown">
      <summary class="join flex items-center border rounded-field border-base-content/25 bg-base-100" @click="isExpanded = !isExpanded">
        <div class="join-item p-2 cursor-pointer swap swap-rotate" :class="{'swap-active': isExpanded}">
          <Icon name="lucide:chevron-down" class="swap-on" />
          <Icon name="lucide:chevron-right" class="swap-off" />
        </div>
        <label class="input input-ghost join-item" @click.stop>
          <input
            v-model="searchInput"
            type="text"
            readonly
            class="truncate"
            :title="searchInput"
          >
          <span class="label">{{ currentSearchSet.length === 0 ? 0 : currentIndex + 1 }}/{{ currentSearchSet.length }}</span>
        </label>
        <div class="join-item flex flex-row" @click.stop>
          <button class="btn btn-ghost px-2" :disabled="!currentSearchSet.length" title="Previous Segment" @click.stop="onPrev">
            <Icon name="lucide:chevron-up" size="20px" />
          </button>
          <button class="btn btn-ghost px-2" :disabled="!currentSearchSet.length" title="Next Segment" @click.stop="onNext">
            <Icon name="lucide:chevron-down" size="20px" />
          </button>
        </div>
      </summary>
      <div class="menu dropdown-content bg-base-300 rounded-b-box z-1 w-full p-2 shadow flex flex-col gap-2 max-h-[90vh]">
        <label class="label space-x-1">
          <input v-model="filterSelectedOnly" type="checkbox" class="toggle toggle-sm toggle-primary">
          <span>Selected only</span>
        </label>
        <fieldset class="fieldset">
          <legend class="fieldset-legend py-1">
            Score filter
          </legend>
          <div class="join items-center bg-base-200 rounded-box p-1">
            <input v-model="scoreFilter.min" type="number" class="input input-sm input-ghost join-item" placeholder="Min">
            <span class="join-item px-2">..</span>
            <input v-model="scoreFilter.max" type="number" class="input input-sm input-ghost join-item" placeholder="Max">
          </div>
        </fieldset>
        <div class="w-full h-100 overflow-auto">
          <table class="table table-xs table-pin-rows w-full">
            <colgroup>
              <col class="col-index">
              <col>
              <col class="col-score">
            </colgroup>
            <thead>
              <tr>
                <th class="text-right">
                  <div class="flex items-center space-x-1">
                    <div class="flex flex-col">
                      <button
                        class="btn btn-ghost btn-xs p-0 m-0 h-2"
                        :class="{'btn-active': ordering.id === 'asc'}"
                        @click="ordering.id = ordering.id === 'asc' ? null : 'asc'; ordering.score = null"
                      >
                        <Icon name="lucide:chevron-up" size="10px" />
                      </button>
                      <button
                        class="btn btn-ghost btn-xs p-0 m-0 h-2"
                        :class="{'btn-active': ordering.id === 'desc'}"
                        @click="ordering.id = ordering.id === 'desc' ? null : 'desc'; ordering.score = null"
                      >
                        <Icon name="lucide:chevron-down" size="10px" />
                      </button>
                    </div>
                    <span>#</span>
                  </div>
                </th>
                <td>Segment</td>
                <td>
                  <div class="flex items-center space-x-1">
                    <div class="flex flex-col">
                      <button
                        class="btn btn-ghost btn-xs p-0 m-0 h-2"
                        :class="{'btn-active': ordering.score === 'asc'}"
                        @click="ordering.score = ordering.score === 'asc' ? null : 'asc'; ordering.id = null"
                      >
                        <Icon name="lucide:chevron-up" size="10px" />
                      </button>
                      <button
                        class="btn btn-ghost btn-xs p-0 m-0 h-2"
                        :class="{'btn-active': ordering.score === 'desc'}"
                        @click="ordering.score = ordering.score === 'desc' ? null : 'desc'; ordering.id = null"
                      >
                        <Icon name="lucide:chevron-down" size="10px" />
                      </button>
                    </div>
                    <span>Pts</span>
                  </div>
                </td>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="(highlight, index) in currentSearchSet"
                :key="highlight.id"
                class="bg-base-200 hover:bg-base-100 select-none cursor-pointer"
                :class="{'bg-primary/30 hover:bg-primary/20': index === currentIndex}"
                @click="() => onSelectFromTable(index)"
              >
                <th class="text-right">
                  {{ highlight.id + 1 }}
                </th>
                <td :title="highlight.text" class="align-top">
                  <div class="truncate">
                    {{ highlight.text }}
                  </div>
                </td>
                <td class="text-right">
                  {{ highlight.score ? (highlight.score * 100).toFixed(2) : 'N/A' }}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </details>
  </div>
</template>

<script setup lang="ts">
import type { Highlight } from "~/types/common";

const props = defineProps<{
  highlights: Highlight[];
  selectedHighlights: Set<number>;
}>();

const { callHook } = useNuxtApp();

const isExpanded = ref(false);
const currentIndex = ref<number>(0);
const searchInput = ref("");
const filterSelectedOnly = ref(false);
const scoreFilter = ref<{ min: number | null; max: number | null }>({ min: null, max: null });
const ordering = ref<{
  id: "asc" | "desc" | null;
  score: "asc" | "desc" | null;
}>({
  id: null,
  score: "desc",
});

function onPrev() {
  const newVal = ((currentIndex.value - 1) + currentSearchSet.value.length) % currentSearchSet.value.length;
  currentIndex.value = newVal;
  onPrevNext();
}

function onNext() {
  const newVal = (currentIndex.value + 1) % currentSearchSet.value.length;
  currentIndex.value = newVal;
  onPrevNext();
}

function onPrevNext() {
  const highlight = currentSearchSet.value[currentIndex.value];
  if (!highlight) {
    return;
  }
  searchInput.value = highlight.text || "";
  callHook("custom:goToHighlight", highlight);
}

function onSelectFromTable(index: number) {
  currentIndex.value = index;
  onPrevNext();
}

const currentSearchSet = computed(() => {
  let set = props.highlights.filter(h => (h.score ?? 0) * 100 >= (scoreFilter.value.min ?? 0) && (h.score ?? 0) * 100 <= (scoreFilter.value.max ?? 100));
  if (filterSelectedOnly.value) {
    set = set.filter(h => props.selectedHighlights.has(h.id));
  }
  if (ordering.value.id) {
    set = [...set].sort((a, b) => ordering.value.id === "asc" ? a.id - b.id : b.id - a.id);
  } else if (ordering.value.score) {
    set = [...set].sort((a, b) => ordering.value.score === "asc" ? (a.score ?? 0) - (b.score ?? 0) : (b.score ?? 0) - (a.score ?? 0));
  }
  return set;
});

function reset() {
  currentIndex.value = 0;
  searchInput.value = "";
}

defineExpose({
  reset,
});
</script>
<style scoped>
.input {
  min-width: 100px;
  max-width: 250px;
}

/* Fixed column widths for index and score; middle column flexes */
col.col-index { width: 2.75rem; }
col.col-score { width: 3.15rem; }

table { table-layout: fixed; }
/* Allow middle cell to actually use remaining space while text truncates */
td { overflow-wrap: anywhere; }
</style>
