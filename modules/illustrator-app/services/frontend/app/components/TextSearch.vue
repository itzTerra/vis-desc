<template>
  <div class="flex items-center">
    <button class="btn btn-ghost" @click="cycleSearchedSets">
      <Icon name="lucide:arrow-up-down" size="20px" />
    </button>
    <div class="join flex items-center border rounded-field border-base-content/25 bg-base-100">
      <label class="input input-ghost join-item">
        <input
          v-model="searchQuery"
          type="text"
          :readonly="input !== undefined && input !== null"
          class="truncate"
          :title="searchQuery"
          @input="onSearchInput"
        >
        <span class="label">{{ found === 0 ? 0 : currentIndex + 1 }}/{{ found }}</span>
      </label>
      <div class="join-item flex flex-row">
        <button class="btn btn-ghost" :disabled="!found" @click="onPrev">
          <Icon name="lucide:chevron-up" size="20px" />
        </button>
        <button class="btn btn-ghost" :disabled="!found" @click="onNext">
          <Icon name="lucide:chevron-down" size="20px" />
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { debounce } from "lodash-es";

const DEBOUNCE_DELAY_BASE_MS = 2000;

const props = withDefaults(defineProps<{
  input?: string;
  searchedSetCount?: number;
}>(), {
  input: undefined,
  searchedSetCount: 1,
});

const currentIndex = defineModel<number>("index", {default: 0});
const found = defineModel<number>("found", {default: 0});
const currentSetIndex = defineModel<number>("currentSetIndex", {default: 0});

const emit = defineEmits<{
  "search": [string];
  "prev": [number];
  "next": [number];
  "cycle": [number];
}>();

const searchQuery = ref("");
const debouncedSearch = ref(debounce(onSearch, DEBOUNCE_DELAY_BASE_MS));

watch(() => props.input, (newInput) => {
  searchQuery.value = newInput || "";
});

function onSearchInput() {
  const x = Math.max(0, 10 - searchQuery.value.length) / 10.0;
  const delay = searchQuery.value.length > 0 ? Math.max(easeInExpo(x) * DEBOUNCE_DELAY_BASE_MS, 0) : 0;
  debouncedSearch.value.cancel();
  debouncedSearch.value = debounce(onSearch, delay);
  debouncedSearch.value();
}

function onSearch() {
  found.value = searchQuery.value.length;
  emit("search", searchQuery.value);
}

function onPrev() {
  const newVal = ((currentIndex.value - 1) + found.value) % found.value;
  currentIndex.value = newVal;
  emit("prev", newVal);
}

function onNext() {
  const newVal = (currentIndex.value + 1) % found.value;
  currentIndex.value = newVal;
  emit("next", newVal);
}

function cycleSearchedSets() {
  currentSetIndex.value = (currentSetIndex.value + 1) % props.searchedSetCount;
  emit("cycle", currentSetIndex.value);
}
</script>

<style scoped>
.input {
  min-width: 100px;
  max-width: 250px;
}

.btn-ghost {
  padding-inline: 0.25rem;
}
</style>
