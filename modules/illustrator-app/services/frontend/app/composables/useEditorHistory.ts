import { computed, ref } from "vue";
import type { EditorHistoryItem } from "~/types/common";

export function useEditorHistory() {
  const history = ref<EditorHistoryItem[]>([]);
  const historyIndex = ref(0);

  const currentHistoryItem = computed(() =>
    history.value[historyIndex.value]
  );

  const isAtStart = computed(() => historyIndex.value <= 0);
  const isAtEnd = computed(() =>
    historyIndex.value >= history.value.length - 1
  );

  function addToHistory(text: string, imageUrl?: string) {
    // Deduplication: skip if text matches immediately previous entry
    const lastItem = history.value[history.value.length - 1];
    if (lastItem && lastItem.text === text && (lastItem.imageUrl === imageUrl || !imageUrl)) {
      return;
    }

    if (lastItem && lastItem.text === text && !lastItem.imageUrl && imageUrl) {
      // If only imageUrl is new, update the last entry instead of adding a new one
      lastItem.imageUrl = imageUrl;
      return;
    }

    const newItem: EditorHistoryItem = { text, imageUrl };

    history.value.push(newItem);
    historyIndex.value = history.value.length - 1;
  }

  function navigatePrevious() {
    if (historyIndex.value > 0) {
      historyIndex.value--;
    } else {
      historyIndex.value = history.value.length - 1;
    }
  }

  function navigateNext() {
    if (historyIndex.value < history.value.length - 1) {
      historyIndex.value++;
    } else {
      historyIndex.value = 0;
    }
  }

  return {
    history,
    historyIndex,
    currentHistoryItem,
    isAtStart,
    isAtEnd,
    addToHistory,
    navigatePrevious,
    navigateNext,
  };
}
