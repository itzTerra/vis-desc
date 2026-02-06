import { computed, ref } from "vue";
import type { EditorHistoryItem } from "~/types/common";

export function useEditorHistory() {
  const history = ref<EditorHistoryItem[]>([]);
  const historyIndex = ref(-1);

  const currentHistoryItem = computed(() => {
    if (historyIndex.value === -1) return null;
    return history.value[historyIndex.value] || null;
  });

  const isAtStart = computed(() => historyIndex.value <= 0);
  const isAtEnd = computed(() => historyIndex.value >= history.value.length - 1);

  function addToHistory(text: string, imageUrl?: string) {
    // Deduplication: skip if text matches immediately previous entry
    const lastItem = history.value[history.value.length - 1];
    if (lastItem && lastItem.text === text && lastItem.imageUrl === imageUrl) {
      return;
    }

    const newItem: EditorHistoryItem = { text, imageUrl };

    // If we're in history view, truncate history from current position
    if (historyIndex.value !== -1) {
      history.value = history.value.slice(0, historyIndex.value + 1);
    }

    history.value.push(newItem);
    historyIndex.value = -1; // Return to editing mode
  }

  function navigatePrevious() {
    if (historyIndex.value === -1) {
      // From edit mode, go to last history item
      historyIndex.value = history.value.length - 1;
    } else if (historyIndex.value > 0) {
      historyIndex.value--;
    }
  }

  function navigateNext() {
    if (historyIndex.value < history.value.length - 1) {
      historyIndex.value++;
    } else if (historyIndex.value === history.value.length - 1) {
      // From last item, return to edit mode
      historyIndex.value = -1;
    }
  }

  function clearHistory() {
    history.value = [];
    historyIndex.value = -1;
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
    clearHistory,
  };
}
