export type AppReadyState = Reactive<{
  apiReady: boolean;
  apiError: string | null;
  scorerWorkerReady: boolean;
}>;

export type Segment = { text: string, score: number };

export type ActionState = "idle" | "queued" | "processing";

export type TextMapping = {
  start: number;
  end: number;
  node: Node;
}

export type ParagraphPosition = {
  startPosition: number;
  endPosition: number;
}

export type Highlight = {
  id: number;
  text: string;
  // page->polygons map
  polygons: Record<number, number[][]>;
  score?: number; // may be attached later when scoring arrives
  score_received_at?: number;
};

export type EditorHistoryItem = {
  text: string;
  imageUrl?: string;
};

export type EditorState = {
  highlightId: number;
  isExpanded: boolean;
  currentPrompt: string;
  imageUrl: string | null;
  history: EditorHistoryItem[];
  historyIndex: number;
  enhanceLoading: boolean;
  generateLoading: boolean;
};

export type EditorImageState = {
  highlightId: number;
  imageUrl: string | null;
  hasImage: boolean;
};

/**
 * Props for the HeatmapViewer component.
 */
export type HeatmapViewerProps = {
  highlights: Highlight[];
  currentPage: number;
  pageAspectRatio: number;
  pageRefs: Element[];
  editorStates: EditorImageState[];
};

export interface AlertOptions {
  type: "info" | "success" | "warning" | "error";
  message: string;
  duration?: number;
}

// progress: 0 to 1
export type ProgressCallback = (progress: number) => void;
