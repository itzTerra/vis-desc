export type Segment = { text: string, score: number };

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
  polygons: Record<number, number[][]>; // page -> list of polygons, each polygon a list of [x,y]
  score?: number; // may be attached later when scoring arrives
  score_received_at?: number;
};

export type EditorHistoryItem = {
  text: string;
  imageUrl?: string;
  imageBlob?: Blob;
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
