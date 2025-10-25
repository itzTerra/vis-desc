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
  imageLoading?: boolean;
  imageUrl?: string;
};
