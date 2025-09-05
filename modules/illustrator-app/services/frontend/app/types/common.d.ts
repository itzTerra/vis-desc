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
  text: string;
  polygons: Record<number, number[][]>; // page -> list of polygons, each polygon a list of [x,y]
  score?: number; // may be attached later when scoring arrives
  imageLoading?: boolean;
  imageUrl?: string;
};

