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
  text?: string;
  x: number;
  y: number;
  width: number;
  height: number;
  score?: number;
  imageLoading?: boolean;
  imageUrl?: string;
};

// Backend aligned segment page span with polygon geometry (normalized coordinates 0-1)
export type SegmentPageSpan = {
  page: number;
  start: number;
  end: number;
  polygons: number[][][]; // list of polygons, each polygon a list of [x,y]
};

export type AlignedSegment = {
  text: string;
  found: boolean;
  page_spans: SegmentPageSpan[];
  score?: number; // may be attached later when scoring arrives
};

