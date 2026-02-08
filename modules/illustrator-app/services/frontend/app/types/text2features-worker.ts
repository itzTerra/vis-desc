export interface LoadMessage {
  type: "load";
  payload: {
    huggingFaceId: string;
  };
}

export interface ExtractMessage {
  type: "extract";
  payload: {
    texts: string[];
    batchSize: number;
  };
}

export interface WorkerMessage {
  type: "load" | "extract" | "progress" | "ready" | "complete" | "error";
  payload: any;
}

export interface FeatureExtractionResult {
  text: string;
  features: number[];
  embeddings: number[];
}

export interface ExtractProgressPayload {
  batchIndex: number;
  totalBatches: number;
  results: FeatureExtractionResult[];
}

export interface BookNLPContext {
  tokens: TokenData[];
  entities: EntityData[];
  supersense: Array<[number, number, string, string]>;
  sentences: SentenceData[];
}

export interface TokenData {
  text: string;
  itext: string;
  pos: string;
  finePOS: string;
  lemma: string;
  sentenceId: number;
  withinSentenceId: number;
  event: boolean;
  isStop: boolean;
  likeNum: boolean;
  morphTense: string | null;
}

export interface EntityData {
  startToken: number;
  endToken: number;
  cat: string;
  text: string;
  coref: number;
  prop: string;
}

export interface SentenceToken {
  text: string;
  pos: string;
  dep: string;
  children: SentenceToken[];
}

export interface SentenceData {
  root: SentenceToken;
}
