export interface LoadMessage {
  type: "load";
  payload: {
    huggingFaceId: string;
    cacheName?: string;
    modelUrl?: string; // Optional URL for the ONNX model, used in scorer.worker
    spacyCtxUrl: string;
  };
}

export interface ExtractMessage {
  type: "evaluate";
  payload: {
    texts: string[];
    batchSize: number;
  };
}

export interface WorkerMessage {
  type: "load" | "evaluate" | "progress" | "ready" | "complete" | "error";
  payload: any;
}

export interface FeatureExtractionResult {
  text: string;
  features: number[];
}

export interface ExtractProgressPayload {
  batchIndex: number;
  totalBatches: number;
  results: TextEvaluationResult[];
}

export type TextEvaluationResult = {
  text: string;
  score: number;
}
