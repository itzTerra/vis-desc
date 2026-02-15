import type { ExecutionProvider } from "booknlp-ts";
import type { HFPipelineConfig } from "~/types/cache";

export type ScorerType = "minilm_catboost" | "nli_roberta";

export interface LoadMessage {
  type: "load";
  payload: {
    scorerId: ScorerType;
  };
}

export interface NLILoadMessage extends LoadMessage {
  payload: {
    scorerId: "nli_roberta";
    pipelineConfig: HFPipelineConfig;
  };
}

export interface CatboostLoadMessage extends LoadMessage {
  payload: {
    scorerId: "minilm_catboost";
    featureServiceEmbeddingConfig: HFPipelineConfig;
    catboostProvider: ExecutionProvider;
    spacyCtxUrl: string;
  };
}

export interface EvaluateMessage {
  type: "evaluate";
  payload: {
    scorerId: ScorerType;
    texts?: string[];
    segments?: string[];
    candidateLabels?: string[];
    hypothesisTemplate?: string;
    batchSize: number;
  };
}

export interface WorkerMessage {
  type: "load" | "evaluate" | "progress" | "ready" | "complete" | "error";
  payload: any;
}

export interface TextEvaluationResult {
  text: string;
  score: number;
}

export interface ExtractProgressPayload {
  batchIndex: number;
  totalBatches: number;
  results: TextEvaluationResult[];
}
