import type { ExecutionProvider } from "booknlp-ts";
import type { HFPipelineConfig } from "~/types/cache";

export type ScorerType = "minilm_catboost" | "nli_roberta";

export interface TextEvaluationResult {
  text: string;
  score: number;
}

export interface ExtractProgressPayload {
  batchIndex: number;
  totalBatches: number;
  results: TextEvaluationResult[];
}

// --- Messages sent TO the worker ---
export type NLILoadPayload = {
  scorerId: "nli_roberta";
  pipelineConfig: HFPipelineConfig;
};

export type CatboostLoadPayload = {
  scorerId: "minilm_catboost";
  featureServiceEmbeddingConfig: HFPipelineConfig;
  // For both BookNLP and Catboost
  provider: ExecutionProvider;
  feBaseUrl: string;
  spacyCtxUrl: string;
  // For SpaCy context preload
  texts: string[];
  batchSize: number;
};

export type LoadPayload = NLILoadPayload | CatboostLoadPayload;

export type LoadMessage = {
  type: "load";
  payload: LoadPayload;
};

export type NLIEvaluatePayload = {
  scorerId: "nli_roberta";
  texts: string[];
  candidateLabels: string[];
  hypothesisTemplate: string;
  batchSize: number;
};

export type CatboostEvaluatePayload = {
  scorerId: "minilm_catboost";
  texts: string[];
  batchSize: number;
};

export type EvaluatePayload = NLIEvaluatePayload | CatboostEvaluatePayload;

export type EvaluateMessage = {
  type: "evaluate";
  payload: EvaluatePayload;
};

type PauseMessage = { type: "pause" };
type ContinueMessage = { type: "continue" };

export type SendMessage = LoadMessage | EvaluateMessage | PauseMessage | ContinueMessage;

// --- Messages received FROM the worker ---
export type ReadyMessage = { type: "ready"; payload: { success: boolean } };
export type ProgressEvalMessage = { type: "progress"; payload: ExtractProgressPayload };
export type CompleteMessage = { type: "complete"; payload: { success: boolean } };
export type ErrorMessage = { type: "error"; payload: { message: string; stack?: string } };

export type RecvMessage = ReadyMessage | ProgressEvalMessage | CompleteMessage | ErrorMessage;

export type { LoadMessage as _LoadMessage, EvaluateMessage as _EvaluateMessage };
