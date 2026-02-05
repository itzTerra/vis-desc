import type { PipelineType } from "@huggingface/transformers";

type MessageType = "load" | "evaluate" | "ready" | "progress" | "complete" | "error";

interface WorkerMessage {
  type: MessageType;
  payload?: any;
}

interface LoadMessage extends WorkerMessage {
  type: "load";
  payload: {
    huggingFaceId: string;
    modelFileName?: string;
    pipeline: PipelineType;
  };
}

interface EvaluateMessage extends WorkerMessage {
  type: "evaluate";
  payload: {
    segments: string[];
    candidateLabels: string[];
    hypothesisTemplate: string;
    batchSize: number;
  };
}
