import type { DataType, PipelineType } from "@huggingface/transformers";
import type { ExecutionProvider } from "booknlp-ts";
import type { Segment } from "~/types/common";

export interface Downloadable {
  id: string;
  label: string;
  sizeMb: number;
  download(onProgress: (progress: number) => void): Promise<void>;
}

export interface ModelGroup {
  id: string;
  name: string;
  downloadables: Downloadable[];
  supportsGpu?: boolean;
}

export interface ScorerStage {
  label: string;
  isMain?: boolean;
}

export interface ScorerProgress {
  stage: string;
  startedAt: number;
  results?: Segment[];
}

export interface HFPipelineConfig {
  model: string;
  type: PipelineType;
  dtype: DataType;
  device?: ExecutionProvider;
  subfolder?: string;
}
