/**
 * Download/cache related types for the cache and scorer system.
 */

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

// export interface Scorer {
//   id: string;
//   label: string;
//   description: string;
//   speed: number;
//   quality: number;
//   disabled: boolean;
//   stages: ScorerStage[];
//   score(
//     data: any,
//     onProgress: (progress: ScorerProgress) => void,
//     socket?: any
//   ): Promise<any[]>;
//   dispose(): void;
// }

export interface HFPipelineConfig {
  model: string;
  type: PipelineType;
  dtype: DataType;
  device?: ExecutionProvider;
  subfolder?: string;
}
