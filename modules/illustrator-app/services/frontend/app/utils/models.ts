import { pipeline, env } from "@huggingface/transformers";
import { FeatureService } from "~/utils/text2features";
import type { Downloadable, ModelGroup, ScorerStage, ScorerProgress, HFPipelineConfig } from "~/types/cache";
import type { Segment } from "~/types/common";
import type { SendMessage, RecvMessage } from "~/types/scorer-worker";
import type { ExecutionProvider } from "booknlp-ts";

env.allowLocalModels = false;
env.allowRemoteModels = true;
env.useBrowserCache = true;

abstract class BaseDownloadable implements Downloadable {
  id: string;
  label: string;
  sizeMb: number;

  constructor(id: string, label: string, sizeMb: number) {
    this.id = id;
    this.label = label;
    this.sizeMb = sizeMb;
  }

  abstract download(onProgress: (progress: number) => void): Promise<void>;
}

class FeatureServiceDownloadable extends BaseDownloadable {
  embeddingPipelineConfig: HFPipelineConfig = {
    model: "Xenova/all-MiniLM-L6-v2",
    type: "feature-extraction",
    dtype: "fp32",
    device: "wasm",
  };

  async download(onProgress: (progress: number) => void): Promise<void> {
    const featureService = new FeatureService(""); // spacyCtxUrl is not needed for downloading to cache
    await featureService.init(this.embeddingPipelineConfig, { progressCallback: (progress) => {
      onProgress(progress * 100);
    } });
  }
}

class HuggingFacePipelineDownloadable extends BaseDownloadable {
  pipelineConfig: HFPipelineConfig;

  constructor(
    id: string,
    label: string,
    sizeMb: number,
    pipelineConfig: HFPipelineConfig
  ) {
    super(id, label, sizeMb);
    this.pipelineConfig = pipelineConfig;
  }

  async download(onProgress: (progress: number) => void): Promise<void> {
    await pipeline(this.pipelineConfig.type, this.pipelineConfig.model, {
      subfolder: this.pipelineConfig.subfolder || "",
      progress_callback: (data: any) => {
        if (data.progress !== undefined && data.file.endsWith(".onnx")) {
          onProgress(data.progress * 100);
        }
      },
      dtype: this.pipelineConfig.dtype,
      device: this.pipelineConfig.device as any,
    });
  }
}

class OnnxDownloadable extends BaseDownloadable {
  modelUrl: string;

  constructor(
    id: string,
    label: string,
    sizeMb: number,
    modelUrl: string
  ) {
    super(id, label, sizeMb);
    this.modelUrl = modelUrl;
  }

  async download(onProgress: (progress: number) => void): Promise<void> {
    const response = await fetch(this.modelUrl);
    if (!response.body) {
      throw new Error(`Failed to fetch model from ${this.modelUrl}`);
    }

    const total = parseInt(response.headers.get("content-length") || "0", 10);
    let loaded = 0;

    const reader = response.body.getReader();
    const chunks: Uint8Array[] = [];

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      chunks.push(value);
      loaded += value.length;

      if (total > 0) {
        onProgress((loaded / total) * 100);
      }
    }

    const _buffer = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) {
      _buffer.set(chunk, offset);
      offset += chunk.length;
    }
  }
}

abstract class Scorer {
  id: string;
  label: string;
  description: string;
  speed: number;
  quality: number;
  disabled: boolean;
  socketBased: boolean;

  constructor(
    id: string,
    label: string,
    description: string,
    speed: number,
    quality: number,
    disabled: boolean = false,
    socketBased: boolean = false
  ) {
    this.id = id;
    this.label = label;
    this.description = description;
    this.speed = speed;
    this.quality = quality;
    this.disabled = disabled;
    this.socketBased = socketBased;
  }

  get stages(): ScorerStage[] {
    return [
      { label: "Initializing..." },
      { label: "Scoring..." },
    ];
  }

  protected getExecutionProvider(): ExecutionProvider {
    try {
      return JSON.parse(localStorage.getItem("onnx_provider") ?? "{}")[this.id] || "wasm";
    } catch {
      return "wasm";
    }
  }

  dispose(): void {
    if ((this as any).worker) {
      (this as any).worker.terminate();
      (this as any).worker = null;
    }
  }

  abstract score(
    data: any,
    onProgress: (progress: ScorerProgress) => void,
    socket?: any
  ): Promise<Segment[]>;
}

class MiniLMCatBoostScorer extends Scorer {
  private isLoaded = false;
  private scoring = false;
  private batchSize = 16;
  private spacyCtxUrl?: string;

  constructor() {
    super(
      "minilm_catboost",
      "CatBoost",
      "Fast local inference with feature extraction",
      4,
      4,
      false,
      false
    );
  }

  async score(
    data: any,
    onProgress: (progress: ScorerProgress) => void
  ): Promise<Segment[]> {
    if (this.scoring) {
      throw new Error("Scoring already in progress");
    }
    this.scoring = true;
    try {
      const segments: Segment[] = data.segments || [];
      const texts = segments.map((s: any) => s.text);

      const scorerWorker = getScorerWorker();

      if (!this.isLoaded) {
        if (!this.spacyCtxUrl) {
          this.spacyCtxUrl = new URL("/api/spacy-ctx", useRuntimeConfig().public.apiBaseUrl).toString();
        }
        onProgress({
          stage: "Initializing...",
          startedAt: Date.now(),
        });
        this.isLoaded = true;

        const provider = this.getExecutionProvider();
        await new Promise<void>((resolve, reject) => {
          const handler = (event: MessageEvent<RecvMessage>) => {
            if (event.data.type === "ready") {
              resolve();
            } else if (event.data.type === "error") {
              reject(new Error(event.data.payload.message));
              this.isLoaded = false;
            }
          };

          scorerWorker.addEventListener("message", handler, { once: true });
          const loadMsg: SendMessage = {
            type: "load",
            payload: {
              scorerId: "minilm_catboost",
              spacyCtxUrl: this.spacyCtxUrl!,
              provider,
              featureServiceEmbeddingConfig: DOWNLOADABLES.featureService.embeddingPipelineConfig,
              texts,
              batchSize: this.batchSize,
            },
          };
          scorerWorker.postMessage(loadMsg);
        });
      }

      const scoringStartTime = Date.now();
      onProgress({
        stage: "Scoring...",
        startedAt: scoringStartTime,
        results: [],
      });

      return new Promise<Segment[]>((resolve, reject) => {
        const results: Segment[] = [];

        const handler = (event: MessageEvent<RecvMessage>) => {
          const { type, payload } = event.data as RecvMessage;

          if (type === "progress") {
            onProgress({
              stage: "Scoring...",
              startedAt: scoringStartTime,
              results: payload.results,
            });

            for (const result of payload.results) {
              results.push(result);
            }
          } else if (type === "complete") {
            scorerWorker.removeEventListener("message", handler);
            resolve(results);
          } else if (type === "error") {
            scorerWorker.removeEventListener("message", handler);
            reject(new Error(payload.message));
          }
        };

        scorerWorker.addEventListener("message", handler);
        const evalMsg: SendMessage = {
          type: "evaluate",
          payload: {
            scorerId: "minilm_catboost",
            texts,
            batchSize: this.batchSize,
          },
        };
        scorerWorker.postMessage(evalMsg);
      });
    } catch (e) {
      useNotifier().error("An error occurred during scoring: " + (e as Error).message);
      console.error("Error in MiniLMCatBoostScorer:", e);
      return [];
    } finally {
      this.scoring = false;
    }
  }
}

class NLIRobertaScorer extends Scorer {
  private isLoaded = false;
  private scoring = false;
  private candidateLabels: string[] = ["not detailed", "detailed"];
  private hypothesisTemplate: string = "This text is {} in terms of visual details of characters, setting, or environment.";
  private batchSize = 16;

  constructor() {
    super(
      "nli_roberta",
      "NLI-RoBERTa",
      "Zero-shot classification with NLI",
      3,
      3,
      false,
      false
    );
  }

  async score(
    data: any,
    onProgress: (progress: ScorerProgress) => void
  ): Promise<Segment[]> {
    if (this.scoring) {
      throw new Error("Scoring already in progress");
    }
    this.scoring = true;
    try {
      const segments: Segment[] = data.segments || [];

      const scorerWorker = getScorerWorker();

      if (!this.isLoaded) {
        onProgress({
          stage: "Initializing...",
          startedAt: Date.now(),
        });
        this.isLoaded = true;

        const provider = this.getExecutionProvider();
        await new Promise<void>((resolve, reject) => {
          const handler = (event: MessageEvent<RecvMessage>) => {
            if (event.data.type === "ready") {
              resolve();
            } else if (event.data.type === "error") {
              reject(new Error(event.data.payload.message));
              this.isLoaded = false;
            }
          };

          scorerWorker.addEventListener("message", handler, { once: true });
          const loadMsg: SendMessage = {
            type: "load",
            payload: {
              scorerId: "nli_roberta",
              pipelineConfig: {
                model: "richardr1126/roberta-base-zeroshot-v2.0-c-ONNX",
                type: "zero-shot-classification",
                dtype: "q8",
                device: provider,
              }
            },
          };
          scorerWorker.postMessage(loadMsg);
        });
      }

      const scoringStartTime = Date.now();
      onProgress({
        stage: "Scoring...",
        startedAt: scoringStartTime,
        results: [],
      });

      return new Promise<Segment[]>((resolve, reject) => {
        const results: Segment[] = [];

        const handler = (event: MessageEvent<RecvMessage>) => {
          const { type, payload } = event.data as RecvMessage;

          if (type === "progress") {
            onProgress({
              stage: "Scoring...",
              startedAt: scoringStartTime,
              results: payload.results,
            });

            for (const result of payload.results) {
              results.push(result);
            }
          } else if (type === "complete") {
            scorerWorker.removeEventListener("message", handler);
            resolve(results);
          } else if (type === "error") {
            scorerWorker.removeEventListener("message", handler);
            reject(new Error(payload.message));
          }
        };

        scorerWorker.addEventListener("message", handler);
        const evalMsg: SendMessage = {
          type: "evaluate",
          payload: {
            scorerId: "nli_roberta",
            texts: segments.map((s: any) => s.text),
            candidateLabels: this.candidateLabels,
            hypothesisTemplate: this.hypothesisTemplate,
            batchSize: this.batchSize,
          },
        };
        scorerWorker.postMessage(evalMsg);
      });
    } finally {
      this.scoring = false;
    }
  }
}

class RandomScorer extends Scorer {
  constructor() {
    super(
      "random",
      "Random",
      "Server-side random scoring (demo)",
      5,
      0,
      false,
      true
    );
  }

  async score(): Promise<Segment[]> {
    return [];
  }
}

export const DOWNLOADABLES = {
  featureService: new FeatureServiceDownloadable(
    "feature_service",
    "Feature Service",
    155
  ),
  nliRoberta: new HuggingFacePipelineDownloadable(
    "roberta_pipeline",
    "RoBERTa Pipeline",
    125,
    { model: "richardr1126/roberta-base-zeroshot-v2.0-c-ONNX", type: "zero-shot-classification", dtype: "q8" }
  ),
  catboost: new OnnxDownloadable(
    "catboost_model",
    "CatBoost Model",
    0,
    "/assets/data/models/catboost.onnx"
  ),
};

export const MODEL_GROUPS: ModelGroup[] = [
  {
    id: "minilm_catboost",
    name: "CatBoost",
    downloadables: [
      DOWNLOADABLES.featureService,
      DOWNLOADABLES.catboost,
    ],
  },
  {
    id: "nli_roberta",
    name: "NLI-RoBERTa",
    downloadables: [
      DOWNLOADABLES.nliRoberta,
    ],
  }
];

export const SCORERS: Scorer[] = [
  new MiniLMCatBoostScorer(),
  new NLIRobertaScorer(),
  new RandomScorer(),
];

export type WordNetInfo = {
  id: string;
  downloadUrl: string;
};

export const WORDNETS: WordNetInfo[] = [
  {
    id: "oewn",
    downloadUrl: "/english-wordnet-2025-json.zip",
  },
];

export { Scorer, FeatureServiceDownloadable, HuggingFacePipelineDownloadable, OnnxDownloadable };
