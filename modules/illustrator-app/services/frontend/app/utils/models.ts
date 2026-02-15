import { pipeline, env } from "@huggingface/transformers";
import { FeatureService } from "~/utils/text2features";
import type { Downloadable, ModelGroup, ScorerStage, ScorerProgress, HFPipelineConfig } from "~/types/cache";
import type { Segment } from "~/types/common";
import type { paths } from "~/types/schema";
import type { EvaluateMessage, CatboostLoadMessage, NLILoadMessage } from "~/types/scorer-worker";
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
  spacyCtxUrl: keyof paths = "/api/spacy-ctx";
  embeddingPipelineConfig: HFPipelineConfig = {
    model: "Xenova/all-MiniLM-L6-v2",
    type: "feature-extraction",
    dtype: "fp32",
    device: "wasm",
  };

  async download(onProgress: (progress: number) => void): Promise<void> {
    const featureService = new FeatureService(this.spacyCtxUrl);
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

  constructor(
    id: string,
    label: string,
    description: string,
    speed: number,
    quality: number,
    disabled: boolean = false
  ) {
    this.id = id;
    this.label = label;
    this.description = description;
    this.speed = speed;
    this.quality = quality;
    this.disabled = disabled;
  }

  get stages(): ScorerStage[] {
    return [
      { label: "Initializing..." },
      { label: "Scoring..." },
    ];
  }

  protected getExecutionProvider(): ExecutionProvider {
    try {
      return JSON.parse(localStorage.getItem("onnx_providers") ?? "{}")[this.id] || "wasm";
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
  private worker: Worker | null = null;
  private scoring = false;

  constructor() {
    super(
      "minilm_catboost",
      "MiniLM-CatBoost",
      "Fast local inference with feature extraction",
      4,
      4,
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

      if (!this.worker) {
        onProgress({
          stage: "Initializing...",
          progress: 0,
        });

        this.worker = new Worker(new URL("~/workers/scorer.worker.ts", import.meta.url), {
          type: "module",
        });

        const provider = this.getExecutionProvider();
        await new Promise<void>((resolve, reject) => {
          const handler = (event: MessageEvent) => {
            if (event.data.type === "ready") {
              resolve();
            } else if (event.data.type === "error") {
              reject(new Error(event.data.payload.message));
            }
          };

          this.worker!.addEventListener("message", handler, { once: true });
          const loadMsg: CatboostLoadMessage = {
            type: "load",
            payload: {
              scorerId: "minilm_catboost",
              spacyCtxUrl: sharedFeatureServiceDownloadable.spacyCtxUrl,
              catboostProvider: provider,
              featureServiceEmbeddingConfig: minilmDownloadable.pipelineConfig,
            },
          };
          this.worker!.postMessage(loadMsg);
        });
      }

      return new Promise<Segment[]>((resolve, reject) => {
        const results: Segment[] = [];

        const handler = (event: MessageEvent) => {
          const { type, payload } = event.data;

          if (type === "progress") {
            onProgress({
              stage: "Scoring...",
              progress: ((payload.batchIndex / payload.totalBatches) * 100),
              eta: payload.eta,
            });

            for (const result of payload.results) {
              results.push(result);
            }
          } else if (type === "complete") {
            this.worker!.removeEventListener("message", handler);
            resolve(results);
          } else if (type === "error") {
            this.worker!.removeEventListener("message", handler);
            reject(new Error(payload.message));
          }
        };

        this.worker!.addEventListener("message", handler);
        const evalMsg: EvaluateMessage = {
          type: "evaluate",
          payload: {
            scorerId: "minilm_catboost",
            texts,
            batchSize: 16,
          },
        };
        this.worker!.postMessage(evalMsg);
      });
    } finally {
      this.scoring = false;
    }
  }
}

class NLIRobertaScorer extends Scorer {
  private worker: Worker | null = null;
  private scoring = false;
  private candidateLabels: string[] = ["not detailed", "detailed"];
  private hypothesisTemplate: string = "This text is {} in terms of visual details of characters, setting, or environment.";

  constructor() {
    super(
      "nli_roberta",
      "NLI-RoBERTa",
      "Zero-shot classification with NLI",
      3,
      3,
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

      if (!this.worker) {
        onProgress({
          stage: "Initializing...",
          progress: 0,
        });

        this.worker = new Worker(new URL("~/workers/scorer.worker.ts", import.meta.url), {
          type: "module",
        });

        const provider = this.getExecutionProvider();
        await new Promise<void>((resolve, reject) => {
          const handler = (event: MessageEvent) => {
            if (event.data.type === "ready") {
              resolve();
            } else if (event.data.type === "error") {
              reject(new Error(event.data.payload.message));
            } else if (event.data.type === "progress") {
              onProgress({
                stage: "Initializing...",
                progress: (event.data.payload.progress ?? 0) * 100,
              });
            }
          };

          this.worker!.addEventListener("message", handler, { once: true });
          const loadMsg: NLILoadMessage = {
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
          this.worker!.postMessage(loadMsg);
        });
      }

      return new Promise<Segment[]>((resolve, reject) => {
        const results: Segment[] = [];

        const handler = (event: MessageEvent) => {
          const { type, payload } = event.data;

          if (type === "progress") {
            onProgress({
              stage: "Scoring...",
              progress: ((payload.batchIndex / payload.totalBatches) * 100),
              eta: payload.eta,
            });

            for (const result of payload.results) {
              results.push(result);
            }
          } else if (type === "complete") {
            this.worker!.removeEventListener("message", handler);
            resolve(results);
          } else if (type === "error") {
            this.worker!.removeEventListener("message", handler);
            reject(new Error(payload.message));
          }
        };

        this.worker!.addEventListener("message", handler);
        const evalMsg: EvaluateMessage = {
          type: "evaluate",
          payload: {
            scorerId: "nli_roberta",
            segments: segments.map((s: any) => s.text),
            candidateLabels: this.candidateLabels,
            hypothesisTemplate: this.hypothesisTemplate,
            batchSize: 16,
          },
        };
        this.worker!.postMessage(evalMsg);
      });
    } finally {
      this.scoring = false;
    }
  }
}

class RandomScorer extends Scorer {
  private socket: any = null;
  private scoring = false;

  constructor() {
    super(
      "random",
      "Random [debug]",
      "Server-side random scoring (demo)",
      5,
      0,
      false
    );
  }

  async score(
    data: any,
    onProgress: (progress: ScorerProgress) => void,
    socket?: any
  ): Promise<Segment[]> {
    if (this.scoring) {
      throw new Error("Scoring already in progress");
    }
    this.scoring = true;
    try {
      const segments: Segment[] = data.segments || [];

      onProgress({
        stage: "Initializing...",
        progress: 0,
      });

      if (!socket) {
        throw new Error("WebSocket connection required for RandomScorer");
      }

      this.socket = socket;

      return new Promise<Segment[]>((resolve, reject) => {
        const results: Segment[] = [];

        const progressHandler = (payload: any) => {
          onProgress({
            stage: "Scoring...",
            progress: (payload.progress ?? 0) * 100,
            eta: payload.eta,
          });
        };

        const completeHandler = (payload: any) => {
          this.socket.off("score:progress", progressHandler);
          this.socket.off("score:complete", completeHandler);
          this.socket.off("score:error", errorHandler);

          for (const result of payload.results || []) {
            results.push(result);
          }

          resolve(results);
        };

        const errorHandler = (payload: any) => {
          this.socket.off("score:progress", progressHandler);
          this.socket.off("score:complete", completeHandler);
          this.socket.off("score:error", errorHandler);

          reject(new Error(payload.message || "Unknown error"));
        };

        this.socket.on("score:progress", progressHandler);
        this.socket.on("score:complete", completeHandler);
        this.socket.on("score:error", errorHandler);

        this.socket.emit("score:request", {
          scorer_id: this.id,
          segments: segments.map((s: any) => ({ text: s.text })),
        });
      });
    } finally {
      this.scoring = false;
    }
  }
}

const sharedFeatureServiceDownloadable = new FeatureServiceDownloadable(
  "feature_service",
  "Feature Service",
  65
);
const minilmDownloadable = new HuggingFacePipelineDownloadable(
  "minilm_pipeline",
  "MiniLM Pipeline",
  90,
  { model: "Xenova/all-MiniLM-L6-v2", type: "feature-extraction", subfolder: "onnx", dtype: "fp32" }
);
const nliRobertaDownloadable = new HuggingFacePipelineDownloadable(
  "roberta_pipeline",
  "RoBERTa Pipeline",
  125,
  { model: "richardr1126/roberta-base-zeroshot-v2.0-c-ONNX", type: "zero-shot-classification", dtype: "q8" }
);

export const MODEL_GROUPS: ModelGroup[] = [
  {
    id: "minilm_catboost",
    name: "MiniLM-CatBoost",
    downloadables: [
      sharedFeatureServiceDownloadable,
      minilmDownloadable,
      new OnnxDownloadable(
        "catboost_model",
        "CatBoost Model",
        0,
        "/assets/data/models/catboost.onnx"
      ),
    ],
  },
  {
    id: "nli_roberta",
    name: "NLI-RoBERTa",
    downloadables: [
      sharedFeatureServiceDownloadable,
      nliRobertaDownloadable,
    ],
  },
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
