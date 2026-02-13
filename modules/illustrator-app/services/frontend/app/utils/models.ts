import type { PipelineType } from "@huggingface/transformers";
import type { Segment } from "~/types/common";
import type { paths } from "~/types/schema";

type SegmentResponse = paths["/api/process/seg/pdf"]["post"]["responses"]["200"]["content"]["application/json"];

export type TransformersModelConfig = {
  huggingFaceId: string;
  modelFileName?: string;
  pipeline: PipelineType;
  sizeMb: number;
  workerName: string;
}

export type ModelInfo = {
  id: string;
  label: string;
  description: string;
  speed: number;
  quality: number;
  apiUrl?: keyof paths;
  onSuccess?: (data: SegmentResponse, model: ModelInfo, socket: any, scoreSegment: (s: Segment) => void) => void | Promise<void>;
  onCancel?: (model: ModelInfo) => void | Promise<void>;
  disabled?: boolean;
  transformersConfig?: TransformersModelConfig;
}

export type TModelInfo = ModelInfo & {
  transformersConfig: TransformersModelConfig;
};

export const MODELS: ModelInfo[] = [
  {
    id: "minilm_catboost",
    label: "MiniLM-CatBoost",
    speed: 4,
    quality: 3,
    disabled: false,
    description: "A fast model with medium quality for general use.",
    transformersConfig: {
      huggingFaceId: "richardr1126/roberta-base-zeroshot-v2.0-c-ONNX",
      modelFileName: "model_quantized.onnx",
      pipeline: "zero-shot-classification",
      sizeMb: 130,
      workerName: "booknlp",
    },
    apiUrl: "/api/process/seg/pdf",
    onSuccess: async (_data: SegmentResponse, model, _socket, _scoreSegment) => {
      const { getOrLoadWorker, terminateWorker } = useModelLoader();
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      const worker = await getOrLoadWorker(model.transformersConfig!);
      // const segments = (data as any).segments.map((seg: any) => seg.text);
      // const nliClassifier = NLI_CLASSIFIERS[model.id];
      // if (!nliClassifier) {
      //   return;
      // }
      // await nliClassifier.evaluateSegmentsWithWorker(worker, segments, (batchIndex, totalBatches, batchResults) => {
      //   console.log(`NLI batch ${batchIndex + 1}/${totalBatches} completed.`);
      //   for (const result of batchResults) {
      //     scoreSegment(result);
      //   }
      // });
      terminateWorker(model.transformersConfig!.huggingFaceId);
    },
    onCancel: (model) => {
      const { terminateWorker } = useModelLoader();
      terminateWorker(model.transformersConfig!.huggingFaceId);
    }
  },
  {
    id: "modernbert_finetuned",
    label: "ModernBERT-Finetuned",
    speed: 3,
    quality: 4,
    disabled: true,
    description: "Balanced model fine-tuned for better performance.",
  },
  {
    id: "nli_roberta",
    label: "NLI-RoBERTa",
    speed: 3,
    quality: 3,
    disabled: false,
    description: "Runs in-browser using transformers.js. High-quality zero-shot classification model.",
    transformersConfig: {
      huggingFaceId: "richardr1126/roberta-base-zeroshot-v2.0-c-ONNX",
      modelFileName: "model_quantized.onnx",
      pipeline: "zero-shot-classification",
      sizeMb: 130,
      workerName: "nli",
    },
    apiUrl: "/api/process/seg/pdf",
    onSuccess: async (data: SegmentResponse, model, _socket, scoreSegment) => {
      const { getOrLoadWorker, terminateWorker } = useModelLoader();
      const worker = await getOrLoadWorker(model.transformersConfig!);
      const segments = (data as any).segments.map((seg: any) => seg.text);
      const nliClassifier = NLI_CLASSIFIERS[model.id];
      if (!nliClassifier) {
        return;
      }
      await nliClassifier.evaluateSegmentsWithWorker(worker, segments, (batchIndex, totalBatches, batchResults) => {
        console.log(`NLI batch ${batchIndex + 1}/${totalBatches} completed.`);
        for (const result of batchResults) {
          scoreSegment(result);
        }
      });
      terminateWorker(model.transformersConfig!.huggingFaceId);
    },
    onCancel: (model) => {
      const { terminateWorker } = useModelLoader();
      terminateWorker(model.transformersConfig!.huggingFaceId);
    }
  },
  {
    id: "random",
    label: "Random [debug]",
    speed: 5,
    quality: 0,
    disabled: false,
    description: "Instant random selection for debugging.",
  },
  {
    id: "xenova_minilm",
    label: "Xenova MiniLM",
    speed: 4,
    quality: 3,
    disabled: false,
    description: "Xenova all-MiniLM-L6-v2 feature-extraction model (browser-friendly).",
    transformersConfig: {
      huggingFaceId: "Xenova/all-MiniLM-L6-v2",
      pipeline: "feature-extraction",
      sizeMb: 30,
      workerName: "text2features",
    },
  },
];

export type ModelId = typeof MODELS[number]["id"];

export const getModelById = (id: ModelId): ModelInfo => {
  return MODELS.find(model => model.id === id)!;
};

export type WordNetInfo = {
  id: string;
  label: string;
  description?: string;
  sizeMb: number;
  downloadUrl: string;
}

export const WORDNETS: WordNetInfo[] = [
  {
    id: "oewn",
    label: "Open English WordNet (oewn:2025)",
    description: "English WordNet JSON dump (zipped) for polysemy counting.",
    sizeMb: 180,
    downloadUrl: "https://en-word.net/static/english-wordnet-2025-json.zip",
  },
];

export class NLIZeroshotClassifier {
  private candidateLabels: string[];
  private hypothesisTemplate: string;
  private batchSize: number;

  constructor(
    candidateLabels: string[],
    hypothesisTemplate: string,
    batchSize: number = 16,
  ) {
    this.candidateLabels = candidateLabels;
    this.hypothesisTemplate = hypothesisTemplate;
    this.batchSize = batchSize;
  }

  async evaluateSegmentsWithWorker(
    worker: Worker,
    segments: string[],
    onBatchComplete?: (batchIndex: number, totalBatches: number, batchResults: Array<Segment>) => void
  ): Promise<void> {
    return new Promise((resolve, reject) => {
      const messageHandler = (event: MessageEvent) => {
        const { type, payload } = event.data;

        switch (type) {
        case "progress":
          onBatchComplete?.(payload.batchIndex, payload.totalBatches, payload.results);
          break;
        case "complete":
          worker.removeEventListener("message", messageHandler);
          resolve();
          break;
        case "error":
          worker.removeEventListener("message", messageHandler);
          reject(new Error(payload.message));
          break;
        }
      };

      worker.addEventListener("message", messageHandler);

      worker.postMessage({
        type: "evaluate",
        payload: {
          segments,
          candidateLabels: this.candidateLabels,
          hypothesisTemplate: this.hypothesisTemplate,
          batchSize: this.batchSize,
        }
      });
    });
  }
}

const NLI_CLASSIFIERS: Record<ModelId, NLIZeroshotClassifier> = {
  "nli_roberta": new NLIZeroshotClassifier(["not detailed", "detailed"], "This text is {} in terms of visual details of characters, setting, or environment."),
};

export enum PromptBridgeMode {
  Expand = "Expand the prompt.",
  Compress = "Compress the prompt into keyword format."
}

class PromptEnhancer {
  id: string;
  transformersConfig: TransformersModelConfig;

  constructor(id: string, transformersConfig: TransformersModelConfig) {
    this.id = id;
    this.transformersConfig = transformersConfig;
  }

  async enhance(text: string, mode: PromptBridgeMode = PromptBridgeMode.Expand): Promise<string> {
    const { getOrLoadWorker } = useModelLoader();
    const worker = await getOrLoadWorker(this.transformersConfig);

    return new Promise((resolve, reject) => {
      const messageHandler = (event: MessageEvent) => {
        const { type, payload } = event.data;

        switch (type) {
        case "complete":
          worker.removeEventListener("message", messageHandler);
          resolve(payload.enhancedText || text);
          break;
        case "error":
          worker.removeEventListener("message", messageHandler);
          reject(new Error(payload.message));
          break;
        }
      };

      worker.addEventListener("message", messageHandler);

      worker.postMessage({
        type: "enhance",
        payload: {
          text,
          mode,
        }
      });
    });
  }

  dispose(): void {
    const { terminateWorker } = useModelLoader();
    terminateWorker(this.transformersConfig.huggingFaceId);
  }
}

export const promptEnhancer = new PromptEnhancer(
  "prompt-bridge",
  {
    huggingFaceId: "Terraa/prompt-enhancer-gemma-3-270m-it-int4-ONNX",
    pipeline: "text-generation",
    sizeMb: 1100,
    workerName: "prompt-enhancer",
  }
);

/**
 * Run the `text2features` worker to extract features for the given texts.
 * Currently the results are ignored (placeholder for future handling).
 */
export async function runText2FeaturesOnTexts(texts: string[], batchSize = 8): Promise<void> {
  const model = getModelById("xenova_minilm");
  if (!model.transformersConfig) return;

  const { getOrLoadWorker, terminateWorker } = useModelLoader();
  const worker = await getOrLoadWorker(model.transformersConfig);

  return new Promise((resolve, reject) => {
    const handler = (event: MessageEvent) => {
      const { type, payload } = event.data;
      switch (type) {
      case "progress":
        // placeholder: could inspect payload.results
        break;
      case "complete":
        worker.removeEventListener("message", handler);
        // For now do nothing with results
        terminateWorker(model.transformersConfig!.huggingFaceId);
        resolve();
        break;
      case "error":
        worker.removeEventListener("message", handler);
        terminateWorker(model.transformersConfig!.huggingFaceId);
        reject(new Error(payload.message));
        break;
      }
    };

    worker.addEventListener("message", handler);

    worker.postMessage({
      type: "extract",
      payload: {
        texts,
        batchSize,
      }
    });
  });
}
