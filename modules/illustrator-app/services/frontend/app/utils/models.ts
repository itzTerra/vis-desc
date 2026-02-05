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
    onSuccess: async (data: SegmentResponse, model, socket, scoreSegment) => {
      const { getOrLoadWorker, terminateWorker } = useModelLoader();
      const worker = await getOrLoadWorker(model as TModelInfo);
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
      terminateWorker(model as TModelInfo);
    },
    onCancel: (model) => {
      const { terminateWorker } = useModelLoader();
      terminateWorker(model as TModelInfo);
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
    onSuccess: async (data: SegmentResponse, model, socket, scoreSegment) => {
      const { getOrLoadWorker, terminateWorker } = useModelLoader();
      const worker = await getOrLoadWorker(model as TModelInfo);
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
      terminateWorker(model as TModelInfo);
    },
    onCancel: (model) => {
      const { terminateWorker } = useModelLoader();
      terminateWorker(model as TModelInfo);
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
];

export type ModelId = typeof MODELS[number]["id"];

export const getModelById = (id: ModelId): ModelInfo => {
  return MODELS.find(model => model.id === id)!;
};

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
