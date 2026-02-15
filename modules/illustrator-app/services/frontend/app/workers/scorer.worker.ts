import type {
  LoadMessage,
  WorkerMessage,
  ExtractMessage,
  ExtractProgressPayload,
  TextEvaluationResult,
} from "~/types/text2features-worker";
import { FeatureService } from "~/utils/text2features";
import * as ort from "onnxruntime-web";
import catboostModel from "~/assets/data/models/catboost.onnx?url";

let featureService: FeatureService | null = null;
let ortSession: ort.InferenceSession | null = null;

async function loadModel(config: LoadMessage["payload"]): Promise<void> {
  try {
    featureService = new FeatureService(config.spacyCtxUrl);

    await featureService.init(config.cacheName, (progress) => {
      self.postMessage({
        type: "load-progress",
        payload: { progress },
      });
    });

    try {
      ortSession = await ort.InferenceSession.create(catboostModel, {
        executionProviders: config.providers?.length ? config.providers : ["wasm"],
      } as any);
    } catch (err) {
      self.postMessage({
        type: "error",
        payload: { message: `ONNX model load failed: ${(err as Error).message}` },
      });
      return;
    }

    self.postMessage({ type: "ready", payload: { success: true } });
  } catch (error) {
    self.postMessage({
      type: "error",
      payload: { message: (error as Error).message },
    });
  }
}

/**
 * Evaluate a batch of texts: extract features and run the ONNX session to get scores.
 */
async function evaluateSegments(data: ExtractMessage["payload"]): Promise<void> {
  if (!featureService) {
    self.postMessage({
      type: "error",
      payload: { message: "Pipeline not loaded" },
    });
    return;
  }

  if (!ortSession) {
    self.postMessage({
      type: "error",
      payload: { message: "ONNX session not loaded" },
    });
    return;
  }

  const { texts, batchSize } = data;
  const totalBatches = Math.ceil(texts.length / batchSize);

  try {
    const batches = Array.from({ length: totalBatches }, (_, i) => texts.slice(i * batchSize, (i + 1) * batchSize));

    // Do not await here - we want to start preloading while we process batches sequentially
    featureService.getFeaturesAllBatchPreload(batches);

    for (let i = 0; i < texts.length; i += batchSize) {
      const batchTexts = texts.slice(i, i + batchSize);

      // Extract features for each text in the batch
      const featureArrays: Float32Array[] = (await featureService.getFeatures(batchTexts)).map((featVec) => new Float32Array(featVec));

      const batchCount = featureArrays.length;
      const featureDim = featureArrays[0].length;

      // Concatenate into single Float32Array of shape [batchCount, featureDim]
      const concat = new Float32Array(batchCount * featureDim);
      for (let r = 0; r < batchCount; r++) {
        concat.set(featureArrays[r], r * featureDim);
      }

      const inputName = (ortSession as any).inputNames?.[0] as string;
      const outputName = (ortSession as any).outputNames?.[0] as string;

      const feeds: Record<string, ort.Tensor> = {};
      feeds[inputName] = new ort.Tensor("float32", concat, [batchCount, featureDim]);

      const outputs = await ortSession.run(feeds as any);
      const outTensor = outputs[outputName] as ort.Tensor;
      const scoresArray = outTensor.data as Float32Array;

      // Support outputs shaped [batch] or [batch, 1]
      const results = batchTexts.map((text, idx) => {
        const score = scoresArray.length === batchCount ? scoresArray[idx] : scoresArray[idx * (scoresArray.length / batchCount)];
        return { text, score: score / 5 } as TextEvaluationResult;
      });

      const progressPayload: ExtractProgressPayload = {
        batchIndex: Math.floor(i / batchSize) + 1,
        totalBatches,
        results,
      };

      self.postMessage({ type: "progress", payload: progressPayload });
    }

    self.postMessage({ type: "complete", payload: { success: true } });
  } catch (error) {
    self.postMessage({
      type: "error",
      payload: { message: (error as Error).message + (error as Error).stack },
    });
  }
}

/**
 * Handle incoming worker messages.
 */
self.onmessage = async (event: MessageEvent<WorkerMessage>) => {
  switch (event.data.type) {
  case "init":
  case "load":
    await loadModel((event.data as LoadMessage).payload);
    break;
  case "evaluate":
    await evaluateSegments((event.data as ExtractMessage).payload);
    break;
  }
};
