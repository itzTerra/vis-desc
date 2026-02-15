import { pipeline, env, type Pipeline } from "@huggingface/transformers";
import type { EvaluateMessage, LoadMessage, WorkerMessage, ScorerType, CatboostLoadMessage, NLILoadMessage } from "~/types/scorer-worker";
import { FeatureService } from "~/utils/text2features";
import * as ort from "onnxruntime-web";
import catboostModel from "~/assets/data/models/catboost.onnx?url";

env.allowLocalModels = false;
env.allowRemoteModels = true;
env.useBrowserCache = true;

interface ScorerState {
  type: ScorerType;
  nliPipeline?: Pipeline | null;
  featureService?: FeatureService | null;
  catboostOrt?: ort.InferenceSession | null;
}

const scorers = new Map<ScorerType, ScorerState>();

function probsToScore(probs: number[]): number {
  const k = probs.length;
  if (k === 0) {
    return 0;
  }

  const total = probs.reduce((sum, p) => sum + p, 0);
  if (total <= 0) {
    return 0;
  }

  const normalizedProbs = probs.map(p => p / total);

  if (k === 1) {
    return normalizedProbs[0];
  }

  const step = 1.0 / (k - 1);
  const expectedPos = normalizedProbs.reduce(
    (sum, p, i) => sum + p * (i * step),
    0
  );

  if (expectedPos < 0) {
    return 0;
  }
  if (expectedPos > 1) {
    return 1;
  }
  return expectedPos;
}

async function loadMiniLMCatBoost(config: CatboostLoadMessage["payload"]): Promise<void> {
  try {
    const scorerState: ScorerState = { type: "minilm_catboost" };

    const featureService = new FeatureService(config.spacyCtxUrl!);
    await featureService.init(config.featureServiceEmbeddingConfig, { progressCallback: (progress) => {
      self.postMessage({
        type: "progress",
        payload: { progress },
      });
    } });

    const ortSession = await ort.InferenceSession.create(catboostModel, {
      executionProviders: [config.catboostProvider],
    } as any);

    scorerState.featureService = featureService;
    scorerState.catboostOrt = ortSession;
    scorers.set("minilm_catboost", scorerState);

    self.postMessage({ type: "ready", payload: { success: true } });
  } catch (error) {
    self.postMessage({
      type: "error",
      payload: { message: (error as Error).message },
    });
  }
}

async function loadNLIRoberta(config: NLILoadMessage["payload"]): Promise<void> {
  try {
    const nliPipeline = (await pipeline(
      config.pipelineConfig.type,
      config.pipelineConfig.model,
      {
        subfolder: config.pipelineConfig.subfolder || "",
        progress_callback: (data: any) => {
          if (data.progress !== undefined && data.file.endsWith(".onnx")) {
            self.postMessage({
              type: "progress",
              payload: {
                progress: data.progress
              }
            });
          }
        },
        dtype: config.pipelineConfig.dtype,
        device: config.pipelineConfig.device as any,
      }
    )) as any;

    const scorerState: ScorerState = {
      type: "nli_roberta",
      nliPipeline,
    };

    scorers.set("nli_roberta", scorerState);

    self.postMessage({
      type: "ready",
      payload: { success: true }
    });
  } catch (error) {
    self.postMessage({
      type: "error",
      payload: { message: (error as Error).message }
    });
  }
}

async function loadModel(config: LoadMessage["payload"]): Promise<void> {
  const scorerId = config.scorerId;

  if (scorerId === "minilm_catboost") {
    await loadMiniLMCatBoost(config as CatboostLoadMessage["payload"]);
  } else if (scorerId === "nli_roberta") {
    await loadNLIRoberta(config as NLILoadMessage["payload"]);
  } else {
    self.postMessage({
      type: "error",
      payload: { message: `Unknown scorer type: ${scorerId}` }
    });
  }
}

async function evaluateMiniLMCatBoost(data: EvaluateMessage["payload"]): Promise<void> {
  const scorerState = scorers.get("minilm_catboost");
  if (!scorerState || !scorerState.featureService || !scorerState.catboostOrt) {
    self.postMessage({
      type: "error",
      payload: { message: "MiniLM-CatBoost scorer not loaded" }
    });
    return;
  }

  const { texts = [], batchSize } = data;
  const totalBatches = Math.ceil(texts.length / batchSize);

  try {
    const batches = Array.from({ length: totalBatches }, (_, i) =>
      texts.slice(i * batchSize, (i + 1) * batchSize)
    );

    scorerState.featureService.getFeaturesAllBatchPreload(batches);

    for (let i = 0; i < texts.length; i += batchSize) {
      const batchTexts = texts.slice(i, i + batchSize);

      const featureArrays = (await scorerState.featureService.getFeatures(batchTexts)).map(
        (featVec) => new Float32Array(featVec)
      );

      const batchCount = featureArrays.length;
      const featureDim = featureArrays[0].length;

      const concat = new Float32Array(batchCount * featureDim);
      for (let r = 0; r < batchCount; r++) {
        concat.set(featureArrays[r], r * featureDim);
      }

      const inputName = (scorerState.catboostOrt as any).inputNames?.[0] as string;
      const outputName = (scorerState.catboostOrt as any).outputNames?.[0] as string;

      const feeds: Record<string, ort.Tensor> = {};
      feeds[inputName] = new ort.Tensor("float32", concat, [batchCount, featureDim]);

      const outputs = await scorerState.catboostOrt.run(feeds as any);
      const outTensor = outputs[outputName] as ort.Tensor;
      const scoresArray = outTensor.data as Float32Array;

      const results = batchTexts.map((text, idx) => {
        const score = scoresArray.length === batchCount ? scoresArray[idx] : scoresArray[idx * (scoresArray.length / batchCount)];
        return { text, score: score / 5 };
      });

      self.postMessage({
        type: "progress",
        payload: {
          batchIndex: Math.floor(i / batchSize) + 1,
          totalBatches,
          results,
        }
      });
    }

    self.postMessage({ type: "complete", payload: { success: true } });
  } catch (error) {
    self.postMessage({
      type: "error",
      payload: { message: (error as Error).message + (error as Error).stack },
    });
  }
}

async function evaluateNLIRoberta(data: EvaluateMessage["payload"]): Promise<void> {
  const scorerState = scorers.get("nli_roberta");
  if (!scorerState || !scorerState.nliPipeline) {
    self.postMessage({
      type: "error",
      payload: { message: "NLI-RoBERTa scorer not loaded" }
    });
    return;
  }

  const { segments = [], candidateLabels = [], hypothesisTemplate = "", batchSize } = data;
  const totalBatches = Math.ceil(segments.length / batchSize);

  try {
    for (let i = 0; i < segments.length; i += batchSize) {
      const batchSegments = segments.slice(i, i + batchSize);

      const results = await scorerState.nliPipeline(batchSegments, candidateLabels, {
        hypothesis_template: hypothesisTemplate,
      });

      const segmentProbs = results.map((result: any) => {
        const scores = candidateLabels.map((label) => {
          const index = result.labels.indexOf(label);
          return result.scores[index];
        });
        return scores;
      });

      const scores = segmentProbs.map((probs: any) => probsToScore(probs));

      const batchResults = batchSegments.map((segment, index) => ({
        text: segment,
        score: scores[index]
      }));

      self.postMessage({
        type: "progress",
        payload: {
          batchIndex: i / batchSize + 1,
          totalBatches,
          results: batchResults
        }
      });
    }

    self.postMessage({
      type: "complete",
      payload: { success: true }
    });
  } catch (error) {
    self.postMessage({
      type: "error",
      payload: { message: (error as Error).message }
    });
  }
}

async function evaluateSegments(data: EvaluateMessage["payload"]): Promise<void> {
  const scorerId = data.scorerId;

  if (scorerId === "minilm_catboost") {
    await evaluateMiniLMCatBoost(data);
  } else if (scorerId === "nli_roberta") {
    await evaluateNLIRoberta(data);
  } else {
    self.postMessage({
      type: "error",
      payload: { message: `Unknown scorer type: ${scorerId}` }
    });
  }
}

self.onmessage = async (event: MessageEvent<WorkerMessage>) => {
  switch (event.data.type) {
  case "load":
    await loadModel((event.data as LoadMessage).payload);
    break;
  case "evaluate":
    await evaluateSegments((event.data as EvaluateMessage).payload);
    break;
  }
};
