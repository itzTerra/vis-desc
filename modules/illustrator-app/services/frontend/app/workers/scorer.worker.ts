import { pipeline, env, type Pipeline } from "@huggingface/transformers";
import type { SendMessage, RecvMessage, ScorerType, EvaluateMessage, LoadMessage, NLILoadPayload, CatboostLoadPayload, NLIEvaluatePayload, CatboostEvaluatePayload } from "~/types/scorer-worker";
import { FeatureService } from "~/utils/text2features";
import * as ort from "onnxruntime-web";
import catboostModel from "~/assets/data/models/catboost.onnx?url";

env.allowLocalModels = false;
env.allowRemoteModels = true;
env.useBrowserCache = true;

console.info("Scorer worker spawned");

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

async function loadMiniLMCatBoost(config: CatboostLoadPayload): Promise<void> {
  try {
    let scorerState = scorers.get("minilm_catboost");
    if (!scorerState) {
      scorerState = { type: "minilm_catboost" };

      const featureService = new FeatureService(config.spacyCtxUrl!);
      await featureService.init(config.featureServiceEmbeddingConfig, { provider: config.provider });
      scorerState.featureService = featureService;

      const ortSession = await ort.InferenceSession.create(catboostModel, {
        executionProviders: [config.provider],
      } as any);
      scorerState.catboostOrt = ortSession;

      scorers.set("minilm_catboost", scorerState);
    }

    // Preload SpaCy contexts without awaiting
    const totalBatches = Math.ceil(config.texts.length / config.batchSize);
    const batches = Array.from({ length: totalBatches }, (_, i) =>
      config.texts.slice(i * config.batchSize, (i + 1) * config.batchSize)
    );
    scorerState.featureService!.getFeaturesAllBatchPreload(batches);

    postRecv({ type: "ready", payload: { success: true } });
  } catch (error) {
    postRecv({ type: "error", payload: { message: (error as Error).message, stack: (error as Error).stack } });
  }
}

async function loadNLIRoberta(config: NLILoadPayload): Promise<void> {
  try {
    let scorerState = scorers.get("nli_roberta");
    if (!scorerState) {
      const nliPipeline = (await pipeline(
        config.pipelineConfig.type,
        config.pipelineConfig.model,
        {
          subfolder: config.pipelineConfig.subfolder || "",
          dtype: config.pipelineConfig.dtype,
          device: config.pipelineConfig.device as any,
        }
      )) as any;

      scorerState = {
        type: "nli_roberta",
        nliPipeline,
      };

      scorers.set("nli_roberta", scorerState);
    }

    postRecv({ type: "ready", payload: { success: true } });
  } catch (error) {
    postRecv({ type: "error", payload: { message: (error as Error).message, stack: (error as Error).stack } });
  }
}

async function loadModel(config: LoadMessage["payload"]): Promise<void> {
  const scorerId = config.scorerId;

  if (scorerId === "minilm_catboost") {
    await loadMiniLMCatBoost(config as CatboostLoadPayload);
  } else if (scorerId === "nli_roberta") {
    await loadNLIRoberta(config as NLILoadPayload);
  } else {
    postRecv({ type: "error", payload: { message: `Unknown scorer type: ${scorerId}` } });
  }
}

async function evaluateMiniLMCatBoost(data: CatboostEvaluatePayload): Promise<void> {
  const scorerState = scorers.get("minilm_catboost");
  if (!scorerState || !scorerState.featureService || !scorerState.catboostOrt) {
    postRecv({ type: "error", payload: { message: "MiniLM-CatBoost scorer not loaded" } });
    return;
  }

  const { texts = [], batchSize } = data;
  const totalBatches = Math.ceil(texts.length / batchSize);

  try {
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

      postRecv({ type: "progress", payload: { batchIndex: Math.floor(i / batchSize) + 1, totalBatches, results } });
    }

    postRecv({ type: "complete", payload: { success: true } });
  } catch (error) {
    postRecv({ type: "error", payload: { message: (error as Error).message, stack: (error as Error).stack } });
  }
}

async function evaluateNLIRoberta(data: NLIEvaluatePayload): Promise<void> {
  const scorerState = scorers.get("nli_roberta");
  if (!scorerState || !scorerState.nliPipeline) {
    postRecv({ type: "error", payload: { message: "NLI-RoBERTa scorer not loaded" } });
    return;
  }

  const { texts = [], candidateLabels = [], hypothesisTemplate = "", batchSize } = data;
  const totalBatches = Math.ceil(texts.length / batchSize);

  try {
    for (let i = 0; i < texts.length; i += batchSize) {
      const batchTexts = texts.slice(i, i + batchSize);

      const results = await scorerState.nliPipeline(batchTexts, candidateLabels, {
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

      const batchResults = batchTexts.map((text, index) => ({
        text,
        score: scores[index]
      }));

      postRecv({ type: "progress", payload: { batchIndex: i / batchSize + 1, totalBatches, results: batchResults } });
    }

    postRecv({ type: "complete", payload: { success: true } });
  } catch (error) {
    postRecv({ type: "error", payload: { message: (error as Error).message, stack: (error as Error).stack } });
  }
}

async function evaluateSegments(data: EvaluateMessage["payload"]): Promise<void> {
  const scorerId = data.scorerId;

  if (scorerId === "minilm_catboost") {
    await evaluateMiniLMCatBoost(data as CatboostEvaluatePayload);
  } else if (scorerId === "nli_roberta") {
    await evaluateNLIRoberta(data as NLIEvaluatePayload);
  } else {
    postRecv({ type: "error", payload: { message: `Unknown scorer type: ${scorerId}` } });
  }
}

function postRecv(msg: RecvMessage) {
  self.postMessage(msg);
}

self.onmessage = async (event: MessageEvent<SendMessage>) => {
  switch (event.data.type) {
  case "load":
    await loadModel((event.data as LoadMessage).payload);
    break;
  case "evaluate":
    await evaluateSegments((event.data as EvaluateMessage).payload);
    break;
  }
};
