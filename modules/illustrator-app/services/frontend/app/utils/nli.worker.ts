import { pipeline, env, type Pipeline, type PipelineType } from "@huggingface/transformers";

env.allowLocalModels = false;
env.allowRemoteModels = true;
env.useBrowserCache = true;

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

let loadedPipeline: Pipeline | null = null;

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

async function loadModel(config: LoadMessage["payload"]) {
  try {
    loadedPipeline = await pipeline(
      config.pipeline,
      config.huggingFaceId,
      {
        subfolder: "",
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
        dtype: "q8",
      }
    ) as any;

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

async function evaluateSegments(data: EvaluateMessage["payload"]) {
  if (!loadedPipeline) {
    self.postMessage({
      type: "error",
      payload: { message: "Pipeline not loaded" }
    });
    return;
  }

  const { segments, candidateLabels, hypothesisTemplate, batchSize } = data;
  const totalBatches = Math.ceil(segments.length / batchSize);

  try {
    for (let i = 0; i < segments.length; i += batchSize) {
      const batchSegments = segments.slice(i, i + batchSize);

      const results = await loadedPipeline(batchSegments, candidateLabels, {
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
