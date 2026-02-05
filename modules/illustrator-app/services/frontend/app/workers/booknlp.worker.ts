import { env, PreTrainedModel } from "@huggingface/transformers";
import type { LoadMessage, WorkerMessage } from "~/types/workers";

env.allowLocalModels = false;
env.allowRemoteModels = true;
env.useBrowserCache = true;

// const loadedPipeline: Pipeline | null = null;

async function loadModel(_config: LoadMessage["payload"]) {
  try {
    // loadedPipeline = await pipeline(
    //   config.pipeline,
    //   config.huggingFaceId,
    //   {
    //     subfolder: "",
    //     progress_callback: (data: any) => {
    //       if (data.progress !== undefined && data.file.endsWith(".onnx")) {
    //         self.postMessage({
    //           type: "progress",
    //           payload: {
    //             progress: data.progress
    //           }
    //         });
    //       }
    //     },
    //     dtype: "q8",
    //   }
    // ) as any;
    try {
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      const onnxSession = (await PreTrainedModel.from_pretrained("Terraa/entities_google_bert_uncased_L-4_H-256_A-4-v1.0-ONNX", {
        subfolder: "onnx",
        dtype: "fp16",
        session_options: {
          externalData: [
            {
              path: "model_fp16.onnx.data",
              data: "onnx/model_fp16.onnx.data"
            }
          ]
        }
      })).sessions["model"];
    } catch (e) {
      console.error("Failed to load ONNX model:", e);
    }

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

self.onmessage = async (event: MessageEvent<WorkerMessage>) => {
  switch (event.data.type) {
  case "load":
    await loadModel((event.data as LoadMessage).payload);
    break;
  }
};
