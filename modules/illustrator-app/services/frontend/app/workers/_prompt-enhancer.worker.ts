import { pipeline, env, type TextGenerationPipeline } from "@huggingface/transformers";

env.allowLocalModels = false;
env.allowRemoteModels = true;
env.useBrowserCache = true;

let loadedPipeline: TextGenerationPipeline | null = null;

const DEFAULT_PROMPT_KEYWORDS = "concept art, highly detailed, 4K, UHD, cinematic lighting, vivid, vibrant, artstation";

type LoadMessage = {
  type: "load";
  payload: {
    huggingFaceId: string;
    pipeline: string;
  };
};

type EnhanceMessage = {
  type: "enhance";
  payload: {
    text: string;
    mode: string;
  };
};

type WorkerMessage = LoadMessage | EnhanceMessage;

async function loadModel(config: LoadMessage["payload"]) {
  try {
    loadedPipeline = await pipeline(
      config.pipeline as any,
      config.huggingFaceId,
      {
        subfolder: "onnx",
        session_options: {
          executionProviders: ["wasm"],
          externalData: [
            {
              path: "model_q4.onnx.data",
              data: "onnx/model_q4.onnx.data"
            },
          ]
        },
        dtype: "q4",
        device: "wasm",
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

async function enhancePrompt(data: EnhanceMessage["payload"]) {
  if (!loadedPipeline) {
    self.postMessage({
      type: "error",
      payload: { message: "Pipeline not loaded" }
    });
    return;
  }

  try {
    const { text } = data;

    const messages = [
      { role: "system", content: "Enhance and expand the following prompt with more details and context:" },
      // { role: "system", content: mode },
      { role: "user", content: text }
    ];

    const result = await loadedPipeline(messages, {
      max_new_tokens: 512,
      temperature: 0.2,
      top_p: 0.9,
      do_sample: true,
      return_full_text: false,
    });

    console.log("Raw model output:", result);
    const generatedText = (result[0] as any).generated_text;
    const enhancedText = (Array.isArray(generatedText) ? generatedText.filter(m => m.role === "assistant").map(m => m.content).join(" ") : generatedText) + ` ${DEFAULT_PROMPT_KEYWORDS}`;

    self.postMessage({
      type: "complete",
      payload: {
        success: true,
        enhancedText
      }
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
  case "enhance":
    await enhancePrompt((event.data as EnhanceMessage).payload);
    break;
  }
};
