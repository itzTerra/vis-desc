const workers = new Map<string, Worker | Promise<Worker>>();

export async function initWorkers() {
  console.info("Initializing workers...");
  const { getOrLoadWorker } = useModelLoader();

  workers.set("scorer", getOrLoadWorker({
    workerName: "scorer",
    modelUrl: "/assets/data/models/catboost.onnx",
    subfolder: "onnx",
    spacyCtxUrl: new URL("/api/spacy-ctx", useRuntimeConfig().public.apiBaseUrl).toString()
  } as any).then(worker => {
    workers.set("scorer", worker);
    return worker;
  }));
  console.info("Scorer worker initialized");
}

export function terminateWorkers() {
  const { terminateWorker } = useModelLoader();
  terminateWorker("scorer");
  terminateWorker("nli");
}

export async function getWorker(name: string): Promise<Worker | undefined> {
  const worker = workers.get(name);
  return worker ? (await worker) : undefined;
}

export async function initializeApp() {
  // Ping API to wake it up (if it's asleep) and check connectivity
  const { $api } = useNuxtApp();
  console.info("Pinging API to wake it up...");
  $api("/api/ping", { method: "GET" }).then(() => {
    console.info("API woke up successfully");
  }).catch((err) => {
    console.error("Error pinging API:", err);
  });

  // Startup workers
  await initWorkers();
}
