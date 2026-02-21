import { setScorerWorker } from "~/utils/scorerWorker";
import ScorerWorker from "~/workers/scorer.worker.ts?worker";

let scorerWorkerSet = false;

export function initApp() {
  // Ping API to wake it up
  const { $api } = useNuxtApp();
  $api("/api/ping", { method: "GET" });

  if (!scorerWorkerSet) {
    const scorerWorker = new ScorerWorker();
    setScorerWorker(scorerWorker);
    scorerWorkerSet = true;
  }
}

