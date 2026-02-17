// Spawn the scorer worker

let scorerWorker: Worker | null = null;

export function initApp() {
  if (scorerWorker === null) {
    scorerWorker = new Worker(new URL("~/workers/scorer.worker.ts", import.meta.url), {
      type: "module",
    });
  }
}

export function getScorerWorker(): Worker {
  if (scorerWorker === null) {
    throw new Error("Scorer worker not initialized. Call initApp() first.");
  }
  return scorerWorker as Worker;
}
