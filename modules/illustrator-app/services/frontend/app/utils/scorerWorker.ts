let scorerWorker: Worker | null = null;

export function setScorerWorker(worker: Worker) {
  scorerWorker = worker;
}

export function getScorerWorker(): Worker {
  if (scorerWorker === null) {
    throw new Error("Scorer worker not initialized. Call initApp() first.");
  }
  return scorerWorker as Worker;
}
