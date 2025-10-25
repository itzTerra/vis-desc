export const MODELS = [
  { value: "minilm_svm", label: "MiniLM-SVM", speed: 4, quality: 3, disabled: true, description: "A fast model with medium quality for general use." },
  { value: "modernbert_finetuned", label: "ModernBERT-Finetuned", speed: 3, quality: 4, disabled: false, description: "Balanced model fine-tuned for better performance." },
  { value: "nli_roberta", label: "NLI-RoBERTa", speed: 2, quality: 4, disabled: true, description: "Slow but high-quality model for precise tasks." },
  { value: "llm", label: "LLM [experimental]", speed: 1, quality: 5, disabled: true, description: "Experimental model using large language models for best results." },
  { value: "random", label: "Random [debug]", speed: 5, quality: 0, disabled: false, description: "Instant random selection for debugging." },
] as const;

export type Model = (typeof MODELS)[number];
export type ModelValue = typeof MODELS[number]["value"];
