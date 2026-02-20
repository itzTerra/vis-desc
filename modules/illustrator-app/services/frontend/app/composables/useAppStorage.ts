import { SCORERS } from "~/utils/models";

const _firstScorer = SCORERS.find(scorer => !scorer.disabled);
const DEFAULT_SELECTED_MODEL = _firstScorer ? _firstScorer.id : (SCORERS[0]?.id ?? "random");

export const AUTO_ILLUSTRATION_DEFAULTS = {
  enabled: false,
  minGapPages: 1.5,
  maxGapPages: 8,
  minScore: null as number | null,
  enableEnhance: true,
  enableGenerate: true,
};

export function useAppStorage() {
  const selectedModel = useLocalStorage("selectedModel", DEFAULT_SELECTED_MODEL);

  const autoEnabled = useLocalStorage("autoIllustration.enabled", AUTO_ILLUSTRATION_DEFAULTS.enabled);
  const autoMinGapPages = useLocalStorage("autoIllustration.minGapPages", AUTO_ILLUSTRATION_DEFAULTS.minGapPages);
  const autoMaxGapPages = useLocalStorage("autoIllustration.maxGapPages", AUTO_ILLUSTRATION_DEFAULTS.maxGapPages);
  const autoMinScore = useLocalStorage<number | null>("autoIllustration.minScore", AUTO_ILLUSTRATION_DEFAULTS.minScore);
  const autoEnableEnhance = useLocalStorage("autoIllustration.enableEnhance", AUTO_ILLUSTRATION_DEFAULTS.enableEnhance);
  const autoEnableGenerate = useLocalStorage("autoIllustration.enableGenerate", AUTO_ILLUSTRATION_DEFAULTS.enableGenerate);

  return {
    selectedModel,
    autoIllustration: {
      enabled: autoEnabled,
      minGapPages: autoMinGapPages,
      maxGapPages: autoMaxGapPages,
      minScore: autoMinScore,
      enableEnhance: autoEnableEnhance,
      enableGenerate: autoEnableGenerate,
    },
  };
}
