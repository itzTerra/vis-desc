# Cache and Scorer Refactor - Planning

## Project Classification
- **Type**: Refactor
- **Size**: L (~10 hours)
- **Date**: February 15, 2026

## Goal
Complete separation of cache/download concerns from scoring/inference logic using OOP. Simplify stage management and unify CacheManager UI.

## Success Criteria
1. Downloadables and Scorers are separate classes
2. CacheManager shows groups (same UI as old ModelManager)
3. Only 2 scoring stages: "Initializing..." and "Scoring..."
4. CPU/GPU toggle per model in CacheManager
5. NO backward compatibility preserved

---

## Architecture

### Core Separation

**Downloadables**: Handle download/cache only
- Abstract base class with `download(onProgress)` method
- `FeatureServiceDownloadable`: Initializes FeatureService
- `HuggingFacePipelineDownloadable`: Downloads transformers.js models
- `OnnxDownloadable`: Downloads ONNX models

**Scorers**: Handle inference only
- Abstract base class with:
  - Metadata: `speed` (1-5), `quality` (1-5), `disabled` (bool), `description` (string)
  - 2 stages (hard-coded): "Initializing..." (no ETA), "Scoring..." (with ETA)
  - Abstract `score(data, onProgress)` method
  - Support both worker-based and WebSocket-based inference
- `MiniLMCatBoostScorer` (worker), `NLIRobertaScorer` (worker), `RandomScorer` (WebSocket demo)

**Model Groups**: Link downloadables (cache) to scorers (inference)
```typescript
{
  id: "minilm_catboost",
  name: "MiniLM-CatBoost",
  downloadables: [sharedFeatureService, miniLMPipeline, catboostModel],
}

// Scorers reference groups by ID
const miniLMScorer = new MiniLMCatBoostScorer(
  "minilm_catboost",
  "MiniLM-CatBoost",
  "Fast local inference with feature extraction",
  5,  // speed
  4,  // quality
  false  // disabled
);
```

### Inference Strategies

**Worker-based** (MiniLM, NLI-RoBERTa):
- Spawns Web Worker
- Runs inference in separate thread
- Reports progress via worker messages

**WebSocket-based** (Random/Demo, future server-side models):
- Connects to API via WebSocket
- Server-side inference
- Reports progress via socket messages

### Stage Flow

**Download** (in CacheManager):
- Shows: "Downloading... 45%" with progress bar
- NO stages

**Scoring** (in EvalProgress):
- Stage 1: "Initializing..." (worker spawn + setup, no ETA)
- Stage 2: "Scoring..." (batch processing with ETA)

### CacheController

Replaces useModelLoader functionality:
- `CacheController` class: Manages cache operations
- `useCacheController()` composable: Vue integration
- EventBus plugin: `cache:model-needed` event only

### CPU/GPU Toggle

In CacheManager.vue:
- Toggle switch per model
- Stored in localStorage as `onnx_providers`
- Passed to worker during initialization
- Controls ONNX ExecutionProvider (wasm vs webgpu)

---

## Files

### DELETE
- `composables/useModelLoader.ts`

### CREATE
- `types/cache.d.ts`
- `utils/CacheController.ts`
- `plugins/cache-event-bus.ts`
- `composables/useCacheController.ts`

### REWRITE
- `utils/models.ts` → Complete OOP rewrite

### MODIFY
- `components/ModelManager.vue` → Rename to CacheManager.vue
- `components/EvalProgress.vue` → 2 stages only
- `pages/index.vue` → Use Scorer classes

---

## Implementation (10 hours)

1. **Types** (30 min): Create cache.d.ts
2. **CacheController** (1 hour): Core cache class + composable + event plugin
3. **models.ts Rewrite** (2 hours): OOP classes for Downloadables and Scorers
4. **CacheManager** (1.5 hours): Refactor ModelManager to work with groups
5. **EvalProgress** (30 min): Simplify to 2 stages  
6. **index.vue Integration** (1 hour): Use Scorer classes
7. **CPU/GPU Toggle** (1 hour): Add localStorage-backed toggle
8. **Testing** (2 hours): End-to-end verification
