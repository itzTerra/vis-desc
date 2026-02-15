# Cache and Scorer Refactor - Tasks

Total: ~10 hours over 8 phases. NO backward compatibility.

---

## Phase 1: Types (30 min) ✓ COMPLETED

**File**: `types/cache.d.ts` (new)

- [x] Create `Downloadable` interface: `id`, `label`, `sizeMb`, `download(onProgress)`
- [x] Create `ModelGroup` interface: `id`, `name`, `downloadables[]`
- [x] Create `ScorerStage` interface: `label`, `isMain?`
- [x] Create `ScorerProgress` interface: `stage`, `progress`, `eta?`

---

## Phase 2: CacheController (1 hour) ✓ COMPLETED

**File**: `utils/CacheController.ts` (new)

- [x] Class with constructor: `namespace = "transformers-cache"`
- [x] `async init()`: Open cache
- [x] `async exists(key)`: Check if cached
- [x] `async downloadGroup(group, onProgress)`: Loop downloadables, aggregate progress

**File**: `composables/useCacheController.ts` (new)

- [x] Singleton instance
- [x] Export composable with `downloadGroup()`, `checkGroupCached()`

**File**: `plugins/cache-event-bus.ts` (new)

- [x] Create event bus (SimpleEventBus - no mitt dependency)
- [x] Export `useCacheEvents()`
- [x] Event: `cache:model-needed` with `{ groupId }`

**Correctness Fixes** ✓ 

- [x] Fix progress calculation: divide callback range (0-100) by 100 in `downloadGroup()`
- [x] Add error handling in event bus `emit()` to prevent handler errors blocking subsequent handlers
- [x] Fix race condition in `init()` by caching the initialization Promise

---

## Phase 3: models.ts Rewrite (2 hours) ✅ COMPLETED & FIXED

**File**: `utils/models.ts` (complete rewrite)

- [x] Delete ALL existing code
- [x] Create `abstract class Downloadable` with abstract `download(onProgress)`
- [x] Create `FeatureServiceDownloadable` with FeatureService.init()
- [x] Create `HuggingFacePipelineDownloadable` with transformers.js pipeline()
- [x] Create `OnnxDownloadable` for ONNX models
- [x] Create `abstract class Scorer` with:
  - Constructor: `id`, `label`, `description`, `speed`, `quality`, `disabled`
  - `stages` property returning 2 stages: "Initializing..." and "Scoring..."
  - Abstract `score(data, onProgress, socket?)` method
  - `protected getProviders()` method for localStorage access
  - `dispose()` method for cleanup
- [x] Create `MiniLMCatBoostScorer` (worker-based):
  - Metadata: speed=5, quality=4, disabled=false
  - Spawns scorer.worker.ts, communicates via postMessage
  - Concurrent scoring prevention with try-finally
- [x] Create `NLIRobertaScorer` (worker-based):
  - Metadata: speed=3, quality=5, disabled=false
  - Spawns nli.worker.ts, zero-shot classification
  - Concurrent scoring prevention with try-finally
- [x] Create `RandomScorer` (WebSocket-based):
  - Metadata: speed=5, quality=1, disabled=false
  - Uses existing WebSocket connection, server-side random scores
  - Concurrent scoring prevention with try-finally
- [x] Create shared instances: `sharedFeatureService`
- [x] Define `MODEL_GROUPS` array
- [x] Export `SCORERS` array

**Phase 3 Bug Fixes**:
- [x] Fix memory leak: Add `{ once: true }` to addEventListener in init handlers
- [x] Fix abstract method contract: Add optional `socket?: any` parameter
- [x] Fix localStorage key format: Change from `onnx_providers_${id}` to `onnx_providers[id]`
- [x] Fix DRY violation: Extract `getProviders()` and `dispose()` to base Scorer class
- [x] Fix concurrent calls: Add `private scoring` flag with try-finally protection

---

## Phase 4: CacheManager (1.5 hours) ✅ COMPLETED & FIXED

**File**: Rename `ModelManager.vue` → `CacheManager.vue`

- [x] Update component name and imports
- [x] Import `MODEL_GROUPS` instead of `MODELS`
- [x] Loop: `v-for="group in MODEL_GROUPS"`
- [x] Display: `group.name` as title
- [x] Compute size: `group.downloadables.reduce((s, d) => s + d.sizeMb, 0)`
- [x] Update `queueDownload()` to accept `ModelGroup`
- [x] Use `useCacheController().downloadGroup()` 
- [x] Add CPU/GPU toggle:
  - [x] Create `providers` ref: `Record<string, string>` (now stores "webgpu" or "wasm")
  - [x] Load from localStorage: `onnx_providers`
  - [x] Add toggle UI per group
  - [x] Save to localStorage on change
- [x] Add remove and clear methods to CacheController
- [x] All linting checks passing

**Phase 4 Bug Fixes** ✅:
- [x] Remove redundant watch block with `{ deep: true }` that duplicates updateProvider() logic
- [x] Wrap JSON.parse() in try-catch on mount for invalid JSON handling
- [x] Add double-check pattern in queueDownload() after async checkCachedGroups() call
- [x] Add try-catch around cacheController.init() and checkCachedGroups() in onMounted
- [x] Fix queue position display from 0-indexed to 1-indexed
- [x] Change provider storage from boolean to string: "webgpu" or "wasm"
- [x] Update checkbox binding to check for "webgpu" value
- [x] Update providers type annotation

---

## Phase 5: EvalProgress (30 min) ✅ COMPLETED

**File**: `components/EvalProgress.vue`

- [x] Add prop: `currentStage?: string`
- [x] If stage === "Initializing...": Show text only, no ETA
- [x] If stage === "Scoring...": Show progress bar + ETA
- [x] Remove multi-stage logic

---

## Phase 6: index.vue Integration (1 hour)

**File**: `pages/index.vue`

- [ ] Import `SCORERS` from models.ts
- [ ] Replace `selectedModel` with:
  ```typescript
  const selectedScorer = computed(() => 
    SCORERS.find(s => s.id === selectedModel.value)
  );
  ```
- [ ] Add `currentStage` ref
- [ ] Update `handleProcessPdf()` to call `selectedScorer.value.score()`
  - Pass `data`, `socket` (for WebSocket scorers), and `onProgress` callback
  - `onProgress` updates `currentStage` and segment highlights
- [ ] Pass `currentStage` to EvalProgress
- [ ] Listen for `cache:model-needed` event
- [ ] Auto-open CacheManager when model needed
- [ ] Ensure WebSocket scorers receive socket connection
- [ ] Ensure worker scorers do NOT require socket

---

## Phase 7: CPU/GPU Toggle (1 hour)

**File**: `workers/scorer.worker.ts`

- [ ] Read provider from postMessage payload
- [ ] Pass to ONNX: `executionProviders: [provider]`

**File**: `utils/models.ts` (scorer classes)

- [ ] In scorer `score()` method:
  - Read from localStorage: `onnx_providers[this.id]`
  - Pass to worker in postMessage

---

## Phase 8: Testing (2 hours)

**Cleanup**:
- [ ] Delete `composables/useModelLoader.ts`
- [ ] Remove all imports of useModelLoader
- [ ] Remove old `transformersConfig` references

**End-to-end testing**:
- [ ] Test MiniLM-CatBoost download and scoring
- [ ] Test NLI-RoBERTa download and scoring
- [ ] Test Random scorer
- [ ] Test CPU/GPU toggle
- [ ] Test cache persistence
- [ ] Test stage transitions
- [ ] Test error handling (cache disabled, network errors, worker failures)

**Completion**:
- [ ] No TypeScript errors
- [ ] No console errors
- [ ] All scorers functional
- [ ] CPU/GPU toggle working
- [ ] UI matches old ModelManager
