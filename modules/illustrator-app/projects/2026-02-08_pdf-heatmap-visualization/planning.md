# PDF Heatmap Visualization - Planning

## Project Classification
- **Type**: Feature Addition (visualization enhancement)
- **Size**: Large (XL)
- **Effort Estimate**: 8-12 days
- **Complexity**: High (coordinate transformations, viewport tracking, scroll synchronization)
- **Date**: February 8, 2026

---

## Goals

Create a fixed left-overlay heatmap visualization that gives users an at-a-glance overview of segment importance distribution across the entire PDF document. The heatmap should:
1. Provide rapid visual scanning of where high-scoring segments exist
2. Enable single-click navigation to specific segments
3. Support efficient viewport scrolling via drag-and-drop rectangle
4. Remain responsive and performant with 100+ page documents and 500+ segments
5. Collapse/expand with toggle always visible to maximize document viewing space

**Success Criteria:**
- Heatmap renders in <500ms for 100-page PDFs with 500+ segments
- Visual distinctiveness between high/low scoring regions (brightness encoding)
- Click navigation to segment location completes in <200ms (smooth scroll)
- Image position indicators visible and distinct from heatmap background
- Collapse/expand transitions smooth with toggle button always visible
- Drag-viewport scrolling provides full-page control equivalent to native scrollbar

---

## User Stories

### Story 1: User Views Heatmap Overview of Document
**As** a user analyzing a long PDF with many segments,
**I want to** see a compact heatmap on the left overlay showing density and quality of annotations,
**So that** I can quickly identify high-value regions without reading the entire document.

**Acceptance Criteria**:
- Fixed overlay displays miniaturized PDF representation (~96px width, full height)
- Higher scores appear as darker pixels (brightness-based encoding)
- All pages fit vertically with proper aspect ratio preservation
- Heatmap updates in real-time as new scores arrive from backend
- Heatmap positioning remains stable during normal scroll

### Story 2: User Identifies Image Regions in Heatmap
**As** a user with images added to segments,
**I want to** see distinct visual indicators (dots) on the heatmap showing which segments have associated images,
**So that** I can quickly find image-enhanced content.

**Acceptance Criteria**:
- Image position dots scale appropriately to heatmap canvas
- Dots are visually distinct from background heatmap (e.g., colored overlay, outline, or glow)
- Each dot positioned at normalized coordinate from highlight.polygons
- Dots remain visible at small heatmap scale
- Dots update when imageUrl is added or removed (editor open/closed state does not matter)

### Story 3: User Clicks Heatmap to Jump to Segment
**As** a user reviewing the heatmap,
**I want to** click on any region to smoothly scroll the PDF to that segment,
**So that** I can quickly navigate between high-interest areas.

**Acceptance Criteria**:
- Click anywhere on heatmap triggers scroll to corresponding PDF location
- Scroll behavior is smooth (duration ~300-400ms)
- Viewport centers the clicked region
- Multiple rapid clicks queue smoothly without jarring jumps

### Story 4: User Drags Viewport Rectangle to Scroll
**As** a user viewing the heatmap,
**I want to** drag a rectangle indicating my current viewport to scroll the PDF,
**So that** I can use the heatmap as a full scrollbar replacement.

**Acceptance Criteria**:
- Viewport rectangle shows current visible PDF region (height, position, opacity)
- Rectangle dimensions proportional to visible vs. total document size
- Dragging rectangle triggers smooth scroll to new position
- Position in rectangle maps linearly to PDF scroll position
- Native browser scrollbar behavior is mirrored (drag at top = scroll up, etc.)
- Works during drag operations with proper debouncing

### Story 5: User Collapses Heatmap to Maximize PDF View
**As** a user needing more horizontal space for the PDF document,
**I want to** collapse the heatmap overlay while keeping toggle button visible,
**So that** I can maximize document viewing area temporarily.

**Acceptance Criteria**:
- Collapse button protrudes visibly from edge (similar to ImageEditor pattern)
- Collapsed state hides entire heatmap canvas
- Toggle button remains clickable and always visible
- Expand/collapse transitions are smooth
- Expand animation reveals heatmap smoothly without layout shift

---

## Functional Requirements

### Component Architecture

#### HeatmapViewer.vue (New Component)
**Purpose**: Fixed left overlay containing miniature PDF representation with segment scoring overlay

**Props**:
```typescript
highlights: Highlight[]          // All segments with polygons and scores
currentPage: number              // 1-indexed current page for viewport rect positioning
pageAspectRatio: number          // A4 ratio for proper scaling
pageRefs: Element[]             // Array of page DOM elements (0-indexed)
editorStates: EditorState[]      // Image editor state for imageUrl tracking
```

**Emits**:
```typescript
navigate: (page: number, normalizedY: number) => void  // User clicked heatmap
```

**State**:
```typescript
isExpanded: boolean              // Visibility toggle
viewportRect: SVGRect           // Current visible region indicator
heatmapCanvas: HTMLCanvasElement // Pre-rendered heatmap for performance
segmentDots: Array<{            // Image position indicators
  highlightId: number
  pageNum: number
  normalizedX: number
  normalizedY: number
  hasImage: boolean
}>
```

#### Key Features

**1. Heatmap Rendering**
- Render each page as a horizontal stripe in the heatmap (proportional height)
- For each page, iterate through segments with normalized polygons
- For each polygon point, calculate pixel position in heatmap space
- Apply brightness encoding with per-point opacity to allow blending
- Clamp brightness to visible range (avoid pure white/black)
- Use HTMLCanvasElement rendering with batched updates

**2. Coordinate System**
- Heatmap maintains page aspect ratio in miniature form
- Normalized coordinates [0,1] map directly to heatmap pixels
- Page boundaries calculated from cumulative height (sum of page heights)
- All calculations use 0-indexed pages internally, convert 1-indexed in interface

**3. Viewport Indicator**
- Track visible region via window scroll position
- Calculate scroll percentage: scrollY / totalHeight
- Draw semi-transparent rectangle showing current viewport
- Rectangle height = (viewportHeight / totalHeight) * heatmapHeight
- Allow drag-to-scroll via rectangle interaction

**4. Image Position Dots**
- Use editorStates to identify segments with active imageUrl
- Render small symbols (~4-6px) at polygon centroid position
- Use two symbols: one for hasImage, one for no-image
- Update reactively when imageUrl changes
**5. Click Navigation**
- Listen for click events on heatmap canvas
- Parent (PdfViewer) scrolls to location smoothly

**6. Scroll Synchronization**
- Listen for scroll events on window
- Update viewport rectangle position in real-time
- Use requestAnimationFrame to avoid jank
- Debounce at 60fps

### Integration Points

#### With PdfViewer.vue
- Mount HeatmapViewer as fixed overlay on the left edge of the viewport
- Pass: highlights, currentPage, pageAspectRatio, pageRefs, editorStates
- Handle navigate event to trigger scrollIntoView with smooth behavior

#### With ImageEditor/ImageLayer
- Use `editorStates` from PdfViewer to identify imageUrl presence
- Update dots reactively when imageUrl changes
- If props cannot be used cleanly, rely on existing global state (no direct component querying)

#### With Scroll Events
- Integration via window scroll tracking (existing pattern)
- Viewport indicator powered by scroll position calculations
- Drag-to-scroll powered by heatmap coordinate → scroll position mapping

### Visual Design

**Layout:**
```
┌─ 96px ─┐
│ Heatmap │ Fixed overlay (semi-transparent)
│ Canvas  │
│  SVG    │
│ Overlay │
│         │
│  [◀]    │ Toggle button (protrudes left)
└─────────┘
```
Height aligns to the PDF viewer area (not the full document height).
Overlay background uses partial transparency to reduce occlusion.

**Encoding:**
- **Brightness**: pixel darkness = 0.2 + (0.8 * score)
  - Low scores (0.0-0.2): very light gray (#E8E8E8)
  - Medium scores (0.4-0.6): mid gray (#808080)
  - High scores (0.8-1.0): dark gray (#1A1A1A)
- **Image Dots**: Two symbols for image vs no-image
  - Has imageUrl: filled gold circle with dark stroke
  - No imageUrl: hollow circle or diamond with lighter stroke
  - Size: 4-5px radius for 96px wide heatmap

- **Viewport Rectangle**: Semi-transparent overlay with border
  - Fill: rgba(100, 150, 255, 0.15) (light blue, 15% opacity)
  - Stroke: 2px #4A90E2 (medium blue)
  - Updates smoothly on scroll/drag

### Performance Considerations

**Rendering Strategy**:
1. Render on HTMLCanvasElement
2. Cache rendered canvas as static image
3. Update cache only when:
   - Scores change (batch updates, debounce 200ms)
   - New segments added (batch updates)
4. Viewport rectangle and dots update via SVG overlay (low cost)

**Optimization Targets**:
- Target: Render heatmap for 100-page, 500-segment PDF in <500ms
- Segment deduplication: avoid processing overlapping segments twice
- Color quantization: reduce color space to 256 levels (8-bit)
- Lazy rendering: only update visible heatmap region for very large PDFs (future enhancement)

**Memory**:
- Canvas: ~(96px * 1131px * 4 bytes) = ~416KB for A4 100-page
- Segment dot array: ~500 * 16 bytes = ~8KB
- SVG overlay: <10KB
- Total: <700KB - acceptable for most systems

### Data Structures

**Highlight Type** (existing, reference):
```typescript
type Highlight = {
  id: number;
  text: string;
  polygons: Record<number, number[][]>;  // 0-indexed page → list of polygons
  score?: number;                        // Added when scoring completes
};
```

**Heatmap Segment** (internal):
```typescript
interface HeatmapSegment {
  highlightId: number;
  pageNum: number;           // 0-indexed
  polygonPoints: number[][];
  score: number;
  hasImage: boolean;
}
```

**Segment Dot** (internal):
```typescript
interface SegmentDot {
  highlightId: number;
  pageNum: number;
  normalizedX: number;
  normalizedY: number;
  hasImage: boolean;
}
```

---

## Non-Functional Requirements

### Performance
- Heatmap render: <500ms for 100-page PDFs with 500+ segments
- Viewport rect update: 60 FPS (16ms frame budget)
- Navigation scroll: complete in <300ms with smooth easing
- Memory footprint: <1MB per document
- No main-thread blocking during heatmap rendering

### Accessibility
- Collapse/expand button has clear `aria-label`
- Heatmap canvas has `role="img"` with descriptive title
- Viewport rectangle has `aria-label` with scroll position description
- Keyboard support: Tab to toggle button, hotkey for collapse (e.g., Ctrl+H)
- Color contrast: heatmap colors meet WCAG AA standards

### Browser Support
- Chrome/Chromium 90+
- Firefox 88+
- Safari 14+

### Responsiveness
- Adapts to different window sizes (<1024px screens may hide heatmap by default)
- Mobile consideration: Optional touch drag support for viewport rect

---

## Current State Analysis

### What Exists (from codebase study)
1. **PdfViewer.vue**: Core document viewer with page tracking
   - IntersectionObserver-based page visibility detection
   - Real-time scroll position via `currentPage` tracking
  - `pdfEmbedWrapper` ref for PDF content area measuring
   - `pageAspectRatio` computed (A4 default 1.4142)
  - `pdfWidth` for responsive sizing
   - Debounced page intersection updates (150ms)

2. **Highlight Structure**: Segments with normalized polygons
   - `highlight.polygons`: Record<pageIndex, polygon[][]>
   - `highlight.score`: Optional, arrives via backend
   - Already in use by HighlightLayer for rendering

3. **SVG Rendering Pattern**: Established in HighlightLayer
   - Absolute positioned page containers
   - SVG overlay approach for interactive elements
   - Viewport-aware rendering (only render visible regions)

4. **ImageLayer Integration**: Image positioning system
  - `editorStates`: array tracking imageUrl state per highlight
   - Viewport-aware styling
   - Coordinate transformation utilities (hints in ImageLayer.vue)

5. **Coordinate System**: Normalized to pixel space
   - Standard pattern: normalized [0,1] → pixel multiplication
   - Page refs provide actual element dimensions
   - Aspect ratio calculations established

### What Needs to Be Built
1. **HeatmapViewer.vue**: Complete new component
2. **Coordinate mapping utilities**: Normalized → heatmap pixel space
3. **Canvas heatmap rendering**: Pre-render background
4. **Viewport rectangle tracking**: Current view indicator
5. **Image position dots**: Visual indicators for image-bearing segments
6. **Click-to-navigate handler**: Smooth scroll integration
7. **Drag-to-scroll handler**: Viewport rect dragging
8. **Integration in PdfViewer template**: Add fixed overlay without layout changes

---

## Key Files & Patterns

### Reference Components

**[PdfViewer.vue](services/frontend/app/components/PdfViewer.vue)**
- **Lines 108-118**: Scroll tracking via IntersectionObserver
  - `pageVisibility`: Record<number (1-indexed), boolean>
  - `currentPage`: ref<number> (1-indexed, updated by intersection observer)
  - `pageIntersectionRatios`: Map<number, number> for determining dominant page
  - **Pattern**: IntersectionObserver with high-frequency threshold thresholds ([0, 0.1, 0.25, 0.5, 0.75, 1.0])
  - Debounced update: 150ms to reduce state churn

- **Lines 129-137**: Page size and aspect ratio
  - `pdfPage Height = pdfWidthScaled.value * pageAspectRatio.value`
  - Pre-fetches first page to get accurate aspect ratio
  - Updates on PDF load completion

- **Lines 80-85**: Flex layout with ImageLayer
  ```vue
  <div class="flex bg-base-100">
    <div :style="{ width: `${pdfWidth}px`}" />  <!-- PDF container -->
    <ImageLayer :style="{ width: IMAGES_WIDTH + 'px' }" />  <!-- Fixed sidebar -->
  </div>
  ```

- **Lines 1-10**: Top layout structure
  - pdfViewer ref tracks outer container
  - pdfEmbedWrapper tracks PDF content area (not scroll container)

**[HighlightLayer.vue](services/frontend/app/components/HighlightLayer.vue)** (SVG rendering reference)
- **Lines 30-43**: Normalized → pixel coordinate transformation
  - Reads element.getBoundingClientRect() for page dimensions
  - Multiplies normalized polygon coords: `normalizedCoord * actualPageWidth`
  - Creates page-relative positioning with absolute container

- **Lines 70-120**: Page-based SVG rendering
  - Separates highlights by pageNum
  - Positions each page's SVG absolutely relative to page element
  - Uses clipPath for polygon rendering

**[ImageLayer.vue](services/frontend/app/components/ImageLayer.vue)** (viewport awareness pattern)
- **Lines 50-100**: Editor position calculation considering viewport
  - Multi-step calculation: normalized coords → actual page height → global position → relative to container
  - Uses pageRefs to get actual element dimensions
  - Fallback to estimated height if element unavailable

- **Lines 150-200**: Drag handling with scroll sync
  - Tracks page coordinates (e.pageX/pageY) during drag
  - Recalculates on scroll events for proper tracking
  - Maintains invariant that position stays valid during scroll

**[pages/index.vue](services/frontend/app/pages/index.vue)** (state management)
- **Lines 136-138**: Component references
  - `pdfViewer` ref for accessing internal state
  - Exposes `imageLayer` and `highlightLayer` via template ref access

### Unique Utilities to Create

**File: `app/utils/heatmapUtils.ts`**
```typescript
// Render heatmap to HTMLCanvasElement
function renderHeatmapCanvas(
  segments: HeatmapSegment[],
  heatmapWidth: number,
  pageAspectRatio: number,
  totalPageCount: number
): HTMLCanvasElement

// Convert score to brightness value (0-255)
function scoreToBrightness(score: number): number
```

---

## Existing Patterns to Follow

### 1. Coordinate Transformation Pattern
**In the codebase**: HighlightLayer.vue (lines 60-75), ImageLayer.vue (lines 60-100)

**Pattern**:
- Store data in normalized [0,1] space
- Transform to pixel space only when rendering/measuring
- Use actual DOM element dimensions (getBoundingClientRect) for accuracy
- Cache aspect ratio to avoid recalculation

**Application in HeatmapViewer**:
- Segments stored with normalized polygons (already done in Highlight type)
- Transform to heatmap pixel space for rendering
- Use pageAspectRatio * page count for total heatmap height

### 2. Overlay Pattern
**In the codebase**: HighlightLayer.vue, ImageEditor.vue

**Pattern**:
- Create absolute-positioned container aligned to parent page regions
- Use SVG for vector rendering (crisp, scalable)
- Keep separate from main PDF content for z-index control

**Application in HeatmapViewer**:
- HeatmapViewer is fixed overlay, independent of flex layout
- Use SVG for viewport rectangle and image dots
- Use Canvas for raster heatmap background
- Layer: Canvas background → SVG overlay (dots + viewport rect)

### 3. Scroll Synchronization Pattern
**In the codebase**: PdfViewer.vue IntersectionObserver, ImageLayer.vue scroll tracking

**Pattern**:
- Track scroll via DOM events + element dimensions
- Use requestAnimationFrame for smooth updates
- Maintain invariant: visual position matches scroll state

**Application in HeatmapViewer**:
- Listen to window scroll events
- Calculate viewport rect height/position from scrollY
- Update SVG viewport rect on scroll (not cache-busting main heatmap)
- Use RAF callback for 60fps updates

### 4. Reactive State Pattern
**In the codebase**: PdfViewer.vue reactive refs, ImageEditor.vue Composition API

**Pattern**:
- Use `ref()` for mutable state
- Use `computed()` for derived data
- Use `watch()` for side effects (API calls, DOM updates)

**Application in HeatmapViewer**:
- `isExpanded` ref for visibility
- `heatmapCanvas` ref for rendered image
- `segmentDots` computed from highlights and editorStates
- Watch on highlights + editorStates changes to invalidate cache

### 5. Component Exposure Pattern
**In the codebase**: ImageLayer.vue lines 270-280 (defineExpose)

**Pattern**:
- Use `defineExpose()` to expose internal methods to parent
- Typically for data queries or actions required by parent

**Application in HeatmapViewer**:
- No component exposure required for v1

---

## Architectural Decisions Made

### 1. Fixed Overlay vs. Sidebar
**Decision**: Implement as a fixed left overlay, not a layout-changing sidebar

**Rationale**:
- Do not alter existing layout sizing
- Keeps integration changes minimal
- Collapse control mitigates overlay coverage
- Consistent with fixed-position controls elsewhere

### 2. Brightness-Based Encoding
**Decision**: Use brightness (darkness) to encode segment scores, not color hue

**Rationale**:
- Works better for large datasets (500+ segments) - avoids color banding
- Reduces visual complexity compared to color gradients
- Accessible to colorblind users
- Easier to render efficiently (grayscale = single channel)
- Proven UX pattern in thermal imaging/heatmaps

### 3. Opacity-Based Blending
**Decision**: Use per-point opacity so overlaps blend visually without complex accumulation

**Rationale**:
- Overlaps darken naturally, matching heatmap expectations
- Simple to implement with canvas globalAlpha
- Avoids expensive per-pixel accumulation logic

### 4. SVG Overlay for Dynamic Elements
**Decision**: Use SVG for viewport rectangle and image dots, not Canvas

**Rationale**:
- Cheaper to update frequently (viewport rect on scroll)
- Easier to apply styling (opacity, transitions, hover effects)
- Simpler to debug (inspect SVG elements)
- Cleaner separation: static heatmap (Canvas) + dynamic UI (SVG)

### 5. Separate Image Dots Array
**Decision**: Compute image dots in component, not query ImageLayer directly per-frame

**Rationale**:
- Decouples HeatmapViewer from ImageLayer internals
- Cleaner computed property pattern
- Easier to test and reason about
- Cached until editorStates imageUrl changes

### 6. Click-to-Navigate via Emit
**Decision**: Emit navigate event to parent, parent handles scroll

**Rationale**:
- Keeps scroll logic centralized in PdfViewer
- Reuses existing scrollIntoView utility function
- Parent controls animation timing/easing
- Easier to test independently

### 7. Drag-to-Scroll Coordination
**Decision**: Map viewport rectangle Y position linearly to PDF scroll position

**Rationale**:
- Intuitive 1:1 mapping (users expect proportional scrolling)
- Matches native scrollbar behavior
- Simple math (avoid complex animation curves)
- Consistent with viewport rect height calculation

---

## Technical Considerations

### Canvas Support
Use HTMLCanvasElement only for v1

### Scroll Event Performance
- window scroll fires frequently (multiple times per second)
- Debounce heatmap cache invalidation to avoid thrashing
- Use RAF for viewport rect updates (naturally throttled to 60fps)

### Coordinate System Edge Cases
- Segments spanning multiple pages: show a single dot (first polygon only)
- Single-point segments: render as dot (no area)
- Floating-point rounding: quantize to pixel boundaries to avoid sub-pixel rendering
- Page boundary alignment: ensure cumulative heights match total document height

### Touch Support (Future Enhancement)
- Current design assumes mouse/pointer events
- Touch drag-to-scroll can be added later via PointerEvent API
- Same API surface (pointermove, pointerup listeners)

---

## Success Metrics

### User-Facing
1. ✓ "I can see at a glance where the important segments are" (heatmap overview)
2. ✓ "I can click and jump to any segment instantly" (navigation latency <300ms)
3. ✓ "I can scroll with the heatmap like a scrollbar" (viewport rect dragging)
4. ✓ "I can hide it when I need space" (collapse/expand mechanism)

### Technical
1. ✓ Render heatmap for 100-page PDF in <500ms (performance target)
2. ✓ Viewport rect updates at 60 FPS during scroll (smoothness)
3. ✓ Memory footprint <1MB per document (efficiency)
4. ✓ No TypeScript errors, comprehensive prop validation (type safety)

---

## Implementation Constraints

### Scope Boundaries
**In Scope**:
- HeatmapViewer component and SVG/Canvas rendering
- Click-to-navigate and drag-to-scroll interactions
- Collapse/expand UI toggle
- Image position dots
- Integration with existing PdfViewer/ImageLayer

**Out of Scope** (future enhancements):
- Custom color schemes/theming for heatmap
- Heatmap data export/saving
- Touch gesture support (pinch, multi-finger)
- Server-side heatmap rendering (pre-computed from API)

### Dependencies
- Existing: Vue 3, TypeScript, Tailwind CSS
- New: None (use native Canvas, SVG)
- Assumptions: window scroll position accessible, highlights always have polygons

---

## Definitions & Terminology

- **Normalized Coordinates**: [0,1] range relative to PDF page dimensions (independent of zoom)
- **Heatmap Pixel Space**: Absolute pixel coordinates in the heatmap canvas (96px width, N*aspect_ratio height)
- **Viewport Rectangle**: Semi-transparent overlay showing current PDF scroll region
- **Image Position Dots**: Visual indicators of segments with associated illustrations
- **Brightness Encoding**: Score-to-pixel-value mapping where higher score = darker pixel
- **Page Aspect Ratio**: Height/Width ratio (typically 1.4142 for A4)
