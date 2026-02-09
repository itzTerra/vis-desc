# PDF Heatmap Visualization - Implementation Tasks

## Overview
This document contains ordered, atomic tasks for implementing the PDF heatmap visualization feature. Tasks are grouped by implementation phase and mapped to acceptance criteria.

---

## Phase 1: Component Foundation & Structure

### Task 1.1: Create HeatmapViewer.vue component scaffold
- [x] Create file `services/frontend/app/components/HeatmapViewer.vue`
- [x] Define props: `highlights`, `currentPage`, `pageAspectRatio`, `pageRefs`, `editorStates`
- [x] Define emits: `navigate` event
- [x] Set up template with:
  - Outer container div with fixed width (96px) and fixed positioning
  - Canvas element for heatmap background
  - SVG overlay container for viewport rect + image dots
  - Toggle button (protrudes left with negative left value)
- [x] Implement basic Composition API setup with refs for `isExpanded`, `heatmapCanvas`
- [x] Add TypeScript types for internal state (HeatmapSegment, SegmentDot interfaces)
- **Scope**: ~80 LoC, no rendering logic yet
- **Acceptance**: Component mounts without errors, props are typed

### Task 1.2: Add HeatmapViewer to PdfViewer template
- [x] Import HeatmapViewer in [PdfViewer.vue](services/frontend/app/components/PdfViewer.vue#L1)
- [x] Do not change existing layout sizing or flex structure
- [x] Add HeatmapViewer as a viewport-fixed overlay on the left edge
  - Positioning: `fixed`, `top` aligned with PDF viewer area
  - Height: match the PDF viewer area height (not full document)
  - Slim semi-transparent banner that can overlay the PDF content
  - Use high z-index to stay above the PDF canvas
- [x] Pass required props: `highlights`, `currentPage`, `pageAspectRatio`, `pageRefs`, `editorStates`
- [x] Verify layout doesn't break existing functionality
- **Scope**: ~20 LoC modifications
- **Acceptance**: PdfViewer renders with heatmap overlay on left, PDF and ImageLayer layout unchanged

### Task 1.3: Implement collapse/expand toggle button
- [x] Add toggle button styled with negative `left` property (similar to ImageEditor pattern)
  - Button outer container: `position: absolute`, `left: -26px`, `top: 8px`
  - Button size: `btn btn-sm btn-circle`
- [x] Style toggle based on `isExpanded` state
  - Show chevron-right when collapsed, chevron-left when expanded
- [x] Bind click handler: `@click="isExpanded = !isExpanded"`
- [x] Add CSS transition for smooth collapse/expand (200ms)
  - Use `transition-all` and `overflow: hidden` on content
  - Or wrap content in `<Transition name="collapse">`
- [x] Verify toggle button always visible (protrudes from edge)
- **Scope**: ~30 LoC (template + styles)
- **Acceptance**: Button visible, toggle works, smooth animation

---

## Phase 2: Coordinate System & Utilities

### Task 2.1: Create heatmapUtils.ts utility file
- [x] Create file `services/frontend/app/utils/heatmapUtils.ts`
- [x] Export function `normalizedToHeatmapPixel()`
  - Input: normalizedX, normalizedY, pageNum (0-indexed), pageAspectRatio, heatmapWidth, heatmapHeight, totalPageCount
  - Calculate single page height: `pageAspectRatio * heatmapWidth`
  - Calculate cumulative page height: `pageNum * singlePageHeight`
  - Add offset within page: `normalizedY * pageHeight`
  - Return: `{ x: normalizedX * heatmapWidth, y: cumulativeOffset + withinPageOffset }`
  - Handle edge case: return undefined if pageNum >= totalPageCount
- [x] Export function `getViewportPercentage()`
  - Input: scrollY (from window.scrollY), totalHeight (cumulative page heights), viewportHeight
  - Calculate: `topPercent = scrollY / totalHeight`, `heightPercent = viewportHeight / totalHeight`
  - Return: `{ topPercent, heightPercent, topPixels, heightPixels }`
  - Clamp values to [0, 1]
- [x] Export function `heatmapPixelToNormalized()`
  - Inverse of normalizedToHeatmapPixel
  - Input: pixelX, pixelY, pageAspectRatio, heatmapHeight, totalPageCount
  - Calculate: which page from cumulative height, normalized coords within page
  - Return: `{ page (0-indexed), normalizedX, normalizedY }`
- [x] Export function `scoreToOpacity()`
  - Input: score (0-1 range)
  - Formula: `opacity = 0.15 + (score * 0.75)` (results in range ~0.15-0.9)
  - Clamp result to [0.15, 0.9]
- [x] Export function `renderHeatmapCanvas()`
  - Input: `segments: HeatmapSegment[]`, `heatmapWidth`, `pageAspectRatio`, `totalPageCount`
  - Create 2D context from HTMLCanvasElement
  - Initialize canvas: dimensions = heatmapWidth × (totalPageCount * pageAspectRatio * heatmapWidth)
  - Fill default background: light gray (#E8E8E8)
  - For each segment:
    - For each polygon point in segment.polygons:
      - Transform to heatmap pixel space (use normalizedToHeatmapPixel)
      - Set `ctx.globalAlpha` from scoreToOpacity(score)
      - Draw small rectangle (2x2) to allow opacity-based blending
  - Reset `globalAlpha` after render
  - Return: Canvas element
  - Error handling: catch rendering errors, log warnings, return blank canvas
  - Use ImageData batching if needed for performance
- [x] Export function `scoreToBrightness()`
  - Input: score (0-1 range)
  - Formula: `brightness = 220 - (score * 180)` (results in range ~40-220)
  - Return brightness value (0-255 integer)
  - Clamp result to [40, 220] to avoid pure black/white
- [x] Add helper: `createSegmentArray(highlights: Highlight[]): HeatmapSegment[]`
  - Transform highlights into renderable segments with scores
  - Filter out segments with score === undefined
  - Return sorted array for consistent rendering
- **Scope**: ~140 LoC
- **Acceptance**: Renders test heatmap without errors, benchmark shows <500ms for 500 segments

---

## Phase 3: Heatmap Rendering

### Task 3.1: Implement canvas rendering in HeatmapViewer
- [x] Add canvas element to template: `<canvas ref="canvasElement" :width="HEATMAP_WIDTH" :height="computedHeight" />`
- [x] Add computed property `computedHeight()`: `totalPages * pageAspectRatio * HEATMAP_WIDTH`
- [x] Implement `renderHeatmap()` function:
  - Call `renderHeatmapCanvas()` from heatmapUtils.ts with normalized highlights
  - Draw result to canvas context: `ctx.drawImage(resultCanvas, 0, 0)`
  - Store reference in state for later reuse
- [x] Add watcher on highlights (debounced 200ms):
  - When highlights change, invalidate heatmap cache
  - Trigger re-render if expanded
  - Debounce to avoid thrashing during batch score updates
  - Use `deep: true` so score updates on existing items trigger re-render
- [x] Verify canvas displays correctly in expanded state
- **Scope**: ~60 LoC
- **Acceptance**: Heatmap renders visible in UI, updates when scores change
- **Implementation Notes**: Used watchDebounced from @vueuse/core, proper totalPages calculation accounting for 0-indexed pages

### Task 3.3: Add background styling and container layout
- [x] Style heatmap container:
  - Background: light gray with slight transparency (e.g., rgba(245,245,245,0.7)) when collapsed
  - Border: subtle (#D0D0D0) on right edge
  - Width: fixed 96px
  - Height: match PDF viewer area height
  - Position: fixed (entire component, including toggle)
  - Top offset: align to PDF viewer area top
- [x] Style canvas element:
  - Display: none when collapsed, block when expanded
  - Width/height: 100% fill container
  - Cursor: pointer for interaction feedback
  - Smooth filter transition on hover (optional: slight brightness increase)
- [x] Keep the toggle button and heatmap fixed in the viewport
- **Scope**: ~25 LoC (template + styles)
- **Acceptance**: Canvas displays with correct dimensions, respects collapse state
- **Implementation Notes**: Used inline styles for dynamic background color, Tailwind classes for layout and transitions

---

## Phase 4: Image Position Indicators

### Task 4.1: Track image positions from ImageLayer
- [x] In HeatmapViewer, derive image state from reactive props
  - Decision: use `editorStates` prop (array of EditorState) for image presence
  - If props are insufficient, use global state (no direct component querying)
- [x] Create computed property `segmentDots()`:
  - Input: highlights, editorStates, heatmapWidth
  - For each highlight in highlights:
    - Find editor state by highlightId
    - Determine `hasImage`: `Boolean(editorState?.imageUrl)`
    - Compute centroid of the first polygon on the first page only
    - Transform centroid to heatmap pixel coordinates
    - Add to segmentDots array: `{ highlightId, pageNum, normalizedX, normalizedY, hasImage }`
  - Return array of SegmentDot objects
- [x] Add defensive checks:
  - Handle undefined polygons (skip segments)
  - Handle empty polygon arrays (skip segments)
  - Clamp normalized coordinates to [0, 1] range
- **Scope**: ~60 LoC
- **Acceptance**: Computed property correctly identifies segments with/without images

### Task 4.2: Render image position dots to SVG
- [x] Add SVG element to template:
  ```vue
  <svg v-if="isExpanded" :viewBox="`0 0 ${HEATMAP_WIDTH} ${computedHeight}`" class="absolute inset-0 w-full h-full pointer-events-none">
    <circle v-for="dot in segmentDots" :key="dot.highlightId" :cx="dot.x" :cy="dot.y" r="5" class="image-dot" />
  </svg>
  ```
- [x] Transform segmentDots coordinates to SVG pixel space (already in heatmap pixels from Task 4.1)
- [x] Style image-dot class:
-  - Use different symbols for image vs no-image:
    - With imageUrl: filled circle (gold) with dark stroke
    - Without imageUrl: hollow circle or diamond with lighter stroke
  - Opacity: 0.85 (slightly transparent)
  - Hover: opacity 1.0, larger radius (6px)
- [x] Verify dots appear in correct positions on heatmap
- [x] Add accessibility: set `aria-label` on dots describing segment/image count
- **Scope**: ~35 LoC
- **Acceptance**: Dots render visibly on heatmap, positioned correctly

### Task 4.3: Update dots reactively on imageEditor changes
- [x] Reactivity is automatic through `editorStates` prop updates
- [x] Test workflow:
  - Open PDF, score segments
  - Click ImageEditor button on segment
  - Verify dot appears on heatmap at correct position
  - Close or collapse editor (dot should persist if imageUrl exists)
  - Remove image from editor, dot updates to no-image symbol
- **Scope**: ~15 LoC (auto-handled by reactivity)
- **Acceptance**: Dots sync with editor state

---

## Phase 5: Viewport Indicator

### Task 5.1: Render viewport rectangle to SVG
- [x] Add SVG rectangle to template:
  ```vue
  <rect v-if="isExpanded" :x="0" :y="viewportY" :width="HEATMAP_WIDTH" :height="viewportHeight" class="viewport-rect" />
  ```
- [x] Create computed property `viewportY()`:
  - Call `getViewportPercentage()` with current scroll position
  - Convert percentage to pixel: `topPercent * computedHeight`
  - Return Y position
- [x] Create computed property `viewportHeight()`:
  - Call `getViewportPercentage()` 
  - Convert to pixel: `heightPercent * computedHeight`
  - Add minimum height to ensure visibility (e.g., 20px minimum)
  - Return height
- [x] Style viewport-rect:
  - Fill: rgba(74, 144, 226, 0.15) (light blue, 15% opacity)
  - Stroke: 2px #4A90E2 (medium blue)
  - Pointer-events: all (for dragging, handled in Task 5.3)
  - Smooth transition (but disable during drag for responsiveness)
- [x] Verify rectangle displays at correct position matching PDF scroll
- **Scope**: ~40 LoC
- **Acceptance**: Rectangle visible and positioned correctly

### Task 5.2: Sync viewport rectangle with scroll events
- [x] Add scroll event listener to window:
  - Listen for scroll events on window
  - Update viewportY and viewportHeight computed values
  - Use requestAnimationFrame for smooth updates (60 FPS)
  - Store scroll position in reactive state: `scrollY`
  - Derive `viewportHeight` from PDF viewer area height
  - Derive `totalHeight` from cumulative page heights (pageRefs)
- [x] Add lifecycle management:
  - `onMounted()`: register scroll listener
  - `onBeforeUnmount()`: remove scroll listener
- [x] Optimization:
  - Use passive event listener: `{ passive: true }`
  - Utilize requestAnimationFrame callback for natural FPS throttling
  - Avoid excessive state updates with debounce (16ms)
- [x] Test:
  - Scroll PDF slowly, verify rectangle follows smoothly
  - Rapid scroll, verify no jank or lag
- **Scope**: ~45 LoC
- **Acceptance**: Rectangle updates smoothly on scroll, no main-thread blocking

### Task 5.3: Implement drag-to-scroll for viewport rectangle
- [x] Add pointer event handlers to SVG rect:
  - `@pointerdown="startDrag"` 
  - Listen for `pointermove` on window during drag
  - Listen for `pointerup` to end drag
- [x] Implement `startDrag()`:
  - Record initial pointer position: `startY = e.clientY`
  - Record initial rectangle position
  - Set capture: `e.target.setPointerCapture(e.pointerId)`
  - Store dragging state
- [x] Implement `onDrag()`:
  - Calculate delta: `deltaY = currentClientY - startY`
  - Map to scroll position: `newScrollY = dragStartScrollY + (deltaY / heatmapHeight * totalHeight)`
  - Clamp to valid range: [0, totalScrollableHeight - viewportHeight]
  - Update `window.scrollTo({ top: newScrollY })`
- [x] Implement `endDrag()`:
  - Release pointer capture
  - Remove listeners
  - Restore smooth scroll behavior
- [x] Testing:
  - Drag rectangle up/down
  - Verify PDF scrolls proportionally
  - Test edge cases (drag at top, bottom, rapid drags)
  - Verify cursor feedback (grab/grabbing)
- **Scope**: ~80 LoC
- **Acceptance**: Dragging scrolls PDF smoothly and proportionally

---

## Phase 6: Click Navigation

### Task 6.1: Implement click-to-navigate handler
- [x] Add click handler to canvas:
  - `@click="handleHeatmapClick"`
- [x] Implement `handleHeatmapClick()`:
  - Get click coordinates relative to canvas: `e.offsetX`, `e.offsetY`
  - Convert to normalized coordinates: call `heatmapPixelToNormalized(offsetX, offsetY, ...)`
  - Extract page and normalizedY from result
  - Emit navigate event: `emit('navigate', page, normalizedY)`
- [x] In parent (PdfViewer.vue):
  - Listen for navigate event: `@navigate="handleNavigate"`
  - Implement `handleNavigate(page, normalizedY)`:
    - Find corresponding highlight closest to page + normalizedY
    - Call existing `scrollIntoView()` utility with smooth behavior
    - Duration: ~300-400ms
- [x] Add visual feedback:
  - Change canvas cursor to pointer: `cursor: pointer`
  - Optional: add brief highlight/flash on clicked region
- **Scope**: ~50 LoC in HeatmapViewer + ~20 LoC in PdfViewer
- **Acceptance**: Click navigates PDF, smooth scroll completes in <300ms



---

## Phase 7: Collapse/Expand

### Task 7.1: Implement smooth collapse/expand animation
- [x] Use Vue Transition or CSS transitions:
  - Wrap main content in `<Transition name="expand">`
  - Define CSS transitions in component:
    ```css
    .expand-enter-active, .expand-leave-active {
      transition: opacity 200ms ease, width 200ms ease;
    }
    .expand-enter-from, .expand-leave-to {
      opacity: 0;
      width: 0;
    }
    ```
  - Alternative: use `overflow: hidden` + max-height transition
- [x] Toggle visibility:
  - `v-show="isExpanded"` or conditional rendering
  - Decision: use v-show for smoother animation (no re-mount)
- [x] Button animation:
  - Icon rotates: chevron-right (collapsed) → chevron-left (expanded)
  - Use CSS transform rotate or icon component prop
- [x] Test animation smoothness:
  - Visual inspection at various zoom levels
  - No layout shift or jank
  - Completes in ~200ms
- **Scope**: ~30 LoC
- **Acceptance**: Smooth 200ms animation, no visual artifacts

---

## Phase 8: Integration

### Task 8.1: Integration with PdfViewer state management
- [x] Verify all prop passing is correct:
  - highlights: should update when new scores arrive
  - currentPage: tracks visible page via IntersectionObserver
  - pageAspectRatio: computed correctly in PdfViewer
  - window scroll: accessible and scroll position readable
  - pageRefs: passed correctly for dimension calculations
- [x] Wire up navigate event:
  - PdfViewer listens to HeatmapViewer navigate event
  - Calls `scrollIntoView()` to smoothly navigate
- [x] Test data flow:
  - Add new highlights
  - Scores update in real-time (from backend)
  - Heatmap updates reflect new scores immediately
- **Scope**: ~30 LoC verification/testing
- **Acceptance**: Full integration works end-to-end

### Task 8.2: Integration with ImageEditor state
- [x] Verify editor state access via props:
  - PdfViewer passes editorStates to HeatmapViewer
- [x] Test image dots:
  - Open ImageEditor on a segment
  - Verify dot appears on heatmap at correct position
  - Close editor, verify dot remains if imageUrl exists
  - Remove image, verify dot switches to no-image symbol
  - Open multiple editors, verify multiple dots
- [x] Handle edge cases:
  - Segment with no polygon (skip)
  - Segment spanning multiple pages (show one dot only)
  - No images: heatmap appears without dots
- **Scope**: ~25 LoC integration code
- **Acceptance**: Image dots sync correctly with editor state

### Task 8.5: Address review feedback (image state + event naming)
- [x] Emit minimal image state payload (highlightId + imageUrl/hasImage)
- [x] Store minimal image state array in PdfViewer and pass to HeatmapViewer
- [x] Standardize kebab-case event naming for editor state updates

### Task 8.3: Manual testing checklist
- [ ] Desktop testing:
  - Chrome/Chromium
  - Firefox
  - Safari (if available)
- [ ] Test scenarios:
  - Load multi-page PDF, verify heatmap height matches
  - Add/update scores, verify heatmap updates
  - Click heatmap regions, verify navigation
  - Drag viewport rect, verify smooth scrolling
  - Collapse/expand multiple times, verify animation
  - Open multiple ImageEditors, verify dots update
  - Scroll PDF manually, verify viewport rect follows
- [ ] Performance checks:
  - 100-page PDF: heatmap renders <500ms
  - No lag during scroll (60 FPS)
  - Memory profiler: heatmap <1MB
- [ ] Accessibility:
  - Tab navigation works
  - Screen reader announces toggle button
  - Color contrast meets WCAG AA
- **Scope**: Comprehensive manual testing (no code)
- **Acceptance**: All tests pass, no regressions

### Task 8.4: Update documentation and types
- [ ] Update type definitions:
  - Ensure HeatmapViewer props are exported in types
  - Add JSDoc comments to utility functions
- [ ] Update README or component documentation:
  - Brief overview of feature
  - Props and events API
  - Known limitations/browser support (Canvas)
- [ ] Add comments in code:
  - Explain coordinate transformation logic
  - Document assumptions (e.g., normalized coords stored in highlights)
- [ ] Update AGENTS.md with HeatmapViewer integration details
- **Scope**: ~50 LoC documentation
- **Acceptance**: Documentation complete and accurate

---


---

## Summary of Task Organization

| Phase | Focus | Tasks | Est. LoC | Est. Days |
|-------|-------|-------|---------|----------|
| 1 | Setup | 1.1-1.3 | 125 | 1 |
| 2 | Utilities | 2.1 | 140 | 1 |
| 3 | Rendering | 3.1-3.3 | 85 | 1 |
| 4 | Image Dots | 4.1-4.3 | 110 | 1 |
| 5 | Viewport | 5.1-5.3 | 165 | 1.5 |
| 6 | Navigation | 6.1 | 70 | 0.5 |
| 7 | UI Polish | 7.1 | 30 | 0.5 |
| 8 | Integration | 8.1-8.4 | 135 | 1 |
| **Total** | | **15 tasks** | **860 LoC** | **6.5 days** |

---

## Task Dependency Graph

```
1.1 --> 1.2 --> 1.3
     |
     +--> 2.1 --> 3.1 --> 3.3 --> 8.1
        |
        +--> 4.1 --> 4.2 --> 4.3 --> 8.2
          |
          +--> 5.1 --> 5.2 --> 5.3
                  |
                  +--> 6.1
                    |
                    +--> 7.1 --> 8.3 --> 8.4
```

**Critical Path** (minimum time, no parallelization):
1.1 → 1.2 → 2.1 → 3.1 → 3.3 → 4.1 → 4.2 → 5.1 → 5.2 → 5.3 → 6.1 → 7.1 → 8.1 → 8.3

**Parallelizable Work Groups**:
- Phase 2 tasks (utilities) can start immediately after 1.2
- Phase 4 and 5 tasks (dots, viewport) can run in parallel after Phase 2
- Phase 6 (navigation) can start after Phase 3
- Phase 8 (testing) requires all implementation phases complete
- Phase 8 depends only on Phase 8.1

---

## Completion Criteria

All tasks in this checklist must be completed and validated before feature is production-ready:
- [ ] All 15 tasks completed and passing acceptance criteria  
- [ ] Manual testing checklist passed
- [ ] TypeScript compilation with zero errors
- [ ] Pre-commit hooks pass (linting, formatting)
- [ ] No regressions in existing functionality (PdfViewer, ImageLayer, HighlightLayer)
- [ ] Performance benchmarks met (<500ms render for 100-page, 500-segment PDFs)
- [ ] Accessibility review passed (WCAG AA)
- [ ] Documentation complete and reviewed
