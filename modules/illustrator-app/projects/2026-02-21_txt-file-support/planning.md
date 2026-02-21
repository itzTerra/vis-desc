---
type: feature
size: M
---

# TXT File Support

## Overview

Add `.txt` file upload support alongside existing PDF support. When a user uploads a `.txt` file, the backend creates a PDF from the raw text, aligns segments with the generated PDF pages, and returns both the segment data and the PDF (as base64) so the frontend can display it in the existing PdfViewer.

## Goals

- Allow users to upload `.txt` files (e.g., Project Gutenberg plain-text books) and get the same segmentation, scoring, and visualization experience as with PDFs
- Keep the existing PDF workflow completely unchanged (beyond a thread-safety refactor)
- Minimize code duplication between PDF and TXT processing paths by delegating alignment from TxtBookPreprocessor to PdfBookPreprocessor
- Eliminate per-request mutable state on singleton preprocessors (thread-safety fix)

## User Stories

### As a user, I want to upload a .txt file so that I can analyze plain-text books

**Acceptance criteria:**

- [ ] Both file inputs (top-bar and Hero drop zone) accept `.txt` files in addition to PDFs
- [ ] After uploading a `.txt` file, the generated PDF renders in PdfViewer with segment polygons
- [ ] Segment scoring works identically to the PDF flow (socket-based or client-side)
- [ ] The loading stage text says "Processing TXT..." (not "Processing PDF...")
- [ ] During TXT processing, the Hero component remains visible (since both `pdfUrl` and `pdfFile` are only set after the API response returns the generated PDF, unlike the PDF flow where they are set immediately from the local file)

### As a user, I want to export an analyzed .txt book the same way I export a PDF

**Acceptance criteria:**

- [ ] After uploading a `.txt` file, the Export button works normally
- [ ] The exported filename derives from the original `.txt` filename (e.g., `moby-dick.txt` becomes `moby-dick-export`)

### As a user, I want clear guidance that both file types are accepted

**Acceptance criteria:**

- [ ] Hero heading mentions both PDF and TXT
- [ ] The file input accept attribute includes both types
- [ ] Drop zone hint text reflects both formats

## Requirements

### Functional

- New `POST /api/segment/txt` endpoint that accepts a `.txt` upload, returns segments with polygons plus `pdf_base64`
- New `POST /api/process/txt` endpoint that accepts a `.txt` upload, returns `ws_key`, segments with polygons, plus `pdf_base64`
- `TxtBookPreprocessor` gains `extract_from_memory(txt_file)` and `create_pdf_and_align(segments, txt_ctx, pdf_preprocessor)` methods
- `TxtBookPreprocessor.create_pdf_and_align` delegates alignment to `PdfBookPreprocessor` -- it generates PDF bytes from cleaned text, constructs an `ExtractionContext` directly from the already-computed normalization data (bypassing `pdf_preprocessor.extract_from_memory`), and calls `pdf_preprocessor.align_segments_with_pages(segments, ctx)`. No alignment code is duplicated or moved to the base class.
- `PdfBookPreprocessor` is refactored to be stateless: `extract_from_memory` returns an `ExtractionContext` dataclass instead of storing state on `self`, and `align_segments_with_pages` accepts that context as a parameter
- Dead code `PdfBookPreprocessor.extract_from_path` is removed (zero call sites in codebase, incompatible with stateless refactor)
- Frontend detects file type by extension or MIME type and calls the appropriate endpoint
- For `.txt` uploads, frontend waits for the API response to get the PDF bytes before setting `pdfUrl` and `pdfFile` (neither is set before the API call)
- `pdfFile` is constructed as `new File([decodedPdfBytes], 'original-name.pdf', { type: 'application/pdf' })` so the export flow works unchanged
- Keep existing `get_segments_with_boxes` name unchanged; add `get_segments_boxes_from_txt` as a new parallel helper

### Non-functional

- TXT encoding: try UTF-8 first, fall back to Latin-1 if decoding fails
- PDF generation should use a monospaced or readable font at a reasonable size so the generated PDF is legible
- Base64-encoded PDFs in JSON responses will increase payload size ~33% over raw bytes; acceptable for book-length texts. No file size limit is enforced, but large `.txt` files will produce large base64 payloads and slower responses.
- Use `base64.b64encode(pdf_bytes).decode()` (not `binascii.b2a_base64` which appends newlines) because the browser's `atob()` throws on embedded newlines

## Scope

### Included

- Backend: Thread-safety refactor of PdfBookPreprocessor (ExtractionContext), dead code removal (extract_from_path), TxtBookPreprocessor methods, new schemas, new endpoints
- Frontend: file input accept changes, file-type detection in handleFileUpload, base64 PDF decoding, Hero copy update
- OpenAPI type regeneration

### Excluded

- Other file formats (EPUB, DOCX, HTML)
- Drag-and-drop file upload (beyond existing drop-zone behavior)
- Encoding auto-detection beyond UTF-8/Latin-1
- Changes to the scoring pipeline, WebSocket flow, or PdfViewer component
- File size limits or upload validation beyond what Django provides by default

---

## Current State

The app only accepts PDF uploads. `PdfBookPreprocessor` handles extraction, cleaning, and polygon alignment. `TxtBookPreprocessor` exists but only overrides `_before_clean` for Gutenberg-style newline handling -- it has no extraction or alignment methods.

The polygon alignment pipeline (`_segments_to_page_polygons`, `_normalize`, `_smooth_boundary`, `_word_overlaps_removed_ranges`) lives entirely on `PdfBookPreprocessor`. Per-request state (`_last_pdf_bytes`, `_normalized_full_text`, `_removed_ranges`) is stored as instance attributes on a module-level singleton, which is a race condition under concurrent requests.

`PdfBookPreprocessor.extract_from_path` exists but has zero call sites in the codebase. It will become incompatible with the stateless refactor (it expects `_get_from_doc` to return `str` but the refactor changes it to return a tuple). It should be deleted.

Both `PdfBookPreprocessor` and `TxtBookPreprocessor` are duck-typed -- they share a conceptual `extract_from_memory` interface but there is no abstract base method enforcing it. This is intentional and will remain so. After the refactor, the return types differ: `PdfBookPreprocessor.extract_from_memory` returns `ExtractionContext`, while `TxtBookPreprocessor.extract_from_memory` returns a `(cleaned_text, normalized_full_text, removed_ranges)` tuple. The call sites (`get_segments_with_boxes` and `get_segments_boxes_from_txt`) are separate functions that know which preprocessor they are using.

## Key Files

- `services/api/core/tools/book_preprocessing.py` - BookPreprocessor hierarchy; PdfBookPreprocessor has all alignment logic and per-request state that needs refactoring; TxtBookPreprocessor needs new methods; `extract_from_path` (dead code) to be removed
- `services/api/core/api.py` - Endpoint definitions, `get_segments_with_boxes` helper, singleton preprocessors
- `services/api/core/schemas.py` - Pydantic/Ninja response DTOs; needs new txt response schemas
- `services/frontend/app/pages/index.vue` - `handleFileUpload`, `pdfFile`/`pdfUrl` state, export flow
- `services/frontend/app/components/Hero.vue` - Landing page file input and copy
- `services/frontend/app/types/schema.d.ts` - Auto-generated OpenAPI types (regenerated, not hand-edited)
- `services/frontend/app/types/common.d.ts` - Highlight and Segment types (no changes needed)

## Existing Patterns

**Endpoint pattern** (from `api.py`):
```python
@api.post("/segment/pdf", response=ProcessPdfSegmentsOnlyResponse)
def segment_pdf(request, pdf: File[UploadedFile], model: Form[ProcessPdfBody]):
    segments, segments_with_pos = get_segments_with_boxes(pdf)
    return {"segment_count": len(segments), "segments": segments_with_pos}
```

**Schema pattern** (from `schemas.py`):
```python
class ProcessPdfResponse(Schema):
    ws_key: str
    expires_in: int
    segment_count: int
    segments: list[SegmentWithPos]
```

**Frontend API call pattern** (from `index.vue`):
```ts
const formData = new FormData();
formData.append("pdf", file, file.name);
formData.append("model", selectedModel.value);
const data = await call(scorer.socketBased ? "/api/process/pdf" : "/api/segment/pdf", {
  method: "POST", body: formData as any
});
```

**Singleton preprocessor pattern** (from `api.py`):
```python
pdf_preprocessor = PdfBookPreprocessor(
    polygon_x_smooth_max_gap_px=settings.POLYGON_X_SMOOTH_MAX_GAP_PX,
    polygon_padding_px=settings.POLYGON_PADDING_PX,
)
```

## Decisions

### Alignment via Delegation (not inheritance)

**Decision**: `TxtBookPreprocessor.create_pdf_and_align` generates PDF bytes from the cleaned text, then delegates alignment to `PdfBookPreprocessor.align_segments_with_pages` by constructing an `ExtractionContext` directly. Zero changes to `BookPreprocessor` base class. No alignment code is moved or duplicated.

**Rationale**: The alignment pipeline is complex (~200 lines) and tightly coupled to `PdfBookPreprocessor`'s internal state. Moving it to the base class would be a large refactor with risk. Delegation keeps the blast radius small -- `TxtBookPreprocessor` produces a PDF and an `ExtractionContext`, then hands them to `PdfBookPreprocessor.align_segments_with_pages`.

### Text Consistency: Bypass Double-Cleaning

**Decision**: `TxtBookPreprocessor.extract_from_memory` computes and returns the normalization data (`cleaned_text`, `normalized_full_text`, `removed_ranges`) in a single pass. `create_pdf_and_align` constructs `ExtractionContext` directly from this already-computed data plus the generated `pdf_bytes` -- it does NOT call `pdf_preprocessor.extract_from_memory` to re-extract from the generated PDF.

**Rationale**: `PdfBookPreprocessor._get_from_doc` processes text page-by-page, applying `clean_text` per page with cross-page sentence-joining logic. `TxtBookPreprocessor.extract_from_memory` processes the full text at once. These two code paths can produce different `normalized_full_text` values for the same content, which would cause word-matching desynchronization in `_segments_to_page_polygons`. By reusing the normalization data from the original TXT extraction pass, we guarantee exact consistency.

**Implementation**:
```python
# TxtBookPreprocessor
def extract_from_memory(self, txt_file) -> tuple[str, str, list[tuple[int, int]]]:
    """Returns (cleaned_text, normalized_full_text, removed_ranges)"""
    ...

def create_pdf_and_align(
    self,
    segments: list[str],
    txt_ctx: tuple[str, str, list[tuple[int, int]]],
    pdf_preprocessor: PdfBookPreprocessor,
) -> tuple[list[dict], bytes]:
    cleaned_text, normalized_full_text, removed_ranges = txt_ctx
    pdf_bytes = self._text_to_pdf(cleaned_text)
    # Construct ExtractionContext directly -- no re-cleaning!
    ctx = ExtractionContext(
        pdf_bytes=pdf_bytes,
        cleaned_text=cleaned_text,
        normalized_full_text=normalized_full_text,
        removed_ranges=removed_ranges,
    )
    segments_with_pos = pdf_preprocessor.align_segments_with_pages(segments, ctx)
    return segments_with_pos, pdf_bytes

# In api.py
def get_segments_boxes_from_txt(txt):
    txt_ctx = txt_preprocessor.extract_from_memory(txt)
    cleaned_text = txt_ctx[0]
    segments = text_segmenter.segment_text(cleaned_text)
    segments_with_pos, pdf_bytes = txt_preprocessor.create_pdf_and_align(
        segments, txt_ctx, pdf_preprocessor
    )
    return segments, segments_with_pos, pdf_bytes
```

### Thread-Safety: ExtractionContext

**Decision**: Refactor `PdfBookPreprocessor` to be stateless. `extract_from_memory` returns an `ExtractionContext` dataclass instead of storing `_last_pdf_bytes`, `_normalized_full_text`, and `_removed_ranges` on `self`. `align_segments_with_pages` takes `ctx: ExtractionContext` as a parameter.

**Rationale**: Both `pdf_preprocessor` and `txt_preprocessor` are module-level singletons in `api.py`. Under concurrent requests (Daphne/ASGI), the per-request mutable state stored on `self` is a race condition -- one request can overwrite another's `_last_pdf_bytes`. Making the methods accept and return state eliminates this.

**Implementation**: Define `ExtractionContext` as a dataclass holding:
- `pdf_bytes: bytes` -- the raw PDF bytes, used by `align_segments_with_pages` to reopen the PDF for word extraction
- `cleaned_text: str` -- the full cleaned text, consumed by `get_segments_with_boxes` as input to `text_segmenter.segment_text()` (not used by alignment methods, but carried in the context so callers do not need a separate return channel)
- `normalized_full_text: str` -- the normalized concatenation of all text, used by `_segments_to_page_polygons` for word matching
- `removed_ranges: list[tuple[int, int]]` -- ranges within `normalized_full_text` that were removed during cleaning, used by `_word_overlaps_removed_ranges` to skip removed words

Update `get_segments_with_boxes` to thread `ctx` through locally:

```python
def get_segments_with_boxes(pdf):
    ctx = pdf_preprocessor.extract_from_memory(pdf)
    segments = text_segmenter.segment_text(ctx.cleaned_text)
    segments_with_pos = pdf_preprocessor.align_segments_with_pages(segments, ctx)
    return segments, segments_with_pos
```

### Dead Code Removal: extract_from_path

**Decision**: Delete `PdfBookPreprocessor.extract_from_path` as part of the thread-safety refactor.

**Rationale**: Zero call sites in the codebase (confirmed by grep). After the refactor, `_get_from_doc` returns a tuple instead of `str`, which would silently break `extract_from_path`'s return type. Removing it avoids a latent bug.

### Base64 PDF in JSON Response

**Decision**: TXT endpoints return the generated PDF as a `pdf_base64: str` field in the JSON response body alongside segments. Use `base64.b64encode(pdf_bytes).decode()` specifically (not `binascii.b2a_base64` which appends newlines that break `atob()` in the browser).

**Rationale**: Simplest approach -- single request, single response, no multipart complexity. The ~33% size overhead from base64 is acceptable for text-to-PDF conversions which produce small PDFs. Alternatives considered: multipart response (complex parsing), separate PDF-fetch call (extra round-trip), binary response with headers (fragile).

### pdfFile Construction from Returned Bytes

**Decision**: Frontend constructs `pdfFile` as `new File([decodedBytes], 'original-name.pdf', { type: 'application/pdf' })` from the base64 response.

**Rationale**: Keeps the export flow (`handleExportConfirm`) working without changes since it reads `pdfFile.value` and derives the filename from it. The `.txt` extension is replaced with `.pdf` in the constructed File name.

### Deferred pdfFile and pdfUrl for TXT Uploads

**Decision**: For TXT uploads, neither `pdfFile.value` nor `pdfUrl.value` is set before the API call. Both are set only after the response returns successfully. This differs from the PDF flow where both are set immediately from the local file.

**Rationale**: For TXT uploads there is no local PDF to display. Setting `pdfFile.value` to the raw `.txt` File before the API call would break any code path that reads it (export, PdfViewer). If the API call fails, both values remain `null`, leaving the UI in a clean state for retry.

### Schema Inheritance for TXT Responses

**Decision**: `ProcessTxtSegmentsOnlyResponse` inherits from `ProcessPdfSegmentsOnlyResponse` adding only `pdf_base64: str`. `ProcessTxtResponse` inherits from `ProcessPdfResponse` adding only `pdf_base64: str`. No separate `ProcessTxtBody` -- reuse `ProcessPdfBody` for both txt endpoints.

**Rationale**: Schema inheritance prevents future drift if PDF schemas gain fields. Reusing `ProcessPdfBody` avoids an unnecessary duplicate schema since the body only contains the `model` field which is the same for both file types.

### TXT Encoding Fallback

**Decision**: `TxtBookPreprocessor.extract_from_memory` tries UTF-8 decoding first, falls back to Latin-1 if a `UnicodeDecodeError` is raised.

**Rationale**: Covers the vast majority of plain-text files. Latin-1 never raises decode errors (it maps all 256 byte values), making it a safe catch-all. No external dependency needed.

### Duck-Typed extract_from_memory

**Decision**: Both `PdfBookPreprocessor.extract_from_memory` and `TxtBookPreprocessor.extract_from_memory` are duck-typed. There is no shared abstract base method on `BookPreprocessor`.

**Rationale**: The two methods have different return types (`ExtractionContext` vs `tuple[str, str, list[tuple[int, int]]]`) and different internal behavior. Forcing a shared interface would require awkward generics or union returns for no practical benefit. The call sites (`get_segments_with_boxes` and `get_segments_boxes_from_txt`) are already separate functions that know which preprocessor they are using.

### Keep get_segments_with_boxes Name

**Decision**: Keep the existing `get_segments_with_boxes` function name unchanged. Add `get_segments_boxes_from_txt` as a new parallel helper.

**Rationale**: No reason to rename existing code. The two helpers are called from separate endpoints and their names are sufficiently descriptive in context.
