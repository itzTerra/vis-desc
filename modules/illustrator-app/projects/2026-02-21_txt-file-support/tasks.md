# Tasks

## Backend: Thread-Safety Refactor (PdfBookPreprocessor)

- [x] Define `ExtractionContext` dataclass in `services/api/core/tools/book_preprocessing.py` with fields: `pdf_bytes: bytes`, `cleaned_text: str` (consumed by callers for segmentation, not by alignment methods), `normalized_full_text: str`, `removed_ranges: list[tuple[int, int]]`
- [x] Refactor `PdfBookPreprocessor._get_from_doc` to return `(cleaned_text, normalized_full_text, removed_ranges)` tuple instead of storing on `self`
- [x] Delete `PdfBookPreprocessor.extract_from_path` -- it is dead code with zero call sites in the codebase and will break after `_get_from_doc` returns a tuple instead of `str`
- [x] Refactor `PdfBookPreprocessor.extract_from_memory` to return `ExtractionContext` instead of `str`. Build context from `_get_from_doc` return values plus `pdf_bytes`. Remove instance attributes `_last_pdf_bytes`, `_normalized_full_text`, `_removed_ranges` from `__init__` and method bodies.
- [x] Refactor `PdfBookPreprocessor._word_overlaps_removed_ranges` to accept `removed_ranges: list[tuple[int, int]]` as an explicit parameter instead of reading `self._removed_ranges`
- [x] Refactor `PdfBookPreprocessor._segments_to_page_polygons` to accept `normalized_full_text: str` and `removed_ranges: list[tuple[int, int]]` as parameters instead of reading from `self`. Update the call to `_word_overlaps_removed_ranges` to pass `removed_ranges` through.
- [x] Refactor `PdfBookPreprocessor.align_segments_with_pages` to accept `ctx: ExtractionContext` as a second parameter -- read `ctx.pdf_bytes`, `ctx.normalized_full_text`, `ctx.removed_ranges` instead of `self._*`. Pass `ctx.normalized_full_text` and `ctx.removed_ranges` to `_segments_to_page_polygons`.
- [x] Update `get_segments_with_boxes` in `services/api/core/api.py` to thread `ctx` locally: `ctx = pdf_preprocessor.extract_from_memory(pdf)`, `segments = text_segmenter.segment_text(ctx.cleaned_text)`, `segments_with_pos = pdf_preprocessor.align_segments_with_pages(segments, ctx)`
- [ ] Verify existing PDF endpoints (`segment_pdf`, `process_pdf`) still work after refactor

## Backend: TxtBookPreprocessor

- [x] Add `extract_from_memory(self, txt_file) -> tuple[str, str, list[tuple[int, int]]]` to `TxtBookPreprocessor` in `services/api/core/tools/book_preprocessing.py` -- reads bytes from `txt_file.read()`, decodes with UTF-8 (falling back to Latin-1 on `UnicodeDecodeError`), runs `self.clean_text(..., return_split_with_removed=True)` on the decoded string, computes `normalized_full_text` and `removed_ranges` (same logic as `PdfBookPreprocessor._get_from_doc` but for the full text in one pass), returns `(cleaned_text, normalized_full_text, removed_ranges)`
- [x] Add `create_pdf_and_align(self, segments: list[str], txt_ctx: tuple[str, str, list[tuple[int, int]]], pdf_preprocessor: PdfBookPreprocessor) -> tuple[list[dict], bytes]` -- unpacks `txt_ctx` into `(cleaned_text, normalized_full_text, removed_ranges)`, creates a pymupdf PDF from `cleaned_text` (readable font, reasonable size), saves to `pdf_bytes`. Constructs `ExtractionContext(pdf_bytes=pdf_bytes, cleaned_text=cleaned_text, normalized_full_text=normalized_full_text, removed_ranges=removed_ranges)` directly from the already-computed normalization data -- does NOT call `pdf_preprocessor.extract_from_memory` (avoids double-cleaning risk where page-by-page extraction could produce different normalization). Calls `pdf_preprocessor.align_segments_with_pages(segments, ctx)`. Returns `(segments_with_pos, pdf_bytes)`.

## Backend: Schemas

- [x] Add `ProcessTxtSegmentsOnlyResponse(ProcessPdfSegmentsOnlyResponse)` schema in `services/api/core/schemas.py` -- inherits all fields from `ProcessPdfSegmentsOnlyResponse`, adds only `pdf_base64: str`
- [x] Add `ProcessTxtResponse(ProcessPdfResponse)` schema -- inherits all fields from `ProcessPdfResponse`, adds only `pdf_base64: str`
- [x] Import new schemas in `services/api/core/api.py`

## Backend: API Endpoints

- [x] Instantiate `TxtBookPreprocessor` singleton in `services/api/core/api.py` (no polygon settings needed -- it delegates to `pdf_preprocessor`)
- [x] Add `get_segments_boxes_from_txt(txt: UploadedFile) -> tuple[list[str], list[dict], bytes]` helper in `services/api/core/api.py` -- calls `txt_ctx = txt_preprocessor.extract_from_memory(txt)`, then `segments = text_segmenter.segment_text(txt_ctx[0])`, then `segments_with_pos, pdf_bytes = txt_preprocessor.create_pdf_and_align(segments, txt_ctx, pdf_preprocessor)`. Returns `(segments, segments_with_pos, pdf_bytes)`.
- [x] Add `POST /api/segment/txt` endpoint with `response=ProcessTxtSegmentsOnlyResponse` -- accepts `txt: File[UploadedFile]` and `model: Form[ProcessPdfBody]` (reuses ProcessPdfBody). Calls `get_segments_boxes_from_txt(txt)`. Returns `{"segment_count": ..., "segments": segments_with_pos, "pdf_base64": base64.b64encode(pdf_bytes).decode()}`
- [x] Add `POST /api/process/txt` endpoint with `response=ProcessTxtResponse` -- accepts `txt: File[UploadedFile]` and `model: Form[ProcessPdfBody]`. Calls `get_segments_boxes_from_txt(txt)`, stores segments + model in Redis under ws_key. Returns `{"ws_key": ..., "expires_in": ..., "segment_count": ..., "segments": segments_with_pos, "pdf_base64": base64.b64encode(pdf_bytes).decode()}`
- [x] Import `base64` at top of `services/api/core/api.py`

## Frontend: File Inputs

- [x] Update top-bar file input `accept` attribute in `services/frontend/app/pages/index.vue` from `"application/pdf"` to `"application/pdf,.txt,text/plain"`
- [x] Update Hero file input `accept` attribute in `services/frontend/app/components/Hero.vue` from `"application/pdf"` to `"application/pdf,.txt,text/plain"`
- [x] Update Hero heading from "Upload a file in PDF format to get started" to "Upload a PDF or TXT file to get started"
- [x] Update Hero drop zone hint text to mention both formats (e.g., "Drop a PDF or TXT file here or click in this area")

## Frontend: handleFileUpload

- [x] Add file type detection at the start of `handleFileUpload` -- `const isTxt = file.name.endsWith('.txt') || file.type === 'text/plain'`
- [x] For `.txt` files: set `currentStage.value` to `"Processing TXT..."` instead of `"Processing PDF..."`
- [x] For `.txt` files: do NOT set `pdfFile.value = file` before the API call (the raw `.txt` File would break export/PdfViewer if the call fails). Do NOT set `pdfUrl.value` either. Hero stays visible during processing.
- [x] For `.txt` files: build FormData with `formData.append("txt", file, file.name)` and call `/api/process/txt` or `/api/segment/txt`
- [x] For `.txt` files: after receiving the response, decode `data.pdf_base64` -- `const pdfBytes = Uint8Array.from(atob(data.pdf_base64), c => c.charCodeAt(0))`
- [x] For `.txt` files: construct `pdfFile.value = new File([pdfBytes], file.name.replace(/\.txt$/i, '.pdf'), { type: 'application/pdf' })` and set `pdfUrl.value = URL.createObjectURL(pdfFile.value)` -- both set only after successful response
- [x] For PDF files: keep existing behavior unchanged (set `pdfFile.value = file` and `pdfUrl.value` immediately from the local file before the API call)

## Sync and Verify

- [ ] Regenerate OpenAPI types: `docker compose run --rm frontend pnpm gen-types`
- [ ] Verify `services/frontend/app/types/schema.d.ts` includes the new `ProcessTxtSegmentsOnlyResponse` and `ProcessTxtResponse` types (no `ProcessTxtBody` -- reuses `ProcessPdfBody`)
- [ ] Run `pre-commit run --all-files` to verify linting and formatting
- [ ] Manual test: upload a PDF and confirm existing flow is unchanged (thread-safety refactor is transparent)
- [ ] Manual test: upload a `.txt` file (e.g., a Project Gutenberg book) and confirm generated PDF renders with segment highlights
- [ ] Manual test: upload a `.txt` file and confirm export works correctly (filename derives from original `.txt` name)
- [ ] Manual test: verify Hero is visible during `.txt` processing and transitions to PdfViewer after response
- [ ] Manual test: verify that if a `.txt` upload API call fails, `pdfFile` and `pdfUrl` remain `null` (clean state for retry)
