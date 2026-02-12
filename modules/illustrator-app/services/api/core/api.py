import uuid
import json

from ninja import File, Form, NinjaAPI, UploadedFile
from django.conf import settings

from core.tools.book_segmenting import TextSegmenter
from core.schemas import (
    ProcessPdfBody,
    ProcessPdfResponse,
    ProcessPdfSegmentsOnlyResponse,
    BatchTextsBody,
)
from core.tools.book_preprocessing import PdfBookPreprocessor
from core.tools.redis import get_redis_client
from core.tools.text2image import (
    ProviderError,
    generate_image_bytes_batch,
)
from core.tools.llm import enhance_text_with_llm


api = NinjaAPI()
redis_client = get_redis_client()
pdf_preprocessor = PdfBookPreprocessor(
    polygon_x_smooth_max_gap_px=settings.POLYGON_X_SMOOTH_MAX_GAP_PX,
    polygon_padding_px=settings.POLYGON_PADDING_PX,
)
text_segmenter = TextSegmenter((settings.SEGMENT_CHARS_MIN, settings.SEGMENT_CHARS_MAX))


def get_segments_with_boxes(pdf: UploadedFile) -> tuple[list[str], list[dict]]:
    book_text = pdf_preprocessor.extract_from_memory(pdf)
    segments = text_segmenter.segment_text(book_text)
    segments_with_pos = pdf_preprocessor.align_segments_with_pages(segments)
    return segments, segments_with_pos


@api.get("/health")
def health(request):
    return "ok"


@api.post("/process/seg/pdf", response=ProcessPdfSegmentsOnlyResponse)
def process_pdf_segments_only(
    request, pdf: File[UploadedFile], model: Form[ProcessPdfBody]
):
    segments, segments_with_pos = get_segments_with_boxes(pdf)
    return {
        "segment_count": len(segments),
        "segments": segments_with_pos,
    }


@api.post("/process/pdf", response=ProcessPdfResponse)
def process_pdf(request, pdf: File[UploadedFile], model: Form[ProcessPdfBody]):
    ws_key = str(uuid.uuid4())

    segments, segments_with_pos = get_segments_with_boxes(pdf)

    redis_client.set(
        f"ws_key:{ws_key}",
        value=json.dumps({"segments": segments, "model": model.model}),
        ex=settings.WS_KEY_EXPIRY_SEC,
    )
    return {
        "ws_key": ws_key,
        "expires_in": settings.WS_KEY_EXPIRY_SEC,
        "segment_count": len(segments),
        "segments": segments_with_pos,
    }


@api.post("/gen-image-bytes")
def gen_image_bytes_endpoint(request, body: BatchTextsBody):
    # Validate input
    texts = body.texts
    if not isinstance(texts, (list, tuple)):
        return api.create_response(
            request, {"error": "texts must be a list"}, status=400
        )

    # Enforce server-side max batch size early for clearer errors
    max_batch = getattr(settings, "IMAGE_GENERATION_MAX_BATCH_SIZE", None)
    if max_batch is not None and len(texts) > max_batch:
        return api.create_response(
            request,
            {"error": f"Batch size exceeds maximum {max_batch}"},
            status=400,
        )

    # Enforce server-side max batch size via helper
    try:
        results = generate_image_bytes_batch(texts)
        return api.create_response(request, results)
    except ValueError as e:
        return api.create_response(request, {"error": str(e)}, status=400)
    except ProviderError:
        return api.create_response(
            request, {"error": "Image generation failed"}, status=500
        )


@api.post("/enhance")
def enhance_text(request, body: BatchTextsBody):
    texts = body.texts
    if not isinstance(texts, (list, tuple)):
        return api.create_response(
            request, {"error": "texts must be a list"}, status=400
        )

    results: list[dict] = []
    for t in texts:
        try:
            enhanced = enhance_text_with_llm(t)
            results.append({"ok": True, "text": enhanced})
        except Exception:
            results.append({"ok": False, "error": "enhance failed"})

    return api.create_response(request, results)
