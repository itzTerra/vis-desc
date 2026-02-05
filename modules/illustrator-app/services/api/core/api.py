from core.tools.book_segmenting import TextSegmenter
from ninja import File, Form, NinjaAPI, UploadedFile
from django.http import HttpResponse
from django.conf import settings
import uuid
from core.schemas import (
    ProcessPdfBody,
    ProcessPdfResponse,
    ProcessPdfSegmentsOnlyResponse,
    TextBody,
)
from core.tools.book_preprocessing import PdfBookPreprocessor
from core.tools.redis import get_redis_client
from core.tools.text2image import Provider, get_image_url, get_image_bytes
import json


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


@api.post("/gen-image")
def gen_image(request, body: TextBody):
    return {"image": get_image_url(request, body.text, Provider.POLLINATIONS)}


@api.post("/gen-image-bytes")
def gen_image_bytes(request, body: TextBody):
    image_bytes = get_image_bytes(body.text, Provider.POLLINATIONS)
    return HttpResponse(image_bytes, content_type="image/png")
