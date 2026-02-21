import base64
import json
import uuid
from typing import List, Sequence
import spacy

from ninja import File, Form, NinjaAPI, UploadedFile
from django.conf import settings
from django.http import HttpResponse

# The generated Python protobuf module must be created with protoc:
# protoc --proto_path=services/api/core/protos --python_out=services/api/core/protos services/api/core/protos/spacy.proto
try:
    from core.protos import spacy_pb2  # type: ignore

    _HAS_SPACY_PB2 = True
except Exception as e:
    spacy_pb2 = None  # type: ignore
    _HAS_SPACY_PB2 = False
    print(f"Error importing spacy_pb2: {e}")

from core.tools.book_segmenting import TextSegmenter
from core.schemas import (
    ProcessPdfBody,
    ProcessPdfResponse,
    ProcessPdfSegmentsOnlyResponse,
    ProcessTxtSegmentsOnlyResponse,
    ProcessTxtResponse,
    BatchTextsBody,
    BatchEnhanceItem,
    BatchImageItem,
    SpacyContextResponse,
    TextScorerBody,
)
from core.tools.book_preprocessing import PdfBookPreprocessor, TxtBookPreprocessor
from core.tools.redis import get_redis_client
from core.tools.text2image import (
    ProviderError,
    generate_image_bytes_batch,
)
from core.tools.llm import enhance_text_with_llm
from core.tools.spacy_pipeline import SpacyPipeline, Token
from core.schemas import TextBody, SpaCyContext
from core.tools.preprocess import preprocess


api = NinjaAPI()
redis_client = get_redis_client()
pdf_preprocessor = PdfBookPreprocessor(
    polygon_x_smooth_max_gap_px=settings.POLYGON_X_SMOOTH_MAX_GAP_PX,
    polygon_padding_px=settings.POLYGON_PADDING_PX,
)
txt_preprocessor = TxtBookPreprocessor()
text_segmenter = TextSegmenter((settings.SEGMENT_CHARS_MIN, settings.SEGMENT_CHARS_MAX))

# Load spaCy model (disable NER per request) and wrap with our pipeline helper
spacy_nlp = spacy.load(settings.SPACY_MODEL, disable=["ner"])
spacy_pipeline = SpacyPipeline(spacy_nlp)


def get_segments_with_boxes(pdf: UploadedFile) -> tuple[list[str], list[dict]]:
    ctx = pdf_preprocessor.extract_from_memory(pdf)
    segments = text_segmenter.segment_text(ctx.cleaned_text)
    segments_with_pos = pdf_preprocessor.align_segments_with_pages(segments, ctx)
    return segments, segments_with_pos


@api.get("/health")
def health(request):
    return "ok"


@api.get("/ping")
def ping(request):
    return "pong"


@api.post("/segment/pdf", response=ProcessPdfSegmentsOnlyResponse)
def segment_pdf(request, pdf: File[UploadedFile], model: Form[ProcessPdfBody]):
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


def get_segments_boxes_from_txt(
    txt: UploadedFile,
) -> tuple[list[str], list[dict], bytes]:
    full_text, cleaned_text, normalized_full_text, removed_ranges = (
        txt_preprocessor.extract_from_memory(txt)
    )
    segments = text_segmenter.segment_text(cleaned_text)
    segments_with_pos, pdf_bytes = txt_preprocessor.create_pdf_and_align(
        segments,
        (full_text, cleaned_text, normalized_full_text, removed_ranges),
        pdf_preprocessor,
    )
    return segments, segments_with_pos, pdf_bytes


@api.post("/segment/txt", response=ProcessTxtSegmentsOnlyResponse)
def segment_txt(request, txt: File[UploadedFile], model: Form[ProcessPdfBody]):
    segments, segments_with_pos, pdf_bytes = get_segments_boxes_from_txt(txt)
    return {
        "segment_count": len(segments),
        "segments": segments_with_pos,
        "pdf_base64": base64.b64encode(pdf_bytes).decode(),
    }


@api.post("/process/txt", response=ProcessTxtResponse)
def process_txt(request, txt: File[UploadedFile], model: Form[ProcessPdfBody]):
    ws_key = str(uuid.uuid4())

    segments, segments_with_pos, pdf_bytes = get_segments_boxes_from_txt(txt)

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
        "pdf_base64": base64.b64encode(pdf_bytes).decode(),
    }


@api.post("/scorer/segment/text", response=ProcessPdfSegmentsOnlyResponse)
def scorer_segment_text(request, body: TextScorerBody):
    text = body.text
    if not text:
        return api.create_response(request, {"error": "text is required"}, status=400)

    segments = text_segmenter.segment_text(text) if body.split else [text]
    segments_data = [
        {
            "id": idx + 1,
            "text": seg,
            "polygons": {},
        }
        for idx, seg in enumerate(segments)
    ]

    return {
        "segment_count": len(segments),
        "segments": segments_data,
    }


@api.post("/scorer/process/text", response=ProcessPdfResponse)
def scorer_process_text(request, body: TextScorerBody):
    ws_key = str(uuid.uuid4())

    text = body.text
    if not text:
        return api.create_response(request, {"error": "text is required"}, status=400)

    model = body.model if hasattr(body, "model") and body.model else "random"

    segments = text_segmenter.segment_text(text) if body.split else [text]
    segments_data = [
        {
            "id": idx + 1,
            "text": seg,
            "polygons": {},
        }
        for idx, seg in enumerate(segments)
    ]

    redis_client.set(
        f"ws_key:{ws_key}",
        value=json.dumps({"segments": segments, "model": model}),
        ex=settings.WS_KEY_EXPIRY_SEC,
    )
    return {
        "ws_key": ws_key,
        "expires_in": settings.WS_KEY_EXPIRY_SEC,
        "segment_count": len(segments),
        "segments": segments_data,
    }


@api.post("/gen-image-bytes", response=List[BatchImageItem])
def gen_image_bytes_endpoint(request, body: BatchTextsBody) -> List[BatchImageItem]:
    # Validate input
    texts = body.texts
    if not isinstance(texts, (list, tuple)):
        return api.create_response(
            request, {"error": "texts must be a list"}, status=400
        )

    # Enforce server-side max batch size early for clearer errors
    if len(texts) > settings.IMAGE_GENERATION_MAX_BATCH_SIZE:
        return api.create_response(
            request,
            {
                "error": f"Batch size exceeds maximum {settings.IMAGE_GENERATION_MAX_BATCH_SIZE}"
            },
            status=400,
        )

    # Enforce server-side max batch size via helper
    try:
        results = generate_image_bytes_batch(texts)
        return api.create_response(request, results, status=200)
    except ValueError as e:
        return api.create_response(request, {"error": str(e)}, status=400)
    except ProviderError:
        return api.create_response(
            request, {"error": "Image generation failed"}, status=500
        )


@api.post("/enhance", response=List[BatchEnhanceItem])
def enhance_text(request, body: BatchTextsBody) -> List[BatchEnhanceItem]:
    texts = body.texts
    if not isinstance(texts, (list, tuple)):
        return api.create_response(
            request, {"error": "texts must be a list"}, status=400
        )

    results: List[BatchEnhanceItem] = []
    for t in texts:
        try:
            enhanced = enhance_text_with_llm(t)
            results.append({"ok": True, "text": enhanced})
        except Exception as e:
            results.append({"ok": False, "error": f"enhance failed: {str(e)}"})

    return api.create_response(request, results, status=200)


def serialize_tag_result(
    toks: List[Token], sents: Sequence, noun_chunks: Sequence
) -> SpaCyContext:
    # Map internal Token -> SpaCyToken shape
    mapped_tokens = []
    for t in toks:
        # Normalize morph to a plain dict
        morph = {}
        if t.morph is not None:
            if hasattr(t.morph, "to_dict"):
                morph = t.morph.to_dict()
            else:
                morph = dict(t.morph)

        mapped_tokens.append(
            {
                "paragraphId": t.paragraph_id,
                "sentenceId": t.sentence_id,
                "withinSentenceId": t.within_sentence_id,
                "tokenId": t.token_id,
                "text": t.text,
                "pos": t.pos,
                "finePos": t.fine_pos,
                "lemma": t.lemma,
                "deprel": t.deprel,
                "dephead": t.dephead,
                "ner": t.ner if hasattr(t, "ner") else None,
                "startByte": t.startByte,
                "endByte": t.endByte,
                "morph": morph,
                "likeNum": t.like_num,
                "isStop": t.is_stop,
                "itext": getattr(t, "itext", t.text.casefold()),
                "inQuote": getattr(t, "inQuote", False),
                "event": getattr(t, "event", False),
            }
        )

    # Build hierarchical sentence representation expected by frontend types.
    def build_sent_token(tok, sent_start, sent_end, visited=None):
        if visited is None:
            visited = set()
        if tok.i in visited:
            return None
        visited.add(tok.i)

        children = []
        for child in tok.children:
            if child.i >= sent_start and child.i < sent_end:
                child_node = build_sent_token(child, sent_start, sent_end, visited)
                if child_node is not None:
                    children.append(child_node)

        return {
            "text": tok.text,
            "pos_": tok.pos_,
            "dep_": tok.dep_,
            "children": children,
        }

    sentences_out = []
    for s in sents:
        root = s.root
        root_node = build_sent_token(root, s.start, s.end)
        sentences_out.append({"root": root_node, "start": s.start, "end": s.end})

    noun_chunks_out = [
        {"start": nc.start, "end": nc.end, "text": nc.text} for nc in noun_chunks
    ]

    result = {
        "tokens": mapped_tokens,
        "sentences": sentences_out,
        "nounChunks": noun_chunks_out,
    }

    return result


@api.post("/spacy-ctx", response=SpacyContextResponse)
def spacy_context(request, body: TextBody):
    """Return a JSON object matching the SpaCyContext shape defined in the frontend types."""
    texts = body.texts

    try:
        preprocessed_texts = [preprocess(text) for text in texts]
        batch_results = spacy_pipeline.batch_tag(preprocessed_texts)
    except Exception as e:
        return api.create_response(request, {"error": str(e)}, status=500)

    return api.create_response(
        request,
        {"contexts": [serialize_tag_result(*res) for res in batch_results]},
        status=200,
    )


@api.post("/spacy-ctx/proto")
def spacy_context_proto(request, body: TextBody):
    """Binary protobuf endpoint. This is a new, separate endpoint from `/spacy-ctx`.

    Requires generated Python protobuf bindings (`core.protos.spacy_pb2`).
    Generate them with protoc before enabling this endpoint.
    """
    if not _HAS_SPACY_PB2:
        return api.create_response(
            request,
            {
                "error": "Protobuf bindings not generated. Run protoc to create core.protos.spacy_pb2."
            },
            status=500,
        )

    texts = body.texts
    try:
        preprocessed_texts = [preprocess(text) for text in texts]
        batch_results = spacy_pipeline.batch_tag(preprocessed_texts)
    except Exception as e:
        return api.create_response(request, {"error": str(e)}, status=500)

    msg = spacy_pb2.SpaCyContexts()
    for c in [serialize_tag_result(*res) for res in batch_results]:
        ctx = msg.contexts.add()
        for t in c.get("tokens", []):
            tok = ctx.tokens.add()
            tok.paragraph_id = int(t.get("paragraphId") or 0)
            tok.sentence_id = int(t.get("sentenceId") or 0)
            tok.within_sentence_id = int(t.get("withinSentenceId") or 0)
            tok.token_id = int(t.get("tokenId") or 0)
            tok.text = t.get("text") or ""
            tok.pos = t.get("pos") or ""
            tok.fine_pos = t.get("finePos") or ""
            tok.lemma = t.get("lemma") or ""
            tok.deprel = t.get("deprel") or ""
            tok.dephead = int(t.get("dephead") or 0)
            tok.ner = t.get("ner") or ""
            tok.start_byte = int(t.get("startByte") or 0)
            tok.end_byte = int(t.get("endByte") or 0)
            morph = t.get("morph") or {}
            if isinstance(morph, dict):
                for mk, mv in morph.items():
                    try:
                        tok.morph[mk] = str(mv)
                    except Exception:
                        pass
            tok.like_num = bool(t.get("likeNum"))
            tok.is_stop = bool(t.get("isStop"))
            tok.itext = t.get("itext") or ""
            tok.in_quote = bool(t.get("inQuote"))
            tok.event = bool(t.get("event"))

        for s in c.get("sentences", []):
            sent = ctx.sentences.add()

            def _fill_senttoken(dst, src):
                if not src:
                    return
                dst.text = src.get("text") or ""
                dst.pos_ = src.get("pos_") or ""
                dst.dep_ = src.get("dep_") or ""
                for child in src.get("children", []):
                    ch = dst.children.add()
                    _fill_senttoken(ch, child)

            _fill_senttoken(sent.root, s.get("root"))
            sent.start = int(s.get("start") or 0)
            sent.end = int(s.get("end") or 0)

        for nc in c.get("nounChunks", []):
            nn = ctx.noun_chunks.add()
            nn.start = int(nc.get("start") or 0)
            nn.end = int(nc.get("end") or 0)
            nn.text = nc.get("text") or ""

    data = msg.SerializeToString()
    return HttpResponse(data, content_type="application/x-protobuf")
