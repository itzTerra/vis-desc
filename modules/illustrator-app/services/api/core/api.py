import uuid
import json
from typing import List, Sequence
import spacy

from ninja import File, Form, NinjaAPI, UploadedFile
from django.conf import settings

from core.tools.book_segmenting import TextSegmenter
from core.schemas import (
    ProcessPdfBody,
    ProcessPdfResponse,
    ProcessPdfSegmentsOnlyResponse,
    BatchTextsBody,
    BatchEnhanceItem,
    BatchImageItem,
    SpacyContextResponse,
)
from core.tools.book_preprocessing import PdfBookPreprocessor
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
text_segmenter = TextSegmenter((settings.SEGMENT_CHARS_MIN, settings.SEGMENT_CHARS_MAX))

# Load spaCy model (disable NER per request) and wrap with our pipeline helper
spacy_nlp = spacy.load(settings.SPACY_MODEL, disable=["ner"])
spacy_pipeline = SpacyPipeline(spacy_nlp)


def get_segments_with_boxes(pdf: UploadedFile) -> tuple[list[str], list[dict]]:
    book_text = pdf_preprocessor.extract_from_memory(pdf)
    segments = text_segmenter.segment_text(book_text)
    segments_with_pos = pdf_preprocessor.align_segments_with_pages(segments)
    return segments, segments_with_pos


@api.get("/health")
def health(request):
    return "ok"


@api.get("/ping")
def ping(request):
    return "pong"


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
        except Exception:
            results.append({"ok": False, "error": "enhance failed"})

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
