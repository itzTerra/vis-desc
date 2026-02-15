from ninja import Schema
from enum import Enum
from typing import Optional


class TextBody(Schema):
    texts: list[str]


class Evaluator(str, Enum):
    minilm_catboost = "minilm_catboost"
    nli_roberta = "nli_roberta"
    random = "random"


class RedisToProcessCtx(Schema):
    model: Evaluator
    segments: list[str]


class SegmentWithPos(Schema):
    id: int
    text: str
    polygons: dict[
        int, list[tuple[float, float]]
    ]  # page: polygon (list of [x,y] points normalized)


class ProcessPdfSegmentsOnlyResponse(Schema):
    segment_count: int
    segments: list[SegmentWithPos]


class ProcessPdfResponse(Schema):
    ws_key: str
    expires_in: int
    segment_count: int
    segments: list[SegmentWithPos]


class ProcessPdfBody(Schema):
    model: Evaluator


class EnhanceTextBody(Schema):
    text: str


class EnhanceTextResponse(Schema):
    text: str


class BatchEnhanceItem(Schema):
    ok: bool
    text: Optional[str] = None
    error: Optional[str] = None


class BatchTextsBody(Schema):
    texts: list[str]


class BatchImageItem(Schema):
    ok: bool
    image_b64: Optional[str] = None
    error: Optional[str] = None


# SpaCy-related types used by the API
class SpaCyToken(Schema):
    text: str
    startByte: int
    endByte: int
    pos: str
    finePos: str
    lemma: str
    deprel: str
    dephead: int
    morph: dict[str, str]
    likeNum: bool
    isStop: bool
    sentenceId: int
    withinSentenceId: int


class SentToken(Schema):
    text: str
    pos_: str
    dep_: str
    children: list["SentToken"]


class SpacySent(Schema):
    root: SentToken
    start: int
    end: int


class NounChunk(Schema):
    start: int
    end: int
    text: str


class SpaCyContext(Schema):
    tokens: list[SpaCyToken]
    sentences: list[SpacySent]
    nounChunks: list[NounChunk]


class SpacyContextResponse(Schema):
    contexts: list[SpaCyContext]
