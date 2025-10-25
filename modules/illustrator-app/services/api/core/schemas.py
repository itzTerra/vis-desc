from ninja import Schema
from enum import Enum


class TextBody(Schema):
    text: str


class Evaluator(str, Enum):
    minilm_svm = "minilm_svm"
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


class ProcessPdfResponse(Schema):
    ws_key: str
    expires_in: int
    segment_count: int
    segments: list[SegmentWithPos]


class ProcessPdfBody(Schema):
    model: Evaluator
