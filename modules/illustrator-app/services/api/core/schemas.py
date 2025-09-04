from ninja import Schema
from enum import Enum


class TextBody(Schema):
    text: str


class Evaluator(str, Enum):
    deberta_mnli = "deberta_mnli"
    all_minilm_l6_v2 = "all_minilm_l6_v2"
    random = "random"


class RedisToProcessCtx(Schema):
    model: Evaluator
    segments: list[str]


class SegmentWithPos(Schema):
    text: str
    page: int
    polygon: list[tuple[float, float]]  # list of [x,y] points normalized


class ProcessPdfResponse(Schema):
    ws_key: str
    expires_in: int
    segment_count: int
    segments: list[SegmentWithPos]


class ProcessPdfBody(Schema):
    model: Evaluator
