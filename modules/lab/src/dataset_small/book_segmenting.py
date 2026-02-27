from semantic_text_splitter import TextSplitter


SEGMENT_CHARS_MIN = 150
SEGMENT_CHARS_MAX = 500


class TextSegmenter:
    def __init__(
        self, segment_size: tuple[int, int] = (SEGMENT_CHARS_MIN, SEGMENT_CHARS_MAX)
    ):
        self.segment_size = segment_size
        self.splitter = TextSplitter(segment_size)

    def segment_text(self, text: str) -> list[str]:
        return self.splitter.chunks(text)
