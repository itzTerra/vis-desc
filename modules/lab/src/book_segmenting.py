from semantic_text_splitter import TextSplitter


class TextSegmenter:
    def __init__(self, segment_size: tuple[int, int]):
        self.segment_size = segment_size
        self.splitter = TextSplitter(segment_size)

    def segment_text(self, text: str) -> list[str]:
        return self.splitter.chunks(text)
