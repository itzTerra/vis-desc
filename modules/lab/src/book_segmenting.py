from semantic_text_splitter import TextSplitter


class TextSegmenter:
    def __init__(self, chunk_size: tuple[int, int]):
        self.chunk_size = chunk_size
        self.splitter = TextSplitter(chunk_size)

    def segment_text(self, text: str) -> list[str]:
        return self.splitter.chunks(text)
