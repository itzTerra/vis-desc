from pathlib import Path
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import pymupdf
from dataclasses import dataclass
import regex


class BookPreprocessor:
    def __init__(self):
        self.patterns = {
            "page_numbers": re.compile(
                r"^\s*(?:page\s+)?(?:\d+|\[?\d+\]?)\s*$", re.IGNORECASE
            ),
            "chapter_headers": re.compile(
                r"^\s*(?:chapter|ch\.?|part|section|§|act|volume)\s+(?:[ivx]+|\d+)(?:\.|:|\s|$)",
                re.IGNORECASE,
            ),
            "roman_numerals": re.compile(r"^\s*[ivxlcdm]{1,7}\s*$", re.IGNORECASE),
            "copyright": re.compile(
                r"©|\bcopyright\b|\ball rights reserved\b", re.IGNORECASE
            ),
            "isbn": re.compile(r"\bisbn[-:\s]*(?:\d[-\s]*){9}[\dx]\b", re.IGNORECASE),
            "footnote_refs": re.compile(r"^\s*(?:\d+|\*+|\†+|\‡+)\s+"),
            "table_of_contents": re.compile(
                r"(table of contents)|\.{5,}|\s{5,}\d+$", re.IGNORECASE
            ),
            "website_urls": re.compile(
                r"https?://\S+|www\.\S+|\S+\.com\b", re.IGNORECASE
            ),
            "brackets_content": re.compile(r"^\[[^\]]*\]$"),
            "illustration": re.compile(
                r"^\s*(\[illustration\])|(image\d*)\s*$", re.IGNORECASE
            ),
            "numeric_only": re.compile(r"^\s*[\d\s\.\-\/]+\s*$"),
            "line_breaked_sentence": regex.compile(
                r"(?<=[^.!?。！？:\"'’”–—-])(?:\n(?=[–—-]?(?:\p{L}|[\"'’”])))|(?:\s+(?=[–—-]?(?:\p{Ll}|I[ ']|[\"'’”])))"
            ),
            "line_breaked_sentence_a_end": regex.compile(r"[^.!?。！？:\"'’”–—-]$"),
            "line_breaked_sentence_b_start": regex.compile(r"[–—-]?(?:\p{L}|[\"'’”])"),
            "hyphenated_sentence": regex.compile(r"(?<=\p{Ll})-\n(?=\p{Ll})"),
        }

        self.metadata_keywords = {
            "acknowledgment",
            "acknowledgement",
            "preface",
            "foreword",
            "introduction",
            "table of contents",
            "contents",
            "list of chapters",
            "index",
            "bibliography",
            "references",
            "appendix",
            "about the author",
            "author’s note",
            "biography",
            "glossary",
            "notes",
            "epigraph",
            "prologue",
            "epilogue",
            "afterword",
            "etext",
            "published",
            "publisher",
            "publications",
            "was updated by",
        }
        self.end_of_sentence_chars = {
            ".",
            "!",
            "?",
            "。",
            "！",
            "？",
            ":",
            '"',
            "'",
            "’",
            "”",
        }
        self.text_keywords = {"!", "?", '"', ":", "”"}
        self.short_line_threshold = 10

    def clean_text(self, text: str, prev_line: str = "") -> str:
        """prev_line is for cases where you are using clean_text on chunks of connected text"""

        text = self._before_clean(text)

        if not text or len(text) < 10:
            return text

        lines = text.split("\n")

        # Parallel processing for large texts
        if len(lines) > 1000:
            return self._parallel_clean(lines, prev_line)
        else:
            return self._sequential_clean(lines, prev_line)

    def _before_clean(self, text: str) -> str:
        """Hook for preprocessing before cleaning"""

        return text.strip().replace("\r\n", "\n")

    def _parallel_clean(self, lines: list[str], prev_line: str = "") -> str:
        """Parallel processing for large texts"""
        chunk_size = max(100, len(lines) // mp.cpu_count())
        chunks = [
            (lines[i : i + chunk_size], lines[i - 1] if i > 0 else prev_line)
            for i in range(0, len(lines), chunk_size)
        ]

        with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
            processed_chunks = list(
                executor.map(
                    lambda ctx: self._process_chunk(ctx[0], ctx[1]),
                    chunks,
                )
            )

        # Flatten results
        processed_lines = []
        for chunk_lines in processed_chunks:
            processed_lines.extend(chunk_lines)

        return self._after_clean(processed_lines)

    def _sequential_clean(self, lines: list[str], prev_line: str = "") -> str:
        """Sequential processing for smaller texts"""
        processed_lines = self._process_chunk(lines, prev_line)
        return self._after_clean(processed_lines)

    def _process_chunk(self, lines: list[str], saved_prev_line="") -> list[str]:
        """Process a chunk of lines and filter them"""
        processed = []
        total_lines = len(lines)
        for i, line in enumerate(lines):
            prev_line = lines[i - 1] if i > 0 else saved_prev_line
            result = self.process_line(line, prev_line, i, total_lines)
            if result is not None:
                processed.append(result)
        return processed

    def process_line(
        self, line: str, prev_line: str, line_num: int, total_lines: int
    ) -> str | None:
        """Process a line: return line if it should be kept, else None"""
        line = line.strip()
        # Keep empty lines for further segmenting
        if not line:
            return line
        prev_line = prev_line.strip()

        if any(
            pattern.search(line)
            for pattern in (
                self.patterns["chapter_headers"],
                self.patterns["roman_numerals"],
                self.patterns["copyright"],
                self.patterns["isbn"],
                self.patterns["footnote_refs"],
                self.patterns["table_of_contents"],
                self.patterns["website_urls"],
                self.patterns["brackets_content"],
                self.patterns["numeric_only"],
            )
        ):
            return "\n"

        if self.patterns["page_numbers"].search(line):
            return ""

        line_lower = line.lower()
        words_lower = line_lower.split()
        word_count = len(words_lower)

        # All-caps short lines
        if line.isupper() and word_count < self.short_line_threshold:
            return "\n"

        # Other short lines without typical sentence characters
        is_end_of_paragraph = (
            prev_line
            and (
                len(prev_line.split()) > 5
                or (prev_line and prev_line[-1] in self.end_of_sentence_chars)
            )
            and line[-1] in self.end_of_sentence_chars
        )
        if (
            not is_end_of_paragraph
            and not any(kw in line_lower for kw in self.text_keywords)
            and (
                metadata_keywords := len(
                    [kw for kw in self.metadata_keywords if kw in line_lower]
                )
            )
            and (
                word_count < self.short_line_threshold + metadata_keywords
                or line.isupper()
            )
        ):
            return "\n"

        return line

    def _after_clean(self, processed_lines: list[str]) -> str:
        text = "\n".join(processed_lines)

        # Unhyphenate
        text = self.patterns["hyphenated_sentence"].sub("", text)

        # Remove functional newlines for page width control - they break the text splitter semantics
        # -> replace newline in the middles of sentences with a white space, e.g. 'Soon\nanother clock began, on a hard, decisive note.'
        text = self.patterns["line_breaked_sentence"].sub(" ", text)

        # Remove None values and normalize
        return unicodedata.normalize("NFKC", text)


class TxtBookPreprocessor(BookPreprocessor):
    """Most txt books from Gutenberg dataset have extra newlines for readability, which are undesirable for segmenting"""

    def _before_clean(self, text):
        """Remove extra newlines"""
        text = super()._before_clean(text)
        return re.sub(r"\S\n\n[ \t]*", lambda m: m.group(0).rstrip() + " ", text)

    # def process_line(self, line, prev_line, line_num, total_lines):
    #     if prev_line.strip() and not line.strip():
    #         return None
    #     return super().process_line(line, prev_line, line_num, total_lines)


@dataclass
class PdfExtractionConfig:
    max_pages: int | None = None  # limit for debugging


class PdfBookPreprocessor(BookPreprocessor):
    def __init__(self):
        super().__init__()

    def _get_from_doc(
        self, doc: pymupdf.Document, page_limit: int | None = None
    ) -> str:
        if page_limit is None:
            page_limit = len(doc)
        pages: list[str] = []
        last_line_of_last_page = ""
        for i in range(page_limit):
            try:
                page = doc[i]
                page_text = page.get_textpage().extractText(sort=True)
                cleaned_page_text = self.clean_text(page_text, last_line_of_last_page)

                # Join pages that cut a sentence in half
                if (
                    last_line_of_last_page
                    and self.patterns["line_breaked_sentence_a_end"].match(
                        last_line_of_last_page
                    )
                    and cleaned_page_text
                    and self.patterns["line_breaked_sentence_b_start"].match(
                        cleaned_page_text
                    )
                ):
                    last_page = f"{pages[-1].rstrip()} {cleaned_page_text.lstrip()}"
                    pages[-1] = last_page
                else:
                    pages.append(cleaned_page_text)
                    last_page = cleaned_page_text

                last_line_of_last_page = last_page.splitlines()[-1] if last_page else ""
            except Exception as e:
                print("Skipping page %d due to error: %s", i, e)
        return "".join(pages)

    def extract_from_path(
        self, pdf_path: str | Path, config: PdfExtractionConfig | None = None
    ) -> str:
        if config is None:
            config = PdfExtractionConfig()

        try:
            with pymupdf.open(pdf_path) as doc:
                raw_text = self._get_from_doc(doc, config.max_pages)
        except Exception:
            print("Failed to open PDF")
            raise

        return self.clean_text(raw_text)


def show_diff(original: str, cleaned: str) -> str:
    import difflib

    original_lines = original.splitlines()
    cleaned_lines = cleaned.splitlines()
    diff = difflib.unified_diff(
        original_lines,
        cleaned_lines,
        fromfile="original.txt",
        tofile="cleaned.txt",
        lineterm="",
    )
    return "\n".join(diff)


if __name__ == "__main__":
    sample_text = """
    Chapter 1

    This is the beginning of the book.

    Page 1

    The quick brown fox jumps over the lazy dog.

    © 2024 Some Publisher

    Visit us at www.example.com

    The end.
    """

    preprocessor = TxtBookPreprocessor()
    cleaned_text = preprocessor.clean_text(sample_text)
    print(cleaned_text)
