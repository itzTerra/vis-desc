from __future__ import annotations
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import logging
from ninja import UploadedFile
import pymupdf
from dataclasses import dataclass
import regex
from typing import Optional


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
        """prev_line is for cases where you are using clean_text on segments of connected text"""

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
        batch_size = max(100, len(lines) // mp.cpu_count())
        batches = [
            (lines[i : i + batch_size], lines[i - 1] if i > 0 else prev_line)
            for i in range(0, len(lines), batch_size)
        ]

        with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
            processed_batches = list(
                executor.map(
                    lambda ctx: self._process_batch(ctx[0], ctx[1]),
                    batches,
                )
            )

        # Flatten results
        processed_lines = []
        for batch_lines in processed_batches:
            processed_lines.extend(batch_lines)

        return self._after_clean(processed_lines)

    def _sequential_clean(self, lines: list[str], prev_line: str = "") -> str:
        """Sequential processing for smaller texts"""
        processed_lines = self._process_batch(lines, prev_line)
        return self._after_clean(processed_lines)

    def _process_batch(self, lines: list[str], saved_prev_line="") -> list[str]:
        """Process a batch of lines and filter them"""
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


@dataclass
class PdfExtractionConfig:
    max_pages: int | None = None  # limit for debugging


class PdfBookPreprocessor(BookPreprocessor):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger("django")
        # Raw unmodified page texts (extracted directly)
        self._raw_pages: list[str] = []
        # Alignment index (built after extraction)
        self._alignment_index: TextAlignmentIndex | None = None

    def _get_from_doc(self, doc, page_limit: int | None = None) -> str:
        if page_limit is None:
            page_limit = len(doc)
        pages: list[str] = []
        last_line_of_last_page = ""
        for i in range(page_limit):
            try:
                page = doc[i]
                page_text = page.get_textpage().extractText(sort=True)  # type: ignore
                self._raw_pages.append(page_text)
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
                self.logger.warning("Skipping page %d due to error: %s", i, e)

        try:
            self._alignment_index = TextAlignmentIndex(self._raw_pages)
        except Exception:
            self.logger.exception("Failed building alignment index")
            self._alignment_index = None

        return "".join(pages)

    def extract_from_memory(
        self, pdf_file: UploadedFile, config: PdfExtractionConfig | None = None
    ) -> str:
        if config is None:
            config = PdfExtractionConfig()

        try:
            pdf_bytes = pdf_file.read()
            self._last_pdf_bytes = pdf_bytes
            with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
                cleaned = self._get_from_doc(doc, config.max_pages)
        except Exception:
            self.logger.exception("Failed to open PDF")
            raise

        return cleaned

    def extract_from_path(
        self, pdf_path: str, config: PdfExtractionConfig | None = None
    ) -> str:
        if config is None:
            config = PdfExtractionConfig()

        try:
            with pymupdf.open(pdf_path) as doc:
                cleaned = self._get_from_doc(doc, config.max_pages)
        except Exception:
            self.logger.exception("Failed to open PDF")
            raise

        return cleaned

    def align_segments_with_pages(self, segments: list[str]) -> list[dict]:
        """Map cleaned segments to original page polygons.
        Returns a list of dicts:
        {
          'text': segment_text,
          'polygons': list[{
            'page': int,  # 0-based page index
            'points': list[tuple[float, float]]
           }]
        }
        """
        if not self._alignment_index or not self._last_pdf_bytes:
            raise RuntimeError("A book must be processed before alignment")

        doc = None
        try:
            doc = pymupdf.open(stream=self._last_pdf_bytes, filetype="pdf")
        except Exception:
            self.logger.exception("Failed to reopen PDF for polygons")
            return [
                {
                    "text": seg,
                    "polygons": [],
                }
                for seg in segments
            ]

        out = self._alignment_index.align_segments_and_get_polygons(segments, doc)
        if doc is not None:
            doc.close()
        return out


REMOVED_CHARS = set([" ", "\t", "\r", "\n", "-", "‐", "‑", "‒", "–", "—"])


def _keep_char(ch: str) -> bool:
    return ch not in REMOVED_CHARS


def _normalize(s: str) -> str:
    return "".join(c.lower() for c in s if _keep_char(c))


@dataclass
class NormalizedCharOrigin:
    page: int
    page_offset: int  # offset in the original raw page text
    char: str  # original character kept in normalized stream


@dataclass
class SegmentPageSpan:
    page: int
    start: int  # inclusive (raw page offset)
    end: int  # exclusive (raw page offset)


@dataclass
class AlignedSegment:
    segment_text: str
    page_spans: list[SegmentPageSpan]  # ordered, non-overlapping


class TextAlignmentIndex:
    """
    Builds a normalized searchable string over ALL raw pages while keeping
    a mapping from each normalized character back to (page, page_offset).
    """

    def __init__(self, raw_pages: list[str]):
        self.raw_pages = raw_pages
        self.normalized_chars: list[str] = []
        self.origins: list[NormalizedCharOrigin] = []
        for page_idx, page_text in enumerate(raw_pages):
            for i, ch in enumerate(page_text):
                if _keep_char(ch):
                    self.normalized_chars.append(ch.lower())
                    self.origins.append(
                        NormalizedCharOrigin(
                            page=page_idx,
                            page_offset=i,
                            char=ch,
                        )
                    )
        self.normalized = "".join(self.normalized_chars)
        self.page_words_cache: dict[int, list[dict]] = {}

    def normalize_fragment(self, fragment: str) -> str:
        return _normalize(fragment)

    def _locate(
        self, cleaned_segment: str, start_hint: int = 0
    ) -> Optional[tuple[int, int]]:
        """
        Returns (norm_start, norm_end_exclusive) in normalized space
        or None if not found.
        """
        norm_seg = self.normalize_fragment(cleaned_segment)
        if not norm_seg:
            return None
        idx = self.normalized.find(norm_seg, start_hint)
        if idx == -1:
            # fallback full search
            idx = self.normalized.find(norm_seg)
            if idx == -1:
                return None
        return idx, idx + len(norm_seg)

    def _map_norm_range_to_page_spans(
        self, norm_start: int, norm_end: int
    ) -> list[SegmentPageSpan]:
        origins_slice = self.origins[norm_start:norm_end]
        if not origins_slice:
            return []
        spans: list[SegmentPageSpan] = []
        current_page = origins_slice[0].page
        start_offset = origins_slice[0].page_offset
        prev_offset = origins_slice[0].page_offset
        for o in origins_slice[1:]:
            if o.page == current_page and o.page_offset == prev_offset + 1:
                prev_offset = o.page_offset
                continue
            spans.append(
                SegmentPageSpan(
                    page=current_page, start=start_offset, end=prev_offset + 1
                )
            )
            current_page = o.page
            start_offset = o.page_offset
            prev_offset = o.page_offset
        spans.append(
            SegmentPageSpan(page=current_page, start=start_offset, end=prev_offset + 1)
        )
        return spans

    def align_segments(self, segments: list[str]) -> list[AlignedSegment]:
        results: list[AlignedSegment] = []
        search_cursor = 0
        for seg in segments:
            loc = self._locate(seg, search_cursor)
            if not loc:
                results.append(
                    AlignedSegment(
                        segment_text=seg,
                        page_spans=[],
                    )
                )
                continue
            norm_start, norm_end = loc
            page_spans = self._map_norm_range_to_page_spans(norm_start, norm_end)
            search_cursor = norm_end
            results.append(
                AlignedSegment(
                    segment_text=seg,
                    page_spans=page_spans,
                )
            )
        return results

    def spans_to_page_polygons(
        self, doc, page_spans: list[SegmentPageSpan]
    ) -> dict[int, list[tuple[float, float]]]:
        """Return precise paragraph polygons per page (at most 2 pages).

        Requirements:
        - Respect exact start/end offsets (can begin/end mid-line).
        - Produce a single polygon per page following ragged left/right paragraph edges.
        - Avoid earlier bottom-right glitches (deduplicate sequential identical points).
        - Normalized coordinates (0..1) with final point repeated to close.
        """

        if not page_spans:
            return {}

        # Group spans by page (normally at most one span per page for a continuous segment).
        spans_by_page: dict[int, list[tuple[int, int]]] = {}
        for ps in page_spans:
            spans_by_page.setdefault(ps.page, []).append((ps.start, ps.end))

        page_to_polygon: dict[int, list[tuple[float, float]]] = {}

        for page_idx in sorted(spans_by_page.keys())[:2]:  # limit to two pages
            if page_idx >= len(self.raw_pages):
                continue
            page_raw = self.raw_pages[page_idx]
            if not page_raw:
                continue

            ranges = spans_by_page[page_idx]
            # Merge to single continuous inclusive range covering text for this segment on the page.
            span_start = min(r[0] for r in ranges)
            span_end = max(r[1] for r in ranges)
            p_len = len(page_raw)
            span_start = max(0, min(span_start, p_len))
            span_end = max(span_start, min(span_end, p_len))
            if span_start == span_end:
                continue

            # Ensure word cache for page
            if page_idx not in self.page_words_cache:
                page = doc[page_idx]
                tp = page.get_textpage()
                try:
                    words = tp.extractWORDS()
                except Exception:
                    self.page_words_cache[page_idx] = []
                else:
                    pointer = 0
                    items: list[dict] = []
                    for x0, y0, x1, y1, wtext, _block, line_no, _wno in words:
                        search_from = pointer
                        while (
                            search_from < len(page_raw)
                            and page_raw[search_from].isspace()
                        ):
                            search_from += 1
                        candidate_start = page_raw.find(
                            wtext, search_from, min(len(page_raw), search_from + 500)
                        )
                        if candidate_start == -1:
                            continue
                        start_off = candidate_start
                        end_off = start_off + len(wtext)
                        pointer = end_off
                        items.append(
                            {
                                "start": start_off,
                                "end": end_off,
                                "line": line_no,
                                "rect": (x0, y0, x1, y1),
                                "text": wtext,
                            }
                        )
                    self.page_words_cache[page_idx] = items

            words = [
                w
                for w in self.page_words_cache.get(page_idx, [])
                if not (w["end"] <= span_start or w["start"] >= span_end)
            ]
            if not words:
                continue

            # Group by line number preserving order (PyMuPDF words already top->bottom, left->right)
            lines: dict[int, list[dict]] = {}
            for w in words:
                lines.setdefault(w["line"], []).append(w)
            # Sort each line's words by x0
            for arr in lines.values():
                arr.sort(key=lambda w: w["rect"][0])

            # Derive per-line left/right bounds with partial-word cropping
            line_entries: list[
                tuple[float, float, float, float]
            ] = []  # (y_top,y_bottom,left_x,right_x)
            for line_no in sorted(
                lines.keys(), key=lambda ln: min(w["rect"][1] for w in lines[ln])
            ):
                wlist = lines[line_no]
                # Compute left boundary
                left_x = None
                right_x = None
                y_top = min(w["rect"][1] for w in wlist)
                y_bottom = max(w["rect"][3] for w in wlist)
                for w in wlist:
                    if w["end"] > span_start:  # first word intersecting start
                        x0, y0, x1, y1 = w["rect"]
                        if span_start > w["start"] and span_start < w["end"]:
                            # partial word start
                            frac = (span_start - w["start"]) / (w["end"] - w["start"])
                            left_x = x0 + (x1 - x0) * frac
                        else:
                            left_x = x0
                        break
                # Compute right boundary
                for w in reversed(wlist):
                    if w["start"] < span_end:  # last word intersecting end
                        x0, y0, x1, y1 = w["rect"]
                        if span_end < w["end"] and span_end > w["start"]:
                            frac = (span_end - w["start"]) / (w["end"] - w["start"])
                            right_x = x0 + (x1 - x0) * frac
                        else:
                            right_x = x1
                        break
                if left_x is None or right_x is None:
                    continue
                if right_x < left_x:  # safety
                    continue
                line_entries.append((y_top, y_bottom, left_x, right_x))

            if not line_entries:
                continue

            # Sort by vertical position (already roughly sorted but ensure)
            line_entries.sort(key=lambda t: t[0])

            # Build a simple non-self-intersecting outline.
            # Left boundary top->bottom with horizontal steps when left_x changes.
            left_boundary: list[tuple[float, float]] = []
            for i, (y_top, y_bottom, left_x, right_x) in enumerate(line_entries):
                left_boundary.append((left_x, y_top))
                left_boundary.append((left_x, y_bottom))

            # Right boundary bottom->top mirrored with steps where width changes.
            right_boundary: list[tuple[float, float]] = []
            for i, (y_top, y_bottom, left_x, right_x) in enumerate(
                reversed(line_entries)
            ):
                # Ascend to top of this line
                right_boundary.append((right_x, y_bottom))
                right_boundary.append((right_x, y_top))

            # Combine: left boundary (top->bottom), then right boundary (bottom->top), then close via top-left.
            outline = left_boundary + right_boundary
            if outline and outline[0] != outline[-1]:
                outline.append(outline[0])

            # Remove consecutive duplicates just in case
            cleaned: list[tuple[float, float]] = []
            for pt in outline:
                if not cleaned or cleaned[-1] != pt:
                    cleaned.append(pt)
            outline = cleaned

            # Normalize
            page = doc[page_idx]
            width = page.rect.width or 1.0
            height = page.rect.height or 1.0
            polygon = [(x / width, y / height) for (x, y) in outline]
            page_to_polygon[page_idx] = polygon

        return page_to_polygon

    def align_segments_and_get_polygons(self, segments: list[str], doc) -> list[dict]:
        aligned: list[AlignedSegment] = self.align_segments(segments)
        out: list[dict] = []
        for a in aligned:
            page_map = self.spans_to_page_polygons(doc, a.page_spans)
            out.append(
                {
                    "text": a.segment_text,
                    "polygons": page_map,
                }
            )
        return out
