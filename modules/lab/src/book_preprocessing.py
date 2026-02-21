from __future__ import annotations
import re
import logging
import pymupdf
from dataclasses import dataclass
import regex


class BookPreprocessor:
    def __init__(self):
        self.before_clean_translation_table = str.maketrans(
            {
                "\r": None,
                "’": "'",
                "“": '"',
                "”": '"',
                "？": "?",
                "！": "!",
            }
        )

        self.patterns = {
            "page_numbers": re.compile(
                r"[\n\r][^\S\r\n]*(?:page\s+)?(?:\d+|\[?\d+\]?)[^\S\r\n]*(?=[\n\r]|$)",
                re.IGNORECASE,
            ),
            "chapter_headers": re.compile(
                r"^\s*(?:chapter|ch\.?|part|section|§|act|volume)\s+(?:[ivx]+|\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)(?:\.|:|\s|$)",
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
                r"(?<=[^.!?。:\"'–—-])(?:\n(?=[–—-]?(?:\p{L}|[\"'])))|(?:\s+(?=[–—-]?(?:\p{Ll}|I[ ']|[\"'])))"
            ),
            "line_breaked_sentence_a_end": regex.compile(r"[^.!?。:\"'–—-]$"),
            "line_breaked_sentence_b_start": regex.compile(r"[–—-]?(?:\p{L}|[\"'])"),
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
            ":",
            '"',
            "'",
        }
        self.text_keywords = {"!", "?", '"', ":"}
        self.short_line_threshold = 10

    def clean_text(
        self, text: str, prev_line: str = "", return_split_with_removed=False
    ) -> str | tuple[str, list[str], set[int]]:
        """prev_line is for cases where you are using clean_text on segments of connected text"""

        text = self._before_clean(text)

        if not text or len(text) < 10:
            if return_split_with_removed:
                return text, [text], []
            return text

        lines = text.split("\n")

        cleaned, removed_lines = self._sequential_clean(lines, prev_line)
        if return_split_with_removed:
            return cleaned, lines, removed_lines
        return cleaned
        # if len(lines) > 1000:
        #     return self._parallel_clean(lines, prev_line)
        # else:
        #     return self._sequential_clean(lines, prev_line)

    def _before_clean(self, text: str) -> str:
        """Hook for preprocessing before cleaning"""

        text = text.strip().translate(self.before_clean_translation_table)

        text = self.patterns["page_numbers"].sub("", text)

        # Unhyphenate
        text = self.patterns["hyphenated_sentence"].sub("", text)

        # Remove functional newlines for page width control - they break the text splitter semantics
        # -> replace newline in the middles of sentences with a white space, e.g. 'Soon\nanother clock began, on a hard, decisive note.'
        text = self.patterns["line_breaked_sentence"].sub(" ", text)

        return text

    # def _parallel_clean(self, lines: list[str], prev_line: str = "") -> str:
    #     """Parallel processing for large texts"""
    #     batch_size = max(100, len(lines) // mp.cpu_count())
    #     batches = [
    #         (lines[i : i + batch_size], lines[i - 1] if i > 0 else prev_line)
    #         for i in range(0, len(lines), batch_size)
    #     ]

    #     with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
    #         processed_batches = list(
    #             executor.map(
    #                 lambda ctx: self._process_batch(ctx[0], ctx[1]),
    #                 batches,
    #             )
    #         )

    #     # Flatten results
    #     processed_lines = []
    #     for batch_lines in processed_batches:
    #         processed_lines.extend(batch_lines)

    #     return self._after_clean(processed_lines)

    def _sequential_clean(
        self, lines: list[str], prev_line: str = ""
    ) -> tuple[str, set[int]]:
        """Sequential processing for smaller texts"""
        processed_lines, removed_lines = self._process_batch(lines, prev_line)
        return self._after_clean(processed_lines), removed_lines

    def _process_batch(
        self, lines: list[str], saved_prev_line=""
    ) -> tuple[list[str], set[int]]:
        """Process a batch of lines and filter them"""
        processed: list[str] = []
        removed_lines: set[int] = set()
        total_lines = len(lines)

        for i, line in enumerate(lines):
            prev_line = lines[i - 1] if i > 0 else saved_prev_line
            result = self.process_line(line, prev_line, i, total_lines)
            if not result or result.isspace():
                removed_lines.add(i)
            if result is not None:
                processed.append(result)

        return processed, removed_lines

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
        # text = self.patterns["hyphenated_sentence"].sub("", text)

        # Remove functional newlines for page width control - they break the text splitter semantics
        # -> replace newline in the middles of sentences with a white space, e.g. 'Soon\nanother clock began, on a hard, decisive note.'
        # text = self.patterns["line_breaked_sentence"].sub(" ", text)

        # text = unicodedata.normalize("NFKC", text)
        return text


@dataclass
class PdfExtractionConfig:
    max_pages: int | None = None  # limit for debugging


@dataclass
class ExtractionContext:
    pdf_bytes: bytes
    cleaned_text: str
    normalized_full_text: str
    removed_ranges: list[tuple[int, int]]


class PdfBookPreprocessor(BookPreprocessor):
    def __init__(
        self, polygon_x_smooth_max_gap_px: int = 12, polygon_padding_px: int = 1
    ):
        super().__init__()
        self.polygon_x_smooth_max_gap_px = polygon_x_smooth_max_gap_px
        self.polygon_padding_px = polygon_padding_px
        self.logger = logging.getLogger("django")

    def _get_from_doc(
        self, doc, page_limit: int | None = None
    ) -> tuple[str, str, list[tuple[int, int]]]:
        if page_limit is None:
            page_limit = len(doc)
        pages: list[str] = []
        last_line_of_last_page = ""
        normalized_full_text = ""
        removed_ranges: list[tuple[int, int]] = []
        for i in range(page_limit):
            try:
                page = doc[i]
                page_text = page.get_textpage().extractText(sort=True)  # type: ignore
                cleaned_page_text, lines, removed_lines = self.clean_text(
                    page_text, last_line_of_last_page, return_split_with_removed=True
                )

                # Fill normalized_full_text and removed_ranges
                for j, line in enumerate(lines):
                    normalized_line = self._normalize(line)
                    normalized_full_text += normalized_line
                    if j in removed_lines:
                        start = len(normalized_full_text) - len(normalized_line)
                        end = len(normalized_full_text)
                        removed_ranges.append((start, end))

                # Join pages that cut a sentence in half
                if (
                    last_line_of_last_page
                    and self.patterns["line_breaked_sentence_a_end"].search(
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

        cleaned_text = "".join(pages)
        return cleaned_text, normalized_full_text, removed_ranges

    def extract_from_memory(
        self, pdf_file, config: PdfExtractionConfig | None = None
    ) -> ExtractionContext:
        if config is None:
            config = PdfExtractionConfig()

        try:
            pdf_bytes = pdf_file.read()
            with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
                cleaned_text, normalized_full_text, removed_ranges = self._get_from_doc(
                    doc, config.max_pages
                )
        except Exception:
            self.logger.exception("Failed to open PDF")
            raise

        return ExtractionContext(
            pdf_bytes=pdf_bytes,
            cleaned_text=cleaned_text,
            normalized_full_text=normalized_full_text,
            removed_ranges=removed_ranges,
        )

    def align_segments_with_pages(
        self, segments: list[str], ctx: ExtractionContext
    ) -> list[dict]:
        """Map cleaned segments to original page polygons.
        Returns a list of dicts:
        {
            'id': int,  # segment index
            'text': segment_text,
            'polygons': { page_index: [(x1, y1), (x2, y2), ...]  }  # normalized coordinates
        }
        """
        doc = None
        try:
            doc = pymupdf.open(stream=ctx.pdf_bytes, filetype="pdf")
        except Exception:
            self.logger.exception("Failed to reopen PDF for polygons")
            return [
                {
                    "id": i,
                    "text": seg,
                    "polygons": {},
                }
                for i, seg in enumerate(segments)
            ]

        seg_to_page_to_polygon = self._segments_to_page_polygons(
            doc, segments, ctx.normalized_full_text, ctx.removed_ranges
        )
        if doc is not None:
            doc.close()
        return [
            {
                "id": i,
                "text": seg,
                "polygons": seg_to_page_to_polygon.get(i, {}),
            }
            for i, seg in enumerate(segments)
        ]

    def _segments_to_page_polygons(
        self,
        doc,
        segments: list[str],
        normalized_full_text: str,
        removed_ranges: list[tuple[int, int]],
    ) -> dict[int, list[tuple[float, float]]]:
        """Convert a segment of text into page polygons."""
        segments = [self._normalize(s) for s in segments]
        concat_segments = "".join(segments)
        segment_idx = 0
        per_segment_idx = 0
        concat_segments_idx = 0
        original_text_idx = 0
        seg_to_page_to_lines = {}

        for page_idx, page in enumerate(doc):
            try:
                words = page.get_textpage().extractWORDS()
            except Exception:
                continue
            for x0, y0, x1, y1, wtext, block, line_no, _wno in words:
                norm_wtext = self._normalize(wtext)
                if not norm_wtext:
                    continue

                if not normalized_full_text.startswith(norm_wtext, original_text_idx):
                    self.logger.warning(
                        f"Word '{wtext}' does not match word at original_text_idx {original_text_idx}"
                    )
                    continue

                original_text_idx += len(norm_wtext)
                word_in_removed_range = self._word_overlaps_removed_ranges(
                    original_text_idx - len(norm_wtext),
                    original_text_idx,
                    removed_ranges,
                )
                if word_in_removed_range:
                    continue

                if concat_segments.startswith(norm_wtext, concat_segments_idx):
                    per_segment_idx += len(norm_wtext)
                    concat_segments_idx += len(norm_wtext)

                    if per_segment_idx > len(segments[segment_idx]):
                        # add part of word rect to current segment, other part to next segment
                        split_at = len(norm_wtext) - (
                            per_segment_idx - len(segments[segment_idx])
                        )
                        avg_letter_width = (x1 - x0) / len(wtext)

                        seg_to_page_to_lines.setdefault(segment_idx, {}).setdefault(
                            page_idx, {}
                        ).setdefault(f"{block}.{line_no}", []).append(
                            {
                                "text": wtext,
                                "rect": (x0, y0, x0 + split_at * avg_letter_width, y1),
                            }
                        )
                        per_segment_idx -= len(segments[segment_idx])
                        segment_idx += 1
                        if segment_idx >= len(segments):
                            break
                        seg_to_page_to_lines.setdefault(segment_idx, {}).setdefault(
                            page_idx, {}
                        ).setdefault(f"{block}.{line_no}", []).append(
                            {
                                "text": wtext,
                                "rect": (x0 + split_at * avg_letter_width, y0, x1, y1),
                            }
                        )
                        continue

                    seg_to_page_to_lines.setdefault(segment_idx, {}).setdefault(
                        page_idx, {}
                    ).setdefault(f"{block}.{line_no}", []).append(
                        {
                            "text": wtext,
                            "rect": (x0, y0, x1, y1),
                        }
                    )
                    if per_segment_idx == len(segments[segment_idx]):
                        segment_idx += 1
                        per_segment_idx = 0
                        if segment_idx >= len(segments):
                            break

        seg_to_page_to_polygon = {}

        for segment_idx, page_to_lines in seg_to_page_to_lines.items():
            for page_idx, lines in page_to_lines.items():
                for arr in lines.values():
                    arr.sort(key=lambda w: w["rect"][0])

                # Derive per-line left/right bounds
                line_entries: list[
                    tuple[float, float, float, float]
                ] = []  # (y_top, y_bottom, left_x, right_x)
                for line_no in sorted(
                    lines.keys(), key=lambda ln: min(w["rect"][1] for w in lines[ln])
                ):
                    wlist = lines[line_no]
                    left_x = None
                    right_x = None
                    y_top = min(w["rect"][1] for w in wlist)
                    y_bottom = max(w["rect"][3] for w in wlist)
                    # Compute left line boundary (first word intersecting start)
                    for w in wlist:
                        x0, y0, x1, y1 = w["rect"]
                        left_x = x0
                        break
                    # Compute right line boundary (last word intersecting end)
                    for w in reversed(wlist):
                        x0, y0, x1, y1 = w["rect"]
                        right_x = x1
                        break
                    if left_x is None or right_x is None or right_x < left_x:
                        continue
                    line_entries.append((y_top, y_bottom, left_x, right_x))

                if not line_entries:
                    continue

                # Build a simple non-self-intersecting outline.
                # Left boundary top->bottom with horizontal steps when left_x changes.
                left_boundary: list[tuple[float, float]] = []
                for y_top, y_bottom, left_x, right_x in line_entries:
                    # Skip if y_bottom <= last y_bottom to negate pymupdf parsing errors
                    if left_boundary and y_bottom <= left_boundary[-1][1]:
                        continue
                    left_boundary.append((left_x, y_top))
                    left_boundary.append((left_x, y_bottom))
                # Right boundary bottom->top mirrored with steps where width changes.
                right_boundary: list[tuple[float, float]] = []
                for y_top, y_bottom, left_x, right_x in reversed(line_entries):
                    # Skip if y_top >= last y_top to negate pymupdf parsing errors
                    if right_boundary and y_top >= right_boundary[-1][1]:
                        continue
                    right_boundary.append((right_x, y_bottom))
                    right_boundary.append((right_x, y_top))

                # Smooth and deduplicate boundaries
                self._smooth_boundary(left_boundary, is_left=True)
                self._smooth_boundary(right_boundary, is_left=False)

                # Combine: left boundary (top->bottom), then right boundary (bottom->top), then close via top-left.
                # Average y between boundary joins to straighten horizontal joins
                avg_y_bottom = (left_boundary[-1][1] + right_boundary[0][1]) / 2
                left_boundary[-1] = (left_boundary[-1][0], avg_y_bottom)
                right_boundary[0] = (right_boundary[0][0], avg_y_bottom)
                avg_y_top = (left_boundary[0][1] + right_boundary[-1][1]) / 2
                left_boundary[0] = (left_boundary[0][0], avg_y_top)
                right_boundary[-1] = (right_boundary[-1][0], avg_y_top)
                outline = left_boundary + right_boundary

                # Normalize
                page = doc[page_idx]
                width = page.rect.width or 1.0
                height = page.rect.height or 1.0
                polygon = [(x / width, y / height) for (x, y) in outline]

                # Scale polygon outwards by PADDING_PX
                cx = sum(x for x, _ in polygon) / len(polygon)
                cy = sum(y for _, y in polygon) / len(
                    polygon
                )  # compute pixel half-width/height
                min_x = min(p[0] for p in polygon)
                max_x = max(p[0] for p in polygon)
                min_y = min(p[1] for p in polygon)
                max_y = max(p[1] for p in polygon)
                cur_w_px = (max_x - min_x) * width
                cur_h_px = (max_y - min_y) * height
                sx = 1.0 + (self.polygon_padding_px * 2) / max(cur_w_px, 1e-6)
                sy = 1.0 + (self.polygon_padding_px * 2) / max(cur_h_px, 1e-6)
                scaled = []
                for x, y in polygon:
                    x2 = cx + (x - cx) * sx
                    y2 = cy + (y - cy) * sy
                    scaled.append((min(1.0, max(0.0, x2)), min(1.0, max(0.0, y2))))
                polygon = scaled

                seg_to_page_to_polygon.setdefault(segment_idx, {})[page_idx] = polygon

        return seg_to_page_to_polygon

    def _word_overlaps_removed_ranges(
        self,
        word_start: int,
        word_end: int,
        removed_ranges: list[tuple[int, int]],
    ) -> bool:
        """Efficiently check if a word range overlaps with any removed range."""
        if not removed_ranges:
            return False

        left, right = 0, len(removed_ranges)
        while left < right:
            mid = (left + right) // 2
            range_start, range_end = removed_ranges[mid]

            if range_end <= word_start:
                # This range ends before our word starts, look right
                left = mid + 1
            else:
                # This range might overlap, look left
                right = mid

        for i in range(left, len(removed_ranges)):
            range_start, range_end = removed_ranges[i]
            if range_start >= word_end:
                break
            if range_start < word_end and range_end > word_start:
                return True

        return False

    def _normalize(self, s: str) -> str:
        return re.sub(
            r"[\s\-‐‑‒–—]+", "", s.translate(self.before_clean_translation_table)
        )

    def _smooth_boundary(
        self,
        boundary: list[tuple[float, float]],
        is_left: bool,
    ):
        """When two consecutive lines start or end at slightly different x or y, smooth the step.
        Then, deduplicate points with identical x and close y.
        """
        if not boundary or len(boundary) < 2:
            return

        # Iteratively smooth smallest horizontal deltas until exceeding threshold.
        max_iterations = len(boundary) * 2  # safety guard
        for _ in range(max_iterations):
            min_idx = -1
            min_diff = float("inf")
            prev_x = boundary[0][0]
            for i in range(1, len(boundary)):
                cur_x = boundary[i][0]
                if cur_x != prev_x:
                    d = abs(cur_x - prev_x)
                    if d < min_diff:
                        min_diff = d
                        min_idx = i
                prev_x = cur_x

            if min_idx == -1 or min_diff > self.polygon_x_smooth_max_gap_px:
                break

            last_x, last_y = boundary[min_idx - 1]
            new_x, new_y = boundary[min_idx]
            changed = False

            if (is_left and new_x < last_x) or (not is_left and new_x > last_x):
                if last_x != new_x:
                    boundary[min_idx - 1] = (new_x, last_y)
                    changed = True
                if (
                    min_idx > 1
                    and boundary[min_idx - 2][0] == last_x
                    and boundary[min_idx - 2][0] != new_x
                ):
                    boundary[min_idx - 2] = (new_x, boundary[min_idx - 2][1])
                    changed = True
            else:
                if new_x != last_x:
                    boundary[min_idx] = (last_x, new_y)
                    changed = True
                if (
                    min_idx + 1 < len(boundary)
                    and boundary[min_idx + 1][0] == new_x
                    and boundary[min_idx + 1][0] != last_x
                ):
                    boundary[min_idx + 1] = (last_x, boundary[min_idx + 1][1])
                    changed = True

            if not changed:
                break

        # Deduplicate sequential points with identical x and close y
        deduped: list[tuple[float, float]] = []
        for pt in boundary:
            if deduped and pt[0] == deduped[-1][0] and abs(pt[1] - deduped[-1][1]) < 2:
                continue
            deduped.append(pt)

        # Update boundary in place
        boundary[:] = deduped

        # Straighten horizontal joins by averaging y between sequential points
        for i in range(1, len(boundary)):
            if boundary[i][0] == boundary[i - 1][0]:
                continue
            avg_y = (boundary[i - 1][1] + boundary[i][1]) / 2
            boundary[i - 1] = (boundary[i - 1][0], avg_y)
            boundary[i] = (boundary[i][0], avg_y)


class TxtBookPreprocessor(BookPreprocessor):
    """Most txt books from Gutenberg dataset have extra newlines for readability, which are undesirable for segmenting"""

    def _before_clean(self, text):
        """Remove extra newlines"""
        text = super()._before_clean(text)
        return re.sub(r"\S\n\n[ \t]*", lambda m: m.group(0).rstrip() + " ", text)
