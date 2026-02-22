# Optimization Checklist

> **Hard constraint:** The Django Ninja process — including spaCy model, all libraries, and
> per-request working set — must stay under **512 MB total RSS**. A 10 MB PDF with ~2 MB extracted
> text can saturate the heap if allocations are not tightly controlled.
>
> Within each section, RAM items appear first, CPU items second. `[BREAKING]` items require changes
> to public interfaces or call sites. Items marked **Verify:** are duplicates of an earlier item —
> complete the related implementation first, then check these off.

---

## Process-level / startup

*Address these first — they apply before any request is processed.*

- [ ] **Load the smallest viable spaCy model and disable all unused pipeline components.**
  `en_core_web_sm` ~30 MB RSS; `en_core_web_md` ~90 MB; `en_core_web_lg` cannot fit in 512 MB.
  Confirm the smallest model satisfying all required pipeline components is used. Audit whether
  `"tok2vec"` can be added to the `disable` list if no retained component depends on it.
  **Saves 30–100 MB RSS — the single highest-leverage startup knob.**

- [ ] **Set `MALLOC_TRIM_THRESHOLD_=131072` and/or preload `jemalloc`/`mimalloc`.**
  CPython's pymalloc arena allocator never returns freed pages to the OS, causing RSS to climb
  monotonically after large requests. Set `MALLOC_TRIM_THRESHOLD_=131072` as an environment
  variable to tell glibc `malloc` to trim more aggressively; alternatively preload `jemalloc` or
  `mimalloc` via `LD_PRELOAD`. Optionally call `ctypes.CDLL("libc.so.6").malloc_trim(0)` at the
  end of each request handler.
  **Saves 20–80 MB of apparent RSS freed but not returned to the OS between requests.**

- [ ] **Configure gunicorn/daphne with `--max-requests` to recycle workers periodically.**
  Set `--max-requests 50` (tune to observed RSS growth) and `--max-requests-jitter 10` to prevent
  simultaneous recycling of all workers.
  **Saves 20–100 MB of fragmentation that accumulates across successive large requests.**

---

## `_get_from_doc` / `extract_from_memory`

- [ ] **[BREAKING] Drop `ExtractionContext.full_text` — stop collecting raw page text entirely.**
  `full_text` is written to an `io.StringIO`, converted with `str(full_text)`, stored in
  `ExtractionContext`, and never read again by `get_segments_with_boxes`. For a 10 MB PDF this
  wastes ~4 MB peak (the `StringIO` buffer and the final `str` are simultaneously live). Remove
  `full_text = io.StringIO()`, `full_text.write(page_text)`, and `return str(full_text)` from
  `_get_from_doc`, and drop `ExtractionContext.full_text`. Update callers unpacking the four-tuple
  return (`TxtBookPreprocessor.create_pdf_and_align`, `get_segments_boxes_from_txt`) to the new
  three-tuple.
  **Saves ~2–8 MB peak.**

- [ ] **Replace `normalized_full_text += normalized_line` with a list joined once; track length with a running counter.**
  Each `+=` on an immutable `str` copies the entire accumulated string — O(n²) total byte copies
  across all lines. Change to `nft_parts: list[str] = []` / `nft_parts.append(normalized_line)`,
  then after the loop: `normalized_full_text = "".join(nft_parts)`. At the same time, replace the
  two `len(normalized_full_text)` calls per removed line with a running `nft_len` integer
  incremented by `len(normalized_line)` each iteration (avoids scanning the string for its length).
  Apply the same list fix in `TxtBookPreprocessor.extract_from_memory`.
  **Saves ~1–2 MB peak (eliminates all intermediate string copies during accumulation).**

- [ ] **Call `page.get_textpage()` once per page and reuse the object for both text and word extraction.**
  `_get_from_doc` calls it for `.extractText(sort=True)` and `_segments_to_page_polygons` calls it
  again for `.extractWORDS()` on the same pages — two full C-level page parses per page. Store the
  `TextPage` in a local variable and reuse it within `_get_from_doc`; if the two-pass merge below
  is implemented, pre-extract word lists in `_get_from_doc` and pass them through `ExtractionContext`
  so `_segments_to_page_polygons` never needs to reparse.
  **Saves tens of KB per page of C-heap; up to a few MB across a 300-page book.**

- [ ] **Stream `cleaned_page_text` directly into a `StringIO` instead of accumulating in a `pages` list.**
  `pages` holds one string per page until `"".join(pages)` makes a full copy at the end, keeping
  all page strings live simultaneously. Write each cleaned page directly into an `io.StringIO` with
  the cross-page join logic applied inline; call `getvalue()` once at the end.
  **Saves ~1 copy of `cleaned_text` (~1–2 MB) during the join step.**

- [ ] **Delete `page_text` immediately after `clean_text` returns.**
  Add `del page_text` right after `self.clean_text(page_text, ...)` so CPython can reclaim it
  before the next page's allocation.
  **Saves ~5–50 KB per page from being live concurrently with its cleaned copy.**

- [ ] **Skip cross-page join regex checks when `pages` is empty (first page only).**
  On page 0, `last_line_of_last_page` is `""` and `pages` is empty, so evaluating both
  `line_breaked_sentence_a_end.search(...)` and `line_breaked_sentence_b_start.match(...)` is
  pointless. Guard with `if pages and last_line_of_last_page and ...`.

- [ ] Verify: **`io.StringIO` + `str(full_text)` accumulation.** If the [BREAKING] `full_text` drop
  above is implemented, the `io.StringIO` is already gone. If `full_text` must be retained for
  other callers, replace `io.StringIO` with plain list accumulation
  (`full_text_parts = []` / `.append(page_text)` / `"".join(full_text_parts)`).

---

## `BookPreprocessor.__init__` / class-level setup

- [ ] **Pre-compile the `_normalize` regex at class level.**
  `_normalize` calls `re.sub(r"[\s\-‐‑‒–—]+", "", ...)` on every invocation, recompiling the
  pattern each time. It is called once per line in `_get_from_doc` and once per word in
  `_segments_to_page_polygons` — the hottest Python-level call site in the pipeline. Add
  `self.patterns["normalize"] = re.compile(r"[\s\-‐‑‒–—]+")` in `__init__` and replace the body
  with `return self.patterns["normalize"].sub("", s.translate(...))`.

- [ ] **Expand the `str.maketrans` table to delete dash variants directly, simplifying the normalize regex.**
  Map `\u2010` (‐), `\u2011` (‑), `\u2012` (‒), `\u2013` (–), `\u2014` (—) → `None` in
  `before_clean_translation_table`. After translation all dashes are already removed, so the regex
  in `_normalize` only needs to strip `\s`, simplifying it from the multi-alternation character
  class to the faster `r"\s+"`.

- [ ] **Compile `metadata_keywords` into a single regex.**
  `process_line` iterates over all 25 keywords per line. Compile once in `__init__`:
  `self._metadata_re = re.compile("|".join(re.escape(kw) for kw in self.metadata_keywords))`.
  In `process_line`, replace the list comprehension with
  `sum(1 for _ in self._metadata_re.finditer(line_lower))` — one C-level automaton pass that also
  eliminates the temporary list allocation.

- [ ] **Replace the `text_keywords` set check with a compiled character-class regex.**
  `any(kw in line_lower for kw in self.text_keywords)` → add
  `self._text_kw_re = re.compile(r'[!?":]')` in `__init__` and use
  `bool(self._text_kw_re.search(line_lower))` in `process_line`.

---

## `process_line` / `_process_batch`

- [ ] **Replace `[kw for kw in self.metadata_keywords if kw in line_lower]` list comprehension with a `sum()` generator.**
  The walrus-operator expression `len([kw for kw in ...])` builds a temporary list on every line
  that passes the earlier filters. Replace with `sum(1 for kw in self.metadata_keywords if kw in line_lower)`.
  **Saves one list allocation per line — aggregate saving across all lines in the book.**
  > If the `_metadata_re` regex from `__init__` is implemented, use
  > `sum(1 for _ in self._metadata_re.finditer(line_lower))` for the CPU benefit too.

- [ ] **Replace `words_lower = line_lower.split()` with `line_lower.count(" ") + 1` for the word count.**
  `line_lower.split()` allocates a full list of word strings that are never accessed individually.
  `line_lower.count(" ") + 1` computes the word count in a single C-level scan with zero list allocation.
  **Saves one list-of-strings allocation per non-empty line across the entire book.**

- [ ] **Change `removed_lines` from `set[int]` to a sorted `list[int]` with `bisect` lookups.**
  A Python `set` has ~200 bytes base overhead plus load-factor slack. Line indices are generated
  in ascending order during `_process_batch`, so they can be appended directly to a list.
  Replace `if j in removed_lines` checks (in `_get_from_doc`) with `bisect.bisect_left` for
  O(log n) lookup.

- [ ] **Reorder the nine-pattern `any()` generator to front-load cheapest and most-frequently-matching patterns.**
  The generator short-circuits on the first match. Suggested order: `numeric_only`,
  `roman_numerals`, `footnote_refs`, `copyright`, `brackets_content`, `isbn`,
  `table_of_contents`, `chapter_headers`, `website_urls` (backtracking-heavy `\S+` last).

- [ ] **Reorder `is_end_of_paragraph` to check `line[-1] in self.end_of_sentence_chars` first.**
  This is an O(1) set lookup. If `False`, the whole expression is `False` without evaluating the
  more expensive `len(prev_line.split()) > 5`. Reorganize to:
  `line[-1] in self.end_of_sentence_chars and prev_line and (len(prev_line.split()) > 5 or prev_line[-1] in self.end_of_sentence_chars)`.

- [ ] **Replace `len(prev_line.split()) > 5` with an early-exit counter.**
  `str.split()` allocates a full word list. Use a manual counter that breaks at 6.

- [ ] **Cache `line.isupper()` as a local variable.**
  It is evaluated twice: once at the all-caps guard and again inside the metadata keyword
  conditional. `is_upper = line.isupper()` at the top of `process_line` eliminates the
  duplicate call.

- [ ] **Maintain a rolling `prev_line` variable instead of indexing `lines[i - 1]`.**
  Replace `prev_line = lines[i - 1] if i > 0 else saved_prev_line` with a rolling assignment at
  the end of the loop body, removing the conditional branch and list-index operation on every
  iteration.

- [ ] **Pre-strip all lines in a single pass before entering the loop.**
  `process_line` calls `line.strip()` and `prev_line.strip()` each iteration. A single
  `lines = [l.strip() for l in lines]` before the loop halves these calls.
  *(Note: allocates one extra list per page — pure CPU trade-off with minor RAM cost.)*

- [ ] Verify: **Word count without `.lower().split()`** — the RAM item replacing `split()` with
  `count(" ") + 1` already avoids both the `.lower()` allocation and the word-list allocation.
  No separate action needed.

- [ ] Verify: **Metadata keyword list comprehension** — addressed by the RAM `sum()` item above.
  If the `__init__` regex is also implemented, update the call site accordingly at that point.

---

## `BookPreprocessor._after_clean`

- [ ] **Inline `_after_clean` into `_sequential_clean`.**
  The method body is a single `"\n".join(processed_lines)`. The extra function call per page is
  pure overhead until the commented-out hooks are restored.

---

## `align_segments_with_pages` / `_segments_to_page_polygons`

- [ ] **[BREAKING] Merge the two PDF parsing passes into one.**
  `extract_from_memory` opens the PDF and parses all pages for text. `align_segments_with_pages`
  reopens the same `pdf_bytes` and reparses all pages for word coordinates — two
  simultaneously-open PyMuPDF document objects and two full parse passes. Refactor `_get_from_doc`
  to return (and keep open) the `pymupdf.Document`; hold it open across both calls in
  `get_segments_with_boxes`, closing it only after `align_segments_with_pages` returns.
  **Saves one full PyMuPDF document parse (~10–50 MB C-heap for a large PDF) plus eliminates
  one copy of `pdf_bytes` in PyMuPDF's stream buffer.**

- [ ] **[BREAKING] Remove `ExtractionContext.pdf_bytes`.**
  `pdf_bytes` (up to 10 MB) is stored solely so `align_segments_with_pages` can reopen the PDF.
  Once the two-pass merge above is done, `pdf_bytes` is not needed beyond the first
  `pymupdf.open` call. Drop it from `ExtractionContext` and free it immediately after the document
  is open.
  **Saves 1–10 MB for the entire lifetime of the context object.**

- [ ] **Replace per-word `{"text": wtext, "rect": (x0, y0, x1, y1)}` dicts with plain 4-tuples `(x0, y0, x1, y1)`.**
  `seg_to_page_to_lines` stores one dict per word in the book. A CPython 3.11 dict has ~232 bytes
  minimum overhead; 100,000 words = ~23 MB of dict overhead alone. The `"text"` field is never
  accessed during polygon construction. Replace with `(x0, y0, x1, y1)` 4-tuples (~120 bytes each)
  and update all consumer code to use index access instead of key access.
  **Saves ~10–15 MB for a typical 100k-word book.**

- [ ] **Delete `seg_to_page_to_lines` immediately after the first loop completes.**
  The full word-level nested dict (potentially 10–30 MB) remains live while `seg_to_page_to_polygon`
  is being built in the second loop. Add `del seg_to_page_to_lines` right after the first `for`
  loop ends.
  **Saves ~10–30 MB from being live concurrently with `seg_to_page_to_polygon`.**

- [ ] **Avoid building `concat_segments = "".join(segments)` — use direct segment indexing.**
  `concat_segments` (~1–2 MB) is used only for `concat_segments.startswith(norm_wtext, concat_segments_idx)`.
  Replace with a slice comparison against the active segment:
  `segments[segment_idx][per_segment_idx : per_segment_idx + len(norm_wtext)] == norm_wtext`.
  **Saves ~1 copy of all cleaned text (~1–2 MB).**

- [ ] **Normalize segments lazily one at a time, not all up front.**
  `segments = [self._normalize(s) for s in segments]` at the top creates a full new list of
  normalized strings (~1 MB) before any page is processed. Once `concat_segments` is eliminated
  (above), normalize each segment on demand when `segment_idx` first advances to it, and `del` the
  previous normalized string.
  **Saves ~1–2 MB of normalized text being live for the entire function duration.**

- [ ] **Delete each page's `words` list at the end of its loop body.**
  Add `del words` at the end of the per-page loop so CPython drops the reference before the next
  page's list is allocated.
  **Saves ~50–500 KB per page.**

- [ ] **Replace triple-chained `.setdefault()` per word with cached intermediate dict references.**
  `seg_to_page_to_lines.setdefault(seg, {}).setdefault(page, {}).setdefault(line_key, []).append(...)`
  may allocate up to three empty dicts per word. Cache `seg_dict`, `page_dict`, and `line_list` as
  locals and update them only when `segment_idx` or `page_idx` changes.

- [ ] **Replace the f-string `f"{block}.{line_no}"` dict key with a tuple `(block, line_no)`.**
  String formatting allocates a new heap `str` per word. Since `block` and `line_no` are integers,
  a tuple is cheaper to allocate and hash.

- [ ] **Pre-compute `y_top` sort keys before calling `sorted()`.**
  The `sorted()` key lambda calls `min(w["rect"][1] for w in lines[ln])` O(L log L) times, each
  time rescanning the full word list for that line. Pre-compute
  `{ln: min(w[1] for w in lines[ln]) for ln in lines}` once before sorting (using tuple index
  after the dict→tuple migration above).

- [ ] **Compute `y_top` and `y_bottom` in a single pass.**
  Replace the two separate generator expressions `min(w["rect"][1] for w in wlist)` and
  `max(w["rect"][3] for w in wlist)` with one loop tracking a running min and max simultaneously.

- [ ] **Replace `for w in wlist: ...; break` loops with direct index access.**
  After sorting `wlist` by `x0`, `left_x = wlist[0][0]` and `right_x = wlist[-1][2]` (tuple
  indexing after the migration above) are O(1) direct lookups.

- [ ] **Cache page width/height before the post-processing loop.**
  `doc[page_idx]` is re-fetched for every `(segment_idx, page_idx)` pair. Build
  `page_dims = {i: (doc[i].rect.width or 1.0, doc[i].rect.height or 1.0) for i in range(len(doc))}`
  once before the loop.

- [ ] **Compute polygon centroid, min/max, and scaled coordinates in a single pass or with NumPy.**
  Currently six separate generator expressions each traverse `polygon`. Merge into one loop, or
  convert `polygon` to a `numpy.ndarray` and use vectorized `mean`, `min`, `max`, broadcasting,
  and `clip`. *(NumPy is also more RAM-efficient for float data: ~8 bytes/float vs ~28 bytes for
  Python float objects — saves RAM for books with many segments.)*

- [ ] Verify: **PDF reopened in `align_segments_with_pages`.** Once the [BREAKING] two-pass merge
  above is implemented, the `pymupdf.open(stream=ctx.pdf_bytes, ...)` call in
  `align_segments_with_pages` is gone. Confirm no other code path reopens the PDF from
  `ctx.pdf_bytes`.

- [ ] Verify: **Single `get_textpage()` call per page.** Confirm that after the two-pass merge,
  `_segments_to_page_polygons` consumes pre-extracted word lists from `ExtractionContext` and
  makes no direct `get_textpage()` calls.

---

## `_word_overlaps_removed_ranges`

- [ ] **Replace the hand-rolled binary search with `bisect.bisect_left`.**
  `bisect.bisect_left(removed_ranges, (word_start,))` performs the same lookup in C, eliminating
  the Python while-loop entirely. Tuple comparison by first element correctly positions against
  `(range_start, range_end)` tuples.

- [ ] **Replace the post-bisect linear scan with a single index check.**
  `removed_ranges` is sorted and non-overlapping, so at most one range can overlap after the
  bisect. Replace the `for i in range(left, ...)` loop with:
  `if left < len(removed_ranges): rs, re_ = removed_ranges[left]; return rs < word_end and re_ > word_start`.
  Then `return False`.

---

## `_smooth_boundary`

- [ ] **Add `del deduped` immediately after `boundary[:] = deduped`.**
  The local `deduped` list remains alive in the frame through the y-averaging loop that follows.
  Explicit `del` releases it immediately.
  **Saves ~1 copy of the boundary list per segment.**

- [ ] **Avoid `boundary[:] = deduped` by building the compacted result in-place.**
  In-place compaction eliminates the `deduped` list entirely, making the `del` above moot. Refactor
  the deduplication to write compacted entries back into `boundary` using an index pointer, or
  refactor callers to accept a new list as a return value.

- [ ] **Merge the deduplication pass and the y-averaging pass into one traversal.**
  The two passes after the smoothing loop can be combined into a single `for` loop, halving the
  iteration count.

- [ ] **Replace the O(n²) minimum-delta scan with a `heapq`-based O(n log n) approach.**
  The outer `for _ in range(max_iterations)` loop rescans the entire `boundary` list each iteration
  to find the smallest x-delta. Build a `heapq` of `(abs_delta, index)` pairs once and update only
  the two affected neighbor entries after each merge.
  *(This item appears in both RAM and CPU analysis — one implementation satisfies both.)*

---

## `get_segments_with_boxes` — pipeline

- [ ] **[BREAKING] Delete `ctx` explicitly after `align_segments_with_pages` returns.**
  Add `del ctx` before the `return` statement. Without it, `ctx` (holding `pdf_bytes`, `full_text`,
  `cleaned_text`, `normalized_full_text`, `removed_ranges`) stays live while the caller serializes
  `segments_with_pos` to JSON.
  **Saves ~3–20 MB from being live during response serialization.**

- [ ] **[BREAKING] Free `segments` after the Redis write, before HTTP serialization.**
  In `process_pdf`, add `del segments` immediately after `redis_client.set(...)`. `segments_with_pos`
  already embeds each segment's text, so `segments` is a full duplicate at that point.
  **Saves ~1–2 MB of duplicate text during JSON response encoding.**

---

## `TextSegmenter.segment_text`

- [ ] Verify: **`TextSegmenter` is a module-level singleton, not re-instantiated per request.**
  `TextSplitter` is Rust-backed; construction is non-trivial. Confirm the instance created at
  module load in `api.py` is the only instantiation — check tasks, worker modules, and any
  other entry points.

---

## Cross-cutting

- [ ] **Profile with `py-spy` or `cProfile` against a real book PDF before committing to structural refactors.**
  The dominant cost may lie in the PyMuPDF C layer (`get_textpage`, rendering) or the `regex`
  automaton, not Python-level loops. Confirm which call sites account for >5% of wall time before
  investing in polygon math or string-building changes.
