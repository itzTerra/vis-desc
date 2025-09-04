import json
import random
from pathlib import Path
from collections import defaultdict


class SegmentSampler:
    def __init__(self, segments_dir, book_meta_dir, samples_dir, seed):
        self.segments_dir = Path(segments_dir)
        self.book_meta_dir = Path(book_meta_dir)
        self.samples_dir = Path(samples_dir)
        self.samples_dir.mkdir(exist_ok=True)
        self._all_segments_cache = []
        self._rng = random.Random(seed)
        self._used_segments_cache = set()
        self._book_meta_cache = {}

    def _load_all_segments_with_book_meta(self):
        """Add metadata to each segment"""
        self._all_segments_cache = []
        for segment_file in self.segments_dir.glob("*.json"):
            with open(segment_file, "r") as f:
                segments = json.load(f)
            meta = self._load_book_meta(segment_file.stem)

            segments_with_meta = []
            for i, segment in enumerate(segments):
                book_id = meta.get(
                    "text_id", meta.get("original_metadata", {}).get("text_id")
                )
                segment = {
                    "segment_id": f"{book_id}_{i}",
                    "book_id": book_id,
                    "book_slug": segment_file.stem,
                    "genre": meta["genre"],
                    "length": len(segment),
                    "text": segment,
                }
                segments_with_meta.append(segment)
            self._all_segments_cache.extend(segments_with_meta)
        return self._all_segments_cache

    def _load_used_segments(self):
        if self._used_segments_cache:
            return self._used_segments_cache

        used = set()
        for sample_file in self.samples_dir.glob("*.json"):
            with open(sample_file, "r") as f:
                data = json.load(f)
                for segment in data:
                    used.add(segment["segment_id"])

        self._used_segments_cache = used
        return used

    def _load_book_meta(self, book_slug: str):
        if book_slug in self._book_meta_cache:
            return self._book_meta_cache[book_slug]

        meta_file = self.book_meta_dir / f"{book_slug}.json"
        if not meta_file.exists():
            return {}

        with open(meta_file, "r") as f:
            meta = json.load(f)
            self._book_meta_cache[book_slug] = meta
            return meta

    def sample_balanced(self, num_segments, output_name=None, max_per_book=5):
        """Sample segments trying to balance across genres"""

        all_segments = self._load_all_segments_with_book_meta()
        used_segments = self._load_used_segments()

        available = [s for s in all_segments if s["segment_id"] not in used_segments]

        print(f"Available: {len(available)} segments")

        if len(available) < num_segments:
            num_segments = len(available)

        # Group by genre
        by_genre = defaultdict(list)
        for segment in available:
            by_genre[segment["genre"]].append(segment)

        # Try to sample equally from each genre
        samples = []
        genres = list(by_genre.keys())
        per_genre = num_segments // len(genres)

        for genre in genres:
            genre_segments = by_genre[genre]

            # Limit per book
            by_book = defaultdict(list)
            for segment in genre_segments:
                by_book[segment["book_id"]].append(segment)

            genre_samples = []
            for book_segments in by_book.values():
                book_sample_size = min(max_per_book, len(book_segments))
                genre_samples.extend(self._rng.sample(book_segments, book_sample_size))

            # Sample what we can for this genre
            if len(genre_samples) >= per_genre:
                samples.extend(self._rng.sample(genre_samples, per_genre))
            else:
                samples.extend(genre_samples)
                print(
                    f"  {genre}: only {len(genre_samples)} available (wanted {per_genre})"
                )

        # Fill remaining from any available segments
        if len(samples) < num_segments:
            remaining_needed = num_segments - len(samples)
            used_ids = {s["segment_id"] for s in samples}
            remaining_available = [
                s for s in available if s["segment_id"] not in used_ids
            ]

            if remaining_available:
                additional = self._rng.sample(
                    remaining_available, min(remaining_needed, len(remaining_available))
                )
                samples.extend(additional)

        self._rng.shuffle(samples)

        if output_name:
            output_file = self.samples_dir / f"{output_name}.json"
            with open(output_file, "w") as f:
                json.dump(samples, f, indent=2)
            print(f"\nSaved {len(samples)} segments to {output_file}")

        return samples

    def get_status(self):
        all_segments = self._load_all_segments_with_book_meta()
        used_segments = self._load_used_segments()
        books = set()
        for segment in all_segments:
            books.add(segment["book_slug"])

        by_genre = defaultdict(int)
        for segment in all_segments:
            by_genre[segment["genre"]] += 1
        books_per_genre = defaultdict(int)
        for book in books:
            book_meta = self._load_book_meta(book)
            genre = book_meta.get("genre", "Unknown")
            books_per_genre[genre] += 1

        return {
            "total_books": len(books),
            "total_segments": len(all_segments),
            "used_segments": len(used_segments),
            "available_segments": len(all_segments) - len(used_segments),
            "by_genre": by_genre,
            "books_per_genre": books_per_genre,
        }

    def print_status(self):
        status = self.get_status()

        print("=== CORPUS STATUS ===")
        print(f"Books: {status['total_books']}")
        print(f"Total segments: {status['total_segments']}")
        print(f"Used segments: {status['used_segments']}")
        print(f"Available: {status['available_segments']}")
        print("\nBy genre:")
        for genre, count in sorted(
            status["by_genre"].items(), key=lambda x: x[1], reverse=True
        ):
            print(
                f"  {genre}: {count} ({status['books_per_genre'].get(genre, 0)} books)"
            )
