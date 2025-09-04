from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
BOOK_DIR = ROOT_DIR / "data" / "books"
BOOK_META_DIR = BOOK_DIR / "meta"
SEGMENT_DIR = ROOT_DIR / "data" / "segments"
PROCESSED_BOOKS_PATH = ROOT_DIR / "data" / "processed_books.json"
TO_ANNOTATE_DIR = ROOT_DIR / "data" / "to-annotate"
