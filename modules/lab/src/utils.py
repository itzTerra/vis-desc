from pathlib import Path
import json
import os

ROOT_DIR = Path(__file__).parent.parent
BOOK_DIR = ROOT_DIR / "data" / "books"
BOOK_META_DIR = BOOK_DIR / "meta"
SEGMENT_DIR = ROOT_DIR / "data" / "segments"
PROCESSED_BOOKS_PATH = ROOT_DIR / "data" / "processed_books.json"
TO_ANNOTATE_DIR = ROOT_DIR / "data" / "to-annotate"
DATA_DIR = ROOT_DIR / "data"
IMAG_DATA_DIR = DATA_DIR / "datasets" / "concreteness"


# https://stackoverflow.com/questions/1229068/with-python-can-i-keep-a-persistent-dictionary-and-modify-it
class PersistentDict(dict):
    def __init__(self, filename: str | Path, *args, **kwargs):
        self.filename = filename
        self._load()

    def _load(self):
        if os.path.isfile(self.filename) and os.path.getsize(self.filename) > 0:
            with open(self.filename, "r") as fh:
                super().update(json.load(fh))

    def _dump(self):
        with open(self.filename, "w") as fh:
            json.dump(self, fh)

    def __getitem__(self, key):
        return dict.__getitem__(self, key)

    def __setitem__(self, key, val):
        dict.__setitem__(self, key, val)
        self._dump()

    def __repr__(self):
        dictrepr = dict.__repr__(self)
        return "%s(%s)" % (type(self).__name__, dictrepr)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v
        self._dump()


def get_device_name() -> str:
    """Get the device name (GPU or CPU)."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass

    # Get CPU name from /proc/cpuinfo
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    return line.split(":")[1].strip()
    except Exception:
        pass

    return "CPU"
