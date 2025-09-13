from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence
import json

"""Dataclasses & loaders for Label Studio JSON export files.

Label Studio (task export) produces a JSON array of Task objects. Example element:

{
  "id": 10,
  "annotations": [
    {
      "id": 12,
      "completed_by": 1,
      "result": [
        {"id": "BSUQwmhPb1", "type": "choices", "value": {"choices": ["0"]}, "origin": "manual", "to_name": "text", "from_name": "rating"},
        {"id": "9Zkmdi01OC", "type": "choices", "value": {"choices": ["Other"]}, "origin": "manual", "to_name": "text", "from_name": "type"},
        {"id": "R78cQXkJwM", "type": "choices", "value": {"choices": ["No"]}, "origin": "manual", "to_name": "text", "from_name": "illsuitable"}
      ],
      "was_cancelled": false,
      "ground_truth": false,
      "created_at": "2025-09-09T07:17:33.241355Z",
      "updated_at": "2025-09-09T07:17:33.241370Z",
      "draft_created_at": "2025-09-09T07:16:37.026663Z",
      "lead_time": 43.107,
      "prediction": {},
      "result_count": 3,
      "unique_id": "91ed19be-a230-4443-a0d8-077aca41c866",
      "bulk_created": false,
      "task": 10,
      "project": 1,
      "updated_by": 1
    }
  ],
  "file_upload": "...",
  "data": {"text": "...", "genre": "Fantasy", "length": 108, "book_id": 546, "segment_id": "546_81"},
  "created_at": "2025-09-08T06:57:48.712518Z",
  "updated_at": "2025-09-09T07:17:33.457843Z",
  "inner_id": 10,
  "total_annotations": 1,
  ...
}

We only model the common stable fields explicitly; all unknown/extra keys are preserved
in an ``extras`` dict for forward compatibility.
"""


ISO_FORMATS = (
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%dT%H:%M:%SZ",  # fallback without microseconds
)


def _parse_dt(val: Any) -> datetime | None:
    if not val or not isinstance(val, str):
        return None
    for fmt in ISO_FORMATS:
        try:
            return datetime.strptime(val, fmt)
        except ValueError:
            continue
    # Try fromisoformat after removing trailing Z
    try:
        return datetime.fromisoformat(val.rstrip("Z"))
    except Exception:
        return None


JSONDict = dict[str, Any]


@dataclass(slots=True)
class ChoiceValue:
    """Value payload for a 'choices' result item."""

    choices: list[str]
    extras: JSONDict = field(default_factory=dict)

    @classmethod
    def from_obj(cls, obj: Any) -> "ChoiceValue | None":
        if not isinstance(obj, dict):
            return None
        if "choices" not in obj or not isinstance(obj["choices"], list):
            return None
        extras = {k: v for k, v in obj.items() if k != "choices"}
        return cls(choices=[str(c) for c in obj["choices"]], extras=extras)


@dataclass(slots=True)
class ResultItem:
    id: str
    type: str
    origin: str | None
    to_name: str | None
    from_name: str | None
    value: ChoiceValue | JSONDict | None
    raw: JSONDict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: JSONDict) -> "ResultItem":
        value_obj = data.get("value")
        value: ChoiceValue | JSONDict | None = ChoiceValue.from_obj(value_obj) or (
            value_obj if isinstance(value_obj, dict) else None
        )
        return cls(
            id=str(data.get("id")),
            type=str(data.get("type")),
            origin=data.get("origin"),
            to_name=data.get("to_name"),
            from_name=data.get("from_name"),
            value=value,
            raw=data,
        )


@dataclass(slots=True)
class Annotation:
    id: int
    completed_by: int | None
    result: list[ResultItem]
    was_cancelled: bool | None
    ground_truth: bool | None
    created_at: datetime | None
    updated_at: datetime | None
    draft_created_at: datetime | None
    lead_time: float | None
    prediction: JSONDict
    result_count: int | None
    unique_id: str | None
    bulk_created: bool | None
    task: int | None
    project: int | None
    updated_by: int | None
    extras: JSONDict = field(default_factory=dict)
    raw: JSONDict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: JSONDict) -> "Annotation":
        known = {
            "id",
            "completed_by",
            "result",
            "was_cancelled",
            "ground_truth",
            "created_at",
            "updated_at",
            "draft_created_at",
            "lead_time",
            "prediction",
            "result_count",
            "unique_id",
            "bulk_created",
            "task",
            "project",
            "updated_by",
        }
        extras = {k: v for k, v in data.items() if k not in known}
        return cls(
            id=int(data.get("id")),
            completed_by=(
                int(data["completed_by"])
                if data.get("completed_by") is not None
                else None
            ),
            result=[
                ResultItem.from_dict(r)
                for r in data.get("result", [])
                if isinstance(r, dict)
            ],
            was_cancelled=data.get("was_cancelled"),
            ground_truth=data.get("ground_truth"),
            created_at=_parse_dt(data.get("created_at")),
            updated_at=_parse_dt(data.get("updated_at")),
            draft_created_at=_parse_dt(data.get("draft_created_at")),
            lead_time=(
                float(data["lead_time"]) if data.get("lead_time") is not None else None
            ),
            prediction=data.get("prediction") or {},
            result_count=data.get("result_count"),
            unique_id=data.get("unique_id"),
            bulk_created=data.get("bulk_created"),
            task=data.get("task"),
            project=data.get("project"),
            updated_by=data.get("updated_by"),
            extras=extras,
            raw=data,
        )

    def choices_by_from_name(self) -> dict[str, list[str]]:
        """Convenience: Map from_name -> selected choices (choices results only)."""
        out: dict[str, list[str]] = {}
        for r in self.result:
            if isinstance(r.value, ChoiceValue) and r.from_name:
                out.setdefault(r.from_name, []).extend(r.value.choices)
        return out


@dataclass(slots=True)
class TaskData:
    text: str | None = None
    genre: str | None = None
    length: int | None = None
    book_id: int | None = None
    segment_id: str | None = None
    extras: JSONDict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: JSONDict) -> "TaskData":
        if not isinstance(data, dict):
            return cls()
        known = {"text", "genre", "length", "book_id", "segment_id"}
        extras = {k: v for k, v in data.items() if k not in known}
        return cls(
            text=data.get("text"),
            genre=data.get("genre"),
            length=data.get("length"),
            book_id=data.get("book_id"),
            segment_id=data.get("segment_id"),
            extras=extras,
        )


@dataclass(slots=True)
class Task:
    id: int
    annotations: list[Annotation]
    file_upload: str | None
    drafts: list[Any]
    predictions: list[Any]
    data: TaskData
    meta: JSONDict
    created_at: datetime | None
    updated_at: datetime | None
    inner_id: int | None
    total_annotations: int | None
    cancelled_annotations: int | None
    total_predictions: int | None
    comment_count: int | None
    unresolved_comment_count: int | None
    last_comment_updated_at: datetime | None
    project: int | None
    updated_by: int | None
    comment_authors: list[Any]
    extras: JSONDict = field(default_factory=dict)
    raw: JSONDict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: JSONDict) -> "Task":
        known = {
            "id",
            "annotations",
            "file_upload",
            "drafts",
            "predictions",
            "data",
            "meta",
            "created_at",
            "updated_at",
            "inner_id",
            "total_annotations",
            "cancelled_annotations",
            "total_predictions",
            "comment_count",
            "unresolved_comment_count",
            "last_comment_updated_at",
            "project",
            "updated_by",
            "comment_authors",
        }
        extras = {k: v for k, v in data.items() if k not in known}
        return cls(
            id=int(data.get("id")),
            annotations=[
                Annotation.from_dict(a)
                for a in data.get("annotations", [])
                if isinstance(a, dict)
            ],
            file_upload=data.get("file_upload"),
            drafts=data.get("drafts", []),
            predictions=data.get("predictions", []),
            data=TaskData.from_dict(data.get("data", {})),
            meta=data.get("meta") or {},
            created_at=_parse_dt(data.get("created_at")),
            updated_at=_parse_dt(data.get("updated_at")),
            inner_id=data.get("inner_id"),
            total_annotations=data.get("total_annotations"),
            cancelled_annotations=data.get("cancelled_annotations"),
            total_predictions=data.get("total_predictions"),
            comment_count=data.get("comment_count"),
            unresolved_comment_count=data.get("unresolved_comment_count"),
            last_comment_updated_at=_parse_dt(data.get("last_comment_updated_at")),
            project=data.get("project"),
            updated_by=data.get("updated_by"),
            comment_authors=data.get("comment_authors", []),
            extras=extras,
            raw=data,
        )

    def first_annotation(self) -> Annotation | None:
        return self.annotations[0] if self.annotations else None

    def choices(self) -> dict[str, list[str]]:
        ann = self.first_annotation()
        return ann.choices_by_from_name() if ann else {}


def load_tasks_from_file(path: str | Path) -> list[Task]:
    """Load a single export JSON file (array of task objects)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):  # Some exports wrap inside {"tasks": [...]}
        if isinstance(data, dict):
            # Try common keys
            for key in ("tasks", "results", "data"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
        if not isinstance(data, list):
            raise ValueError("Expected a list of tasks at top-level export")
    return [Task.from_dict(obj) for obj in data if isinstance(obj, dict)]


def load_tasks(path_or_paths: Iterable[str | Path]) -> list[Task]:
    """Load tasks from multiple files (iterator of paths)."""
    tasks: list[Task] = []
    for p in path_or_paths:
        tasks.extend(load_tasks_from_file(p))
    return tasks


def load_tasks_auto(path: str | Path) -> list[Task]:
    """Convenience loader:
    - If given a directory, loads every *.json file inside (non-recursive) and concatenates.
    - If given a file, loads just that file.
    """
    p = Path(path)
    if p.is_dir():
        json_files = sorted(f for f in p.iterdir() if f.suffix == ".json")
        return load_tasks(json_files)
    return load_tasks_from_file(p)


def iter_choices(
    tasks: Sequence[Task], label_name: str
) -> Iterable[tuple[Task, list[str]]]:
    """Yield (task, choices_for_label_name). Skips tasks lacking the label."""
    for t in tasks:
        ch = t.choices().get(label_name)
        if ch:
            yield t, ch


__all__ = [
    "ChoiceValue",
    "ResultItem",
    "Annotation",
    "TaskData",
    "Task",
    "load_tasks_from_file",
    "load_tasks",
    "load_tasks_auto",
    "iter_choices",
]
