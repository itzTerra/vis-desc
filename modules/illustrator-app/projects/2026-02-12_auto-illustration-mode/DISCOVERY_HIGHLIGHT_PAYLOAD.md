Findings: highlight / WS payload format

- Canonical type: `Highlight` is defined in `services/frontend/app/types/common.d.ts` with keys:
  - `id: number` - unique integer identifier for the segment/highlight
  - `text: string` - the segment text
  - `polygons: Record<number, number[][]>` - mapping pageIndex (0-based) → polygon (array of `[x,y]` points). Coordinates are normalized (0.0–1.0) relative to the page width/height.
  - `score?: number` - optional numeric score attached when scoring arrives
  - `score_received_at?: number` - optional timestamp

- Where WS payload is parsed:
  - `services/frontend/app/pages/index.vue` — `socket` `onMessage` handler parses incoming JSON and dispatches by `type` (`segment`, `batch`, `info`, `error`, `success`). In `segment`/`batch` cases the message `content` is treated as a `Segment`/`Segment[]` and fed into `scoreSegment()` (see index.vue socket handling).

- Example payload (from `app/assets/data/example-segments.json`):

```json
{
  "id": 0,
  "text": "Title: Alice's Adventures in Wonderland ...",
  "polygons": {
    "1": [
      [0.14912981128040556, 0.5419381267362419],
      [0.14912981128040556, 0.5681110294662025],
      [0.6458080592171517, 0.7170391661217966]
    ]
  },
  "score": 0.11727176748917029
}
```

- Notes:
  - Polygon keys in example JSON are strings of page indices (pages are typically 0-based in the `polygons` map; viewer converts to 1-based pages for display).
  - Coordinates are normalized floats in range [0,1]. The `ImageLayer` and `HighlightLayer` components compute page pixel positions using page dimensions and `pageAspectRatio`.
