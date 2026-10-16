# Raw document pool

This file is a raw document pool for later router labelling. It is not training data. Do not train on it.

`labels.jsonl` has one row per stored Fork document excerpt. Row count is unchanged from the export: **6580**.

sha256: `29d75c0a89268bc5fa13d21fc75e30e0525ff0623c38359cfdfce5378bb1f02f`

Path tags removed: the Fork export prepended `[source: <drive path>]` tags to 2343 excerpts, and 16 excerpts quoted a file path in their own text. `scripts/router/strip_source_tags.py` removed them from `first_512_tokens` in 2359 rows. Row count, `doc_id`, `filename` and every other column are unchanged. The digest before the strip was `99263feadf1b9a0b6f5182c0b527e9481a323d7f158124bdff9e682344122cc5`.

The previous digest `2ee2b8204feaec33c6cfce0abad0afcc677b166fd1de41f1a69e46890eade903` was the same 6580 rows before column renames. No documents were added or dropped.

## Schema

- `doc_id`
- `project_id`
- `filename`
- `first_512_tokens` — first 512 whitespace tokens. Renamed from `first_512`.
- `fork_declared_type` — stored Fork document type, renamed from `declared_type`. Reference only. Never a training target.
- `content_sha256` — stored content hash when the source had one, otherwise null
- `doc_id_source` — `content_sha256` (4024 rows) or `rejoined_chunks` (2556 rows)

`ingested_at` was not in the source export, so it is not invented.

The source `label` column matched `declared_type` on every row. It was removed so it cannot be read as a training label.

The source `text` column was the same string as `first_512` on every row. It was removed. The pool has one excerpt column, `first_512_tokens`.

`content_sha256` is null on the 2556 `rejoined_chunks` rows.

## Fork reference counts

These are stored Fork types, not router labels.

| fork_declared_type | rows |
| --- | ---: |
| document | 6428 |
| contract | 39 |
| specification | 38 |
| boq | 28 |
| drawing | 26 |
| schedule | 20 |
| bim | 1 |
| **total** | **6580** |

6428 of 6580 rows (about 98%) have reference type `document`. There are 20 distinct `project_id` values.

## Dedupe by doc_id

The raw file keeps every row. Repeats are an integrity note, not a cleaned training set.

- Distinct `doc_id`: 5683
- `doc_id` values that repeat: 848
- Extra rows beyond the first copy: 897

Among the 848 repeating `doc_id` groups:

- excerpt differs: 126
- fork reference type differs: 3
- project id differs: 837

## Not for training

No content label has been assigned. Do not train on `fork_declared_type`. Do not assign a label from the filename, the document id, or the project id.
