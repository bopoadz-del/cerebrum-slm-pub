"""A duplicate document hash must not reach a labelling sheet.

On current main the collector posted two rows of one source document. This test fails
there because keep_for_sheet does not exist and nothing rejects the second hash.
"""

from router.candidate_dedupe import keep_for_sheet


def _sha(char: str) -> str:
    return char * 64


def test_duplicate_file_or_source_sha_never_reaches_the_sheet() -> None:
    first = {"sha256": _sha("a"), "source_sha256": _sha("b")}
    same_source_new_extract = {"sha256": _sha("c"), "source_sha256": _sha("b")}
    already_sealed = {"sha256": _sha("d"), "source_sha256": _sha("e")}
    fresh = {"sha256": _sha("f"), "source_sha256": _sha("1")}
    kept, rejected = keep_for_sheet(
        [first, same_source_new_extract, already_sealed, fresh],
        known={_sha("d")},
    )
    assert kept == [first, fresh]
    assert rejected == [same_source_new_extract, already_sealed]


def test_a_row_with_no_hash_does_not_reach_the_sheet() -> None:
    kept, rejected = keep_for_sheet([{"text": "no hash"}], known=set())
    assert kept == []
    assert len(rejected) == 1
