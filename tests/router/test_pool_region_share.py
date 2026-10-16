from router.pool_region_share import classify


def test_aia_g702_with_us_zip_is_us_practice() -> None:
    text = "AIA Document G702 Application and Certificate for Payment, Austin, TX 78701."
    cls, score_a, score_b, _families_a, _families_b = classify(text)
    assert cls == "a"
    assert score_a >= 3
    assert score_a >= score_b + 2


def test_fidic_subclause_ipc_in_aed_is_gulf_or_fidic_practice() -> None:
    text = "FIDIC Sub-Clause 14.6 IPC certified in the amount of AED 45000."
    cls, score_a, score_b, _families_a, _families_b = classify(text)
    assert cls == "b"
    assert score_b >= 3
    assert score_b >= score_a + 2


def test_all_dimensions_in_mm_alone_is_indeterminate() -> None:
    cls, score_a, score_b, families_a, families_b = classify("ALL DIMENSIONS IN MM")
    assert cls == "c"
    assert score_a == 0
    assert score_b == 0
    assert families_a == []
    assert families_b == []


def test_source_path_tag_alone_is_stripped_and_indeterminate() -> None:
    cls, score_a, score_b, families_a, families_b = classify("[source: X:/Dubai/f.pdf]")
    assert cls == "c"
    assert score_a == 0
    assert score_b == 0
    assert families_a == []
    assert families_b == []
