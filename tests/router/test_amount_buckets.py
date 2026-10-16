"""Magnitude buckets for amounts written in words or as exact digits.

Synthetic strings only. Ingest rows are checked in test_slm_ingest_1015.
"""

from router.amount_buckets import bucket_amounts, magnitude_token


def test_written_amounts_become_magnitude_tokens() -> None:
    cases = (
        ("one hundred", "<AMT:~1e2>"),
        ("fifty thousand", "<AMT:~1e4>"),
        ("one hundred and twenty", "<AMT:~1e2>"),
        ("nine million one hundred ninety eight thousand two hundred ten", "<AMT:~1e6>"),
        ("forty three million, six hundred and fifty two thousand", "<AMT:~1e7>"),
        ("twenty-one", "<AMT:~1e1>"),
        ("eighty nine", "<AMT:~1e1>"),
    )
    for raw, token in cases:
        out, count = bucket_amounts(f"the sum of {raw} dirhams")
        assert out == f"the sum of {token} dirhams", raw
        assert count == 1


def test_fils_fraction_is_part_of_the_written_amount() -> None:
    out, count = bucket_amounts(
        "Dirhams nine million one hundred and fils 56/100 only"
    )
    assert out == "Dirhams <AMT:~1e6> only"
    assert count == 1
    assert "56" not in out


def test_exact_digit_amounts_become_magnitude_tokens() -> None:
    out, count = bucket_amounts("AED 1,234,567.89 then 866.60 and 0.00")
    assert out == "AED <AMT:~1e6> then <AMT:~1e2> and <AMT:~1e0>"
    assert count == 3
    spaced, n = bucket_amounts("price AED 612, 801, 84.00 today")
    assert spaced == "price AED <AMT:~1e7> today"
    assert n == 1


def test_dates_clauses_percentages_and_small_counts_stay() -> None:
    text = (
        "Sub-Clause 14.3 on 05.07.2023 and 30 June 2023. "
        "VAT 5% over 120 days. Application No. 36. one window."
    )
    out, count = bucket_amounts(text)
    assert out == text
    assert count == 0


def test_existing_magnitude_tokens_are_stable() -> None:
    text = "net AED <AMT:~1e6> (<AMT:~1e4>)"
    out, count = bucket_amounts(text)
    assert out == text
    assert count == 0
    again, count2 = bucket_amounts(out)
    assert again == out and count2 == 0


def test_zero_and_fraction_exponents() -> None:
    assert magnitude_token(0) == "<AMT:~1e0>"
    assert magnitude_token(0.56) == "<AMT:~1e-1>"
    assert magnitude_token(100) == "<AMT:~1e2>"


def _exact(raw: str, expected: str, count: int) -> None:
    out, n = bucket_amounts(raw)
    assert out == expected, (raw, out)
    assert n == count, (raw, n, out)


def test_comma_groups_exact() -> None:
    _exact("1,250", "<AMT:~1e3>", 1)
    _exact("12,500", "<AMT:~1e4>", 1)
    _exact("1,234,567.89", "<AMT:~1e6>", 1)
    _exact("see 1,250 today", "see <AMT:~1e3> today", 1)
    _exact("amount 12,500", "amount <AMT:~1e4>", 1)
    _exact("1,250.00", "<AMT:~1e3>", 1)
    _exact("12,500.50", "<AMT:~1e4>", 1)
    # Inconsistent group widths stay whole. They are not split after the first comma.
    _exact("1,23,456", "1,23,456", 0)
    _exact("12,34,567", "12,34,567", 0)


def test_spaced_groups_exact() -> None:
    _exact("8 765 432", "<AMT:~1e6>", 1)
    _exact("1 250", "<AMT:~1e3>", 1)
    _exact("12 500", "<AMT:~1e4>", 1)
    _exact("1 250 000", "<AMT:~1e6>", 1)
    _exact("1 250 000.00", "<AMT:~1e6>", 1)
    _exact("12 500.50", "<AMT:~1e4>", 1)
    _exact("AED 8 765 432", "AED <AMT:~1e6>", 1)
    _exact("8 765 432.10", "<AMT:~1e6>", 1)


def test_nbsp_and_thin_space_groups_exact() -> None:
    _exact("1\u00a0250\u00a0000", "<AMT:~1e6>", 1)
    _exact("1\u2009250\u2009000", "<AMT:~1e6>", 1)
    _exact("1\u202f250\u202f000", "<AMT:~1e6>", 1)
    _exact("1\u00a0250", "<AMT:~1e3>", 1)
    _exact("12\u2009500.50", "<AMT:~1e4>", 1)
    _exact("8\u202f765\u202f432.10", "<AMT:~1e6>", 1)


def test_dot_grouped_amounts_exact() -> None:
    _exact("1.250.000", "<AMT:~1e6>", 1)
    _exact("12.500.000", "<AMT:~1e7>", 1)
    _exact("05.07.2023", "05.07.2023", 0)
    _exact("1.2.3", "1.2.3", 0)
    _exact("14.3", "14.3", 0)


def test_mixed_separators_exact() -> None:
    _exact("1,234 567", "<AMT:~1e6>", 1)
    _exact("1 234,567", "<AMT:~1e6>", 1)
    _exact("1,234 567,890", "<AMT:~1e9>", 1)
    _exact("1,234 567.89", "<AMT:~1e6>", 1)
    _exact("123 456,789", "<AMT:~1e8>", 1)
    _exact("1, 234,567", "<AMT:~1e6>", 1)
    _exact("AED 1,234 567", "AED <AMT:~1e6>", 1)
    _exact("AED 1,234 567.89 75 only", "AED <AMT:~1e6> only", 1)
    _exact("8 765\u00a0432", "<AMT:~1e6>", 1)
    _exact("1,250\u00a0000", "<AMT:~1e6>", 1)
    _exact("1\u00a0234 567", "<AMT:~1e6>", 1)
    # A short group, a slash, a unit, a schedule word, a phone cue, or a leading zero stays.
    stayed = (
        "1,23 456",
        "12,34 567",
        "1,234/567",
        "1,234 567 kg",
        "item 1,234 567",
        "schedule 8 765\u00a0432",
        "tel 123,456 789",
        "01,234 567",
    )
    for raw in stayed:
        _exact(raw, raw, 0)


def test_leading_zero_values_exact() -> None:
    _exact("01,250", "01,250", 0)
    _exact("012,500", "012,500", 0)
    _exact("012,500.00", "012,500.00", 0)
    _exact("00.50", "00.50", 0)
    _exact("04.00", "04.00", 0)
    _exact("00,000", "00,000", 0)
    _exact("0,500", "0,500", 0)
    _exact("0.00", "<AMT:~1e0>", 1)
    _exact("0,000", "<AMT:~1e0>", 1)


def test_unit_quantities_stay_exact() -> None:
    _exact("1,250 kg", "1,250 kg", 0)
    _exact("12,500 m2", "12,500 m2", 0)
    _exact("8 765 432 mm", "8 765 432 mm", 0)
    _exact("10.50 kg", "10.50 kg", 0)
    _exact("1,250 days", "1,250 days", 0)
    _exact("1,250%", "1,250%", 0)
    _exact("5%", "5%", 0)
    _exact("100 mm", "100 mm", 0)
    _exact("3 m³", "3 m³", 0)
    _exact("2.50kg", "2.50kg", 0)


def test_dates_stay_exact() -> None:
    raw = "On 05.07.2023 pay 8 765 432 and on 30 June 2023 the rate is 5%."
    _exact(
        raw,
        "On 05.07.2023 pay <AMT:~1e6> and on 30 June 2023 the rate is 5%.",
        1,
    )
    stayed = (
        "On 05/07/2023 and 2023-07-05 and 12-Apr-17 VAT 5% clause 14.3.",
        "Sub-Clause 14.3 on 05.07.2023 and 30 June 2023.",
        "page 1/2 dated 12/30/1999",
        "on 18-May-22 118 noted",
        "signed 12-Apr-17 500 later",
    )
    for text in stayed:
        _exact(text, text, 0)


def test_phone_numbers_stay_exact() -> None:
    _exact(
        "call +971 50 123 4567 about 8 765 432 today",
        "call +971 50 123 4567 about <AMT:~1e6> today",
        1,
    )
    _exact("tel 04 321 0987 about 12 500", "tel 04 321 0987 about <AMT:~1e4>", 1)
    _exact("(04) 123 4567", "(04) 123 4567", 0)
    _exact("00971 50 123 4567", "00971 50 123 4567", 0)


def test_codes_and_references_exact() -> None:
    _exact(
        "see QX/48/2031/77/6 page 1/2",
        "see <REF> page 1/2",
        1,
    )
    _exact("ticket AB/4821 closed", "ticket <REF> closed", 1)
    _exact("form __/8/120/44/3 attached", "form <REF> attached", 1)
    _exact("on 05/07/2023 ref KLM/12/8704/63", "on 05/07/2023 ref <REF>", 1)
    _exact("lot AB12/CD34.", "lot <REF>.", 1)


def test_schedule_and_boq_row_numbers_stay_exact() -> None:
    _exact("BOQ 2.10", "BOQ 2.10", 0)
    _exact("item 2.10", "item 2.10", 0)
    _exact("schedule 1.01", "schedule 1.01", 0)
    _exact("row 01.02", "row 01.02", 0)
    _exact("bill 4 <AMT:~1e6> net", "bill <ROW> <AMT:~1e6> net", 1)
    _exact("1.2.3 concrete", "1.2.3 concrete", 0)


def test_minor_unit_and_hundredths_tails_exact() -> None:
    _exact("balance <AMT:~1e7> 99 payable", "balance <AMT:~1e7> payable", 1)
    _exact("net <AMT:~1e5> and 99/100 fils due", "net <AMT:~1e5> due", 1)
    _exact("AED 8,765,432 75 only", "AED <AMT:~1e6> only", 1)
    _exact("AED 1,250.00 75 only", "AED <AMT:~1e3> only", 1)
    _exact("USD 3,250.00 and 05/100 cents flat", "USD <AMT:~1e3> flat", 1)
    _exact("fee <AMT:~1e4> and 25/100 halalas net", "fee <AMT:~1e4> net", 1)
    _exact("part <AMT:~1e2> 9/12", "part <AMT:~1e2> 9/12", 0)


def test_minor_column_and_hundredths_together_exact() -> None:
    cases = (
        ("AED 8,765,432 75 and 50/100 only", "AED <AMT:~1e6> only", 1),
        ("USD 3,250.00 05 and 50/100 cents flat", "USD <AMT:~1e3> flat", 1),
        (
            "balance <AMT:~1e7> 99 and 25/100 fils payable",
            "balance <AMT:~1e7> payable",
            1,
        ),
        ("AED 1,234 567 75 and fils 50/100 only", "AED <AMT:~1e6> only", 1),
    )
    for raw, expected, count in cases:
        _exact(raw, expected, count)
        once, n_once = bucket_amounts(raw)
        twice, n_twice = bucket_amounts(once)
        assert twice == once == expected
        assert n_once == count and n_twice == 0


def test_redacting_already_redacted_text_changes_nothing() -> None:
    raw = (
        "pay 1,250 and 8 765 432 and 1\u00a0250\u00a0000 and 1,234 567 "
        "and 01,250 kg on 05/07/2023 ref QX/48/2031/77/6 "
        "bill 4 <AMT:~1e6> 99"
    )
    expected = (
        "pay <AMT:~1e3> and <AMT:~1e6> and <AMT:~1e6> and <AMT:~1e6> "
        "and 01,250 kg on 05/07/2023 ref <REF> "
        "bill <ROW> <AMT:~1e6>"
    )
    _exact(raw, expected, 7)
    twice, second = bucket_amounts(expected)
    assert twice == expected
    assert second == 0
    again, third = bucket_amounts(raw)
    stable, fourth = bucket_amounts(again)
    assert stable == again == expected
    assert third == 7 and fourth == 0
