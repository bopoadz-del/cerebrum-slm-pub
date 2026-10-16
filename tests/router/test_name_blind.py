import inspect
import json
from pathlib import Path

import router.name_blind as nb
from router.name_blind import MASK, blank_fields, flips, name_blind_gate
from router.sealed_gate import MIN_SEALED_PER_LABEL


def test_party_values_are_blanked_by_position_and_labels_stay() -> None:
    text = "From: Alpha Beta\nTo: Gamma Delta Ltd\nAttn: Epsilon\nSubject: Pump station valve swap"
    out, n = blank_fields(text)
    assert n == 3
    for value in ("Alpha Beta", "Gamma Delta Ltd", "Epsilon"):
        assert value not in out
    assert "From:" in out and "To:" in out and "Attn:" in out
    assert "Subject: Pump station valve swap" in out


def test_flat_text_value_stops_at_the_next_label() -> None:
    text = "Project: Riverside Works Contract No: 44 Contractor: Omega Builders Date: 01/02/2025 Description of change: add 3 valves"
    out, n = blank_fields(text)
    assert "Riverside Works" not in out
    assert "Omega Builders" not in out
    assert "No: 44" in out
    assert "Date: 01/02/2025" in out
    assert "Description of change: add 3 valves" in out
    assert out.count(MASK) == n == 2


def test_qualified_and_signature_fields() -> None:
    text = "Project Name: Zeta Hall\nContractor's Name: Kappa Co\nPrepared by: Lambda\nReviewed By: Mu\nApproved by: Nu\nSignature: Xi\nBy: Omicron\nName: Pi Rho\nEmployer: Sigma\nClient: Tau\nConsultant: Upsilon\nSite: North yard"
    out, n = blank_fields(text)
    assert n == 12
    for value in ("Zeta Hall", "Kappa Co", "Lambda", "Mu", "Nu", "Xi", "Omicron", "Pi Rho", "Sigma", "Tau", "Upsilon", "North yard"):
        assert value not in out


def test_words_without_a_field_colon_are_untouched() -> None:
    text = "The contractor shall submit the project programme to the client from week 3. Site clearance by others."
    assert blank_fields(text) == (text, 0)


def test_blanking_is_idempotent_and_empty_fields_stay_empty() -> None:
    once, n1 = blank_fields("To: Alpha\nFrom:\nProject: ")
    twice, n2 = blank_fields(once)
    assert n1 == 1 and n2 == 0 and twice == once


def test_no_value_lists_in_the_module() -> None:
    source = inspect.getsource(nb)
    for constant in ("PARTY_FIELDS", "SIGNATURE_FIELDS"):
        for word in getattr(nb, constant):
            assert len(word.split()) <= 2
            assert word == word.lower()
    assert "MASK" in source


def test_flip_gate_fails_only_on_passable_classes() -> None:
    labels = ["alpha", "beta"]
    gold = ["alpha"] * 12 + ["beta"] * 3
    before = list(gold)
    after = list(gold)
    after[-1] = "alpha"
    table = flips(gold, before, after, labels)
    assert table["beta"]["flips"] == 1
    assert name_blind_gate(table, MIN_SEALED_PER_LABEL)["passed"] is True
    after[0] = "beta"
    table = flips(gold, before, after, labels)
    result = name_blind_gate(table, MIN_SEALED_PER_LABEL)
    assert result["passed"] is False
    assert result["failing_passable_classes"] == ["alpha"]


def test_a_long_value_is_cut_at_a_word_boundary() -> None:
    text = "Contractor: " + " ".join(["word"] * 40)
    out, n = blank_fields(text)
    assert n == 1
    tail = out.split(MASK, 1)[1]
    assert tail.startswith(" word") or tail.startswith(" ")
    assert "wo " not in out.replace("word", "")


def test_signature_names_are_blanked_whole() -> None:
    out, n = blank_fields("Sign : Date : Name : Alpha Beta Gamma Name : Delta Epsilon Title : Manager")
    assert n == 2
    for value in ("Alpha", "Beta", "Gamma", "Delta", "Epsilon"):
        assert value not in out
    assert "Title : Manager" in out


# Check fix (owner order 2026-10-07 22:20 +04): blank the party value only, never the
# headings, table headers, line items or body text that follow it in flattened text.


def test_flat_title_block_keeps_the_next_all_caps_label() -> None:
    text = "NAME OF CONTRACTOR: Omega Paving Inc. ADDRESS OF CONTRACTOR: 12 Alpha Road, Unit Q, Betatown ZZ1 9QQ NAME OF CERTIFIER: Kappa Lambda"
    out, n = blank_fields(text)
    for value in ("Omega Paving", "12 Alpha Road", "Betatown"):
        assert value not in out
    assert "ADDRESS OF CONTRACTOR:" in out and "NAME OF CERTIFIER:" in out
    assert n == 2


def test_a_number_field_blanks_the_code_only_and_keeps_the_table() -> None:
    text = "Project No: 1234/5 ITEM NO. DESCRIPTION UNIT QTY 1.19 f) Water supplies, electric power Sum 1"
    out, n = blank_fields(text)
    assert n == 1 and "1234/5" not in out
    assert "ITEM NO. DESCRIPTION UNIT QTY 1.19 f) Water supplies, electric power Sum 1" in out


def test_body_text_after_a_salutation_or_heading_is_not_a_party_value() -> None:
    for text in (
        "Dear Contractor: Pursuant to your contract dated you are hereby notified to proceed with the work",
        "INSTRUCTIONS TO CONTRACTOR: Please sign both copies of this notice and return one copy to the owner",
    ):
        assert blank_fields(text) == (text, 0)


def test_an_empty_form_field_blanks_nothing() -> None:
    text = "CONTRACTOR: ____________ _ CONTRACTOR ADDRESS: ____________ Signature: ................"
    assert blank_fields(text) == (text, 0)


def test_a_name_before_body_text_is_still_blanked() -> None:
    out, n = blank_fields("ATTN: Alpha Beta Site instructions are issued only for the quantity shown below")
    assert n == 1 and "Alpha" not in out and "Beta" not in out
    assert "Site instructions are issued only for the quantity shown below" in out
    out, n = blank_fields("FROM: ______________ Kappa Lambda, Project Manager")
    assert n == 1 and "Kappa" not in out


def test_capitals_inside_a_name_do_not_cut_it() -> None:
    for text, values in (
        ("Contractor: OMEGA BUILD CO LTD", ("OMEGA", "BUILD", "LTD")),
        ("Client: Gamma RTA Delta", ("Gamma", "RTA", "Delta")),
        ("From: Delta Epsilon, Zeta QEA, LLC", ("Delta", "Zeta", "QEA")),
    ):
        out, n = blank_fields(text)
        assert n == 1
        for value in values:
            assert value not in out
