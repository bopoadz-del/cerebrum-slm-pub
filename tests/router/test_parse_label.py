import json

import router.parse_label as parse_label
from router.labels import load_labels
from router.parse_label import _ALIASES, parse_router_output
from router.teacher_label import eligible_for_train, parse_teacher_response
from router.train_router import _as_label


def test_models_and_baselines_share_one_parser(repo_root) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    samples = [
        "change order",
        "Change Order",
        "CHANGE ORDER",
        "variation order",
        "Variation Order",
        "variation",
        "VARIATION",
        "CO",
        "co",
        "Co",
        "  change   order  ",
        "VO/CO",
        "drawing",
        "payment certificate",
    ]
    for sample in samples:
        parsed = parse_router_output(sample, labels)
        assert _as_label(sample, labels) == parsed
        teacher = parse_teacher_response(
            json.dumps({"label": sample.strip(), "reason": "from text", "confidence": 0.91}),
            labels,
        )
        assert teacher["label"] == parse_router_output(sample.strip(), labels)
        assert teacher["label"] == parsed or sample != sample.strip()
    assert parse_router_output("change order", labels) == "VO/CO"
    assert parse_router_output("drawing", labels) == "drawing"


def test_alias_is_the_canonical_label_and_can_train(repo_root) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    parsed = parse_teacher_response(
        '{"label": "change order", "reason": "owner wording", "confidence": 0.9}',
        labels,
    )
    assert parsed["label"] == "VO/CO"
    assert eligible_for_train(parsed, labels, 0.5) is True
    bare = _as_label("variation order", labels)
    assert bare == "VO/CO"


def test_alias_is_inert_when_the_canonical_label_is_absent() -> None:
    labels = ["alpha-mark", "beta-mark"]
    assert parse_router_output("change order", labels) == "unsure"
    assert parse_router_output("CO", labels) == "unsure"
    parsed = parse_teacher_response('{"label": "variation", "reason": "", "confidence": 0.9}', labels)
    assert parsed["label"] == "unsure"


def test_near_misses_stay_unmatched(repo_root) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    unmatched = [
        "change orders",
        "change-order",
        "change order.",
        "a change order",
        "variation orders",
        "the variation",
        "VOC",
        "C",
        "CO.",
        "vo/co",
        "rfi",
        "Rfi",
        "Payment Certificate",
        "correspondence",
        "",
        "   ",
    ]
    expected = {
        "correspondence": "correspondence",
    }
    for sample in unmatched:
        assert parse_router_output(sample, labels) == expected.get(sample, "unsure")
        teacher = parse_teacher_response(
            json.dumps({"label": sample, "reason": "no", "confidence": 0.99}),
            labels,
        )
        assert teacher["label"] == expected.get(sample, "unsure")


def test_parsing_rules_are_fixed_before_v8_scores() -> None:
    assert "Parsing rules are fixed now, before any v8 scores exist." in (parse_label.__doc__ or "")


def test_every_alias_targets_a_declared_label(repo_root) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    assert set(_ALIASES.values()) <= set(labels)
    for alias, target in _ALIASES.items():
        assert parse_router_output(alias, labels) == target
        assert parse_router_output(alias.upper(), labels) == target
        assert parse_router_output(f"  {alias}  ", labels) == target


def test_approved_synonyms_match_the_whole_string(repo_root) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    expected = {
        "vo": "VO/CO",
        "variation instruction": "VO/CO",
        "pay application": "payment application",
        "pay app": "payment application",
        "application for payment": "payment application",
        "interim application": "payment application",
        "progress claim": "payment application",
        "payment claim": "payment application",
        "ipc": "payment certificate",
        "interim payment certificate": "payment certificate",
        "final payment certificate": "payment certificate",
        "certificate of payment": "payment certificate",
        "spec": "specification",
        "specification": "specification",
        "request for inspection": "inspection request",
        "inspection and test request": "inspection request",
        "itr": "inspection request",
        "wir": "inspection request",
        "mir": "inspection request",
        "work inspection request": "inspection request",
        "material inspection request": "inspection request",
        "register": "register/log",
        "log": "register/log",
        "register/log": "register/log",
        "notice of claim": "claim",
        "eot claim": "claim",
        "request for information": "RFI",
        "non-conformance report": "NCR",
        "non-compliance report": "NCR",
        "bill of quantities": "BOQ",
        "schedule of rates": "BOQ",
        "minutes of meeting": "meeting minutes",
        "VO": "VO/CO",
        "Pay App": "payment application",
        "  IPC  ": "payment certificate",
    }
    for sample, target in expected.items():
        assert parse_router_output(sample, labels) == target


def test_plurals_substrings_and_conditional_names_stay_unmatched(repo_root) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    unmatched = [
        "pay apps",
        "pay applications",
        "payapp",
        "pay-app",
        "applications for payment",
        "interim applications",
        "progress claims",
        "payment claims",
        "ipcs",
        "specs",
        "specifications",
        "registers",
        "logs",
        "a register",
        "the log",
        "notices of claim",
        "eot claims",
        "bills of quantities",
        "schedules of rates",
        "minutes of meetings",
        "non-conformance reports",
        "non-compliance reports",
        "requests for information",
        "request for inspections",
        "work inspection requests",
        "variation instructions",
        "engineer's instruction",
        "architect's instruction",
        "engineers instruction",
        "rfi",
        "wire",
        "mire",
        "notice of inspection",
    ]
    for sample in unmatched:
        assert parse_router_output(sample, labels) == "unsure"
    assert parse_router_output("rfi", ["inspection request"]) == "unsure"
    assert parse_router_output("RFI", ["inspection request"]) == "unsure"
    assert parse_router_output("RFI", labels) == "RFI"
    assert parse_router_output("spec", ["spec"]) == "spec"
    assert parse_router_output("spec", ["alpha-mark"]) == "unsure"


def test_json_model_reply_uses_the_same_alias(repo_root) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    raw = '{"label": "CO", "reason": "short form", "confidence": 0.8}'
    assert _as_label(raw, labels) == "VO/CO"
    assert parse_teacher_response(raw, labels)["label"] == "VO/CO"
    assert parse_router_output("RFI", labels) == "RFI"
    assert _as_label('{"label": "invented-type"}', labels) == "unsure"


def test_v9_change_request_aliases_match_the_whole_string(repo_root) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    expected = {
        "change request": "change request",
        "Change Request": "unsure",
        "change order request": "change request",
        "Variation Request": "change request",
        "potential change order": "change request",
        "PCO": "change request",
        "COR": "change request",
        "request for modification": "change request",
        "RFM": "change request",
        "proposal request": "change request",
        "variation proposal": "change request",
        "change proposal": "change request",
        "construction change directive": "VO/CO",
        "change order": "VO/CO",
        "variation": "VO/CO",
        "change requests": "unsure",
        "variation requests": "unsure",
        "pcos": "unsure",
        "a change request": "unsure",
        "change-request": "unsure",
    }
    for sample, target in expected.items():
        assert parse_router_output(sample, labels) == target, sample
        teacher = parse_teacher_response(
            json.dumps({"label": sample, "reason": "from text", "confidence": 0.9}),
            labels,
        )
        assert teacher["label"] == target, sample
    assert parse_router_output("variation request", labels[:4]) == "unsure"
