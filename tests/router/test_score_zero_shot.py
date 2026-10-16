import json
from pathlib import Path

import pytest

from router import score_zero_shot as z

REPO = Path(__file__).resolve().parents[2]
CFG = z.load_config()
LABELS = z.load_labels(REPO)
DOC = "label set document"


def _rows(n=3):
    return [{"sheet_id": f"T{i}", "set": "t", "gold": LABELS[i % 2], "text": "word " * 60, "blind_text": "word " * 60} for i in range(n)]


def _resp(choice, tokens=1000):
    return {"success": True, "result": {"answers": {"document_type": {"choice": choice}}, "usage": {"prompt_tokens": tokens}}}


def test_question_is_label_doc_verbatim_and_closed_set():
    doc = (REPO / "docs/router/LABEL_SET_v9.md").read_text(encoding="utf-8")
    q = z.build_question(doc, LABELS)
    assert q["type"] == "choice" and q["instructions"] == doc
    assert list(q["criteria"]) == LABELS and all(v is None for v in q["criteria"].values())


def test_request_shape_and_data_url_images():
    q = z.build_question(DOC, LABELS)
    body = z.build_request("clef-flash", "doc text", q, "document_type", [b"\x89PNG"])
    assert set(body) == {"model", "state", "questions", "images"}
    assert body["images"][0].startswith("data:image/png;base64,")
    assert set(z.build_request("clef", "t", q, "n")) == {"model", "state", "questions"}
    with pytest.raises(ValueError):
        z.build_request("clef", "t", q, "n", [b"x"] * 5)


def test_one_label_returned_scored(tmp_path):
    calls = []

    def post(url, body, token, timeout):
        calls.append((url, body))
        return _resp(LABELS[0])

    s = z.run(_rows(2), tmp_path, "clef-flash", "text", False, CFG, LABELS, DOC, "acct", "SECRET-T0KEN-xyz", post=post, log=lambda *_: None)
    recs = z.read_jsonl(Path(s["out"]))
    assert [r["pred"] for r in recs] == [LABELS[0], LABELS[0]]
    assert calls[0][0].endswith("/accounts/acct/ai/run/@cf/cloudflare/clef-flash")
    assert calls[0][1]["model"] == "clef-flash" and "SECRET-T0KEN-xyz" not in json.dumps(recs)
    assert abs(recs[0]["cost_usd"] - 1000 * 0.038 / 1e6) < 1e-12 and not recs[0]["tokens_estimated"]


def test_off_set_label_scored_wrong(tmp_path):
    post = lambda *a: _resp("Request for information")
    s = z.run(_rows(1), tmp_path, "clef", "text", False, CFG, LABELS, DOC, "a", "t", post=post, log=lambda *_: None)
    rec = z.read_jsonl(Path(s["out"]))[0]
    assert rec["pred"] is None and rec["raw_choice"] == "Request for information"
    assert z.metrics([rec["gold"]], [rec["pred"]], LABELS)["accuracy"] == 0.0


def test_budget_abort(tmp_path):
    cfg = json.loads(json.dumps(CFG))
    cfg["budget"]["cap_usd"] = 1e-9
    post = lambda *a: _resp(LABELS[0])
    with pytest.raises(z.BudgetExceeded):
        z.run(_rows(3), tmp_path, "clef", "text", False, cfg, LABELS, DOC, "a", "t", post=post, log=lambda *_: None)
    assert z.read_jsonl(tmp_path / z.out_name("clef", "text", False)) == []


def test_budget_abort_after_spend(tmp_path):
    cfg = json.loads(json.dumps(CFG))
    cfg["budget"]["cap_usd"] = 0.3
    post = lambda *a: _resp(LABELS[0], tokens=1_000_000)
    with pytest.raises(z.BudgetExceeded):
        z.run(_rows(3), tmp_path, "clef", "text", False, cfg, LABELS, DOC, "a", "t", post=post, log=lambda *_: None)
    assert len(z.read_jsonl(tmp_path / z.out_name("clef", "text", False))) == 1


def test_resume_skips_done_rows(tmp_path):
    seen = []
    post = lambda url, body, tok, to: (seen.append(body["state"]), _resp(LABELS[1]))[1]
    rows = _rows(3)
    z.run(rows[:2], tmp_path, "clef-flash", "text", True, CFG, LABELS, DOC, "a", "t", post=post, log=lambda *_: None)
    s = z.run(rows, tmp_path, "clef-flash", "text", True, CFG, LABELS, DOC, "a", "t", post=post, log=lambda *_: None)
    assert s["skipped"] == 2 and s["scored"] == 1 and len(seen) == 3
    assert [r["sheet_id"] for r in z.read_jsonl(Path(s["out"]))] == ["T0", "T1", "T2"]


def test_image_mode_without_image_is_flagged(tmp_path):
    post = lambda url, body, tok, to: (_resp(LABELS[0]) if "images" not in body else None)
    s = z.run(_rows(1), tmp_path, "clef-flash", "image", False, CFG, LABELS, DOC, "a", "t", post=post, log=lambda *_: None)
    rec = z.read_jsonl(Path(s["out"]))[0]
    assert rec["text_only_flag"] and not rec["image_used"]


def test_missing_secrets_parks_cleanly(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("CF_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("CF_API_TOKEN", raising=False)
    assert z.main(["run", "--rows", str(tmp_path / "none.jsonl"), "--out-dir", str(tmp_path)]) == 0
    assert capsys.readouterr().out.strip() == "zero-shot run parked: CF_ACCOUNT_ID / CF_API_TOKEN not set"


def test_parse_answer_rejects_off_set():
    labels = ["RFI", "other"]
    assert z.parse_answer({"answers": {"q": {"choice": "RFI"}}}, "q", labels) == "RFI"
    assert z.parse_answer({"result": {"answers": {"q": {"choice": "rfi"}}}}, "q", labels) is None
    assert z.parse_answer({}, "q", labels) is None


def test_metrics_never_predicted_precision_zero():
    m = z.metrics(["a", "a", "b"], ["a", None, "a"], ["a", "b", "c"])
    assert m["per_class"]["b"]["precision"] == 0.0 and m["per_class"]["b"]["recall"] == 0.0
    assert m["per_class"]["c"]["recall"] is None
    assert abs(m["macro_recall"] - 0.25) < 1e-9 and abs(m["accuracy"] - 1 / 3) < 1e-9


def test_scanned_rule_is_generic():
    assert z.is_scanned("short text")
    assert not z.is_scanned("The contractor shall submit the method statement for review before work starts. " * 5)
    assert z.is_scanned(" ".join(["~#!", "|_l", "%^&", "Ww0r", "¦¦"] * 40))


def test_flips_per_gold_class():
    assert z.flips(["a", "b"], ["a", "b"], ["b", "b"], ["a", "b"]) == {"total": 1, "per_gold_class": {"a": 1, "b": 0}}


def test_report_three_way(tmp_path):
    post = lambda *a: _resp(LABELS[0])
    z.run(_rows(2), tmp_path, "clef-flash", "text", False, CFG, LABELS, DOC, "a", "t", post=post, log=lambda *_: None)
    z.run(_rows(2), tmp_path, "clef-flash", "text", True, CFG, LABELS, DOC, "a", "t", post=post, log=lambda *_: None)
    res = z.report(tmp_path, CFG, LABELS, REPO)
    assert res["runs"]["clef-flash__text"]["name_blind_flips"]["total"] == 0
    assert "| label |" in z.markdown(res, LABELS)


def test_config_ids_prices_cap_and_no_rows():
    assert {m["model_id"] for m in CFG["models"]} == {"@cf/cloudflare/clef", "@cf/cloudflare/clef-flash"}
    assert CFG["budget"]["cap_usd"] == 10.0 and all(m["usd_per_m_input"] > 0 for m in CFG["models"])
    text = json.dumps(CFG)
    assert "V8-" not in text and "S5-" not in text and "V9" not in text
