"""Zero-shot baseline scorer for the router sealed set (evaluation only, never training).

Backend: Cloudflare Workers AI, models @cf/cloudflare/clef-flash and @cf/cloudflare/clef.

Prompt rule (no hardwiring): one `choice` question whose instructions are
docs/router/LABEL_SET_v9.md verbatim and whose options are the 22 v9 label names.
No per-row hint, project name or example document is ever added. `state` is the
document text. Model ids, endpoint, prices and the budget cap come from
configs/router/baselines.json. Credentials come from the environment
(CF_ACCOUNT_ID / CF_API_TOKEN) and are never printed or written.

Subcommands
  rows    build the scoring rows in a private work dir (runner temp): text from a
          sealed source PDF fetched from its source_url and verified against
          file_sha256, else the in-repo excerpt; page-1 image rendered at 150 dpi
          from the verified PDF. Nothing built here is ever committed or uploaded.
  run     score rows against one model and mode; resumable per row; cost-capped.
  report  three-way tables (Clef Flash / Clef / v6), name-blind flips, scanned subset.

Standard library only (plus poppler-utils for the rows step).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


class BudgetExceeded(RuntimeError):
    pass


def load_config(path: Path | None = None) -> dict:
    return json.loads((path or REPO / "configs/router/baselines.json").read_text(encoding="utf-8"))


def load_labels(repo: Path = REPO) -> list[str]:
    labels = json.loads((repo / "configs/router/labels.json").read_text(encoding="utf-8"))["labels"]
    if len(labels) != 22:
        raise ValueError("expected 22 v9 labels")
    return labels


def model_cfg(cfg: dict, name: str) -> dict:
    for m in cfg["models"]:
        if m["name"] == name:
            return m
    raise KeyError(f"model {name!r} not in config")


def build_question(label_doc: str, labels: list[str]) -> dict:
    """The only prompt: the label-set document verbatim plus the closed label set."""
    return {"type": "choice", "instructions": label_doc, "criteria": {label: None for label in labels}}


def build_request(selector: str, state: str, question: dict, name: str, images: list[bytes] | None = None, max_images: int = 4) -> dict:
    body = {"model": selector, "state": state, "questions": {name: question}}
    if images:
        if len(images) > max_images:
            raise ValueError("too many images")
        body["images"] = ["data:image/png;base64," + base64.b64encode(b).decode("ascii") for b in images]
    return body


def unwrap(response: dict) -> dict:
    """REST responses wrap the model output in {"result": ...}; the binding does not."""
    if isinstance(response, dict) and isinstance(response.get("result"), dict):
        return response["result"]
    return response or {}


def parse_answer(response: dict, name: str, labels: list[str]) -> str | None:
    """The answer must be exactly one of the labels; anything else is None (scored wrong)."""
    try:
        ans = unwrap(response)["answers"][name]
        choice = ans["choice"] if isinstance(ans, dict) else ans
    except (KeyError, TypeError):
        return None
    return choice if isinstance(choice, str) and choice in labels else None


def usage_tokens(response: dict) -> tuple[int | None, int | None]:
    u = unwrap(response).get("usage") or (response or {}).get("usage") or {}
    if not isinstance(u, dict):
        return None, None
    tin = u.get("prompt_tokens", u.get("input_tokens"))
    tout = u.get("completion_tokens", u.get("output_tokens", 0))
    return (int(tin) if isinstance(tin, (int, float)) else None, int(tout) if isinstance(tout, (int, float)) else 0)


def estimate_tokens(body: dict, budget: dict) -> int:
    q = json.dumps(body["questions"], ensure_ascii=False)
    chars = len(q) + len(body["state"] if isinstance(body["state"], str) else json.dumps(body["state"]))
    return int(chars / budget["chars_per_token_estimate"]) + 1 + budget["image_tokens_max"] * len(body.get("images", []))


def cost_usd(tokens_in: int, tokens_out: int, m: dict) -> float:
    return tokens_in * m["usd_per_m_input"] / 1e6 + tokens_out * m["usd_per_m_output"] / 1e6


_TOKEN = re.compile(r"\S+")
_WORDLIKE = re.compile(r"^[(\[\"'’]*(?:[A-Za-z][a-z]+|[A-Z]{2,}|[A-Za-z]|\d+(?:[.,/:-]\d+)*)[)\]\"'’.,;:!?]*$")


def is_scanned(text: str | None, min_chars: int = 200, wordlike_max: float = 0.5, min_tokens: int = 20) -> bool:
    """Generic scanned/OCR-garbage measure: too short, or most tokens are not word-like.

    Uses only the text shape; no ids, names or labels.
    """
    text = text or ""
    if len(text.strip()) < min_chars:
        return True
    tokens = _TOKEN.findall(text)
    if len(tokens) < min_tokens:
        return True
    share = sum(1 for t in tokens if _WORDLIKE.match(t)) / len(tokens)
    return share < wordlike_max


def metrics(gold: list[str], pred: list[str | None], labels: list[str]) -> dict:
    """Per-class recall/precision, macro recall over present classes, accuracy.

    A class that is never predicted has precision 0 (as in the v6 table).
    """
    if len(gold) != len(pred):
        raise ValueError("length mismatch")
    per = {}
    present = []
    for label in labels:
        n = sum(1 for g in gold if g == label)
        npred = sum(1 for p in pred if p == label)
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        recall = tp / n if n else None
        precision = tp / npred if npred else 0.0
        per[label] = {"n": n, "predicted": npred, "tp": tp, "recall": recall, "precision": precision}
        if n:
            present.append(recall)
    correct = sum(1 for g, p in zip(gold, pred) if g == p)
    return {
        "n": len(gold),
        "per_class": per,
        "macro_recall": sum(present) / len(present) if present else None,
        "accuracy": correct / len(gold) if gold else None,
        "correct": correct,
    }


def flips(gold: list[str], before: list, after: list, labels: list[str]) -> dict:
    per = {label: 0 for label in labels}
    total = 0
    for g, b, a in zip(gold, before, after):
        if b != a:
            per[g] += 1
            total += 1
    return {"total": total, "per_gold_class": per}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# ---------------------------------------------------------------- rows (CI, private)

def _fetch(url: str, timeout: int = 90) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (cerebrum-slm sealed eval)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def cmd_rows(a) -> int:
    sys.path.insert(0, str(a.repo / "scripts"))
    from router.content import content_text
    from router.name_blind import blank_fields

    cfg = load_config(a.config)
    sealed = json.loads((a.repo / cfg["sealed_eval"]).read_text(encoding="utf-8"))["v9_sealed"]
    work = a.work
    (work / "img").mkdir(parents=True, exist_ok=True)
    out = []
    for s in sealed["sets"]:
        from private_data import locate, missing_message

        sealed_path = locate(a.repo, s["sealed_labels"])
        if not sealed_path.is_file():
            print(missing_message(s["sealed_labels"]), file=sys.stderr)
            return 2
        for g in read_jsonl(sealed_path):
            text, source, img = None, "none", None
            pdf_ok = False
            if g.get("source_url"):
                try:
                    data = _fetch(g["source_url"])
                    pdf_ok = hashlib.sha256(data).hexdigest() == g["file_sha256"] and data[:5] == b"%PDF-"
                except Exception:
                    pdf_ok = False
            if pdf_ok:
                with tempfile.TemporaryDirectory() as td:
                    pdf = Path(td) / "d.pdf"
                    pdf.write_bytes(data)
                    try:
                        raw = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, timeout=120).stdout.decode("utf-8", "replace")
                        text = content_text({"text": " ".join(raw.split())})
                        source = "fetched_pdf"
                    except Exception:
                        text = None
                    try:
                        prefix = work / "img" / g["sheet_id"]
                        subprocess.run(["pdftoppm", "-f", "1", "-l", "1", "-r", str(cfg["image"]["dpi"]), "-png", "-singlefile", str(pdf), str(prefix)], capture_output=True, timeout=120, check=True)
                        if prefix.with_suffix(".png").is_file():
                            img = str(prefix.with_suffix(".png"))
                    except Exception:
                        img = None
            if not text:
                text = content_text(g)
                source = "repo_excerpt" if text else "none"
            blind, nb = blank_fields(text) if text else ("", 0)
            out.append({"set": s["name"], "sheet_id": g["sheet_id"], "content_sha256": g["file_sha256"], "gold": g["label"],
                        "text": text or "", "blind_text": blind or "", "blanked_fields": nb, "text_source": source, "image": img})
    (work / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    from collections import Counter
    print(f"rows {len(out)} text_source {dict(Counter(r['text_source'] for r in out))} images {sum(1 for r in out if r['image'])}")
    return 0


# ---------------------------------------------------------------- run

def out_name(model: str, mode: str, blind: bool) -> str:
    return f"{model}__{mode}{'__blind' if blind else ''}.jsonl"


def spent(out_dir: Path) -> float:
    return sum(r.get("cost_usd") or 0.0 for f in sorted(out_dir.glob("*.jsonl")) for r in read_jsonl(f))


def _post(url: str, body: dict, token: str, timeout: int) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run(rows: list[dict], out_dir: Path, model: str, mode: str, blind: bool, cfg: dict, labels: list[str], label_doc: str,
        account_id: str, token: str, post=_post, timeout: int = 300, log=print) -> dict:
    m = model_cfg(cfg, model)
    qname = cfg["question"]["name"]
    question = build_question(label_doc, labels)
    url = cfg["endpoint"].format(account_id=account_id, model_id=m["model_id"])
    cap = float(cfg["budget"]["cap_usd"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / out_name(model, mode, blind)
    done = {r["sheet_id"] for r in read_jsonl(out)}
    total = spent(out_dir)
    scored = skipped = 0
    worst = max((r.get("cost_usd") or 0.0 for r in read_jsonl(out)), default=0.0)
    with out.open("a", encoding="utf-8") as fh:
        for row in rows:
            if row["sheet_id"] in done:
                skipped += 1
                continue
            state = row["blind_text"] if blind else row["text"]
            imgs = []
            if mode == "image" and row.get("image") and Path(row["image"]).is_file():
                imgs = [Path(row["image"]).read_bytes()]
            body = build_request(m["selector"], state, question, qname, imgs, cfg["image"]["max_images"])
            if imgs and len(json.dumps(body)) > cfg["image"]["max_body_bytes"]:
                body = build_request(m["selector"], state, question, qname, None)
                imgs = []
            est = estimate_tokens(body, cfg["budget"])
            nxt = max(cost_usd(est, 0, m), worst)
            if total + nxt > cap:
                log(f"ABORT: budget cap ${cap:.2f} would be exceeded (spent ${total:.4f}, next row est ${nxt:.4f})")
                raise BudgetExceeded(f"spent {total:.4f}")
            t0 = time.time()
            resp, err = None, None
            for attempt in range(3):
                try:
                    resp = post(url, body, token, timeout)
                    err = None
                    break
                except urllib.error.HTTPError as exc:
                    err = f"HTTP {exc.code}"
                    if exc.code not in (429, 500, 502, 503, 504):
                        break
                    time.sleep(2 * (attempt + 1))
                except Exception as exc:
                    err = type(exc).__name__
                    time.sleep(2 * (attempt + 1))
            secs = time.time() - t0
            tin, tout = usage_tokens(resp or {})
            estimated = tin is None
            if resp is None:
                tin, tout = 0, 0
            elif estimated:
                tin, tout = est, 0
            c = cost_usd(tin, tout or 0, m)
            total += c
            worst = max(worst, c)
            ans = unwrap(resp or {}).get("answers", {}).get(qname)
            raw = ans.get("choice") if isinstance(ans, dict) else ans
            rec = {
                "sheet_id": row["sheet_id"], "set": row["set"], "gold": row["gold"],
                "pred": parse_answer(resp or {}, qname, labels), "raw_choice": raw if isinstance(raw, str) else None,
                "model": m["model_id"], "mode": mode, "name_blind": blind,
                "image_used": bool(imgs), "text_only_flag": mode == "image" and not imgs,
                "text_source": row.get("text_source"), "scanned": is_scanned(row["text"], **_scan_kw(cfg)),
                "seconds": round(secs, 3), "tokens_in": tin, "tokens_out": tout, "tokens_estimated": estimated,
                "cost_usd": round(c, 8), "running_cost_usd": round(total, 6), "error": err,
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            scored += 1
            log(f"{row['sheet_id']} {rec['pred']} {secs:.1f}s ${total:.4f} {err or ''}")
    return {"scored": scored, "skipped": skipped, "spent_usd": total, "out": str(out)}


def _scan_kw(cfg: dict) -> dict:
    s = cfg["scanned_rule"]
    return {"min_chars": s["min_chars"], "wordlike_max": s["garbage_wordlike_share_max"], "min_tokens": s["garbage_min_tokens"]}


def cmd_run(a) -> int:
    account_id = os.environ.get(load_config(a.config)["auth_env"]["account_id"], "")
    token = os.environ.get(load_config(a.config)["auth_env"]["api_token"], "")
    if not account_id or not token:
        print("zero-shot run parked: CF_ACCOUNT_ID / CF_API_TOKEN not set")
        return 0
    cfg = load_config(a.config)
    labels = load_labels(a.repo)
    doc = (a.repo / cfg["label_set_doc"]).read_text(encoding="utf-8")
    rows = read_jsonl(a.rows)
    models = [m["name"] for m in cfg["models"]] if a.model == "all" else [a.model]
    modes = ["text", "image"] if a.mode == "all" else [a.mode]
    blinds = [False, True] if a.blind == "both" else [a.blind == "yes"]
    try:
        for model in models:
            for mode in modes:
                for blind in blinds:
                    print(json.dumps(run(rows, a.out_dir, model, mode, blind, cfg, labels, doc, account_id, token, timeout=a.timeout)))
    except BudgetExceeded:
        return 3
    return 0


# ---------------------------------------------------------------- report

def _fmt(x):
    return "n/a" if x is None else f"{x:.3f}"


def report(out_dir: Path, cfg: dict, labels: list[str], repo: Path = REPO) -> dict:
    v6 = {r["sheet_id"]: r for p in cfg["v6_predictions"] for r in read_jsonl(repo / p)}
    v6b = {r["sheet_id"]: r for p in cfg["v6_name_blind_predictions"] for r in read_jsonl(repo / p)}
    res = {"runs": {}, "v6_rows": len(v6)}
    for f in sorted(out_dir.glob("*.jsonl")):
        recs = read_jsonl(f)
        if not recs:
            continue
        key = f.stem
        gold = [r["gold"] for r in recs]
        pred = [r["pred"] for r in recs]
        ids = [r["sheet_id"] for r in recs]
        both = [i for i, s in enumerate(ids) if s in v6]
        scan = [i for i, r in enumerate(recs) if r.get("scanned")]
        scan6 = [i for i in scan if ids[i] in v6]
        entry = {
            "metrics": metrics(gold, pred, labels),
            "on_v6_rows": metrics([gold[i] for i in both], [pred[i] for i in both], labels),
            "v6_on_same_rows": metrics([gold[i] for i in both], [v6[ids[i]]["pred"] for i in both], labels),
            "scanned_subset": {"n": len(scan), "model": metrics([gold[i] for i in scan], [pred[i] for i in scan], labels),
                               "model_on_v6_rows": metrics([gold[i] for i in scan6], [pred[i] for i in scan6], labels),
                               "v6": metrics([gold[i] for i in scan6], [v6[ids[i]]["pred"] for i in scan6], labels)},
            "cost_usd": sum(r.get("cost_usd") or 0 for r in recs),
            "seconds_per_row": sum(r.get("seconds") or 0 for r in recs) / len(recs),
            "errors": sum(1 for r in recs if r.get("error")),
            "off_set_answers": sum(1 for r in recs if r.get("pred") is None and r.get("raw_choice")),
            "text_only_flagged": sum(1 for r in recs if r.get("text_only_flag")),
            "image_used": sum(1 for r in recs if r.get("image_used")),
            "text_source": {k: sum(1 for r in recs if r.get("text_source") == k) for k in {r.get("text_source") for r in recs}},
        }
        res["runs"][key] = entry
    for key in list(res["runs"]):
        if key.endswith("__blind"):
            continue
        b = out_dir / f"{key}__blind.jsonl"
        if b.exists():
            base = {r["sheet_id"]: r for r in read_jsonl(out_dir / f"{key}.jsonl")}
            bl = {r["sheet_id"]: r for r in read_jsonl(b)}
            common = [s for s in base if s in bl]
            res["runs"][key]["name_blind_flips"] = flips([base[s]["gold"] for s in common], [base[s]["pred"] for s in common], [bl[s]["pred"] for s in common], labels)
    common6 = [s for s in v6 if s in v6b]
    res["v6_name_blind_flips"] = flips([v6[s]["gold"] for s in common6], [v6[s]["pred"] for s in common6], [v6b[s]["pred"] for s in common6], labels)
    return res


def markdown(res: dict, labels: list[str]) -> str:
    keys = [k for k in res["runs"] if not k.endswith("__blind")]
    lines = ["# Zero-shot baselines vs v6 (sealed v9)", "", "Recall / precision per class. v6 columns are on the rows v6 scored.", ""]
    v6m = next((res["runs"][k]["v6_on_same_rows"] for k in keys), None)
    head = "| label | n | " + " | ".join(f"{k} R/P" for k in keys) + " | v6 R/P |"
    lines += [head, "|" + "---|" * (len(keys) + 3)]
    for lab in labels:
        cells = []
        for k in keys:
            pc = res["runs"][k]["metrics"]["per_class"][lab]
            cells.append(f"{_fmt(pc['recall'])}/{_fmt(pc['precision'])}")
        n = res["runs"][keys[0]]["metrics"]["per_class"][lab]["n"] if keys else 0
        v = v6m["per_class"][lab] if v6m else None
        lines.append(f"| {lab} | {n} | " + " | ".join(cells) + f" | {_fmt(v['recall']) if v else 'n/a'}/{_fmt(v['precision']) if v else 'n/a'} |")
    lines += ["", "| run | n | macro | acc | macro on v6 rows | v6 macro same rows | blind flips | cost $ | s/row |", "|---|---|---|---|---|---|---|---|---|"]
    for k in res["runs"]:
        e = res["runs"][k]
        lines.append(f"| {k} | {e['metrics']['n']} | {_fmt(e['metrics']['macro_recall'])} | {_fmt(e['metrics']['accuracy'])} | {_fmt(e['on_v6_rows']['macro_recall'])} | {_fmt(e['v6_on_same_rows']['macro_recall'])} | {e.get('name_blind_flips', {}).get('total', '-')} | {e['cost_usd']:.4f} | {e['seconds_per_row']:.2f} |")
    lines += ["", f"v6 name-blind flips: {res['v6_name_blind_flips']['total']}", "", "## Scanned subset", "", "| run | n | model macro/acc | model on v6 rows macro | v6 macro |", "|---|---|---|---|---|"]
    for k in res["runs"]:
        s = res["runs"][k]["scanned_subset"]
        lines.append(f"| {k} | {s['n']} | {_fmt(s['model']['macro_recall'])}/{_fmt(s['model']['accuracy'])} | {_fmt(s['model_on_v6_rows']['macro_recall'])} | {_fmt(s['v6']['macro_recall'])} |")
    return "\n".join(lines) + "\n"


def cmd_report(a) -> int:
    cfg = load_config(a.config)
    labels = load_labels(a.repo)
    res = report(a.out_dir, cfg, labels, a.repo)
    (a.out_dir / "tables.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (a.out_dir / "TABLES.md").write_text(markdown(res, labels), encoding="utf-8")
    print(markdown(res, labels))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=REPO)
    ap.add_argument("--config", type=Path, default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("rows")
    b.add_argument("--work", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--rows", type=Path, required=True)
    r.add_argument("--out-dir", type=Path, required=True)
    r.add_argument("--model", default="all")
    r.add_argument("--mode", choices=["text", "image", "all"], default="all")
    r.add_argument("--blind", choices=["no", "yes", "both"], default="both")
    r.add_argument("--timeout", type=int, default=300)
    t = sub.add_parser("report")
    t.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args(argv)
    return {"rows": cmd_rows, "run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
