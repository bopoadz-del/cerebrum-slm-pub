"""Router run v4 driver (v9, 22 labels). Same recipe as run v3; only data and labels change.

Data: data/router/train.jsonl and val.jsonl from main at 868e5af, exactly as merged (pr5 not merged).
Each merged row is turned back into (label, excerpt) and rebuilt with the repo's
train_router.training_examples (22-label prompt_for from configs/router/label_set_v9.json,
require_no_paths). The sealed sets are never read or scored.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import REPO, force_httpx, load_env_file

import router.train_router as tr
from router.labels import load_labels, load_sealed_config
from router.teacher_label import prompt_for

_orig_encode = tr._encode


def _encode_compat(tokenizer, messages, *, add_generation_prompt, enable_thinking):
    """transformers>=5 BatchEncoding unwrap (same patch as run v2)."""
    try:
        encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_generation_prompt, enable_thinking=enable_thinking)
    except TypeError:
        encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_generation_prompt)
    if hasattr(encoded, "keys") and "input_ids" in encoded:
        ids = encoded["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(t) for t in ids]
    return _orig_encode(tokenizer, messages, add_generation_prompt=add_generation_prompt, enable_thinking=enable_thinking)


tr._encode = _encode_compat


class _LoggedFuture:
    def __init__(self, fut, log):
        self._fut, self._log = fut, log

    def result(self, *a, **k):
        res = self._fut.result(*a, **k)
        self._log(res)
        return res


class _LoggedClient:
    def __init__(self, client, total_steps):
        self._c, self._step, self._total, self._t0 = client, 0, total_steps, time.time()

    def __getattr__(self, name):
        return getattr(self._c, name)

    def forward_backward(self, batch, loss_fn):
        self._step += 1
        step = self._step

        def log(res):
            if step % 10 == 0 or step == 1 or step == self._total:
                metrics = getattr(res, "metrics", None)
                loss = {k: v for k, v in metrics.items() if "loss" in k} if isinstance(metrics, dict) else None
                print(f"step {step}/{self._total} elapsed_s={time.time()-self._t0:.0f} {loss}", flush=True)

        return _LoggedFuture(self._c.forward_backward(batch, loss_fn), log)

    def save_weights_for_sampler(self, name, **kw):
        return _LoggedFuture(self._c.save_weights_for_sampler(name=name, **kw),
                             lambda res: print(f"saved {name} path={getattr(res, 'path', None)}", flush=True))


class _Service:
    def __init__(self, real, total_steps):
        self._s, self._total = real, total_steps

    def __getattr__(self, name):
        return getattr(self._s, name)

    def create_lora_training_client(self, **kw):
        client = self._s.create_lora_training_client(**kw)
        for attr in ("model_id", "training_run_id"):
            v = getattr(client, attr, None)
            if v:
                print(f"training run {attr}={v}", flush=True)
        try:
            print(f"training run console={client.get_console_url()}", flush=True)
        except Exception:
            pass
        return _LoggedClient(client, self._total)


class _Sdk:
    def __init__(self, tinker, total_steps):
        self.types, self._tinker, self._total = tinker.types, tinker, total_steps

    def ServiceClient(self):
        return _Service(self._tinker.ServiceClient(), self._total)


def predict_parallel(tinker, lora, rows, checkpoint, labels, workers=16):
    service = tinker.ServiceClient()
    try:
        client = service.create_sampling_client(model_path=checkpoint)
        tokenizer = client.get_tokenizer()
        params = tinker.types.SamplingParams(max_tokens=32, temperature=0, top_p=1, seed=int(lora["seed"]))

        def one(row):
            messages = [m for m in row["messages"] if m["role"] != "assistant"]
            for attempt in range(5):
                try:
                    text = tr._complete(client, tokenizer, tinker.types, messages, params)
                    return text, tr._as_label(text, labels)  # repo exact-alias parser
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(2 * (attempt + 1))

        with ThreadPoolExecutor(max_workers=workers) as pool:
            out = list(pool.map(one, rows))
        return [o[0] for o in out], [o[1] for o in out]
    finally:
        try:
            service.close("success").result()
        except Exception:
            pass


def rebuild(path: Path, labels: list[str]) -> tuple[list[dict], dict]:
    merged = tr.rows_from_pool(path)
    src = []
    for r in merged:
        msgs = r["messages"]
        user = [m["content"] for m in msgs if m["role"] == "user"]
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        row = {"label": asst[-1], "excerpt": user[0], "project_id": r.get("project_id")}
        if isinstance(r.get("doc_id"), str):
            row["doc_id"] = r["doc_id"]
        if isinstance(r.get("file_sha256"), str):
            row["file_sha256"] = r["file_sha256"]
        src.append(row)
    rows = tr.training_examples(src, labels)  # runs require_no_paths
    same_prompt = sum(1 for a in merged if a["messages"][0]["content"] == prompt_for(labels))
    same_user = sum(1 for a, b in zip(merged, rows) if a["messages"][1]["content"] == b["messages"][1]["content"])
    info = {"merged_rows": len(merged), "rebuilt_rows": len(rows), "merged_system_prompt_equals_prompt_for": same_prompt,
            "user_text_unchanged": same_user}
    return rows, info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-file", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--max-seq", type=int, default=6144)
    ap.add_argument("--checkpoints", type=Path, default=None, help="Skip training; score these epoch samplers.")
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()
    root = REPO
    train_config = tr.load_train_config(root)
    sealed_cfg = load_sealed_config(root / "configs" / "router" / "sealed_eval.json")  # config only; sealed rows not read
    labels = load_labels(root / "configs" / "router" / "labels.json")
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    w = lambda p, rs: p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs), encoding="utf-8")

    train_rows, tinfo = rebuild(root / "data/router/train.jsonl", labels)
    val_rows, vinfo = rebuild(root / "data/router/val.jsonl", labels)
    gold = [r["messages"][-1]["content"] for r in val_rows]
    split_counts = json.loads((root / "data/router/split_counts.json").read_text(encoding="utf-8"))
    print("DATA " + json.dumps({"train": tinfo, "val": vinfo, "labels": len(labels)}), flush=True)
    if tinfo["rebuilt_rows"] != 1482 or vinfo["rebuilt_rows"] != 848:
        raise SystemExit("WALL: rebuilt row count differs from merged train/val")

    lora = json.loads((root / train_config["lora_config"]).read_text(encoding="utf-8"))
    configured_base = lora["base_model"]
    lora["base_model"] = args.base_model
    lora["checkpoint_name"] = "router-v4-" + args.base_model.split("/")[-1].lower() + "-lora-r16"
    configured_max_seq = int(lora["max_sequence_length"])
    if args.max_seq != configured_max_seq:
        print(f"OVERRIDE max_sequence_length {configured_max_seq} -> {args.max_seq}", flush=True)
        lora["max_sequence_length"] = int(args.max_seq)
    print("RECIPE " + json.dumps({k: lora[k] for k in ("base_model", "rank", "alpha", "epochs", "learning_rate", "batch_size", "seed", "max_sequence_length", "checkpoint_name")}), flush=True)

    names = load_env_file(args.env_file)
    print(f"env loaded names={names}", flush=True)
    force_httpx()
    import tinker

    if args.preflight_only:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.base_model)
        lens = []
        for r in train_rows:
            inputs, _, _ = tr._supervised(tok, r["messages"], enable_thinking=bool(lora["enable_thinking"]))
            lens.append(len(inputs) + 1)
        vl = [len(tr._encode(tok, [m for m in r["messages"] if m["role"] != "assistant"], add_generation_prompt=True, enable_thinking=False)) for r in val_rows]
        rep = {"train_max": max(lens), "train_over_limit": sum(1 for x in lens if x > lora["max_sequence_length"]),
               "val_prompt_max": max(vl), "limit": lora["max_sequence_length"]}
        print("PREFLIGHT " + json.dumps(rep), flush=True)
        (out / "preflight.json").write_text(json.dumps(rep, indent=2) + "\n")
        return 0

    if args.checkpoints:
        paths = json.loads(args.checkpoints.read_text(encoding="utf-8"))
        trained = {"epoch_sampler": paths[-1], "epoch_samplers": paths, "steps": None}
    else:
        steps_per_epoch = -(-len(train_rows) // int(lora["batch_size"]))
        total = steps_per_epoch * int(lora["epochs"])
        print(f"fit start rows={len(train_rows)} steps={total}", flush=True)
        trained = tr.fit_router(lora, train_rows, sdk=_Sdk(tinker, total))
        (out / "checkpoints.json").write_text(json.dumps(trained["epoch_samplers"], indent=2) + "\n", encoding="utf-8")
        print("FIT " + json.dumps(trained), flush=True)

    split = split_counts.get("split", {})
    metrics = {
        "run": "v4", "repo_commit": "868e5af", "label_set": "v9 (22 labels)",
        "base_model": lora["base_model"], "configured_base_model": configured_base,
        "max_sequence_length": lora["max_sequence_length"], "rank": lora["rank"], "alpha": lora["alpha"],
        "epochs": lora["epochs"], "learning_rate": lora["learning_rate"], "batch_size": lora["batch_size"], "seed": lora["seed"],
        "teacher_model": train_config["teacher_model"],
        "train_rows": len(train_rows), "val_rows": len(val_rows),
        "checkpoint": trained["epoch_sampler"], "epoch_samplers": trained["epoch_samplers"], "steps": trained.get("steps"),
        "sealed_read": False, "sealed_scored": False,
        "split_version": split.get("split_version"), "drawing_cap_train": split.get("drawing_cap_train"),
        "other_cap_train": split.get("other_cap_train"),
        "train_label_counts": dict(Counter(r["messages"][-1]["content"] for r in train_rows)),
        "val_label_counts": {l: gold.count(l) for l in labels},
        "data_check": {"train": tinfo, "val": vinfo},
        "per_epoch": {},
    }
    from router.majority_gate import majority_label
    mcf = bool(sealed_cfg["majority_class_fail"])
    maj = majority_label(gold, labels)
    maj_scored = tr.recall_report(gold, [maj] * len(gold), labels, majority_class_fail=mcf)
    mv = [v for v in maj_scored["per_label_recall"].values() if v is not None]
    maj_macro = sum(mv) / len(mv) if mv else None
    metrics["majority_baseline"] = {"label": maj, "gate_passed": maj_scored["majority_gate_passed"],
                                    "gate_reason": maj_scored["majority_gate_reason"], "accuracy": maj_scored["accuracy"],
                                    "macro_recall": maj_macro}
    print("MAJ " + json.dumps(metrics["majority_baseline"]), flush=True)
    for epoch, path in enumerate(trained["epoch_samplers"], 1):
        texts, preds = predict_parallel(tinker, lora, val_rows, path, labels)
        scored = tr.recall_report(gold, preds, labels, majority_class_fail=mcf)
        w(out / f"val_predictions_epoch{epoch}.jsonl", [
            {"doc_id": r.get("doc_id"), "project_id": r.get("project_id"), "gold": g, "pred": p, "raw": t}
            for r, g, p, t in zip(val_rows, gold, preds, texts)])
        vals = [v for v in scored["per_label_recall"].values() if v is not None]
        macro = sum(vals) / len(vals) if vals else None
        beats = macro is not None and maj_macro is not None and macro > maj_macro
        pc = Counter(preds)
        metrics["per_epoch"][str(epoch)] = {
            "checkpoint": path,
            "per_label_recall": scored["per_label_recall"],
            "per_label_n": {l: gold.count(l) for l in labels},
            "macro_recall": macro, "macro_over_labels": len(vals),
            "accuracy": scored["accuracy"],
            "beats_majority_macro": beats,
            "gate_overall_pass": bool(scored["majority_gate_passed"] and beats and not maj_scored["majority_gate_passed"]),
            "majority_gate": {"passed": scored["majority_gate_passed"], "reason": scored["majority_gate_reason"], "accuracy": scored["accuracy"]},
            "pred_counts": dict(pc.most_common()),
            "labels_never_predicted": [l for l in labels if pc.get(l, 0) == 0],
            "unsure_or_offlist": sum(v for k, v in pc.items() if k not in labels),
        }
        print(f"EVAL epoch{epoch} " + json.dumps({k: metrics['per_epoch'][str(epoch)][k] for k in ('macro_recall', 'accuracy', 'gate_overall_pass', 'labels_never_predicted')}), flush=True)
        (out / "metrics_partial.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    final = metrics["per_epoch"][str(len(trained["epoch_samplers"]))]
    metrics["val_recall"] = final["per_label_recall"]
    metrics["majority_gate"] = final["majority_gate"]
    metrics["macro_recall"] = final["macro_recall"]
    metrics["gate_overall_pass"] = final["gate_overall_pass"]
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
