"""Router run v7 driver (v9, 22 labels). Mechanism: configs/router/train_v7.json via router.train_router.fit_router_v7.

Tree: the staged PR-A tree (main 729a358 + bundle A), read-only at /workspace/router-v7/train_tree_A; its git tree hash must equal
STAGED_TREE, which is what the squash-merge of PR A must produce. Data: data/router/train.jsonl (1719) and val.jsonl (901) exactly as on
729a358, plus the committed v7 pair refs materialised by router.v7_pairs.v7_training_examples (980). Train rows are rebuilt with
train_router.training_examples and must equal the merged rows. Class weights: read from data/router/v7/class_weights.json by
router.balance_v7.read_weights (refused unless it equals the recomputation from these examples' counts). Epoch choice and early stop:
val macro recall over classes present in val (never accuracy); tie keeps the earlier epoch; at most 3 epochs. As in v6, the only driver
change to the fit is that predict_labels is swapped for a parallel version with the same sampling params, _complete and _as_label,
which also writes the per-epoch val predictions. The sealed sets are never read. No pre-run note (owner order). Never prints secret values.
"""
from __future__ import annotations

import argparse, hashlib, json, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import REPO, force_httpx, load_env_file

import router.train_router as tr
from router.balance import SELECTION_METRIC, macro_recall_present
from router.balance_v7 import load_v7_config, read_weights
from router.v7_pairs import v7_training_examples
from router.labels import load_labels, load_sealed_config
from router.teacher_label import prompt_for

import os

BASE_SHA = "729a35859a46172b24a1ef53e2398105ff6d270a"
BASE_TREE = "9e0de89aed3b9933e27294e8a5d754e11610c078"  # tree of 729a358
STAGED_TREE = "73e5d4f68c905fb743572e3824c076ab53932dbd"  # 729a358 + bundle A; must equal the tree of the PR-A squash merge


def git_tree(path: Path) -> str:
    entries = []
    for child in path.iterdir():
        if child.name in (".git", "__pycache__", ".pytest_cache"):
            continue
        if child.is_dir():
            entries.append((child.name + "/", b"40000 " + child.name.encode() + b"\0" + bytes.fromhex(git_tree(child))))
        else:
            data = child.read_bytes()
            mode = b"100755" if os.access(child, os.X_OK) else b"100644"
            h = hashlib.sha1(b"blob %d\0" % len(data) + data).digest()
            entries.append((child.name, mode + b" " + child.name.encode() + b"\0" + h))
    body = b"".join(e for _, e in sorted(entries, key=lambda x: x[0].encode()))
    return hashlib.sha1(b"tree %d\0" % len(body) + body).hexdigest()


def blob(b):
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


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
            if step % 10 == 0 or step == 1:
                metrics = getattr(res, "metrics", None)
                loss = {k: v for k, v in metrics.items() if "loss" in k} if isinstance(metrics, dict) else None
                print(f"step {step}/<= {self._total} elapsed_s={time.time()-self._t0:.0f} {loss}", flush=True)

        return _LoggedFuture(self._c.forward_backward(batch, loss_fn), log)

    def save_weights_for_sampler(self, name, **kw):
        return _LoggedFuture(self._c.save_weights_for_sampler(name=name, **kw),
                             lambda res: print(f"saved {name} path={getattr(res, 'path', None)} step={self._step}", flush=True))


class _Service:
    def __init__(self, real, total_steps, info):
        self._s, self._total, self._info = real, total_steps, info

    def __getattr__(self, name):
        return getattr(self._s, name)

    def create_lora_training_client(self, **kw):
        print("create_lora_training_client " + json.dumps({k: v for k, v in kw.items()}), flush=True)
        client = self._s.create_lora_training_client(**kw)
        for attr in ("model_id", "training_run_id"):
            v = getattr(client, attr, None)
            if v:
                self._info[attr] = str(v)
                print(f"training run {attr}={v}", flush=True)
        return _LoggedClient(client, self._total)


class _Sdk:
    def __init__(self, tinker, total_steps, info):
        self.types, self._tinker, self._total, self._info = tinker.types, tinker, total_steps, info

    def ServiceClient(self):
        return _Service(self._tinker.ServiceClient(), self._total, self._info)


def rebuild(path: Path, labels):
    merged = tr.rows_from_pool(path)
    src = []
    for r in merged:
        msgs = r["messages"]
        user = [m["content"] for m in msgs if m["role"] == "user"]
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        row = {"label": asst[-1], "excerpt": user[0], "project_id": r.get("project_id")}
        for k in ("doc_id", "file_sha256"):
            if isinstance(r.get(k), str):
                row[k] = r[k]
        src.append(row)
    rows = tr.training_examples(src, labels)
    same_prompt = sum(1 for a in merged if a["messages"][0]["content"] == prompt_for(labels))
    same_user = sum(1 for a, b in zip(merged, rows) if a["messages"][1]["content"] == b["messages"][1]["content"])
    same_label = sum(1 for a, b in zip(merged, rows) if a["messages"][-1]["content"] == b["messages"][-1]["content"])
    return rows, {"merged_rows": len(merged), "rebuilt_rows": len(rows), "merged_system_prompt_equals_prompt_for": same_prompt,
                  "user_text_unchanged": same_user, "label_unchanged": same_label}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-file", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()
    root, out = REPO, args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    w = lambda p, rs: p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs), encoding="utf-8")

    tree_hash = git_tree(root)
    if tree_hash != STAGED_TREE:
        raise SystemExit(f"WALL: training tree {tree_hash[:7]} is not the staged PR-A tree {STAGED_TREE[:7]}")
    print(f"training tree {tree_hash} verified (= base {BASE_SHA[:7]} tree {BASE_TREE[:7]} + bundle A)", flush=True)

    sealed_cfg = load_sealed_config(root / "configs/router/sealed_eval.json")  # config only; sealed rows not read
    labels = load_labels(root / "configs/router/labels.json")
    assert len(labels) == 22
    system = prompt_for(labels)
    base_rows, tinfo = rebuild(root / "data/router/train.jsonl", labels)
    examples, refs = v7_training_examples(root)
    if len(examples) != len(base_rows) + len(refs) or any(a["messages"] != b["messages"] for a, b in zip(examples, base_rows)):
        raise SystemExit("WALL: v7 examples do not start with the merged train rows")
    train_rows = examples
    pair_counts = dict(Counter(r["pair_type"] for r in refs))
    val_rows, vinfo = rebuild(root / "data/router/val.jsonl", labels)
    split = json.loads((root / "data/router/split_counts.json").read_text(encoding="utf-8"))["split"]
    print("DATA " + json.dumps({"train": tinfo, "val": vinfo, "pairs": pair_counts, "v7_examples": len(train_rows), "labels": len(labels), "system_sha256": hashlib.sha256(system.encode()).hexdigest()}), flush=True)
    for info, n in ((tinfo, split["train"]), (vinfo, split["val"])):
        if not (info["merged_rows"] == info["rebuilt_rows"] == info["merged_system_prompt_equals_prompt_for"] == info["user_text_unchanged"] == info["label_unchanged"] == n):
            raise SystemExit("WALL: rebuilt rows differ from merged train/val")

    v6 = load_v7_config(root)  # name kept from the v6 driver; this is the v7 config
    lora = v6["lora"]
    assert lora["base_model"] == "Qwen/Qwen3.5-4B" and int(lora["max_sequence_length"]) == 6656 and int(lora["rank"]) == 16
    assert v6["max_epochs"] == 3 and v6["exponent"] == 0.5 and v6["share_cap"] == 3.0 and v6["patience"] == 1
    assert lora["learning_rate"] == 0.000473 and lora["batch_size"] == 16 and lora["seed"] == 0 and lora["alpha"] == 16
    train_counts = Counter(r["messages"][-1]["content"] for r in train_rows)
    weights = read_weights(root, [r["messages"][-1]["content"] for r in train_rows])
    print("RECIPE " + json.dumps({k: lora[k] for k in ("base_model", "rank", "alpha", "epochs", "learning_rate", "batch_size", "seed", "max_sequence_length", "checkpoint_name")}), flush=True)
    print("WEIGHTS " + json.dumps({k: round(weights[k], 4) for k in sorted(weights, key=lambda x: -train_counts[x])}), flush=True)

    names = load_env_file(args.env_file)
    print(f"env loaded names={names}", flush=True)
    force_httpx()
    import tinker

    if args.preflight_only:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(lora["base_model"])
        lens = [len(tr._supervised(tok, r["messages"], enable_thinking=bool(lora["enable_thinking"]))[0]) + 1 for r in train_rows]
        vl = [len(tr._encode(tok, [m for m in r["messages"] if m["role"] != "assistant"], add_generation_prompt=True, enable_thinking=False)) for r in val_rows]
        rep = {"base_commit": BASE_SHA, "training_tree": tree_hash, "pairs": pair_counts, "train_rows": len(train_rows), "val_rows": len(val_rows), "train_max": max(lens),
               "train_over_limit": sum(1 for x in lens if x > lora["max_sequence_length"]), "val_prompt_max": max(vl),
               "val_over_limit": sum(1 for x in vl if x > lora["max_sequence_length"]), "limit": lora["max_sequence_length"]}
        print("PREFLIGHT " + json.dumps(rep), flush=True)
        (out / "preflight.json").write_text(json.dumps(rep, indent=2) + "\n")
        return 0

    marker = out / "RUN_STARTED_train.json"
    if marker.exists():
        raise SystemExit("WALL: v7 training already started; no re-runs")
    if not (out / "preflight.json").exists():
        raise SystemExit("WALL: no preflight")
    pf = json.loads((out / "preflight.json").read_text())
    if pf["train_over_limit"] != 0 or pf["val_over_limit"] != 0 or pf["training_tree"] != tree_hash:
        raise SystemExit("WALL: preflight not clean")
    marker.write_text(json.dumps({"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "base_commit": BASE_SHA, "training_tree": tree_hash, "config": "configs/router/train_v7.json",
                                  "base_model": lora["base_model"], "checkpoint_name": lora["checkpoint_name"], "train_rows": len(train_rows),
                                  "val_rows": len(val_rows), "system_sha256": hashlib.sha256(system.encode()).hexdigest(),
                                  "selection_metric": SELECTION_METRIC, "max_epochs": v6["max_epochs"], "patience": v6["patience"],
                                  "weights_file": "data/router/v7/class_weights.json", "weights_file_sha256": hashlib.sha256((root / "data/router/v7/class_weights.json").read_bytes()).hexdigest(),
                                  "pairs": pair_counts, "pre_run_note": None, "sealed_read": False}) + "\n")

    gold = [r["messages"][-1]["content"] for r in val_rows]
    recorded: dict[str, dict] = {}

    def predict_parallel(lora_, rows, checkpoint, labels_, *, sdk=None, workers=16):
        assert rows is val_rows
        service = tinker.ServiceClient()
        try:
            client = service.create_sampling_client(model_path=checkpoint)
            tok = client.get_tokenizer()
            params = tinker.types.SamplingParams(max_tokens=32, temperature=0, top_p=1, seed=int(lora_["seed"]))

            def one(row):
                messages = [m for m in row["messages"] if m["role"] != "assistant"]
                for attempt in range(5):
                    try:
                        text = tr._complete(client, tok, tinker.types, messages, params)
                        return text, tr._as_label(text, labels_)
                    except Exception:
                        if attempt == 4:
                            raise
                        time.sleep(2 * (attempt + 1))

            with ThreadPoolExecutor(max_workers=workers) as pool:
                res = list(pool.map(one, rows))
        finally:
            try:
                service.close("success").result()
            except Exception:
                pass
        texts, preds = [r[0] for r in res], [r[1] for r in res]
        epoch = len(recorded) + 1
        recorded[checkpoint] = {"epoch": epoch, "preds": preds}
        w(out / f"val_predictions_epoch{epoch}.jsonl", [
            {"doc_id": r.get("doc_id"), "project_id": r.get("project_id"), "gold": g, "pred": p, "raw": t}
            for r, g, p, t in zip(val_rows, gold, preds, texts)])
        acc = sum(a == b for a, b in zip(gold, preds)) / len(gold)
        print(f"EVAL epoch{epoch} {SELECTION_METRIC}={macro_recall_present(gold, preds):.6f} accuracy(report only)={acc:.6f}", flush=True)
        return preds

    tr.predict_labels = predict_parallel
    steps_per_epoch = -(-len(train_rows) // int(lora["batch_size"]))
    info: dict = {}
    print(f"fit start rows={len(train_rows)} steps_per_epoch={steps_per_epoch} max_epochs={v6['max_epochs']}", flush=True)
    result = tr.fit_router_v7(root, train_rows, val_rows, labels, sdk=_Sdk(tinker, steps_per_epoch * v6["max_epochs"], info))
    (out / "checkpoints.json").write_text(json.dumps(result["epoch_samplers"], indent=2) + "\n", encoding="utf-8")
    print("FIT " + json.dumps({k: result[k] for k in ("steps", "epoch_samplers", "selected_epoch", "selected_sampler", "selection_metric", "history")}), flush=True)

    mcf = bool(sealed_cfg["majority_class_fail"])
    hist = {str(h["epoch"]): h[SELECTION_METRIC] for h in result["history"]}
    per_epoch = {}
    for path, rec in sorted(recorded.items(), key=lambda kv: kv[1]["epoch"]):
        preds = rec["preds"]
        scored = tr.recall_report(gold, preds, labels, majority_class_fail=mcf)
        pc = Counter(preds)
        per_epoch[str(rec["epoch"])] = {
            "checkpoint": path, SELECTION_METRIC: hist[str(rec["epoch"])],
            "accuracy_report_only": scored["accuracy"], "per_label_recall": scored["per_label_recall"],
            "per_label_n": {l: gold.count(l) for l in labels}, "majority_gate": {"passed": scored["majority_gate_passed"], "reason": scored["majority_gate_reason"]},
            "pred_counts": dict(pc.most_common()), "labels_never_predicted": [l for l in labels if pc.get(l, 0) == 0 and gold.count(l) > 0],
            "unsure_or_offlist": sum(v for k, v in pc.items() if k not in labels)}
    choice = {"run": "v7", "run_id": info.get("model_id"), "rule": "router.balance.select_epoch on " + SELECTION_METRIC + " (tie keeps the earlier epoch), before any sealed score; never accuracy",
              SELECTION_METRIC: hist, "accuracy_report_only": {k: v["accuracy_report_only"] for k, v in per_epoch.items()},
              "epochs_run": len(result["history"]), "early_stopped": len(result["history"]) < v6["max_epochs"],
              "chosen_epoch": result["selected_epoch"], "chosen_checkpoint": result["selected_sampler"], "sealed_read": False, "sealed_scored": False,
              "written": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (out / "epoch_choice.json").write_text(json.dumps(choice, indent=2) + "\n", encoding="utf-8")
    metrics = {"run": "v7", "run_id": info.get("model_id"), "base_commit": BASE_SHA, "training_tree": tree_hash, "label_set": "v9 (22 labels)", "config": "configs/router/train_v7.json",
               "recipe": {k: lora[k] for k in ("base_model", "rank", "alpha", "epochs", "learning_rate", "batch_size", "seed", "max_sequence_length", "checkpoint_name", "enable_thinking", "loss_fn")},
               "class_weighting": {"method": "tempered_inverse_frequency_loss_weight", "exponent": v6["exponent"], "share_cap": v6["share_cap"],
                                   "weights_file": "data/router/v7/class_weights.json", "weights": weights},
               "pairs": pair_counts, "base_train_rows": len(base_rows),
               "train_rows": len(train_rows), "val_rows": len(val_rows), "train_label_counts": dict(train_counts.most_common()),
               "val_label_counts": {l: gold.count(l) for l in labels}, "steps": result["steps"], "epoch_samplers": result["epoch_samplers"],
               "selection_metric": SELECTION_METRIC, "selected_epoch": result["selected_epoch"], "selected_checkpoint": result["selected_sampler"],
               "data_check": {"train": tinfo, "val": vinfo}, "sealed_read": False, "sealed_scored": False, "per_epoch": per_epoch}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print("CHOICE " + json.dumps(choice), flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
