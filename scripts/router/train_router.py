"""Router LoRA entry. Dry-run writes nothing and does not contact Tinker.

The human agreement check is done (0.875, kappa 0.864). The owner order that
allowed training before that check stays in configs/router/train.json. The
majority gate is unchanged.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from router.content import is_mostly_encoded
from router.guard import RouterWall
from router.isolation import load_exclusions, require_isolated
from router.labels import load_labels, load_sealed_config, repo_root
from router.prepare_pool import prepare_pool
from router.split_projects import project_grouped_split
from router.strip_source_tags import row_has_path, strip_source_tags
from router.teacher_label import prompt_for
from router.val_metrics import per_label_recall, refuse_sealed_score

WALL = (
    "WALL: TINKER_API_KEY is not set; the owner key file was not reachable "
    "and the environment has no key. Teacher labelling and LoRA training did not run."
)


def load_train_config(root: Path) -> dict:
    path = root / "configs" / "router" / "train.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    status = data.get("human_agreement_status")
    if status not in {"PENDING_OWNER", "done"}:
        raise RouterWall("WALL: human agreement status is not the pending owner check")
    override = data.get("train_before_human_agreement")
    if not isinstance(override, dict) or override.get("enabled") is not True:
        raise RouterWall("WALL: owner override to train before human agreement is absent")
    if status == "done":
        recorded = data.get("human_agreement")
        if not isinstance(recorded, dict):
            raise RouterWall("WALL: recorded human agreement is absent")
        for key in ("overall", "kappa"):
            value = recorded.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RouterWall("WALL: recorded human agreement is absent")
    return data


def rows_from_pool(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def pool_report(root: Path) -> dict:
    pool = root / "data" / "raw" / "router" / "labels.jsonl"
    config = load_sealed_config(root / "configs" / "router" / "sealed_eval.json")
    if not pool.is_file():
        return {"pool": "missing"}
    prepared = prepare_pool(rows_from_pool(pool), load_exclusions(root / config["sealed_labels"]))
    return {
        "input_rows": prepared.input_rows,
        "empty_text": prepared.empty_text,
        "duplicate_doc_ids": prepared.duplicate_doc_ids,
        "sealed_overlaps": prepared.sealed_overlaps,
        "kept": prepared.kept,
    }


def require_no_paths(rows: list[dict]) -> list[dict]:
    """Refuse training rows whose text still holds a source tag or a file path."""
    bad = sum(1 for row in rows if row_has_path(row))
    if bad:
        raise RouterWall(f"WALL: {bad} training rows hold a file path or [source:] tag")
    return rows


def training_examples(rows: list[dict], labels: list[str]) -> list[dict]:
    """Assistant text is the teacher label only. Path fields are not copied.

    Rows whose excerpt is mostly encoded bytes (``router.content.is_mostly_encoded``)
    have no content and are not made into examples, so they reach neither train nor val.

    Source tags and file paths are stripped from the excerpt before it becomes
    the user message, and the result is checked again by ``require_no_paths``.
    """
    allowed = set(labels)
    system = prompt_for(labels)
    examples: list[dict] = []
    for row in rows:
        label = row.get("label")
        excerpt = row.get("excerpt")
        if isinstance(excerpt, str):
            excerpt = strip_source_tags(excerpt)
        if label not in allowed or not isinstance(excerpt, str) or not excerpt.strip():
            continue
        if is_mostly_encoded(excerpt):
            continue  # encoded bytes are not content (router.content encoded-bytes rule)
        item = {
            "project_id": row.get("project_id"),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": excerpt},
                {"role": "assistant", "content": label},
            ],
        }
        if isinstance(row.get("doc_id"), str):
            item["doc_id"] = row["doc_id"]
        if isinstance(row.get("file_sha256"), str):
            item["file_sha256"] = row["file_sha256"]
        examples.append(item)
    return require_no_paths(examples)


def guarded_split(rows: list[dict], sealed_path: Path, val_fraction: float) -> tuple[list[dict], list[dict]]:
    isolated = require_isolated(rows, load_exclusions(sealed_path))
    return project_grouped_split(isolated, val_fraction=val_fraction)


def recall_report(gold: list[str], pred: list[str], labels: list[str], *, majority_class_fail: bool) -> dict:
    from router.val_metrics import majority_gate_on_val

    gate = majority_gate_on_val(
        gold,
        pred,
        order=labels,
        majority_class_fail=majority_class_fail,
    )
    return {
        "per_label_recall": per_label_recall(gold, pred, labels),
        "majority_gate_passed": gate.passed,
        "majority_gate_reason": gate.reason,
        "accuracy": gate.accuracy,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Router LoRA. Does not score the sealed set.")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--labels", type=Path, default=None, help="Teacher JSONL. Omit when labelling has not run.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--score", type=Path, default=None)
    args = parser.parse_args(argv)
    root = (args.root or repo_root()).resolve()
    try:
        if args.score is not None:
            refuse_sealed_score(args.score, root)
        train_config = load_train_config(root)
        sealed = load_sealed_config(root / "configs" / "router" / "sealed_eval.json")
        labels = load_labels(root / "configs" / "router" / "labels.json")
        report = pool_report(root)
        report["human_agreement_status"] = train_config["human_agreement_status"]
        report["train_before_human_agreement"] = True
        report["checkpoint"] = None
        report["val_recall"] = None
        report["majority_gate"] = None
        if args.labels is None:
            print(WALL)
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
            return 2
        refuse_sealed_score(args.labels, root)
        teacher_rows = rows_from_pool(args.labels)
        examples = training_examples(teacher_rows, labels)
        train_rows, val_rows = guarded_split(
            examples,
            root / sealed["sealed_labels"],
            float(train_config["val_fraction"]),
        )
        report["train_rows"] = len(train_rows)
        report["val_rows"] = len(val_rows)
        report["status"] = "dry-run" if args.dry_run or not _key_present() else "ready"
        if not _key_present():
            print(WALL)
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
            return 2
        if args.dry_run:
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
            return 0
        lora = json.loads((root / train_config["lora_config"]).read_text(encoding="utf-8"))
        trained = fit_router(lora, train_rows)
        preds = predict_labels(lora, val_rows, trained["epoch_sampler"], labels)
        gold = [row["messages"][-1]["content"] for row in val_rows]
        scored = recall_report(
            gold,
            preds,
            labels,
            majority_class_fail=bool(sealed["majority_class_fail"]),
        )
        report["checkpoint"] = trained["epoch_sampler"]
        report["val_recall"] = scored["per_label_recall"]
        report["majority_gate"] = {
            "passed": scored["majority_gate_passed"],
            "reason": scored["majority_gate_reason"],
            "accuracy": scored["accuracy"],
        }
        report["status"] = "trained"
        _record_checkpoint(root, trained["epoch_sampler"])
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0
    except RouterWall as exc:
        print(str(exc))
        return 2



def _optimizer_name(lora: dict) -> str:
    """Optimizer family from config. Default adamw keeps every landed recipe unchanged."""
    name = lora.get("optimizer", "adamw")
    if not isinstance(name, str):
        raise RouterWall("WALL: optimizer must be a string")
    key = name.strip().lower()
    if key in {"adamw", "adam"}:
        return "adamw"
    if key == "dimuon":
        return "dimuon"
    raise RouterWall("WALL: optimizer must be adamw or dimuon")


def _optimizer_config(types, lora: dict):
    """Tinker optimizer config object selected at training-client creation."""
    name = _optimizer_name(lora)
    if name == "dimuon":
        return types.DimuonOptimizerConfig()
    return types.AdamOptimizerConfig()


def _optim_params(types, lora: dict):
    """Per-step optimizer hyperparameters matching the selected family."""
    lr = float(lora["learning_rate"])
    beta1 = float(lora["beta1"])
    clip = float(lora["grad_clip_norm"])
    if _optimizer_name(lora) == "dimuon":
        return types.DimuonParams(learning_rate=lr, beta1=beta1, grad_clip_norm=clip)
    return types.AdamParams(
        learning_rate=lr,
        beta1=beta1,
        beta2=float(lora["beta2"]),
        eps=float(lora["eps"]),
        weight_decay=float(lora["weight_decay"]),
        grad_clip_norm=clip,
    )


def fit_router(lora: dict, train_rows: list[dict], *, sdk=None, class_weight: dict[str, float] | None = None, on_epoch=None) -> dict:
    """Fit the Qwen3-4B LoRA. sdk is a test double. The default import is tinker.

    ``class_weight`` (v6) multiplies each row's label-token loss weights by the weight of its
    label; ``on_epoch(epoch, sampler_path)`` returns True to stop early. Both default to off,
    which is the v5 recipe unchanged.
    """
    if not train_rows:
        raise RouterWall("WALL: no training rows")
    if sdk is None:
        import tinker

        sdk = tinker
    service = sdk.ServiceClient()
    status = "errored"
    try:
        modules = lora["target_modules"]
        client = service.create_lora_training_client(
            base_model=lora["base_model"],
            rank=int(lora["rank"]),
            seed=int(lora["seed"]),
            train_mlp=bool(modules["mlp"]),
            train_attn=bool(modules["attn"]),
            train_unembed=bool(modules["unembed"]),
            optimizer=_optimizer_config(sdk.types, lora),
            user_metadata={"checkpoint": str(lora["checkpoint_name"])},
        )
        trained = _fit(client, sdk.types, lora, train_rows, class_weight=class_weight, on_epoch=on_epoch)
        status = "success"
        return trained
    finally:
        closer = getattr(service, "close", None)
        if closer is not None:
            closer(status).result()


def fit_router_v6(root: Path, train_rows: list[dict], val_rows: list[dict], labels: list[str], *, sdk=None) -> dict:
    """v6: class-weighted loss from train label counts, early stop and epoch choice on val macro recall.

    Nothing here names a label. Weights come from ``router.balance.class_weights`` over the
    training rows' labels. The selection number is ``router.balance.macro_recall_present`` on val.
    """
    from router.balance import SELECTION_METRIC, class_weights, load_v6_config, macro_recall_present, select_epoch, should_stop

    v6 = load_v6_config(root)
    lora = v6["lora"]
    weights = class_weights([row["messages"][-1]["content"] for row in train_rows], v6["cap"])
    gold = [row["messages"][-1]["content"] for row in val_rows]
    history: list[dict] = []

    def on_epoch(epoch: int, path: str) -> bool:
        pred = predict_labels(lora, val_rows, path, labels, sdk=sdk)
        history.append({"epoch": epoch, "sampler": path, SELECTION_METRIC: macro_recall_present(gold, pred)})
        return should_stop(history, v6["patience"], v6["max_epochs"])

    trained = fit_router(lora, train_rows, sdk=sdk, class_weight=weights, on_epoch=on_epoch)
    chosen = select_epoch(history)
    return {
        **trained,
        "class_weights": weights,
        "history": history,
        "selected_epoch": chosen,
        "selected_sampler": next(entry["sampler"] for entry in history if entry["epoch"] == chosen),
        "selection_metric": SELECTION_METRIC,
    }


def fit_router_v7(
    root: Path, train_rows: list[dict], val_rows: list[dict], labels: list[str], *, sdk=None, config_file: str | None = None
) -> dict:
    """v7: tempered class weights read from data/router/v7/class_weights.json; otherwise the v6 loop unchanged.

    ``config_file`` selects a v7-family train config (default configs/router/train_v7.json; v7b passes
    configs/router/train_v7b.json, whose weights live in data/router/v7b/class_weights.json).

    ``train_rows`` are the v7 training examples (router.v7_pairs.v7_training_examples). The weights file must equal a
    recomputation from these examples' label counts (router.balance_v7.read_weights), or training is refused.
    Early stop and epoch choice read val macro recall over present classes only. Nothing here names a label.
    """
    from router.balance import SELECTION_METRIC, macro_recall_present, select_epoch, should_stop
    from router.balance_v7 import CONFIG_FILE, load_v7_config, read_weights

    config_file = config_file or CONFIG_FILE
    v7 = load_v7_config(root, config_file)
    lora = v7["lora"]
    weights = read_weights(root, [row["messages"][-1]["content"] for row in train_rows], config_file)
    gold = [row["messages"][-1]["content"] for row in val_rows]
    history: list[dict] = []

    def on_epoch(epoch: int, path: str) -> bool:
        pred = predict_labels(lora, val_rows, path, labels, sdk=sdk)
        history.append({"epoch": epoch, "sampler": path, SELECTION_METRIC: macro_recall_present(gold, pred)})
        return should_stop(history, v7["patience"], v7["max_epochs"])

    trained = fit_router(lora, train_rows, sdk=sdk, class_weight=weights, on_epoch=on_epoch)
    chosen = select_epoch(history)
    return {
        **trained,
        "class_weights": weights,
        "history": history,
        "selected_epoch": chosen,
        "selected_sampler": next(entry["sampler"] for entry in history if entry["epoch"] == chosen),
        "selection_metric": SELECTION_METRIC,
    }


def predict_labels(
    lora: dict,
    rows: list[dict],
    checkpoint: str,
    labels: list[str],
    *,
    sdk=None,
) -> list[str]:
    """Label val rows from the saved sampler. The sealed set is not an input."""
    if sdk is None:
        import tinker

        sdk = tinker
    service = sdk.ServiceClient()
    try:
        client = service.create_sampling_client(model_path=checkpoint)
        tokenizer = client.get_tokenizer()
        params = sdk.types.SamplingParams(max_tokens=32, temperature=0, top_p=1, seed=int(lora["seed"]))
        predicted: list[str] = []
        for row in rows:
            messages = [message for message in row["messages"] if message["role"] != "assistant"]
            text = _complete(client, tokenizer, sdk.types, messages, params)
            predicted.append(_as_label(text, labels))
        return predicted
    finally:
        closer = getattr(service, "close", None)
        if closer is not None:
            closer("success").result()


def _record_checkpoint(root: Path, checkpoint: str) -> None:
    path = root / "configs" / "serve" / "router-epoch2.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["checkpoint"] = checkpoint
    data["status"] = "trained"
    data.pop("wall", None)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _fit(client, types, lora: dict, train_rows: list[dict], *, class_weight: dict[str, float] | None = None, on_epoch=None) -> dict:
    tokenizer = client.get_tokenizer()
    prepared: list[tuple[list[int], list[int], list[float]]] = []
    limit = int(lora["max_sequence_length"])
    thinking = bool(lora["enable_thinking"])
    for row in train_rows:
        inputs, targets, weights = _supervised(tokenizer, row["messages"], enable_thinking=thinking)
        if class_weight is not None:
            from router.balance import scale_label_weights

            weights = scale_label_weights(weights, class_weight[row["messages"][-1]["content"]])
        if len(inputs) + 1 > limit:
            raise RouterWall("WALL: a training row exceeds max_sequence_length")
        prepared.append((inputs, targets, weights))
    params = _optim_params(types, lora)
    steps = 0
    batch_size = int(lora["batch_size"])
    epochs = int(lora["epochs"])
    name = str(lora["checkpoint_name"])
    epoch_paths: list[str] = []
    for epoch in range(epochs):
        order = list(range(len(prepared)))
        random.Random(int(lora["seed"]) + epoch).shuffle(order)
        for start in range(0, len(order), batch_size):
            batch = [
                types.Datum(
                    model_input=types.ModelInput.from_ints(prepared[index][0]),
                    loss_fn_inputs={
                        "target_tokens": prepared[index][1],
                        "weights": prepared[index][2],
                    },
                )
                for index in order[start : start + batch_size]
            ]
            forward = client.forward_backward(batch, lora["loss_fn"])
            optim = client.optim_step(optim_params=params)
            forward.result()
            optim.result()
            steps += 1
        saved = client.save_weights_for_sampler(name=f"{name}-epoch{epoch + 1}").result()
        path = getattr(saved, "path", None)
        if not isinstance(path, str) or not path:
            raise RouterWall("WALL: epoch checkpoint save did not return a path")
        epoch_paths.append(path)
        if on_epoch is not None and on_epoch(epoch + 1, path):
            break
    if on_epoch is None and len(epoch_paths) != epochs:
        raise RouterWall("WALL: epoch checkpoint count does not match the config")
    return {"steps": steps, "epoch_sampler": epoch_paths[-1], "epoch_samplers": epoch_paths}


def _supervised(tokenizer, messages: list[dict], *, enable_thinking: bool) -> tuple[list[int], list[int], list[float]]:
    if not messages or messages[-1]["role"] != "assistant":
        raise RouterWall("WALL: training row has no assistant label")
    prompt = _encode(tokenizer, messages[:-1], add_generation_prompt=True, enable_thinking=enable_thinking)
    full = _encode(tokenizer, messages, add_generation_prompt=False, enable_thinking=enable_thinking)
    if not prompt or full[: len(prompt)] != prompt:
        raise RouterWall("WALL: chat template is not prefix-stable")
    completion = full[len(prompt) :]
    if not completion or len(full) < 2:
        raise RouterWall("WALL: assistant label produced no tokens")
    weights = [0.0] * (len(prompt) - 1) + [1.0] * len(completion)
    inputs = full[:-1]
    targets = full[1:]
    return inputs, targets, weights


def _encode(tokenizer, messages: list[dict], *, add_generation_prompt: bool, enable_thinking: bool) -> list[int]:
    """Token ids for one chat. transformers 5.x returns a BatchEncoding, not a list of ids."""
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
    if hasattr(encoded, "keys") and "input_ids" in encoded:
        ids = encoded["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(token) for token in ids]
    if hasattr(encoded, "ids"):
        encoded = encoded.ids
    return [int(token) for token in encoded]


def _complete(client, tokenizer, types, messages: list[dict], params) -> str:
    encoded = _encode(tokenizer, messages, add_generation_prompt=True, enable_thinking=False)
    result = client.sample(
        prompt=types.ModelInput.from_ints(encoded),
        num_samples=1,
        sampling_params=params,
    ).result()
    try:
        return tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)
    except TypeError:
        return tokenizer.decode(result.sequences[0].tokens)


def _as_label(text: str, labels: list[str]) -> str:
    """Router-model output. The teacher and a baseline use this same parser."""
    from router.parse_label import parse_router_output

    return parse_router_output(text, labels)


def _key_present() -> bool:
    import os

    return bool(os.environ.get("TINKER_API_KEY", "").strip())


if __name__ == "__main__":
    raise SystemExit(main())
