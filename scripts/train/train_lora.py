"""LoRA entry point for the posture model.

Reads the train split through training_open, renders it with the Qwen3 chat
template, and saves a named checkpoint. --dry-run validates and prints counts.
It does not construct a Tinker client.

Hyperparameter reasoning: configs/lora/README.md.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _bootstrap() -> None:
    scripts = Path(__file__).resolve().parents[1]
    root = scripts.parent
    for entry in (scripts, root / "src", root):
        text = str(entry)
        if text not in sys.path:
            sys.path.append(text)


_bootstrap()

from gates.blind_seal import BlindSealError, training_open
from gates.split import TermsError, leak_kinds, load_retrieval_terms
from train.posture_target import TARGET_FORMAT, TargetError, registry_inputs, target_for

TRAIN_SPLIT = "data/train/pairs.jsonl"
TRAIN_MANIFEST = "data/train/manifest.json"
MODEL_CONFIG = "configs/model/qwen3-8b.json"
LORA_CONFIG = "configs/lora/qwen3-8b-posture.json"
EVAL_SYSTEM_PROMPT = "eval/baselines/prompt_only_system_prompt.txt"
FORMULA_REGISTRY = "data/retrieval/formula_registry.json"
INSTRUCT_ID = "Qwen/Qwen3-8B"
API_KEY_ENV = "TINKER_API_KEY"
ORIGINAL_LOSS_WEIGHT = 3.0  # originals weighted x3 vs paraphrases/hard-negatives

_CONFIG_KEYS = {
    "base_model",
    "train_file",
    "manifest_file",
    "chat_template",
    "enable_thinking",
    "loss_fn",
    "loss_on",
    "rank",
    "alpha",
    "target_modules",
    "learning_rate",
    "optimizer",
    "beta1",
    "beta2",
    "eps",
    "weight_decay",
    "grad_clip_norm",
    "epochs",
    "batch_size",
    "max_sequence_length",
    "seed",
    "checkpoint_name",
    "system_prompt_file",
    "target_format",
}
_MODULE_KEYS = {"mlp", "attn", "unembed"}
_MAX_LEARNING_RATE = 0.0005
_MAX_EPOCHS = 3
_MAX_BATCH = 32
_MAX_RANK = 32
_MAX_SEQUENCE = 2048


class TrainRefused(RuntimeError):
    """The run stopped before any training step."""


class TinkerKeyMissing(RuntimeError):
    """A real run needs TINKER_API_KEY in the environment."""


@dataclass(frozen=True)
class TrainPair:
    line: int
    system: str
    user: str
    assistant: str
    pair_type: str = ""
    formula_id: str = ""
    aug: str = ""


@dataclass(frozen=True)
class TrainSummary:
    pairs: int
    assistant_turns: int
    system_turns: int


@dataclass(frozen=True)
class TrainResult:
    steps: int
    state_path: str
    sampler_path: str
    epoch_sampler_paths: tuple[str, ...] = ()
    losses: tuple[dict, ...] = ()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_text(path: Path, root: Path) -> str:
    """Read one repository file. training_open rejects a sealed path first."""
    try:
        with training_open(path, root, "r") as handle:
            return handle.read()
    except BlindSealError as exc:
        raise TrainRefused("training reader rejected a sealed path") from exc


def load_json(path: Path, root: Path) -> Any:
    try:
        return json.loads(read_text(path, root))
    except json.JSONDecodeError as exc:
        raise TrainRefused(f"{path.name} is not JSON") from exc


def load_model_identity(root: Path) -> dict[str, Any]:
    path = root / MODEL_CONFIG
    config = load_json(path, root)
    if not isinstance(config, dict):
        raise TrainRefused("model identity must be a JSON object")
    return config


def load_lora_config(root: Path, path: Path | None = None) -> dict[str, Any]:
    target = path if path is not None else root / LORA_CONFIG
    config = load_json(target, root)
    if not isinstance(config, dict):
        raise TrainRefused("LoRA config must be a JSON object")
    return config


def validate_lora_config(config: dict[str, Any], model: dict[str, Any]) -> None:
    """Reject a config that is not the posture adapter on the instruct checkpoint."""
    extra = sorted(set(config) - _CONFIG_KEYS)
    missing = sorted(_CONFIG_KEYS - set(config))
    if extra or missing:
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if extra:
            detail.append("unknown " + ", ".join(extra))
        raise TrainRefused("LoRA config fields: " + "; ".join(detail))
    if _secret_key(config):
        raise TrainRefused("LoRA config must not carry a secret")

    identity = model.get("tinker_base_model")
    if model.get("model_id") != INSTRUCT_ID or identity != INSTRUCT_ID:
        raise TrainRefused("model identity must be the instruct checkpoint Qwen/Qwen3-8B")
    sampling = model.get("sampling")
    if not isinstance(sampling, dict) or sampling.get("enable_thinking") is not False:
        raise TrainRefused("model sampling must keep thinking off")
    if config["base_model"] != INSTRUCT_ID:
        raise TrainRefused("base_model must be Qwen/Qwen3-8B")
    if config["train_file"] != TRAIN_SPLIT or config["manifest_file"] != TRAIN_MANIFEST:
        raise TrainRefused("training reads the train split only")
    if config["chat_template"] != "qwen3":
        raise TrainRefused("chat_template must be qwen3")
    if config["enable_thinking"] is not False:
        raise TrainRefused("enable_thinking must be false")
    if config["loss_fn"] != "cross_entropy":
        raise TrainRefused("loss_fn must be cross_entropy")
    if config["loss_on"] != "assistant_turns_only":
        raise TrainRefused("loss_on must be assistant_turns_only")

    rank = _positive_int("rank", config["rank"], _MAX_RANK)
    alpha = _positive_int("alpha", config["alpha"], _MAX_RANK)
    if alpha != rank:
        raise TrainRefused("alpha must equal rank; the training client has no alpha argument")
    modules = config["target_modules"]
    if not isinstance(modules, dict) or set(modules) != _MODULE_KEYS:
        raise TrainRefused("target_modules must set mlp, attn, and unembed")
    if any(not isinstance(modules[key], bool) for key in _MODULE_KEYS):
        raise TrainRefused("target_modules values must be booleans")
    if not modules["mlp"]:
        raise TrainRefused("target_modules.mlp must be true")

    opt = str(config.get("optimizer", "adamw")).strip().lower()
    if opt not in {"adamw", "adam", "dimuon"}:
        raise TrainRefused("optimizer must be adamw or dimuon")
    # Shared AdamW/Dimuon sanity ceiling. The rate still comes from config.
    _positive_float("learning_rate", config["learning_rate"], _MAX_LEARNING_RATE)
    _unit_interval("beta1", config["beta1"])
    _unit_interval("beta2", config["beta2"])
    _positive_float("eps", config["eps"], 1.0)
    weight_decay = config["weight_decay"]
    if isinstance(weight_decay, bool) or not isinstance(weight_decay, (int, float)) or weight_decay < 0:
        raise TrainRefused("weight_decay must be >= 0")
    clip = config["grad_clip_norm"]
    if isinstance(clip, bool) or not isinstance(clip, (int, float)) or clip < 0:
        raise TrainRefused("grad_clip_norm must be >= 0")

    _positive_int("epochs", config["epochs"], _MAX_EPOCHS)
    _positive_int("batch_size", config["batch_size"], _MAX_BATCH)
    length = _positive_int("max_sequence_length", config["max_sequence_length"], _MAX_SEQUENCE)
    if length < 32:
        raise TrainRefused("max_sequence_length must be at least 32")
    seed = config["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise TrainRefused("seed must be an integer >= 0")
    name = config["checkpoint_name"]
    if not isinstance(name, str) or not _checkpoint_name(name):
        raise TrainRefused("checkpoint_name must be a short lowercase slug")
    if config["system_prompt_file"] != EVAL_SYSTEM_PROMPT:
        raise TrainRefused("system_prompt_file must be the eval system prompt " + EVAL_SYSTEM_PROMPT)
    if config["target_format"] != TARGET_FORMAT:
        raise TrainRefused("target_format must be " + TARGET_FORMAT)


def load_system_prompt(root: Path, config: dict[str, Any]) -> str:
    """The same system prompt the eval stacks on the tuned checkpoint."""
    text = read_text(root / config["system_prompt_file"], root)
    if not text.strip():
        raise TrainRefused("system prompt file is empty")
    return text


def load_registry_inputs(root: Path) -> dict[str, tuple[str, ...]]:
    try:
        return registry_inputs(load_json(root / FORMULA_REGISTRY, root))
    except TargetError as exc:
        raise TrainRefused(str(exc)) from exc


def render_pairs(
    pairs: list[TrainPair],
    system_prompt: str,
    inputs: dict[str, tuple[str, ...]],
    root: Path | None = None,
) -> list[TrainPair]:
    """Put the eval system prompt and the reply-contract target on every pair.

    A row that already carries a different system prompt is refused: train and
    eval must see the same one. The rendered target passes the Split check.
    """
    rendered: list[TrainPair] = []
    for pair in pairs:
        if pair.system.strip() and pair.system != system_prompt:
            raise TrainRefused(f"line {pair.line}: system prompt differs from the eval system prompt")
        try:
            target = target_for(pair.pair_type, pair.formula_id, inputs).render()
        except TargetError as exc:
            raise TrainRefused(f"line {pair.line}: {exc}") from exc
        rendered.append(
            TrainPair(
                line=pair.line,
                system=system_prompt,
                user=pair.user,
                assistant=target,
                pair_type=pair.pair_type,
                formula_id=pair.formula_id,
                aug=pair.aug,
            )
        )
    if root is not None:
        _refuse_gates_text(root, rendered)
    return rendered


def prepare_training_pairs(root: Path, config: dict[str, Any]) -> list[TrainPair]:
    """Train pairs as the run sees them: eval system prompt plus contract target."""
    pairs = load_train_pairs(root, config)
    return render_pairs(pairs, load_system_prompt(root, config), load_registry_inputs(root), root)


def load_train_pairs(root: Path, config: dict[str, Any]) -> list[TrainPair]:
    """Open the train split and its manifest. Refuse when the gates fail."""
    manifest_path = root / config["manifest_file"]
    train_path = root / config["train_file"]
    manifest = load_json(manifest_path, root)
    raw = read_text(train_path, root)
    pairs = _parse_pairs(raw)
    _refuse_gates(root, manifest, pairs)
    return pairs


def summarize(pairs: list[TrainPair]) -> TrainSummary:
    return TrainSummary(
        pairs=len(pairs),
        assistant_turns=len(pairs),
        system_turns=sum(1 for pair in pairs if pair.system.strip()),
    )


def step_count(pairs: int, batch_size: int, epochs: int) -> int:
    if pairs <= 0:
        raise TrainRefused("train split is empty")
    return math.ceil(pairs / batch_size) * epochs


def format_dry_run(config: dict[str, Any], summary: TrainSummary) -> str:
    steps = step_count(summary.pairs, int(config["batch_size"]), int(config["epochs"]))
    lines = [
        "dry-run: ok",
        f"base_model: {config['base_model']}",
        f"train_file: {config['train_file']}",
        f"pairs: {summary.pairs}",
        f"assistant_turns: {summary.assistant_turns}",
        f"system_turns: {summary.system_turns}",
        f"epochs: {config['epochs']}",
        f"batch_size: {config['batch_size']}",
        f"steps: {steps}",
        f"rank: {config['rank']}",
        f"alpha: {config['alpha']}",
        f"max_sequence_length: {config['max_sequence_length']}",
        f"seed: {config['seed']}",
        f"loss_fn: {config['loss_fn']}",
        f"loss_on: {config['loss_on']}",
        f"checkpoint_name: {config['checkpoint_name']}",
        f"system_prompt_file: {config['system_prompt_file']}",
        f"target_format: {config['target_format']}",
        "tinker: not contacted",
    ]
    return "\n".join(lines)


def messages_for(pair: TrainPair) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if pair.system.strip():
        messages.append({"role": "system", "content": pair.system})
    messages.append({"role": "user", "content": pair.user})
    messages.append({"role": "assistant", "content": pair.assistant})
    return messages


def encode_chat(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )
    if hasattr(encoded, "ids"):
        encoded = encoded.ids
    return [int(token) for token in encoded]


def supervised_tokens(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    enable_thinking: bool,
    example_weight: float = 1.0,
) -> tuple[list[int], list[int], list[float]]:
    """Shift-by-one datum. Assistant tokens carry example_weight (originals use 3.0)."""
    if not messages or messages[-1]["role"] != "assistant":
        raise TrainRefused("conversation must end with an assistant turn")
    if any(message["role"] == "assistant" for message in messages[:-1]):
        raise TrainRefused("only a final assistant turn is supported")
    if not any(message["role"] == "user" for message in messages):
        raise TrainRefused("conversation must include a user turn")
    prompt = encode_chat(
        tokenizer,
        messages[:-1],
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    full = encode_chat(
        tokenizer,
        messages,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    if not prompt or full[: len(prompt)] != prompt:
        raise TrainRefused("chat template is not prefix-stable; refusing to guess the assistant span")
    completion = full[len(prompt) :]
    if not completion or len(full) < 2:
        raise TrainRefused("assistant turn produced no tokens")
    if not isinstance(example_weight, (int, float)) or isinstance(example_weight, bool) or float(example_weight) <= 0:
        raise TrainRefused("example_weight must be > 0")
    weight = float(example_weight)
    weights = [0.0] * (len(prompt) - 1) + [weight] * len(completion)
    inputs = full[:-1]
    targets = full[1:]
    if len(weights) != len(inputs) or len(targets) != len(inputs):
        raise TrainRefused("loss mask does not align with the shifted sequence")
    if not any(w > 0.0 for w in weights):
        raise TrainRefused("loss mask has no assistant token")
    return inputs, targets, weights


def api_key_present() -> bool:
    return bool(os.environ.get(API_KEY_ENV, "").strip())


def run_training(config: dict[str, Any], pairs: list[TrainPair], *, sdk: Any = None) -> TrainResult:
    """Train and save. Pass sdk only from a test double. The default import is tinker."""
    if sdk is None:
        import tinker

        sdk = tinker
    service = sdk.ServiceClient()
    status = "errored"
    try:
        modules = config["target_modules"]
        client = service.create_lora_training_client(
            base_model=config["base_model"],
            rank=int(config["rank"]),
            seed=int(config["seed"]),
            train_mlp=bool(modules["mlp"]),
            train_attn=bool(modules["attn"]),
            train_unembed=bool(modules["unembed"]),
            optimizer=_optimizer_config(sdk.types, config),
            user_metadata={"checkpoint": str(config["checkpoint_name"])},
        )
        result = _fit(client, sdk.types, config, pairs)
        status = "success"
        return result
    finally:
        service.close(status).result()


def prepare(argv: list[str] | None = None) -> tuple[argparse.Namespace, Path]:
    parser = argparse.ArgumentParser(description="Posture LoRA entry point")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--log-json", type=Path, default=None, help="Write step losses and checkpoint paths")
    args = parser.parse_args(argv)
    root = args.root.resolve() if args.root is not None else repo_root()
    return args, root


def main(argv: list[str] | None = None) -> int:
    try:
        args, root = prepare(argv)
        config_path = args.config
        if config_path is not None and not config_path.is_absolute():
            config_path = root / config_path
        config = load_lora_config(root, config_path)
        model = load_model_identity(root)
        validate_lora_config(config, model)
        pairs = prepare_training_pairs(root, config)
        if args.dry_run:
            print(format_dry_run(config, summarize(pairs)))
            return 0
        if not api_key_present():
            raise TinkerKeyMissing(
                "Training is blocked: TINKER_API_KEY is not set in the environment. "
                "The key is not read from the repo."
            )
        result = run_training(config, pairs)
    except TrainRefused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    except TinkerKeyMissing as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    print(f"steps: {result.steps}")
    for index, path in enumerate(result.epoch_sampler_paths, 1):
        print(f"checkpoint_sampler_epoch{index}: {path}")
    print(f"checkpoint_state: {result.state_path}")
    print(f"checkpoint_sampler: {result.sampler_path}")
    if args.log_json is not None:
        _write_log(args.log_json, config, pairs, result)
    return 0


def _write_log(path: Path, config: dict[str, Any], pairs: list[TrainPair], result: TrainResult) -> None:
    payload = {
        "checkpoint_name": config["checkpoint_name"],
        "base_model": config["base_model"],
        "pairs": len(pairs),
        "system_turns": sum(1 for pair in pairs if pair.system.strip()),
        "target_format": config["target_format"],
        "rank": config["rank"],
        "learning_rate": config["learning_rate"],
        "epochs": config["epochs"],
        "batch_size": config["batch_size"],
        "steps": result.steps,
        "losses": list(result.losses),
        "checkpoint_state": result.state_path,
        "checkpoint_sampler": result.sampler_path,
        "epoch_samplers": list(result.epoch_sampler_paths),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")



def _optimizer_name(config: dict[str, Any]) -> str:
    name = str(config.get("optimizer", "adamw")).strip().lower()
    if name in {"adamw", "adam"}:
        return "adamw"
    if name == "dimuon":
        return "dimuon"
    raise TrainRefused("optimizer must be adamw or dimuon")


def _optimizer_config(types: Any, config: dict[str, Any]) -> Any:
    if _optimizer_name(config) == "dimuon":
        return types.DimuonOptimizerConfig()
    return types.AdamOptimizerConfig()


def _optim_params(types: Any, config: dict[str, Any]) -> Any:
    lr = float(config["learning_rate"])
    beta1 = float(config["beta1"])
    clip = float(config["grad_clip_norm"])
    if _optimizer_name(config) == "dimuon":
        return types.DimuonParams(learning_rate=lr, beta1=beta1, grad_clip_norm=clip)
    return types.AdamParams(
        learning_rate=lr,
        beta1=beta1,
        beta2=float(config["beta2"]),
        eps=float(config["eps"]),
        weight_decay=float(config["weight_decay"]),
        grad_clip_norm=clip,
    )


def _fit(client: Any, types: Any, config: dict[str, Any], pairs: list[TrainPair]) -> TrainResult:
    tokenizer = client.get_tokenizer()
    prepared: list[tuple[list[int], list[int], list[float]]] = []
    limit = int(config["max_sequence_length"])
    thinking = bool(config["enable_thinking"])
    for pair in pairs:
        # Originals (empty aug) get ORIGINAL_LOSS_WEIGHT; paraphrases and
        # hard negatives stay at 1.0. Per-example weight, not duplication.
        example_weight = ORIGINAL_LOSS_WEIGHT if not pair.aug else 1.0
        inputs, targets, weights = supervised_tokens(
            tokenizer,
            messages_for(pair),
            enable_thinking=thinking,
            example_weight=example_weight,
        )
        length = len(inputs) + 1
        if length > limit:
            raise TrainRefused(
                f"line {pair.line}: sequence length {length} exceeds max_sequence_length {limit}"
            )
        prepared.append((inputs, targets, weights))

    adam = _optim_params(types, config)
    steps = 0
    batch_size = int(config["batch_size"])
    epochs = int(config["epochs"])
    name = str(config["checkpoint_name"])
    epoch_paths: list[str] = []
    losses: list[dict] = []
    for epoch in range(epochs):
        order = list(range(len(prepared)))
        random.Random(int(config["seed"]) + epoch).shuffle(order)
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
            forward = client.forward_backward(batch, config["loss_fn"])
            optim = client.optim_step(optim_params=adam)
            output = forward.result()
            optim.result()
            steps += 1
            weighted = sum(sum(prepared[index][2]) for index in order[start : start + batch_size])
            losses.append(_loss_record(steps, epoch + 1, output, len(batch), weighted))
        if epochs > 1:
            saved = client.save_weights_for_sampler(name=f"{name}-epoch{epoch + 1}").result()
            path = getattr(saved, "path", None)
            if not isinstance(path, str):
                raise TrainRefused("epoch checkpoint save did not return a path")
            epoch_paths.append(path)

    state = client.save_state(name=name).result()
    sampler = client.save_weights_for_sampler(name=name).result()
    state_path = getattr(state, "path", None)
    sampler_path = getattr(sampler, "path", None)
    if not isinstance(state_path, str) or not isinstance(sampler_path, str):
        raise TrainRefused("checkpoint save did not return a path")
    return TrainResult(
        steps=steps,
        state_path=state_path,
        sampler_path=sampler_path,
        epoch_sampler_paths=tuple(epoch_paths),
        losses=tuple(losses),
    )


def _loss_record(step: int, epoch: int, output: Any, n_datums: int, weighted: float) -> dict:
    """Loss for one step. Token counts come from the prepared weight lists, not the SDK datum."""
    tokens = float(weighted)
    metrics = getattr(output, "metrics", None)
    loss_sum = None
    if isinstance(metrics, dict):
        value = metrics.get("loss:sum")
        if isinstance(value, (int, float)):
            loss_sum = float(value)
    record: dict = {"step": step, "epoch": epoch, "n_datums": n_datums, "loss_weight_tokens": tokens}
    if loss_sum is not None:
        record["loss_sum"] = round(loss_sum, 6)
        record["loss_mean_per_token"] = round(loss_sum / tokens, 6) if tokens else None
    return record


def _parse_pairs(raw: str) -> list[TrainPair]:
    pairs: list[TrainPair] = []
    for lineno, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TrainRefused(f"line {lineno}: record is not JSON") from exc
        if not isinstance(obj, dict):
            raise TrainRefused(f"line {lineno}: record is not an object")
        system = obj.get("system")
        user = obj.get("user")
        assistant = obj.get("assistant")
        meta = obj.get("meta")
        if not isinstance(system, str) or not isinstance(user, str) or not isinstance(assistant, str):
            raise TrainRefused(f"line {lineno}: system, user, and assistant must be strings")
        if not user.strip() or not assistant.strip():
            raise TrainRefused(f"line {lineno}: user and assistant must be non-empty")
        if not isinstance(meta, dict) or meta.get("split") != "train":
            raise TrainRefused(f"line {lineno}: meta.split must be train")
        pair_type = meta.get("pair_type")
        formula_id = meta.get("formula_id")
        aug = meta.get("aug")
        pairs.append(
            TrainPair(
                line=lineno,
                system=system,
                user=user,
                assistant=assistant,
                pair_type=pair_type.strip() if isinstance(pair_type, str) else "",
                formula_id=formula_id.strip() if isinstance(formula_id, str) else "",
                aug=aug.strip() if isinstance(aug, str) else "",
            )
        )
    return pairs


def _refuse_gates(root: Path, manifest: Any, pairs: list[TrainPair]) -> None:
    if not isinstance(manifest, dict):
        raise TrainRefused("train manifest must be a JSON object")
    if manifest.get("fact_count") != 0:
        raise TrainRefused("train manifest fact_count must be 0")
    if manifest.get("count") != len(pairs):
        raise TrainRefused("train manifest count does not match the train split")
    if not pairs:
        raise TrainRefused("train split is empty")
    _refuse_gates_text(root, pairs)


def _refuse_gates_text(root: Path, pairs: list[TrainPair]) -> None:
    try:
        terms = load_retrieval_terms(root)
    except TermsError as exc:
        raise TrainRefused(f"retrieval terms could not be loaded: {exc}") from exc
    failures: list[str] = []
    for pair in pairs:
        kinds = leak_kinds(pair.assistant, terms)
        if kinds:
            failures.append(f"line {pair.line}: assistant turn contains {', '.join(kinds)}")
    if failures:
        raise TrainRefused("\n".join(failures))


def _secret_key(config: dict[str, Any]) -> bool:
    blocked = {"api_key", "token", "secret", "password"}
    return any(str(key).lower() in blocked for key in config)


def _positive_int(name: str, value: Any, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= upper:
        raise TrainRefused(f"{name} must be an integer from 1 to {upper}")
    return value


def _positive_float(name: str, value: Any, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < float(value) <= upper:
        raise TrainRefused(f"{name} must be > 0 and <= {upper}")
    return float(value)


def _unit_interval(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < float(value) < 1:
        raise TrainRefused(f"{name} must be between 0 and 1")


def _checkpoint_name(name: str) -> bool:
    if not 1 <= len(name) <= 64:
        return False
    if name[0] == "-" or name[-1] == "-":
        return False
    return all(char.islower() or char.isdigit() or char == "-" for char in name)


if __name__ == "__main__":
    raise SystemExit(main())
