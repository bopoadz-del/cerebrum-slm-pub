from router.labels import load_labels, load_sealed_config
from router.majority_gate import evaluate


def test_majority_predictor_fails_at_perfect_accuracy(repo_root) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    config = load_sealed_config(repo_root / "configs" / "router" / "sealed_eval.json")
    gold = [labels[0]] * 4
    pred = [labels[0]] * 4
    result = evaluate(gold, pred, majority_class_fail=config["majority_class_fail"], order=labels)
    assert result.passed is False
    assert result.accuracy == 1.0
    assert result.reason == "majority-class predictor"


def test_majority_predictor_fails_when_accuracy_is_high_but_not_perfect(repo_root) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    gold = [labels[0]] * 9 + [labels[1]]
    pred = [labels[0]] * 10
    result = evaluate(gold, pred, majority_class_fail=True, order=labels)
    assert result.passed is False
    assert result.accuracy == 0.9


def test_majority_predictor_fails_on_a_balanced_set(repo_root) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    gold = list(labels)
    pred = [labels[0]] * len(labels)
    result = evaluate(gold, pred, majority_class_fail=True, order=labels)
    assert result.passed is False
    assert result.accuracy == 1 / len(labels)


def test_a_non_constant_predictor_is_not_the_majority_failure(repo_root) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    gold = [labels[0], labels[1]]
    result = evaluate(gold, gold, majority_class_fail=True, order=labels)
    assert result.passed is True
    assert result.accuracy == 1.0


def test_flag_off_does_not_apply_the_majority_failure(repo_root) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    gold = [labels[0], labels[0]]
    result = evaluate(gold, gold, majority_class_fail=False, order=labels)
    assert result.passed is True


def test_empty_gold_fails(repo_root) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    result = evaluate([], [], majority_class_fail=True, order=labels)
    assert result.passed is False
    assert result.accuracy is None
