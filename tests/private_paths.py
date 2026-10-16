"""Skip a test when private data is not in the tree or at an external path.

The skip is explicit: the test still asserts the same things when the file
is present. Nothing in the message is file contents.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from private_data import locate, missing_message


def skip_if_missing(root: Path, *relative: str, env: str | None = None) -> None:
    """Skip when the repo-relative path is absent.

    These tests read that path. Restoring the gitignored file there runs the
    test with the same assertions. Loaders also accept the environment
    variable named in the skip message.
    """
    missing = [rel for rel in relative if not (root / rel).exists() and not locate(root, rel).exists()]
    if not missing:
        # Present only outside the tree: tests that still open the repo path
        # would fail. Skip those until the file is at the repo-relative path,
        # unless locate() and the repo path are the same file.
        outside = []
        for rel in relative:
            local = root / rel
            found = locate(root, rel)
            if not local.exists() and found.exists():
                outside.append(rel)
        if outside:
            names = ", ".join(outside)
            variable = env or "SLM_PRIVATE_ROOT"
            pytest.skip(
                "needs an external data path at the repo-relative location; "
                f"skipped because the file is absent from the tree: {names}. "
                f"Loaders can read it via {variable}. "
                "Place the file at the repo-relative path (gitignored) to run this test."
            )
        return
    first = missing[0]
    names = ", ".join(missing)
    pytest.skip(
        "needs an external data path; skipped because the file is absent: "
        f"{names}. {missing_message(first, env)}"
    )


# Tests that read archive, sealed, or owner files. Each entry is the
# repo-relative path the test needs. Assertions are unchanged; the test
# runs when locate() finds the file (in the tree or via SLM_PRIVATE_ROOT
# and the specific variables in private_data.py).
EXTERNAL_DATA_TESTS: dict[str, tuple[str, ...]] = {}


def _need(node: str, *paths: str) -> None:
    EXTERNAL_DATA_TESTS[node] = paths


_HELD = (
    "data/held_out/shards.json",
    "data/held_out/pairs.jsonl",
    "data/held_out/manifest.json",
)
_ZIP = ("data/raw/slm-datasets.zip",)
_SEALED = (
    "data/eval/router_unseen/sealed_labels.jsonl",
    "data/eval/router_unseen/v8/sealed_labels.jsonl",
)
_CR = ("data/router/cr_v9_sealed_candidates_isolation.json", "data/eval/router_unseen/v9_cr_addendum/sealed_labels.jsonl")
_THIN = ("data/router/thin_v9_sealed_candidates_isolation.json",)
_WEB_F = (
    "data/eval/router_unseen/v9_web_f_addendum/sealed_labels.jsonl",
    "data/eval/router_unseen/v9_web_f_addendum/held.jsonl",
    "data/router/web_f_sealed_candidates_isolation.json",
    "data/router/sealed_cap_exhaustion.json",
)
_WEB_G = (
    "data/eval/router_unseen/v9_web_g_addendum/sealed_labels.jsonl",
    "data/eval/router_unseen/v9_web_g_addendum/held.jsonl",
    "data/router/web_g_sealed_candidates_isolation.json",
    "data/router/sealed_cap_exhaustion.json",
)
_LABELS = ("data/raw/router/labels.jsonl",)
_PR2 = ("data/router/additions_owner_sources_pr2.jsonl",)
_LAPTOP = ("data/router/additions_owner_laptop.jsonl",)
_SHEET = ("data/sealed/router/labelling_sheet.csv", "data/sealed/router/README.md")
_V2_SEALED = ("docs/router/runs/v2/sealed/sealed_predictions.jsonl",)
_RAW_TABLE = ("docs/audit/raw-reconciliation/raw_lines.jsonl",)
_RATES = ("data/retrieval/rates/rates.jsonl", "data/retrieval/facilities/records.jsonl")

for _name in (
    "test_real_shard_list_names_the_original_and_the_added_shards",
    "test_original_shard_labels_are_the_held_out_labels",
    "test_every_added_shard_labels_with_the_pair_type_rule",
    "test_no_shard_name_or_pair_text_in_code",
    "test_cli_shard_flag_parses",
):
    _need(f"tests/eval/test_held_out_shards_eval.py::{_name}", *_HELD)

for _name in (
    "test_real_splits_use_the_same_label_method",
    "test_label_sources_do_not_embed_eval_item_text",
):
    _need(f"tests/eval/test_labels.py::{_name}", "data/held_out/pairs.jsonl", "data/held_out/manifest.json")

for _name in (
    "test_dry_run_checks_real_data_without_tinker",
    "test_dry_run_can_stack_the_prompt_only_system_prompt",
    "test_missing_key_blocks_before_sampling",
    "test_baselines_label_and_still_do_not_sample",
    "test_cli_dry_run_and_missing_key",
):
    _need(f"tests/eval/test_tuned_eval.py::{_name}", "data/held_out/pairs.jsonl", "data/held_out/manifest.json")

_need("tests/model/test_prompt_only.py::test_baselines_are_not_run", "data/held_out/pairs.jsonl", "data/held_out/manifest.json")
_need("tests/data/test_apply_split.py::CorpusTests::test_committed_split", *_HELD, *_RATES)
for _name in (
    "test_every_line_has_one_destination_and_dropped_rows_have_a_reason",
    "test_split_files_are_fully_accounted_for",
    "test_committed_table_is_current",
    "test_table_carries_no_item_text",
):
    _need(f"tests/data/test_reconcile_raw.py::{_name}", *_ZIP, "data/held_out/pairs.jsonl", *_RAW_TABLE)

_need("tests/gates/test_blind_seal.py::test_seal_json_matches_then_count_hash_and_source_zip_fail", *_ZIP)
_need("tests/gates/test_held_out_shards_gate.py::test_real_repo_shards_are_green_and_match_freeze", *_HELD)
for _name in (
    "test_catalog_seals_eval_and_leaves_posture_unsealed",
    "test_real_probe_and_domain_red_then_removed",
    "test_real_eval_pair_in_training_red_then_removed",
    "test_training_domain_words_and_a_different_case_stay_green",
):
    _need(f"tests/gates/test_no_hardwiring.py::{_name}", *_ZIP)
_need("tests/gates/test_prove.py::test_repo_gates_are_green", *_ZIP, "data/held_out")
_need("tests/gates/test_raw.py::test_archive_matches_sha256_manifest", *_ZIP)
_need("tests/gates/test_raw.py::test_manifest_mismatch_is_red", *_ZIP)
_need("tests/gates/test_split_gate.py::test_archived_posture_passes_and_standard_names_fail", *_ZIP)

_need("tests/router/test_label_set_v9.py::test_sealed_retick_adds_no_rows_and_records_every_change", *_SEALED)
_need("tests/router/test_seal_sealed.py::test_committed_v5_seal", "data/eval/router_unseen/sealed_labels.jsonl")
_need("tests/router/test_sealed_gate.py::test_recorded_sealed_score_matches_its_sha256sums", *_V2_SEALED)
_need("tests/router/test_sealed_gate.py::test_v5_change_order_rows_stay_recorded_as_unsure", *_V2_SEALED)
_need(
    "tests/router/test_sealed_gate.py::test_v9_retick_sets_mark_thin_labels_not_passable",
    "data/eval/router_unseen/v9_retick/v5/sealed_labels.jsonl",
    "data/eval/router_unseen/v9_retick/v8/sealed_labels.jsonl",
)
_need("tests/router/test_sealed_sheet.py::test_owner_line_and_sealed_drop_zone", *_SHEET, "data/eval/router_unseen/sealed_labels.jsonl")
_need("tests/router/test_sealed_sheet.py::test_labelling_sheet_is_sixty_blank_rows", "data/sealed/router/labelling_sheet.csv")
_need("tests/router/test_train_path.py::test_pending_sample_is_a_header_only", "data/router/human_sample_PENDING_OWNER.csv")
_need("tests/router/test_train_path.py::test_raw_pool_prep_counts_and_no_sealed_overlap", *_LABELS)
_need("tests/router/test_train_path.py::test_train_entry_walls_without_labels_and_hides_a_key", "data/eval/router_unseen/sealed_labels.jsonl")
_need("tests/router/test_train_path.py::test_split_v2_reproduces_the_landed_rows", *_LABELS)
_need(
    "tests/router/test_train_path.py::test_proposed_split_stays_isolated_and_still_rejects_sealed_keys",
    *_SEALED,
    *_LAPTOP,
)
_need("tests/router/test_pr5_additions_isolation.py::test_pr2_additions_are_the_unmasked_originals", *_PR2)
_need("tests/router/test_pr5_additions_isolation.py::test_additions_never_overlap_any_sealed_set", *_PR2, *_SEALED)
_need("tests/router/test_pr5_additions_isolation.py::test_change_request_rows_are_isolated_from_sealed_sources", *_CR, *_SEALED)
_need("tests/router/test_pr5_additions_isolation.py::test_train_and_val_hold_zero_sealed_rows", *_CR, *_SEALED)
_need("tests/router/test_pr5_additions_isolation.py::test_templates_train_only_and_projects_not_split", *_PR2)
_need("tests/router/test_slm_ingest_1015.py::test_rows_never_overlap_any_sealed_or_held_set", *_SEALED, "data/held_out/pairs.jsonl", *_THIN, *_CR)
_need("tests/router/test_thin_v9_additions_isolation.py::test_thin_rows_are_isolated_from_sealed_and_pending_sources", *_THIN, *_CR, *_SEALED)
_need("tests/router/test_thin_v9_additions_isolation.py::test_thin_pending_candidates_share_no_source_with_any_train_batch", *_THIN)
_need("tests/router/test_thin_v9_additions_isolation.py::test_train_and_val_hold_zero_sealed_or_pending_rows", *_THIN, *_CR, *_SEALED)
_need("tests/router/test_thin_v9f_claims_isolation.py::test_v9f_rows_never_overlap_any_sealed_set", *_SEALED, *_THIN, *_CR)
_need("tests/router/test_v7_pairs.py::test_v7_examples_never_overlap_any_sealed_set", *_SEALED, "data/eval/router_unseen/v9_cr_addendum/sealed_labels.jsonl")

for _name in (
    "test_addendum_seal_matches_its_file",
    "test_addendum_rows_are_owner_ticks_on_preregistered_candidates",
    "test_addendum_swaps_nothing_in_other_sealed_sets",
    "test_v9_config_lists_the_sets_and_counts_match_the_files",
    "test_addendum_makes_its_label_passable_in_the_combined_set",
):
    _need(f"tests/router/test_v9_cr_addendum.py::{_name}", *_CR, *_SEALED)

for _name in (
    "test_every_file_in_the_addendum_matches_its_seal",
    "test_sealed_rows_are_owner_ticks_with_url_and_sha_provenance",
    "test_held_rows_are_recorded_and_never_sealed_or_trained",
    "test_candidate_hashes_cover_sealed_and_held_and_stay_out_of_training",
    "test_project_cap_holds_for_every_pair_this_addendum_touches",
    "test_exhaustion_record_is_recomputed_from_data",
    "test_no_document_of_this_addendum_is_sealed_elsewhere",
    "test_gap_table_matches_the_registered_sets",
):
    _need(f"tests/router/test_v9_web_f_addendum.py::{_name}", *_WEB_F)
    _need(f"tests/router/test_v9_web_g_addendum.py::{_name}", *_WEB_G)

_need(
    "tests/router/test_v9_web_g_addendum.py::test_one_source_document_is_sealed_at_most_once",
    *_WEB_G,
)
