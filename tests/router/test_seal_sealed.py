import ast
import json
import zipfile
from pathlib import Path

import pytest

from router.guard import RouterWall
from router.isolation import load_exclusions, rejected_rows, split_training_pool
from router.labels import load_labels
from router.seal_sealed import agreement, corpus_project, main, seal_sheet


def _sheet(path: Path, labels: list[str], rows: list[dict[str, str]]) -> None:
    columns = ["sheet_id", "excerpt", "filename_optional_blank", "source_path", "file_sha256", "excluded_from_eval", *labels, "unsure"]
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(row.get(column, "") for column in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_owner_tick_is_copied_and_path_is_not_a_label(tmp_path: Path, repo_root: Path) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    owner = labels[1]
    decoy = labels[0]
    sheet = tmp_path / "sheet.csv"
    _sheet(
        sheet,
        labels,
        [
            {
                "sheet_id": "S1",
                "excerpt": "body-one",
                "filename_optional_blank": f"{decoy}.pdf",
                "source_path": f"Master Folder/DG2 Infra Pack 1/{decoy}.pdf",
                "file_sha256": "a" * 64,
                owner: "x",
            },
            {
                "sheet_id": "S2",
                "excerpt": "body-two",
                "filename_optional_blank": "note.pdf",
                "source_path": "Master Folder/other/note.pdf",
                "file_sha256": "b" * 64,
                "unsure": "1",
            },
            {
                "sheet_id": "S3",
                "excerpt": "unreadable",
                "source_path": "Master Folder/other/blank.pdf",
                "file_sha256": "c" * 64,
                "excluded_from_eval": "true",
                labels[2]: "yes",
            },
        ],
    )
    out = tmp_path / "sealed"
    report = seal_sheet(sheet, out, labels)
    assert report["sealed"] == 1
    assert report["dropped"] == {"unsure": 1, "excluded_from_eval": 1}
    stored = [json.loads(line) for line in (out / "sealed_labels.jsonl").read_text(encoding="utf-8").splitlines()]
    assert stored == [
        {
            "sheet_id": "S1",
            "label": owner,
            "file_sha256": "a" * 64,
            "project": "Master Folder",
            "excerpt": "body-one",
        }
    ]
    blob = (out / "sealed_labels.jsonl").read_text(encoding="utf-8")
    assert f"{decoy}.pdf" not in blob
    assert "DG2 Infra Pack 1" not in blob
    digest = (out / "SHA256SUMS").read_text(encoding="utf-8").split()[0]
    assert digest == __import__("hashlib").sha256((out / "sealed_labels.jsonl").read_bytes()).hexdigest()


def test_two_ticks_write_nothing(tmp_path: Path, repo_root: Path) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    sheet = tmp_path / "sheet.csv"
    _sheet(sheet, labels, [{"sheet_id": "S1", "excerpt": "body", labels[0]: "x", labels[1]: "1"}])
    out = tmp_path / "sealed"
    with pytest.raises(RouterWall, match="exactly one tick"):
        seal_sheet(sheet, out, labels)
    assert not (out / "sealed_labels.jsonl").exists()


def test_agreement_does_not_change_the_sealed_label(tmp_path: Path, repo_root: Path) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    owner = labels[3]
    other = labels[4]
    sheet = tmp_path / "sheet.csv"
    _sheet(
        sheet,
        labels,
        [
            {"sheet_id": "S1", "excerpt": "one", "source_path": "Master Folder/a/a.txt", owner: "x"},
            {"sheet_id": "S2", "excerpt": "two", "source_path": "Master Folder/b/b.txt", other: "x"},
        ],
    )
    proposed = tmp_path / "hidden_proposed.jsonl"
    proposed.write_text(
        "\n".join(
            [
                json.dumps({"sheet_id": "S1", "proposed_label": other}),
                json.dumps({"sheet_id": "S2", "proposed_label": other}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "sealed"
    seal_sheet(sheet, out, labels, proposed=proposed)
    sealed = [json.loads(line) for line in (out / "sealed_labels.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["label"] for row in sealed] == [owner, other]
    assert "proposed_label" not in (out / "sealed_labels.jsonl").read_text(encoding="utf-8")
    stats = json.loads((out / "agreement.json").read_text(encoding="utf-8"))
    assert stats["overall"] == 0.5
    assert stats["per_label"][owner] == 0.0
    assert stats["per_label"][other] == 1.0
    assert stats["compared"] == 2
    again = agreement(sealed, {"S1": other, "S2": other}, labels)
    assert again["overall"] == stats["overall"]
    assert sealed[0]["label"] == owner


def test_xlsx_tick_matches_csv(tmp_path: Path, repo_root: Path) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    columns = ["sheet_id", "excerpt", "filename_optional_blank", "source_path", "file_sha256", "excluded_from_eval", *labels, "unsure"]
    row = {column: "" for column in columns}
    row.update(
        {
            "sheet_id": "S1",
            "excerpt": "from-xlsx",
            "filename_optional_blank": "drawing.pdf",
            "source_path": "Master Folder/pack/drawing.pdf",
            "file_sha256": "d" * 64,
            labels[5]: "1",
        }
    )
    sheet = tmp_path / "sheet.xlsx"
    _write_xlsx(sheet, columns, [row])
    out = tmp_path / "sealed"
    seal_sheet(sheet, out, labels)
    stored = json.loads((out / "sealed_labels.jsonl").read_text(encoding="utf-8"))
    assert stored["label"] == labels[5]
    assert stored["project"] == "Master Folder"
    assert "drawing.pdf" not in json.dumps(stored)


def test_xlsx_uses_the_labelling_worksheet_when_instructions_is_first(tmp_path: Path, repo_root: Path) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    columns = ["sheet_id", "excerpt", "filename_optional_blank", "source_path", "file_sha256", "excluded_from_eval", *labels, "unsure"]
    decoy = {column: "" for column in columns}
    decoy.update({"sheet_id": "S1", "excerpt": "instructions", labels[0]: "1"})
    owner = {column: "" for column in columns}
    owner.update(
        {
            "sheet_id": "S1",
            "excerpt": "from-labelling",
            "source_path": "Master Folder/pack/note.pdf",
            "file_sha256": "e" * 64,
            labels[5]: "1",
        }
    )
    sheet = tmp_path / "book.xlsx"
    _write_workbook(
        sheet,
        [
            ("Instructions", columns, [decoy]),
            ("Labelling", columns, [owner]),
        ],
    )
    out = tmp_path / "sealed"
    seal_sheet(sheet, out, labels)
    stored = json.loads((out / "sealed_labels.jsonl").read_text(encoding="utf-8"))
    assert stored["label"] == labels[5]
    assert stored["excerpt"] == "from-labelling"


def test_xlsx_falls_back_to_the_first_sheet_with_label_columns(tmp_path: Path, repo_root: Path) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    columns = ["sheet_id", "excerpt", "filename_optional_blank", "source_path", "file_sha256", "excluded_from_eval", *labels, "unsure"]
    owner = {column: "" for column in columns}
    owner.update({"sheet_id": "S9", "excerpt": "grid", labels[3]: "x"})
    sheet = tmp_path / "book.xlsx"
    _write_workbook(
        sheet,
        [
            ("Instructions", ["note"], [{"note": "read me"}]),
            ("Grid", columns, [owner]),
        ],
    )
    out = tmp_path / "sealed"
    seal_sheet(sheet, out, labels)
    stored = json.loads((out / "sealed_labels.jsonl").read_text(encoding="utf-8"))
    assert stored["label"] == labels[3]
    assert stored["excerpt"] == "grid"


def test_xlsx_without_a_label_sheet_keeps_the_header_wall(tmp_path: Path, repo_root: Path) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    sheet = tmp_path / "book.xlsx"
    _write_workbook(
        sheet,
        [
            ("Instructions", ["note"], [{"note": "read me"}]),
            ("Notes", ["comment"], [{"comment": "later"}]),
        ],
    )
    out = tmp_path / "sealed"
    with pytest.raises(RouterWall, match="sheet header is missing a declared label column"):
        seal_sheet(sheet, out, labels)
    assert not (out / "sealed_labels.jsonl").exists()


def test_committed_v5_seal(repo_root: Path) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels_v5.json")
    directory = repo_root / "data" / "eval" / "router_unseen"
    rows = [
        json.loads(line)
        for line in (directory / "sealed_labels.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 64
    assert len({row["sheet_id"] for row in rows}) == 64
    for row in rows:
        assert set(row) == {
            "sheet_id",
            "label",
            "file_sha256",
            "project",
            "isolation_key",
            "source_url",
        }
        assert row["label"] in labels
        assert row["project"] == "web_public"
        assert row["isolation_key"] == "web_public"
        assert len(row["file_sha256"]) == 64
        assert row["source_url"].startswith("http")
    blob = (directory / "sealed_labels.jsonl").read_text(encoding="utf-8")
    assert "proposed_label" not in blob
    stats = json.loads((directory / "agreement.json").read_text(encoding="utf-8"))
    assert stats["compared"] == 64
    assert stats["overall"] == 0.875
    assert stats["note"] == "Agreement does not change sealed labels."
    listed = (directory / "SHA256SUMS").read_text(encoding="utf-8")
    assert "sealed_labels.jsonl" in listed
    assert "agreement.json" in listed


def test_sealed_doc_or_project_is_rejected(tmp_path: Path) -> None:
    sealed = tmp_path / "sealed_labels.jsonl"
    sealed.write_text(
        json.dumps({"sheet_id": "S1", "label": "kept", "file_sha256": "abc", "project": "Master Folder"}) + "\n",
        encoding="utf-8",
    )
    pool = [
        {"doc_id": "keep", "excerpt": "ok", "file_sha256": "other", "project": "Elsewhere"},
        {"doc_id": "bad-doc", "excerpt": "no", "file_sha256": "abc"},
        {"doc_id": "bad-project", "excerpt": "no", "project": "Master Folder"},
    ]
    exclusions = load_exclusions(sealed)
    rejected = rejected_rows(pool, exclusions)
    assert {row["doc_id"] for row in rejected} == {"bad-doc", "bad-project"}
    with pytest.raises(RouterWall, match="sealed doc or project"):
        split_training_pool(pool, sealed)
    assert split_training_pool([pool[0]], sealed) == [pool[0]]


def _id_sheet(path: Path, labels: list[str], rows: list[dict[str, str]]) -> None:
    columns = ["id", "document", *labels, "unsure", "notes"]
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(row.get(column, "") for column in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_id_column_is_the_sheet_id_and_document_text_is_omitted(tmp_path: Path, repo_root: Path) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    sheet = tmp_path / "sheet.csv"
    _id_sheet(
        sheet,
        labels,
        [
            {
                "id": "S5-001",
                "document": "SECRET-BODY",
                "notes": "SECRET-NOTE",
                labels[0]: "x",
            }
        ],
    )
    out = tmp_path / "sealed"
    seal_sheet(sheet, out, labels)
    stored = json.loads((out / "sealed_labels.jsonl").read_text(encoding="utf-8"))
    assert stored["sheet_id"] == "S5-001"
    assert stored["label"] == labels[0]
    blob = (out / "sealed_labels.jsonl").read_text(encoding="utf-8")
    assert "SECRET-BODY" not in blob
    assert "SECRET-NOTE" not in blob
    assert "document" not in stored


def test_sheets_merge_and_duplicate_ids_write_nothing(tmp_path: Path, repo_root: Path) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    first = tmp_path / "a.csv"
    second = tmp_path / "b.csv"
    _id_sheet(first, labels, [{"id": "S5-001", labels[1]: "x"}])
    _id_sheet(second, labels, [{"id": "S5-002", labels[2]: "x"}])
    out = tmp_path / "sealed"
    report = seal_sheet([first, second], out, labels)
    assert report["sealed"] == 2
    stored = [json.loads(line) for line in (out / "sealed_labels.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["sheet_id"] for row in stored] == ["S5-001", "S5-002"]
    duplicate = tmp_path / "dup.csv"
    _id_sheet(duplicate, labels, [{"id": "S5-001", labels[3]: "x"}])
    blocked = tmp_path / "blocked"
    with pytest.raises(RouterWall, match="duplicate sheet id S5-001"):
        seal_sheet([first, duplicate], blocked, labels)
    assert not (blocked / "sealed_labels.jsonl").exists()


def test_meta_adds_isolation_fields_and_never_a_label(tmp_path: Path, repo_root: Path) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    owner = labels[0]
    other = labels[1]
    sheet = tmp_path / "sheet.csv"
    _sheet(
        sheet,
        labels,
        [
            {
                "sheet_id": "S5-001",
                "excerpt": "body",
                "source_path": "Master Folder/pack/file.pdf",
                "file_sha256": "f" * 64,
                owner: "x",
            }
        ],
    )
    meta = tmp_path / "meta.jsonl"
    meta.write_text(
        json.dumps(
            {
                "id": "S5-001",
                "sheet_id": "S5-001",
                "proposed_label": other,
                "label": other,
                "content_sha256": "ab" * 32,
                "isolation_key": "web_public",
                "source_url": "https://example.com/doc",
                "text": "HIDDEN-TEXT",
                "source_path": "https://example.com/doc",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "sealed"
    report = seal_sheet(sheet, out, labels, meta=[meta])
    assert report["sealed"] == 1
    stored = json.loads((out / "sealed_labels.jsonl").read_text(encoding="utf-8"))
    assert stored == {
        "sheet_id": "S5-001",
        "label": owner,
        "file_sha256": "ab" * 32,
        "project": "web_public",
        "isolation_key": "web_public",
        "source_url": "https://example.com/doc",
        "excerpt": "body",
    }
    blob = (out / "sealed_labels.jsonl").read_text(encoding="utf-8")
    assert other not in blob
    assert "HIDDEN-TEXT" not in blob
    assert "Master Folder" not in blob
    assert "proposed_label" not in blob
    assert not (out / "agreement.json").exists()


def test_agreement_uses_proposed_and_not_meta(tmp_path: Path, repo_root: Path) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    owner = labels[3]
    other = labels[4]
    sheet = tmp_path / "sheet.csv"
    _id_sheet(sheet, labels, [{"id": "S5-009", owner: "x"}])
    meta = tmp_path / "meta.jsonl"
    meta.write_text(
        json.dumps(
            {
                "sheet_id": "S5-009",
                "proposed_label": owner,
                "content_sha256": "cd" * 32,
                "isolation_key": "web_public",
                "source_url": "https://example.com/a",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    proposed = tmp_path / "proposed.jsonl"
    proposed.write_text(
        json.dumps({"sheet_id": "S5-009", "proposed_label": other}) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "sealed"
    seal_sheet(sheet, out, labels, proposed=proposed, meta=[meta])
    stored = json.loads((out / "sealed_labels.jsonl").read_text(encoding="utf-8"))
    assert stored["label"] == owner
    stats = json.loads((out / "agreement.json").read_text(encoding="utf-8"))
    assert stats["overall"] == 0.0
    assert stats["compared"] == 1


def test_duplicate_meta_id_writes_nothing(tmp_path: Path, repo_root: Path) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    sheet = tmp_path / "sheet.csv"
    _id_sheet(sheet, labels, [{"id": "S5-001", labels[0]: "x"}])
    meta = tmp_path / "meta.jsonl"
    line = json.dumps(
        {
            "sheet_id": "S5-001",
            "content_sha256": "aa" * 32,
            "isolation_key": "web_public",
            "source_url": "https://example.com/a",
        }
    )
    meta.write_text(line + "\n" + line + "\n", encoding="utf-8")
    out = tmp_path / "sealed"
    with pytest.raises(RouterWall, match="duplicate meta id S5-001"):
        seal_sheet(sheet, out, labels, meta=[meta])
    assert not (out / "sealed_labels.jsonl").exists()


def test_cli_accepts_repeatable_sheet_and_meta(tmp_path: Path, repo_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    first = tmp_path / "a.csv"
    second = tmp_path / "b.csv"
    _id_sheet(first, labels, [{"id": "S5-001", labels[0]: "x"}])
    _id_sheet(second, labels, [{"id": "S5-002", labels[1]: "x"}])
    meta_a = tmp_path / "meta-a.jsonl"
    meta_b = tmp_path / "meta-b.jsonl"
    meta_a.write_text(
        json.dumps(
            {
                "id": "S5-001",
                "content_sha256": "11" * 32,
                "isolation_key": "web_public",
                "source_url": "https://example.com/1",
                "proposed_label": labels[2],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    meta_b.write_text(
        json.dumps(
            {
                "sheet_id": "S5-002",
                "content_sha256": "22" * 32,
                "isolation_key": "web_public",
                "source_url": "https://example.com/2",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    proposed = tmp_path / "proposed.jsonl"
    proposed.write_text(
        "\n".join(
            [
                json.dumps({"sheet_id": "S5-001", "proposed_label": labels[0]}),
                json.dumps({"sheet_id": "S5-002", "proposed_label": labels[2]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "sealed"
    code = main(
        [
            "--sheet",
            str(first),
            "--sheet",
            str(second),
            "--meta",
            str(meta_a),
            "--meta",
            str(meta_b),
            "--proposed",
            str(proposed),
            "--output-dir",
            str(out),
            "--root",
            str(repo_root),
        ]
    )
    assert code == 0
    assert "WALL" not in capsys.readouterr().out
    stored = [json.loads(line) for line in (out / "sealed_labels.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["label"] for row in stored] == [labels[0], labels[1]]
    assert [row["file_sha256"] for row in stored] == ["11" * 32, "22" * 32]
    assert all(row["project"] == "web_public" for row in stored)
    stats = json.loads((out / "agreement.json").read_text(encoding="utf-8"))
    assert stats["overall"] == 0.5


def test_corpus_project_ignores_a_bare_filename() -> None:
    assert corpus_project("Master Folder/DG2 Infra Pack 1/file.pdf") == "Master Folder"
    assert corpus_project("file.pdf") is None
    assert corpus_project("Master Folder\\nested\\file.pdf") == "Master Folder"


def test_router_code_does_not_assign_a_declared_label_from_a_path(repo_root: Path) -> None:
    labels = set(load_labels(repo_root / "configs" / "router" / "labels.json"))
    labels.add("unsure")
    path_names = {"filename", "source_path", "folder", "doc_id", "file_name", "filename_optional_blank"}
    for path in (repo_root / "scripts" / "router").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            names = {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}
            if not names & path_names:
                continue
            literals = {
                item.value
                for item in ast.walk(node)
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            }
            assert not literals & labels, f"{path.name}:{node.name}"


def _write_xlsx(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    def column_ref(index: int, row_number: int) -> str:
        letters = ""
        number = index + 1
        while number:
            number, rem = divmod(number - 1, 26)
            letters = chr(65 + rem) + letters
        return f"{letters}{row_number}"

    def row_xml(values: list[str], row_number: int) -> str:
        cells = []
        for index, value in enumerate(values):
            cells.append(
                f'<c r="{column_ref(index, row_number)}" t="inlineStr"><is><t>{value}</t></is></c>'
            )
        return f'<row r="{row_number}">{"".join(cells)}</row>'

    body = [row_xml(columns, 1)]
    for offset, row in enumerate(rows, start=2):
        body.append(row_xml([row.get(column, "") for column in columns], offset))
    sheet = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{''.join(body)}</sheetData></worksheet>"
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/></Relationships>'
    )
    package = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/></Relationships>'
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        "</Types>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", package)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", rels)
        archive.writestr("xl/worksheets/sheet1.xml", sheet)


def _write_workbook(path: Path, sheets: list[tuple[str, list[str], list[dict[str, str]]]]) -> None:
    def column_ref(index: int, row_number: int) -> str:
        letters = ""
        number = index + 1
        while number:
            number, rem = divmod(number - 1, 26)
            letters = chr(65 + rem) + letters
        return f"{letters}{row_number}"

    def sheet_xml(columns: list[str], rows: list[dict[str, str]]) -> str:
        def row_xml(values: list[str], row_number: int) -> str:
            cells = [
                f'<c r="{column_ref(index, row_number)}" t="inlineStr"><is><t>{value}</t></is></c>'
                for index, value in enumerate(values)
            ]
            return f'<row r="{row_number}">{"".join(cells)}</row>'

        body = [row_xml(columns, 1)]
        for offset, row in enumerate(rows, start=2):
            body.append(row_xml([row.get(column, "") for column in columns], offset))
        return (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f"<sheetData>{''.join(body)}</sheetData></worksheet>"
        )

    sheet_tags = []
    relationships = []
    overrides = []
    for index, (name, _columns, _rows) in enumerate(sheets, start=1):
        sheet_tags.append(f'<sheet name="{name}" sheetId="{index}" r:id="rId{index}"/>')
        relationships.append(
            f'<Relationship Id="rId{index}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{index}.xml"/>'
        )
        overrides.append(
            f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheets>{''.join(sheet_tags)}</sheets></workbook>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f"{''.join(relationships)}</Relationships>"
    )
    package = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/></Relationships>'
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        f"{''.join(overrides)}</Types>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", package)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", rels)
        for index, (_name, columns, rows) in enumerate(sheets, start=1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", sheet_xml(columns, rows))
