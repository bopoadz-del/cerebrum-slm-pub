"""Project-diversity rule: <=3 sealed rows per (source project, label)."""

import pytest

from router.guard import RouterWall
from router.seal_sealed import enforce_project_diversity, project_key


def _row(project, label, url=None):
    r = {"project": project, "label": label}
    if url:
        r["source_url"] = url
    return r


def test_project_key_counts_host_and_first_path_segment():
    a = _row("web_public", "NCR", "https://www.nrc.gov/docs/ML0728/x.pdf")
    b = _row("web_public", "NCR", "https://www.nrc.gov/docs/ML0723/y.pdf")
    assert project_key(a) == project_key(b) == "www.nrc.gov/docs"


def test_three_rows_ok_four_rejected():
    rows = [_row("web_public", "NCR", f"https://www.nrc.gov/docs/ML0728/{i}.pdf") for i in range(3)]
    enforce_project_diversity(rows)  # no raise at 3
    rows.append(_row("web_public", "NCR", "https://www.nrc.gov/docs/ML0728/4.pdf"))
    with pytest.raises(RouterWall):
        enforce_project_diversity(rows)


def test_same_host_different_class_is_separate():
    rows = [
        _row("web_public", "NCR", f"https://www.nrc.gov/docs/ML0728/{i}.pdf") for i in range(3)
    ] + [
        _row("web_public", "report", f"https://www.nrc.gov/docs/ML0728/r{i}.pdf") for i in range(3)
    ]
    enforce_project_diversity(rows)  # 3 NCR + 3 report from the same project is fine


def test_different_projects_do_not_sum():
    rows = [
        _row("web_public", "NCR", f"https://www.nrc.gov/docs/ML0728/{i}.pdf") for i in range(3)
    ] + [
        _row("web_public", "NCR", f"https://ncsx.pppl.gov/NCSX_Construction/q/{i}.pdf") for i in range(3)
    ]
    enforce_project_diversity(rows)
