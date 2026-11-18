"""The frozen Source Data tables are complete and build into workbooks; regenerated tables match them."""
import pytest

from draft_analysis import compare, config, paths, source_data


@pytest.mark.parametrize("kind", (*source_data.KINDS, source_data.TABLES))
def test_tables_complete(kind):
    for sheet in source_data.sheets(kind):
        for block in sheet["blocks"]:
            df = source_data.read_table(source_data.frozen_dir(kind) / block["tsv"])
            assert len(df) == block["rows"], block["tsv"]


@pytest.mark.parametrize("kind", (*source_data.KINDS, source_data.TABLES))
def test_workbooks_build(kind, tmp_path):
    import openpyxl

    files = source_data.build_workbooks(kind, source_data.frozen_dir(kind), tmp_path)
    figures = sorted({s["figure"] for s in source_data.sheets(kind)})
    assert len(files) == len(figures)
    for f, fig in zip(files, figures):
        names = openpyxl.load_workbook(f, read_only=True).sheetnames
        expected = {s["sheet"] for s in source_data.sheets(kind) if s["figure"] == fig}
        assert names[0] == "Contents" and expected <= set(names)


def test_frozen_equals_itself():
    assert all(d.ok for d in source_data.compare_dir("main", source_data.frozen_dir("main")))


def test_config_paths_rewritten():
    cfg = config.load("germ_network")
    assert cfg["data"]["rna_h5ad"] == str(paths.DATA / "fly" / "raw" / "A_rna_focus.h5ad")
    assert cfg["network"]["tf_filter_file"].startswith(str(paths.DATA))
    unified = config.load("ccc_unified")
    assert unified["baselines"]["CellChat"]["edges"].startswith(str(paths.DATA / "fly" / "external"))
    assert unified["germ_stage_labels"].startswith(str(config.CONFIGS))


@pytest.mark.skipif(not paths.PANELS["main"].exists(), reason="run `draft-analysis checkpoint` first")
def test_regenerated_main_tables_match():
    results = compare.run("main", verbose=False)
    assert results, "no regenerated tables"
    assert [d.name for d in results if not d.ok] == []


def test_text_tolerance():
    from draft_analysis.source_data import _text_close

    assert _text_close("0.28 (-0.24-0.77)", "0.28 (-0.24-0.78)", 0.01) is not None
    assert _text_close("0.28 (-0.24-0.77)", "0.28 (-0.24-0.78)", 0.001) is None
    assert _text_close("70% (57%-88%)", "70 (57%-88%)", 1) is None  # different wording
