import duckdb
import polars as pl
from aip_data import cmapss, synthetic


def _make(tmp_path):
    return synthetic.generate(tmp_path, n_train_units=6, n_test_units=4, seed=1)


def test_parser_roundtrip(tmp_path):
    files = _make(tmp_path)
    df = cmapss.parse_cmapss_file(files["train"])
    assert df.columns == cmapss.COLUMNS
    assert df["unit_id"].n_unique() == 6
    assert df["cycle"].min() == 1
    assert df.null_count().sum_horizontal().item() == 0


def test_rul_labels_capped(tmp_path):
    files = _make(tmp_path)
    df = cmapss.add_rul_labels(cmapss.parse_cmapss_file(files["train"]), cap=125)
    assert df["rul"].max() <= 125
    # last cycle of every unit has RUL 0
    last = df.filter(pl.col("cycle") == pl.col("cycle").max().over("unit_id"))
    assert (last["rul"] == 0).all()


def test_test_rul_attachment(tmp_path):
    files = _make(tmp_path)
    df = cmapss.attach_test_rul(cmapss.parse_cmapss_file(files["test"]), files["rul"], cap=125)
    ruls = [int(x) for x in files["rul"].read_text().split()]
    last = df.filter(pl.col("cycle") == pl.col("cycle").max().over("unit_id")).sort("unit_id")
    expected = [min(r, 125) for r in ruls]
    assert last["rul"].to_list() == expected


def test_duckdb_load_idempotent(tmp_path):
    files = _make(tmp_path)
    df = cmapss.add_rul_labels(cmapss.parse_cmapss_file(files["train"]))
    con = duckdb.connect()
    cmapss.load_to_duckdb(df, "FDTEST", con, split="train")
    cmapss.load_to_duckdb(df, "FDTEST", con, split="train")  # reload, no dupes
    n = con.execute("SELECT COUNT(*) FROM cycles").fetchone()[0]
    assert n == df.height
    eng = con.execute("SELECT COUNT(*) FROM engines").fetchone()[0]
    assert eng == 6
