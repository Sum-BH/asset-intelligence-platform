from aip_data import cmapss, synthetic
from aip_signal import features


def test_sensor_selection_drops_flat(tmp_path):
    files = synthetic.generate(tmp_path, n_train_units=6, n_test_units=2, seed=3)
    train = cmapss.parse_cmapss_file(files["train"])
    keep = features.select_informative_sensors(train)
    flat = {f"s{i + 1}" for i in synthetic.FLAT_SENSORS}
    assert not flat & set(keep), "flat sensors must be dropped"
    assert len(keep) == 21 - len(flat)


def test_feature_build_shapes(tmp_path):
    files = synthetic.generate(tmp_path, n_train_units=4, n_test_units=2, seed=4)
    train = cmapss.add_rul_labels(cmapss.parse_cmapss_file(files["train"]))
    sensors = features.select_informative_sensors(train)
    out = features.build_features(train, sensors)
    cols = features.feature_columns(sensors)
    assert set(cols) <= set(out.columns)
    assert out.select(cols).null_count().sum_horizontal().item() == 0


def test_scaler_train_only(tmp_path):
    files = synthetic.generate(tmp_path, n_train_units=4, n_test_units=2, seed=5)
    train = cmapss.add_rul_labels(cmapss.parse_cmapss_file(files["train"]))
    sensors = features.select_informative_sensors(train)
    cols = features.feature_columns(sensors)
    tf = features.build_features(train, sensors)
    scaler = features.fit_scaler(tf, cols)
    scaled = features.apply_scaler(tf, scaler)
    mins = scaled.select(cols).min().row(0)
    maxs = scaled.select(cols).max().row(0)
    assert all(m >= -1e-9 for m in mins) and all(m <= 1 + 1e-9 for m in maxs)
