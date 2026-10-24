import numpy as np
from aip_data import cmapss, synthetic
from aip_signal import features, models


def test_nasa_score_asymmetry():
    y = np.array([100.0])
    late = models.nasa_score(y, np.array([110.0]))  # 10 late
    early = models.nasa_score(y, np.array([90.0]))  # 10 early
    assert late > early > 0


def test_baselines_learn_signal(tmp_path):
    files = synthetic.generate(tmp_path, n_train_units=25, n_test_units=10, seed=6)
    train = cmapss.add_rul_labels(cmapss.parse_cmapss_file(files["train"]))
    test = cmapss.attach_test_rul(cmapss.parse_cmapss_file(files["test"]), files["rul"])
    sensors = features.select_informative_sensors(train)
    cols = features.feature_columns(sensors)
    tf = features.apply_scaler(
        features.build_features(train, sensors),
        features.fit_scaler(features.build_features(train, sensors), cols),
    )
    scaler = features.fit_scaler(features.build_features(train, sensors), cols)
    ef = features.apply_scaler(features.build_features(test, sensors), scaler)
    res = models.train_baselines(
        tf,
        ef,
        cols,
        fd_set="FD00S",
        use_mlflow=True,
        tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
    )
    assert len(res) == 2
    naive_rmse = models.rmse(
        models.last_cycle_matrix(ef, cols)[1], np.full(10, 62.5)
    )  # predict mid-cap constant
    for r in res:
        assert r.rmse < naive_rmse, f"{r.model_name} no better than naive"
        assert r.run_id is not None  # MLflow logged
