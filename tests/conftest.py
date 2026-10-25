import numpy as np
import pytest

# A-1, D-1, M-1 are clean. P-2 appears twice with conflicting intervals. T-10 has no row.
LABELS_CSV = """chan_id,spacecraft,anomaly_sequences,class,num_values
A-1,SMAP,"[[10, 19], [40, 49]]","[point, contextual]",60
D-1,SMAP,"[[5, 9]]",[point],50
P-2,SMAP,"[[5, 9]]",[point],50
P-2,SMAP,"[[6, 9]]",[point],50
M-1,MSL,"[[20, 29]]",[contextual],40
"""

# name -> (n_features, train_len, test_len)
CHANNELS = {
    "A-1": (25, 30, 60),
    "D-1": (25, 30, 50),
    "P-2": (25, 30, 50),
    "M-1": (55, 25, 40),
    "T-10": (55, 20, 30),
}


@pytest.fixture
def synthetic_dir(tmp_path):
    rng = np.random.default_rng(0)
    data_dir = tmp_path / "telemanom"
    for name, (n_features, n_train, n_test) in CHANNELS.items():
        for split, n in (("train", n_train), ("test", n_test)):
            (data_dir / split).mkdir(parents=True, exist_ok=True)
            np.save(data_dir / split / f"{name}.npy", rng.normal(size=(n, n_features)))
    (data_dir / "labeled_anomalies.csv").write_text(LABELS_CSV)
    return data_dir
