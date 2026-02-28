import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))

from warp_md.analysis.ti import ti


def test_ti_trapz():
    data = np.array(
        [
            [0.0, 0.0],
            [0.5, 1.0],
            [1.0, 0.0],
        ],
        dtype=float,
    )
    out = ti(data=data, x_col=0, y_col=1, method="trapz")
    assert abs(out["integral"] - 0.5) < 1e-6
    assert len(out["lambda"]) == 3
    assert len(out["dvdl"]) == 3
