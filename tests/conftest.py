import os
import sys
import tempfile

# Use a scratch database so tests never touch a real one.
_TMP = tempfile.mkdtemp(prefix="ict-tests-")
os.environ.setdefault("DB_PATH", os.path.join(_TMP, "test.db"))
os.environ.setdefault("DATA_PROVIDER", "synthetic")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

from backend.data.providers import SyntheticProvider


def frame(bars):
    """Build an OHLCV frame from (o, h, l, c) tuples on a 5-minute grid."""
    rows = []
    t = 1_717_000_000
    for i, (o, h, l, c) in enumerate(bars):
        rows.append({"t": t + i * 300, "o": o, "h": h, "l": l, "c": c, "v": 100.0})
    return pd.DataFrame(rows)


@pytest.fixture
def make_frame():
    return frame


# The fixtures end at a fixed moment - Monday 3 June 2024, 10:15 New York,
# inside the silver bullet window - so that session-dependent behaviour
# (killzones, the weekend veto) is identical whenever the suite is run.
ANCHOR_TS = 1_717_424_100


@pytest.fixture(scope="session")
def anchor_ts():
    return ANCHOR_TS


@pytest.fixture(scope="session")
def synthetic_1m():
    return SyntheticProvider().fetch_1m("NAS100_USD", count=12000, to_ts=ANCHOR_TS)


@pytest.fixture(scope="session")
def synthetic_peer_1m():
    return SyntheticProvider(base_price=5200.0, seed=11).fetch_1m(
        "SPX500_USD", count=12000, to_ts=ANCHOR_TS
    )
