import numpy as np
import pandas as pd

from epic.pandas.numpy import isnan, isnullscalar


def test_isnan():
    assert np.all(isnan([1, 2, np.NaN, 'asdf']) == [False, False, True, False])
    assert isnan(np.NaN)
    assert not isnan(3)


def test_isnullscalar():
    for null in (np.NaN, pd.NA, pd.NaT, None):
        assert isnullscalar(null)
    assert not isnullscalar(0)
    assert not isnullscalar([1, 2, np.NaN])
