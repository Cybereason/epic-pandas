import pytest
import operator
import numpy as np
import pandas as pd

from epic.pandas.utils import sample_with_distribution, fillna


class TestSampleWithDistribution:
    SEED = 42
    DATA = pd.DataFrame({'A': [1, 1, 1, 1, 0, 1, 0, 0, 0, 0]})

    def test_sampling(self):
        assert sample_with_distribution(self.DATA, 'A', 6, {0: 4, 1: 2}, random_state=self.SEED).equals(
            self.DATA.loc[[4, 9, 6, 3, 1, 7]]
        )

    def test_sample_too_large(self):
        with pytest.raises(ValueError):
            sample_with_distribution(self.DATA, 'A', 12, {0: 4, 1: 2})


class TestFillNA:
    NA_ROW = 1
    DATA = pd.DataFrame(np.arange(12).reshape(3, 4), columns=['A', 'B', 'C', 'D'])
    DATA.iloc[NA_ROW, :] = np.NaN

    def common_tests(self, other):
        assert other.index.equals(self.DATA.index)
        assert other.drop(self.NA_ROW).astype(int).equals(self.DATA.drop(self.NA_ROW).astype(int))

    def test_scalar(self):
        scalar = 123
        filled = fillna(self.DATA, scalar)
        self.common_tests(filled)
        assert filled.loc[self.NA_ROW].eq(scalar).all()

    def test_mutable(self):
        mutable = list
        filled = fillna(self.DATA, mutable)
        self.common_tests(filled)
        assert filled.loc[self.NA_ROW].apply(operator.eq, args=(mutable(),)).all()
        assert filled.loc[self.NA_ROW].map(id).nunique() == self.DATA.shape[1]

    def test_different_columns(self):
        filled = fillna(self.DATA, {'A': None, 'B': list, 'C': dict, 'D': -1})
        self.common_tests(filled)
        assert filled.loc[self.NA_ROW, 'A'] is None
        assert filled.loc[self.NA_ROW, 'B'] == []
        assert filled.loc[self.NA_ROW, 'C'] == {}
        assert filled.loc[self.NA_ROW, 'D'] == -1

    def test_dtype_revert_to_object(self):
        dtypes = fillna(self.DATA.astype('Int32'), list).dtypes.unique()
        assert dtypes.size == 1
        assert dtypes[0] == np.dtype(object)
