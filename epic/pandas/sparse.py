import warnings
import numpy as np
import pandas as pd

from operator import mul
from scipy import sparse as sp
from typing import Literal, TypeVar
from collections.abc import Iterable, Hashable
from epic.common.general import to_list, coalesce
from numpy.typing import ArrayLike, DTypeLike, NDArray

from .utils import stack_indices

Shape2D = tuple[int, int]
IndexLike = Iterable[Hashable]
SDF = TypeVar('SDF', bound='SparseDataFrame')
CSRArgType = ArrayLike | sp.spmatrix | Shape2D | \
             tuple[ArrayLike, tuple[ArrayLike, ArrayLike]] | \
             tuple[ArrayLike, ArrayLike, ArrayLike]


class SparseDataFrame(sp.csr_matrix):
    """
    A subclass of scipy.sparse.csr_matrix, adding to it an index and columns.

    Parameters
    ----------
    arg :
        Valid first argument to csr_matrix constructor.

    index : iterable, optional
        Row labels of the matrix.

    columns : iterable, optional
        Column labels of the matrix.

    shape : 2-tuple, optional
        Shape of the matrix.

    dtype : dtype-like, optional
        Data type of the matrix.

    copy : bool, default False
        Whether to force a copy of the data.

    Notes
    -----
    Does NOT support all types of fancy slicing the way a normal CSR matrix does.
    """
    def __init__(self, arg: CSRArgType, /, index: IndexLike | None = None, columns: IndexLike | None = None,
                 *, shape: Shape2D | None = None, dtype: DTypeLike = None, copy: bool = False):
        super().__init__(arg, shape=shape, dtype=dtype, copy=copy)
        self.index = self._make_index(index, 0)
        self.columns = self._make_index(columns, 1)
        if self.shape != (len(self.index), len(self.columns)):
            raise ValueError("Indices inconsistent with matrix shape.")

    def _make_index(self, index, axis) -> pd.Index:
        if index is None:
            return pd.RangeIndex(self.shape[axis])
        if not isinstance(index, pd.Index):
            return pd.Index(index)
        return index

    @property
    def density(self) -> float:
        """
        Fraction of non-zero entries.

        Returns
        -------
        float
        """
        size = float(mul(*self.shape))
        return size and self.nnz / size

    @staticmethod
    def _get_indexer(index: pd.Index, item):
        if isinstance(item, slice):
            return item
        if not isinstance(item, pd.Series):
            item = to_list(item)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', FutureWarning)
                item = index[item]
        except IndexError:
            pass
        indexer = index.get_indexer_for(item)
        if -1 in indexer:
            raise KeyError([x for i, x in zip(indexer, item) if i == -1])
        return indexer

    def __getitem__(self: SDF, item) -> SDF:
        index_item, columns_item = item if isinstance(item, tuple) and len(item) == 2 else (item, slice(None))
        idx = self._get_indexer(self.index, index_item)
        col = self._get_indexer(self.columns, columns_item)
        data: SDF = super().__getitem__((
            idx[:, np.newaxis] if isinstance(idx, np.ndarray) and isinstance(col, np.ndarray) else idx,
            col,
        ))
        data.index = self.index[idx]
        data.columns = self.columns[col]
        return data

    def todense(self, order: Literal['C', 'F'] | None = None, out: NDArray | None = None) -> pd.DataFrame:
        """
        Convert to a dense regular DataFrame.

        Parameters
        ----------
        order : {'C', 'F'}, optional
            Data storage order.
            See `toarray` documentation for details.

        out : ndarray, optional
            Array in which to store the data.
            See `toarray` documentation for details.

        Returns
        -------
        DataFrame
            Dense representation of the data.

        See Also
        --------
        csr_matrix.toarray : Convert to a dense numpy array.
        """
        return pd.DataFrame(data=self.toarray(order, out), index=self.index, columns=self.columns)

    def is_square(self) -> bool:
        """
        Tests whether the matrix is square, i.e. the index and columns are equal.

        Returns
        -------
        bool
        """
        return self.index.equals(self.columns)

    def __repr__(self) -> str:
        details = [
            f"shape={'x'.join(map(str, self.shape))}",
            f"dtype={self.dtype}",
            f"density={self.density * 100:.2g}%",
        ]
        for ind in ('index', 'columns'):
            idx: pd.Index = getattr(self, ind)
            if any(n is not None for n in idx.names):
                details.append(f"{ind}={'.'.join(str(coalesce(n, '_')) for n in idx.names)}")
        return f"{type(self).__name__}({', '.join(details)})"

    def stack(self, name: Hashable | None = None) -> pd.Series:
        """
        Convert to a Series with a MultiIndex.

        Parameters
        ----------
        name : hashable, optional
            Name for the resulting Series.

        Returns
        -------
        Series
        """
        coo = self.tocoo()
        return pd.Series(coo.data, index=stack_indices(self.index[coo.row], self.columns[coo.col]), name=name)
