import os
import inspect
import warnings

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

import pandas as pd
from pandas.core import algorithms
from pandas.core.generic import NDFrame
from pandas.core.sorting import get_compressed_ids
from pandas._typing import NDFrameT, AnyArrayLike, RandomState
from pandas.core.common import random_state as process_random_state

from io import BytesIO
from functools import partial
from scipy import sparse as sp
from decorator import decorator
from itertools import combinations
from typing import overload, Literal, TypeVar
from collections.abc import Mapping, Iterable, Hashable, Callable

from epic.common.io import pload, pdump
from epic.common.general import classproperty
from epic.common.iteration import SizedIterable
from epic.common.pathgeneralizer import PathGeneralizer
from epic.common.assertion import TypeOrTypes, assert_type, assert_types

from .parallel import papply

T = TypeVar('T')

__all__ = [
    'pdload', 'pddump', 'autoload', 'sample_at_most', 'sample_with_distribution', 'value_counts',
    'dfdiag', 'pd_wide_display', 'drop_duplicates_by_index', 'fillna', 'sizeof', 'upsert',
    'alignable', 'stack_indices', 'IdentitySeries', 'canonize_df_and_cols', 'column_stats',
    'unique_row_ids', 'iterrows',
]


class _PandasLoaderDumper:
    """
    A robust loader/dumper of objects (pandas and others) from/to various file formats.
    File format is deduced from the filename extension.
    """
    @classproperty
    def handlers(cls):
        # Default is to use read_suffix and to_suffix, so e.g. csv and json are covered
        return {
            'pkl': (pload, pdump),
            'pklgz': (partial(pd.read_pickle, compression="gzip"), partial(pd.to_pickle, compression="gzip")),
            'pklbz2': (partial(pd.read_pickle, compression="bz2"), partial(pd.to_pickle, compression="bz2")),
            'idx': (cls._load_index, cls._dump_index),
            'npy': (np.load, lambda x, f, **kw: np.save(f, x, **kw)),
            'npz': (np.load, cls._dump_npz),
            'npzsp': (sp.load_npz, cls._dump_spmat),
        }

    @classmethod
    def load(cls, filepath: str, check_instanceof: TypeOrTypes | None = None, **kwargs):
        """
        Load an object from a file.

        File format is deduced from the filename extension.
        Valid extensions include all "pandas.read_*" suffixes, as well as:
        - pkl: pickle
        - pklgz: pickle + gzip compression
        - pklbz2: pickle + bz2 compression
        - idx: for loading pandas.Index objects
        - npy: for loading numpy arrays
        - npz: for loading an archive of numpy arrays (mapping returned)
        - npzsp: for loading scipy sparse matrices

        Parameters
        ----------
        filepath : str
            Name of the file.
            Remote resources are also supported (see `epic.common.pathgeneralizer.PathGeneralizer`).

        check_instanceof : type, tuple of types or UnionType, optional
            If provided, asserts that the loaded object is of these type.

        **kwargs :
            Sent to loading function.

        Returns
        -------
        object
            Loaded object.
        """
        generalized_path = PathGeneralizer.from_path(filepath)
        if not generalized_path.exists():
            raise ValueError(f"File '{filepath}' not found")
        extension = cls._get_extension(filepath)
        handler = cls.handlers.get(extension, extension)
        if isinstance(handler, str):
            loader = getattr(pd, f"read_{handler}", None)
        else:
            loader = handler[0]
        if loader is None:
            raise ValueError(f"Unexpected path extension '{extension}'")
        with generalized_path.read_proxy() as filepath_:
            result = loader(filepath_, **kwargs)
        if check_instanceof is not None:
            assert_type(result, check_instanceof, 'loaded object')
        return result

    @classmethod
    def dump(cls, obj, filepath: str, **kwargs):
        """
        Dump an object to file.

        File format is deduced from the filename extension.
        Valid extensions include all "pandas.to_*" suffixes, any of the object's "to_*" suffixes, as well as:
        - pkl: pickle
        - pklgz: pickle + gzip compression
        - pklbz2: pickle + bz2 compression
        - idx: for dumping pandas.Index objects
        - npy: for dumping numpy arrays
        - npz: for dumping several numpy arrays together.
               Can provide either an iterable of objects or a mapping of names to objects.
               Can also provide a `compressed` flag (default True).
        - npzsp: for dumping scipy sparse matrices.
                 Can provide a `compressed` flag (default True).

        Parameters
        ----------
        obj : object
            Object to dump.

        filepath : str
            Name of the file.
            Remote resources are also supported for suffixes 'pkl', 'pklgz', 'pklbz2' and 'idx'
            (see `epic.common.pathgeneralizer.PathGeneralizer`).

        **kwargs :
            Sent to dumping function.
        """
        extension = cls._get_extension(filepath)
        handler = cls.handlers.get(extension, extension)
        if isinstance(handler, str):
            dumper = getattr(obj, f"to_{handler}", None)
            if dumper is not None:
                return dumper(filepath, **kwargs)
            dumper = getattr(pd, f"to_{handler}", None)
            if dumper is not None:
                return dumper(obj, filepath, **kwargs)
            raise ValueError(f"Unexpected path extension '{extension}'")
        with PathGeneralizer.from_path(filepath).write_proxy() as filepath_:
            return handler[1](obj, filepath_, **kwargs)

    @staticmethod
    def _get_extension(filepath: str) -> str:
        return os.path.splitext(filepath)[1].lower()[1:]

    @staticmethod
    def _load_index(filepath):
        return pd.Index(pd.read_csv(filepath, header=None, usecols=[0]).values.reshape(-1))

    @staticmethod
    def _dump_index(obj, filepath):
        if not isinstance(obj, list | tuple | set | pd.Index):
            raise TypeError(f"Unsupported type for dumping as index file: {type(obj).__name__}")
        with open(filepath, "w") as f:
            f.write("\n".join(map(str, obj)))

    @staticmethod
    def _dump_npz(obj, filepath, compressed=True):
        save = np.savez_compressed if compressed else np.savez
        if isinstance(obj, Mapping):
            save(filepath, **obj)
        elif isinstance(obj, Iterable) and not isinstance(obj, np.ndarray):
            save(filepath, *obj)
        else:
            raise TypeError(
                "Cannot dump a single object to 'npz' file. "
                "Provide either an iterable of objects or a mapping of names to objects. "
                "To dump a single numpy array, use 'npy' format. "
                f"Got {type(obj).__name__}."
            )

    @staticmethod
    def _dump_spmat(spmat, filepath, compressed=True):
        if not sp.isspmatrix(spmat):
            raise TypeError(f"Unsupported type for dumping to 'npzsp' file: {type(spmat).__name__}")
        # sp.save_npz uses np.savez or np.savez_compressed, which both enforce a .npz suffix
        # for a string filename. So we write the file to disk for ourselves.
        with BytesIO() as buffer:
            sp.save_npz(buffer, spmat, compressed=compressed)
            with open(filepath, 'wb') as f:
                f.write(buffer.getbuffer())


pdload = _PandasLoaderDumper.load
pddump = _PandasLoaderDumper.dump


@assert_types(df=pd.DataFrame, distribution=pd.Series | Mapping | None)
def sample_with_distribution(
        df: pd.DataFrame,
        population: Hashable | ArrayLike | AnyArrayLike,
        n_rows: int | None = None,
        distribution: pd.Series | Mapping | None = None,
        *,
        random_state: RandomState | None = None,
) -> pd.DataFrame:
    """
    Sample random rows from a DataFrame with a given distribution on a population.
    The population is a data-series compatible with the DataFrame, possibly one of its columns.

    Parameters
    ----------
    df : DataFrame
        The data from which to sample.

    population : Series, array-like or hashable
        The data whose distribution we wish to control.
        If a hashable, it is the name of one of the columns of `df`.

    n_rows : int, optional
        The number of rows to sample.
        If not given, it is set as the largest number of rows that can be sampled while still
        satisfying the requested distribution.

    distribution : Series or Mapping, optional
        A mapping from each population value to its wanted fraction in the results.
        Fractions should add up to 1.0; otherwise, they will be normalized (divided by sum).
        If not given, the existing distribution of population is used.

    random_state : int, array-like, BitGenerator, np.random.RandomState or np.random.Generator, optional
        Allows reproducibility.

    Returns
    -------
    DataFrame
        Sampled rows from `df`.

    Notes
    -----
    N/A values will never be sampled and are ignored when given in `distribution`.
    """
    if isinstance(population, Hashable):
        population = df[population]
    elif isinstance(population, pd.Series):
        population = population.reindex(df.index, copy=False)
    else:
        population = pd.Series(population, index=df.index)
    if n_rows is None and distribution is None:
        return df.sample(frac=1., random_state=random_state)
    if distribution is None:
        distribution = population.value_counts(normalize=True)
    else:
        if not isinstance(distribution, pd.Series):
            distribution = pd.Series(distribution)
        distribution = distribution.dropna()
        distribution /= distribution.sum()
        max_possible_n = int(population.value_counts(dropna=False).div(distribution).min())
        if n_rows is None:
            n_rows = max_possible_n
        elif n_rows > max_possible_n:
            raise ValueError(
                f"Cannot sample {n_rows} rows; "
                f"max possible is {max_possible_n} with distribution:\n{distribution.to_string()}"
            )
    if n_rows < 0:
        raise ValueError(f"`n_rows` must be non-negative; got {n_rows}")
    # Calculate counts from distribution while keeping sum(counts) =~ n_rows
    cum_counts = distribution.mul(n_rows).cumsum().round()
    counts = cum_counts.diff()
    counts.iloc[0] = cum_counts.iloc[0]
    counts = counts.astype(int)
    # Select the indices
    rand = np.random.default_rng() if random_state is None else process_random_state(random_state)
    chosen = pd.Series(np.arange(len(population))).groupby(population.values).apply(
        lambda x: x.sample(counts.get(x.name, 0), random_state=rand)
    ).values
    rand.shuffle(chosen)
    return df.iloc[chosen]


def value_counts(items: Iterable, sort: bool = True, ascending: bool = False, bins: int | None = None,
                 dropna: bool = True, total: bool = False) -> pd.DataFrame:
    """
    Compute a histogram of the items and return both the counts and fractional parts of the whole.

    Parameters
    ----------
    items : iterable
        Items to count.

    sort : bool, default True
        Whether to sort the values.

    ascending : bool, default False
        Sorting order.

    bins : int, optional
        Number of half-open bins into which to group the values, instead of counting them.

    dropna : bool, default True
        Whether to disregard N/A values.

    total : bool, default False
        Whether to add a total row to the result.

    Returns
    -------
    DataFrame
        A frame with two columns: 'count' and 'fraction'.
        The sum of the 'fraction' column is 1.0.
    """
    vc = pd.value_counts(items, normalize=False, sort=sort, ascending=ascending, bins=bins, dropna=dropna)
    n_items = vc.sum()
    res = pd.DataFrame({'count': vc, 'fraction': vc / n_items})
    if total:
        tot_label = 'TOTAL'
        while tot_label in res.index:
            tot_label = f"={tot_label}="
        res.loc[tot_label] = [n_items, 1]
    return res


@assert_types(df=pd.DataFrame)
def dfdiag(df: pd.DataFrame) -> pd.Series:
    """
    Return the diagonal of a DataFrame, as a Series.
    The result includes entries which appear both in the index and the columns.

    Parameters
    ----------
    df : DataFrame
        Input frame.

    Returns
    -------
    Series
    """
    return pd.Series({x: df.loc[x, x] for x in df.index.intersection(df.columns)})


def pd_wide_display(width: int = 200, max_columns: int = 50, max_colwidth: int = 500) -> pd.option_context:
    """
    A context manager for displaying wide DataFrame objects.
    """
    return pd.option_context(
        'display.width', width,
        'display.max_columns', max_columns,
        'display.max_colwidth', max_colwidth,
    )


@assert_types(obj=NDFrame)
def drop_duplicates_by_index(obj: NDFrameT, keep: Literal["first", "last", False] = 'first') -> NDFrameT:
    """
    Drop duplicated indices from a DataFrame or Series.

    Parameters
    ----------
    obj : DataFrame or Series
        Object to work on.

    keep : {"first", "last", False}, default "first"
        - "first": Keep the first occurrence of each duplication.
        - "last": Keep the last occurrence of each duplication.
        - False: Don't keep any of the duplicated indices.

    Returns
    -------
    DataFrame or Series
        View of the original object (no copy), filtered.
    """
    return obj.loc[~obj.index.duplicated(keep=keep)]


@assert_types(obj=NDFrame)
def fillna(obj: NDFrameT, value) -> NDFrameT:
    """
    A more robust version of `fillna`, allowing to fill with any object (including None, lists and dicts),
    or the result of calling a callable.

    Filling with a mutable object will fill with the *same* object.
    To fill with different instances, use a callable to create a new object for each filled N/A.

    Parameters
    ----------
    obj : DataFrame or Series
        Object to fill.

    value : callable, mapping or object
        - Callable: Should take no arguments. Fills with the result
            of calling the callable once for each filled N/A.
        - Mapping: If `obj` is a DataFrame, this is a mapping of column
            names to values (each value can itself be a callable).
            Columns not appearing in the mapping are left as is.
            If `obj` is a Series, regarded as a value to fill.
        - Any other object: Fills with this value.

    Returns
    -------
    DataFrame or Series
        Filled object.
        If `obj` is a series with no N/A values to fill, the same object
        is returned, not a copy.

    Examples
    --------
    >>> df = pd.DataFrame(np.arange(12).reshape(3, 4),
    ...                   columns=['A', 'B', 'C', 'D'])
    >>> df.iloc[1, :] = np.NaN
    >>> df
         A    B     C     D
    0  0.0  1.0   2.0   3.0
    1  NaN  NaN   NaN   NaN
    2  8.0  9.0  10.0  11.0

    Fill with (different instances of) empty lists:
    >>> fillna(df, list)
         A    B    C    D
    0    0    1    2    3
    1   []   []   []   []
    2    8    9   10   11

    Fill column A with None, B with empty lists, C with dictionaries and D with -1:
    >>> fillna(df, {'A': None, 'B': list, 'C': dict, 'D': -1})
          A    B    C     D
    0     0    1    2   3.0
    1  None   []   {}  -1.0
    2     8    9   10  11.0
    """
    if isinstance(obj, pd.DataFrame):
        if isinstance(value, Mapping):
            return pd.concat([
                obj.drop(columns=value.keys()),
                *[fillna(obj[col], val) for col, val in value.items()],
            ], axis=1).reindex_like(obj, copy=False)
        return obj.apply(fillna, args=(value,))
    ind = obj.index[obj.isnull()]
    if not ind.empty:
        obj = obj.copy()
        if not callable(value):
            value = lambda x=value: x
        filler = ind.map(lambda x: value())
        try:
            obj[ind] = filler
        except (TypeError, ValueError):
            obj = obj.astype(object)
            obj[ind] = filler
    return obj


@assert_types(data=pd.DataFrame | pd.Series | np.ndarray | sp.spmatrix)
def sizeof(data: pd.DataFrame | pd.Series | np.ndarray | sp.spmatrix) -> int:
    """
    Calculate the memory consumption of the data, in bytes.

    Parameters
    ----------
    data : DataFrame, Series, numpy array or scipy sparse matrix

    Returns
    -------
    int
    """
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return np.sum(data.memory_usage(index=True, deep=True)).item()
    return data.size * data.dtype.itemsize


@assert_types(orig_data=NDFrame, new_data=NDFrame)
def upsert(orig_data: NDFrameT, new_data: NDFrameT) -> NDFrameT:
    """
    Insert-or-update rows (items) of one DataFrame (Series) onto another.

    Parameters
    ----------
    orig_data : DataFrame or Series
        Data that will be updated.
        There should be no duplicates in the index (not checked).

    new_data : DataFrame or Series (same as `orig_data`)
        Data that will be appended, overriding `orig_data` wherever the index already exists.

    Returns
    -------
    DataFrame or Series (same as input)
        A new object, in which rows or items from `new_data` overwrite or are appended to
        rows or items from `orig_data`. The order of the results is not guaranteed.
    """
    return pd.concat([orig_data[~orig_data.index.isin(new_data.index)], new_data])


@assert_types(obj1=NDFrame, obj2=NDFrame)
def alignable(obj1: NDFrame, obj2: NDFrame) -> bool:
    """
    Test whether two pandas objects can be exactly aligned together, i.e. whether
    their indices are without duplicates and if sorted, will be equal.
    Equal indices are also considered alignable, even if they contain duplicates.

    Parameters
    ----------
    obj1, obj2 : DataFrame or Series
        Objects to test.

    Returns
    -------
    bool
    """
    return obj1.index.equals(obj2.index) or (
            len(obj1) == len(obj2)
            and obj1.index.is_unique
            and obj2.index.is_unique
            and obj1.index.symmetric_difference(obj2.index).empty
    )


def autoload(*arg_names: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    A decorator factory returning a decorator which, before calling the function,
    first considers the value for each of the provided argument names.
    If it is a string, it is `pdload`ed, calling the function with the loaded object instead.

    Parameters
    ----------
    *arg_names : str
        Names of function arguments to load.

    Returns
    -------
    decorator

    See Also
    --------
    pdload : Function used to load each argument.

    Examples
    --------
    >>> @autoload('data')
    ... def func(data: pd.Series):
    ...   ...

    Call the function with an object:
    >>> func(pd.Series([1, 2, 3]))

    Or call it with the name of a file containing the desired object:
    >>> func("pickled_series.pkl")
    """
    def autoload_decorator(func, *args, **kwargs):
        call_args = inspect.signature(func).bind(*args, **kwargs).arguments
        for arg in arg_names:
            val = call_args.get(arg)
            if isinstance(val, str):
                call_args[arg] = pdload(val)
        return func(**call_args)
    return decorator(autoload_decorator)


@assert_types(obj=NDFrame)
def sample_at_most(
        obj: NDFrameT,
        n_or_frac: int | float | None = 1,
        replace: bool = False,
        weights: Hashable | ArrayLike | None = None,
        random_state: RandomState | None = None,
        axis: Literal[0, 1, 'index', 'columns'] | None = None,
) -> NDFrameT:
    """
    Sample items, rows or columns from a pandas object.

    Unlike the object's `sample` method, if `replace` is False, samples *at most* `n_or_frac` items,
    but not more than the size of `obj`.
    If `replace` is True, identical to the regular `sample` method.

    Parameters
    ----------
    obj : DataFrame or Series
        Input object.

    n_or_frac : int or float, default 1
        Sample size.
        - int: Absolute number to sample, but no more than the length of `obj` if `replace` is False.
        - float: Fraction of `axis` items to sample, but no more than 1.0 if `replace` is False.

    replace : bool, default False
        Whether to sample with replacements.

    weights : hashable or array-like, optional
        If given, perform weighted sampling.
        - Hashable: Name of a column of `obj` (if a DataFrame) containing weights.
        - Array-like: The weights themselves.

    random_state : int, array-like, BitGenerator, np.random.RandomState or np.random.Generator, optional
        Allows reproducibility.

    axis : {0 or ‘index’, 1 or ‘columns’}, optional
        Axis to sample.
        Default is the stat axis for given data type (0 for both Series and DataFrames).

    Returns
    -------
    DataFrame or Series
        A new object of same type as `obj`.
    """
    n = frac = None
    if isinstance(n_or_frac, int):
        n = n_or_frac if replace else min(n_or_frac, len(obj))
    elif isinstance(n_or_frac, float):
        frac = n_or_frac if replace else min(n_or_frac, 1.0)
    return obj.sample(n=n, frac=frac, replace=replace, weights=weights, random_state=random_state, axis=axis)


def stack_indices(*indices: pd.Index) -> pd.MultiIndex:
    """
    Stack indices to form a MultiIndex.

    Input indices can be either regular Index or MultiIndex instances (or a mix of both).
    All must have the same length, and their names are preserved.

    Parameters
    ----------
    *indices : Index
        Index or MultiIndex instances to stack.

    Returns
    -------
    MultiIndex
    """
    return pd.MultiIndex.from_arrays(idx.get_level_values(i) for idx in indices for i in range(idx.nlevels))


class IdentitySeries(Mapping):
    """
    A class representing a Series with values equal to the index.

    Values are saved only once.
    If the values are {0, ..., n-1}, then values are not saved at all, but deduced from the indices.

    While not a subclass of Series, reproduces most of the useful Series interface.

    Parameters
    ----------
    items_or_size : int or iterable
        - int: The items are {0, 1, ..., `items_or_size` - 1}.
        - iterable: The items themselves.

    name : hashable, optional
        Name of the series.

    dtype : dtype-like, optional
        Data type of the values.
        When items are retrieved (using __getitem__, loc or iloc), they are converted to this type.
    """
    def __init__(self, items_or_size: Iterable[Hashable] | int = 0, name: Hashable | None = None,
                 dtype: DTypeLike = None):
        if isinstance(items_or_size, int):
            items_or_size = pd.RangeIndex(items_or_size)
        self.index = items_or_size
        self.name = name
        self.dtype = dtype

    @property
    def index(self) -> pd.Index:
        return self._index

    @index.setter
    def index(self, value):
        if not isinstance(value, pd.Index):
            value = pd.Index(value)
        self._index = value

    @property
    def dtype(self) -> np.dtype | None:
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        self._dtype = None if value is None else np.dtype(value)

    def __len__(self) -> int:
        return len(self.index)

    @property
    def shape(self) -> tuple[int]:
        return len(self),

    def __iter__(self):
        return iter(self.index)

    class _ItemAccessor:
        def __init__(self, parent: "IdentitySeries", by_label: bool):
            self.parent = parent
            self.by_label = by_label

        def __getitem__(self, item):
            if self.by_label:
                if isinstance(item, list | np.ndarray | pd.Index):
                    # Several items (but not a slice)
                    ind = self.parent.index.get_indexer(item)
                    if (bad := ind == -1).any():
                        raise KeyError(f"{np.asarray(item)[bad].tolist()} not in index")
                    item = ind
                else:
                    # Single item
                    item = self.parent.index.get_loc(item)
            value = self.parent.index[item]
            if isinstance(value, pd.Index):
                # Several values
                return value.to_numpy(dtype=self.parent.dtype)
            # Single value
            return value if self.parent.dtype is None else self.parent.dtype.type(value)

    @property
    def loc(self):
        return self._ItemAccessor(self, by_label=True)

    @property
    def iloc(self):
        return self._ItemAccessor(self, by_label=False)

    def __getitem__(self, item):
        is_int_slice = isinstance(item, slice) and all(
            isinstance(getattr(item, x), int | None) for x in ('start', 'stop', 'step')
        )
        return self._ItemAccessor(self, by_label=not is_int_slice)[item]


@overload
def canonize_df_and_cols(arg: pd.DataFrame, /, *items: Hashable) -> tuple: ...
@overload
def canonize_df_and_cols(arg: ArrayLike | AnyArrayLike, /, *items: ArrayLike | AnyArrayLike) -> tuple: ...


def canonize_df_and_cols(arg, /, *items):
    """
    Canonize parameters for functions which accept either a DataFrame and some of its column names,
    or just a collection of Series.

    For either input form:
        (1) canonize_df_and_cols(dataframe, *column_names)
        (2) canonize_df_and_cols(*series)

    The return values are:
        dataframe, *column_names

    where for form (1) the output is the same as the input and for (2) `dataframe` is constructed
    from the input series and `column_names` are the corresponding names.
    For form (2), the inputs are first converted to Series if needed.
    """
    if isinstance(arg, pd.DataFrame):
        if not all(c in arg for c in items):
            raise ValueError("Some column names are not found in DataFrame.")
        return arg, *items
    series = [x if isinstance(x, pd.Series) else pd.Series(x) for x in (arg, *items)]
    for s1, s2 in combinations(series, 2):
        if not alignable(s1, s2):
            raise ValueError("Inputs have inconsistent indices.")
    df = pd.concat(series, axis=1)
    return df, *df.columns


def column_stats(obj: NDFrameT, n_workers: int | float = 0) -> NDFrameT:
    """
    Calculate useful statistics on a Series or the columns of a DataFrame.
    Useful for non-numeric data; Complements the `describe` method.

    Statistics calculated:
        - dtype: The types of objects
        - n_unique: Number of unique values (excluding Nulls)
        - n_null: Number of Null values
        - top_count: Value count for the top (non-Null) value
        - top_val: Top (non-Null) value
        - second_val: Second from the top (non-Null) value

    Parameters
    ----------
    obj : DataFrame or Series
        Input pandas object.

    n_workers : int or float, default 0
        If working on a DataFrame, number of workers to use in parallel.
        Default is 0, meaning no parallelization is performed.
        If working on a Series, the parameter is ignored.
        See `ultima.Workforce` for more details.

    Returns
    -------
    DataFrame or Series
        Same type as `obj` with calculated statistics.
    """
    if isinstance(obj, pd.Series):
        vc = obj.value_counts(dropna=False)
        na = pd.isna(vc.index)
        n_null = vc[na].sum()
        vc = vc[~na]
        types = sorted(obj.dropna().apply(lambda x: type(x).__name__).unique())
        return pd.Series(dict(
            dtype=types if len(types) > 1 else types[0] if types else None,
            n_unique=len(vc),
            n_null=n_null,
            top_count=vc.max(),
            top_val=vc.index[0] if not vc.empty else None,
            second_val=vc.index[1] if len(vc) > 1 else None,
        ))
    return (
        papply(obj, column_stats, n_workers=n_workers)
        .transpose()
        .reindex(columns=column_stats(pd.Series(dtype=object)).index, copy=False)
        .astype(dict(n_unique='Int32', n_null='Int32', top_count='Int32'), copy=False)
    )


def unique_row_ids(df: pd.DataFrame) -> pd.Series:
    """
    Assign to each unique row of a DataFrame an ID and return a Series mapping
    each row index to its unique ID.

    Parameters
    ----------
    df : DataFrame
        Input object.

    Returns
    -------
    Series
        For each index of `df`, its unique ID based on row values.

    See Also
    --------
    DataFrame.duplicated : Return boolean Series denoting duplicate rows.
    """
    if df.empty:
        return pd.Series(dtype=np.int64)

    size_hint = min(len(df), 1 << 20)

    def get_unique_ids(values):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            unique_ids, unique_vals = algorithms.factorize(values, size_hint=size_hint)
        return unique_ids.astype("i8", copy=False), len(unique_vals)

    return pd.Series(get_compressed_ids(*df.apply(get_unique_ids).values)[0], index=df.index)


@overload
def iterrows(df: pd.DataFrame, index: Literal[True]) -> SizedIterable[tuple[Hashable, pd.Series]]: ...
@overload
def iterrows(df: pd.DataFrame, index: Literal[False] = False) -> SizedIterable[pd.Series]: ...
@overload
def iterrows(df: pd.DataFrame, index: bool) -> SizedIterable[tuple[Hashable, pd.Series]] | SizedIterable[pd.Series]: ...


def iterrows(df, index=False):
    """
    A convenience wrapper for iterating over the rows of a DataFrame.

    The returned iterator is Sized (has a len), which is convenient for feeding into a progress bar.

    Parameters
    ----------
    df : DataFrame
        Input object.

    index : bool, default False
        Whether to yield just the rows as Series (the default) or tuples of (index, row) like the `iterrows` method.
        Since the index value is the name of each row Series, it is redundant and can sometimes be inconvenient.

    Returns
    -------
    SizedIterable
        An iterable of either rows or (index, row) tuples, which also has a length.
    """
    rows = df.iterrows()
    if not index:
        rows = (series for ind, series in rows)
    return SizedIterable(rows, len(df))
