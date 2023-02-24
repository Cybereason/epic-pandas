import numpy as np
import pandas as pd

from toolz import compose
from functools import partial
from typing import TypeVar, Literal, Any
from pandas._typing import Dtype, IndexLabel
from collections.abc import Iterable, Mapping, Callable, Hashable

from ultima import ultimap, Args
from ultima.backend import BackendArgument
from epic.common.general import pass_none

KT = TypeVar('KT', bound=Hashable)
DT = TypeVar('DT')
VT = TypeVar('VT')


def _process_single(item, keyfunc, subdict_keyfunc, transform):
    if keyfunc is None:
        try:
            key, data = item
        except (TypeError, ValueError):
            raise ValueError("When items are not (key, data) pairs, a key-making function must be provided.")
    else:
        key = keyfunc(item)
        data = item
    if isinstance(data, dict):
        processed = {}
        for field, value in data.items():
            if isinstance(value, dict) and subdict_keyfunc is not None:
                for subfield, subvalue in value.items():
                    processed[subdict_keyfunc((field, subfield))] = subvalue
            elif value is not None:
                processed[field] = value
        data = processed
    if transform is not None:
        data = transform(data)
        if data is None:
            # Sample is to be ignored
            key = None
    if not isinstance(data, dict):
        # Sample should be empty (full of NaNs)
        data = {}
    return key, data


def df_from_iterable(
        iterable: Iterable[tuple[KT, DT]] | Iterable[DT],
        *,
        collapse_subdict: Literal['multilevel', 'join', None] = None,
        transform: Callable[[DT], dict[Hashable, VT] | None] | None = None,
        keyfunc: Callable[[DT], KT] | None = None,
        sparse_values: Mapping[Hashable, VT] | None = None,
        dtypes: Mapping[Hashable, Dtype] | None = None,
        index_name: IndexLabel | None = None,
        n_workers: int | float = 1,
        backend: BackendArgument = "multiprocessing",
        ordered: bool = False,
        buffering: int | None = None,
) -> pd.DataFrame:
    """
    Read data from an iterable and build a DataFrame efficiently.

    If working in parallel (n_jobs!=1) and ordered=False, the order of the samples may not be kept.
    If a key is None, the sample is skipped.

    Parameters
    ----------
    iterable :
        An iterable yielding (key, datum) pairs, or just data.

    collapse_subdict : {'multilevel', 'join', None}, default None
        How to treat sub-dictionaries.
        'multilevel': Create a tuple (key, subkey) item for each subkey and add to the main dict.
                      If all keys are such tuples, a multilevel index is created for the DataFrame.
        'join':       Create a key_subkey item for each subkey.
        None:         Do nothing.

    transform : callable, optional
        A transformation function for each datum (datum -> dict[column_name, value]).
        If the datum is a dict, will be applied *after* each sub-dict (if exists) has been collapsed.
        If the function returns None, this sample will be skipped.

    keyfunc : callable, optional
        A function creating a key from each datum.
        Must be provided if (and only if) the iterable yields data and not (key, datum) pairs.

    sparse_values : dict, optional
        Specifies the columns that should be sparse, and maps them to their sparse fill values.

    dtypes : dict, optional
        If provided, specifies the exact dtypes of specific columns.

    index_name : string or sequence of strings, optional
        Name of the index.
        In case of a MultiIndex, should be a sequence with length matching the number of levels.

    n_workers : int or float, default 1
        Number of workers (subprocesses or threads) to use.
        If 0 or 1, no parallelization is performed.
        When using the "threading" backend, a positive int must be provided.
        See `ultima.Workforce` for more details.

    backend : 'multiprocessing', 'threading', 'inline' or other options, default 'multiprocessing'
        The backend to use for parallelization.
        See `ultima.Workforce` for more details.

    ordered : bool, default False
        Whether the order of the outputs should correspond to the order of `iterable`.

    buffering : int, optional
        Limit to the number of tasks created simultaneously.

    Returns
    -------
    DataFrame
    """
    match collapse_subdict:
        case 'multilevel':
            subdict_keyfunc = tuple
        case 'join':
            subdict_keyfunc = '_'.join
        case None:
            subdict_keyfunc = None
        case _:
            raise ValueError(f"Unexpected value for `collapse_subdict`: {collapse_subdict}.")
    if sparse_values is None:
        sparse_values = {}
    if dtypes is None:
        dtypes = {}
    if n_workers == 1:
        n_workers = 0
    idx = []
    data = {}
    for key, datum in ultimap(
            partial(_process_single, keyfunc=keyfunc, subdict_keyfunc=subdict_keyfunc, transform=transform),
            map(Args, iterable),
            backend=backend,
            n_workers=n_workers,
            ordered=ordered,
            buffering=buffering,
    ):
        if key is None:
            continue
        for col, val in datum.items():
            if col in sparse_values:
                fill_value = sparse_values[col]
                if col not in data:
                    data[col] = [], []
                if val != fill_value and not (val is fill_value is np.NaN):
                    data[col][0].append(len(idx))
                    data[col][1].append(val)
            else:
                if col not in data:
                    data[col] = [np.NaN] * len(idx)
                data[col].append(val)
        for col in set(data).difference(datum, sparse_values):
            data[col].append(np.NaN)
        idx.append(key)
    index = pd.Index(idx, name=index_name)
    for col in data:
        if col in sparse_values:
            data[col] = pd.SparseArray(
                data=data[col][1],
                sparse_index=pd._libs.sparse.IntIndex(len(idx), data[col][0]),
                fill_value=sparse_values[col],
                dtype=dtypes.get(col),
            )
        elif col in dtypes:
            data[col] = pd.Series(data[col], dtype=dtypes[col], index=index, copy=False)
    return pd.DataFrame(data=data, index=index)


def series_from_iterable(
        iterable: Iterable[tuple[Hashable, DT]] | Iterable[DT],
        *,
        transform: Callable[[DT], Any | None] | None = None,
        **kwargs,
) -> pd.Series:
    """
    Read data from an iterable and build a Series.

    Allows for a lot of flexibility in data transformation and parallelization by utilizing
    the mechanisms of `df_from_iterable`.

    Parameters
    ----------
    iterable :
        An iterable yielding (key, datum) pairs, or just data.

    transform : callable, optional
        A transformation function for each datum (datum -> value).
        If the datum is a dict, will be applied *after* each sub-dict (if exists) has been collapsed.
        If the function returns None, this sample will be skipped.

    **kwargs :
        Sent to `df_from_iterable` as is.

    Returns
    -------
    Series

    See Also
    --------
    df_from_iterable : Create a DataFrame from an iterable.
    """
    make_val = pass_none(lambda v: {'value': v})
    return df_from_iterable(
        iterable,
        transform=make_val if transform is None else compose(make_val, transform),
        **kwargs,
    )['value']
