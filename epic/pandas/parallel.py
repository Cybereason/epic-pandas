import pandas as pd

from typing import Literal
from functools import partial
from collections.abc import Callable
from pandas.core.generic import NDFrame

from ultima import Workforce
from ultima.backend import BackendArgument
from epic.common.assertion import assert_types

from .numpy import gen_slices


@assert_types(obj=NDFrame)
def papply(
        obj: NDFrame,
        func: Callable,
        axis: Literal[0, 1, 'index', 'columns'] = 0,
        n_workers: int | float = -1,
        backend: BackendArgument = "multiprocessing",
        batch_size: int | None = None,
        **kwargs,
) -> NDFrame:
    """
    Apply a function on chunks of a pandas object in parallel.

    Parameters
    ----------
    obj : DataFrame or Series
        The object to which to apply the function.

    func : callable
        The function to apply.

    axis : {0 or 'index', 1 or 'columns'}, default 0
        Same as for DataFrame.apply. Ignored if `obj` is a Series.

    n_workers : int or float, default -1
        Number of workers (subprocesses or threads) to use. Default (-1) is the number of processes on the machine.
        If 0, no parallelization is performed.
        When using the "threading" backend, a positive int must be provided.
        See `ultima.Workforce` for more details.

    backend : 'multiprocessing', 'threading', 'inline' or other options, default 'multiprocessing'
        The backend to use for parallelization.
        See `ultima.Workforce` for more details.

    batch_size : int, optional
        When given, each task will include this many items.
        Otherwise, the input is divided into even batches, one per worker.

    **kwargs :
        Sent to the object's apply method.

    Returns
    -------
    DataFrame or Series
        Type determined based on the result of `func`, as if regular `apply` was called.
    """
    if isinstance(obj, pd.DataFrame):
        if axis == 'index':
            axis = 0
        elif axis == 'columns':
            axis = 1
        if axis not in (0, 1):
            raise ValueError(f"invalid value for `axis`: {axis}")
        kwargs['axis'] = axis
        actual_axis = 1 - axis  # damn pandas and their inconsistent use of axis!!!
    else:
        actual_axis = 0
    n_items = obj.shape[actual_axis]
    with Workforce(backend=backend, n_workers=n_workers) as workforce:
        n_jobs = max(workforce.n_workers, 1) if batch_size is None else None
        results = list(workforce.map(
            func=partial(lambda chunk, **kw: chunk.apply(func, **kw), **kwargs),
            inputs=(obj.iloc[(slice(None), s) if actual_axis else s] for s in gen_slices(n_items, n_jobs, batch_size)),
            ordered=True,
            errors='raise',
        ))
    if not results:
        return type(obj)(dtype=getattr(obj, 'dtype', None))
    if all(isinstance(x, pd.Series) for x in results):
        concat_axis = 0
    elif all(isinstance(x, pd.DataFrame) for x in results):
        concat_axis = actual_axis
    else:
        raise ValueError(
            "Expected results to be either all of type Series or all of type DataFrame; "
            f"got {set(type(x).__name__ for x in results)}."
        )
    return pd.concat(results, axis=concat_axis)
