import io
import numpy as np

from os import PathLike
from pandas import isna
from collections.abc import Iterator
from pandas.core.dtypes.inference import is_scalar
from numpy.typing import ArrayLike, DTypeLike, NDArray
from typing import TypeVar, TypeGuard, Literal, overload

T = TypeVar('T')
S = TypeVar('S')
ExtendedArrayLike = str | bytes | bytearray | ArrayLike | PathLike | io.IOBase


def isnan(array: ArrayLike) -> NDArray | bool:
    """
    A more robust version of numpy.isnan.

    Checks whether the elements of an array are NaN, but works also for non-numerical dtypes.
    If given a scalar, returns a single boolean.

    Parameters
    ----------
    array : array-like
        Array to test.

    Returns
    -------
    Boolean array or bool
    """
    array = np.asarray(array)
    if np.issubdtype(array.dtype, np.number):
        return np.isnan(array)
    result = array.astype(str) == str(np.NaN)
    return result if result.shape else result.item()


def isnullscalar(obj) -> TypeGuard[float | None]:
    """
    Check whether an object is a scalar NaN or None.

    Parameters
    ----------
    obj : object
        Object to test.

    Returns
    -------
    bool
    """
    return is_scalar(obj) and isna(obj)


def fillnullscalar(obj: T, default: S) -> T | S:
    """
    If the given object is a scalar NaN or None, replace it with the default.
    Otherwise, return the given object.

    Parameters
    ----------
    obj : object
        Object to test.

    default : object
        Will be returned if `obj` is NaN or None.

    Returns
    -------
    object
    """
    return default if isnullscalar(obj) else obj


def asnparray(obj: ExtendedArrayLike, dtype: DTypeLike = None) -> NDArray:
    """
    A convenience wrap around loading an array from a file or a buffer.
    Interpretation of the input depends on its type.

    Parameters
    ----------
    obj : str, bytes, bytearray, array-like, path-like or opened file
        Behavior depends on the type:
        - str, path-like or opened file: A filename or file to read using numpy.fromfile.
        - bytes or bytearray: A buffer to read using numpy.frombuffer.
        - array-like: Converted to an array using numpy.asarray.

    dtype : dtype-like, optional
        Data type for array.

    Returns
    -------
    ndarray
    """
    if isinstance(obj, str | PathLike | io.IOBase):
        return np.fromfile(obj, dtype=dtype)
    if isinstance(obj, bytes | bytearray):
        return np.frombuffer(obj, dtype=dtype)
    return np.asarray(obj, dtype=dtype)


def asnpbytearray(obj: ExtendedArrayLike) -> NDArray[np.uint8]:
    """
    Load the input from a file or a buffer and return an array of dtype uint8.
    Interpretation of the input depends on its type.

    Parameters
    ----------
    obj : str, bytes, bytearray, array-like, path-like or opened file
        Behavior depends on the type:
        - str, path-like or opened file: A filename or file to read using numpy.fromfile.
        - bytes or bytearray: A buffer to read using numpy.frombuffer.
        - array-like: Converted to an array using numpy.asarray.

    Returns
    -------
    ndarray
        Array of bytes, with dtype uint8.

    See Also
    --------
    asnparray : Read input with any dtype.
    """
    return asnparray(obj, dtype=np.uint8)


def dropna(array: ArrayLike, axis: int | None = None, how: Literal['any', 'all'] = 'any') -> NDArray:
    """
    Remove NaNs from an array.

    Parameters
    ----------
    array : array-like
        Input array.

    axis : int, optional
        If provided, consider values along this axis (see `how` below).
        Can also be negative (counted from the end).
        Otherwise, the array is flattened and the `how` parameter is ignored.

    how : {"any", "all"}, default "any"
        Entries along the `axis` are removed if any or all of the values are NaN.

    Returns
    -------
    ndarray
        Input array with NaNs removed.

    Raises
    ------
    ValueError
        If `axis` is not valid (either positive or negative) for the number of dimensions of `array`.
    """
    array = np.asarray(array)
    isna = isnan(array)
    if axis is None:
        return array[~isna]
    if not -array.ndim <= axis < array.ndim:
        raise ValueError(f"invalid axis {axis} for a {array.ndim}-dimensional array")
    if axis < 0:
        axis += array.ndim
    assert how in ('any', 'all')
    return array.take(
        np.nonzero(~getattr(isna, how)(axis=tuple(x for x in range(array.ndim) if x != axis)))[0],
        axis=axis,
    )


def unique_destroy(array: ArrayLike) -> NDArray:
    """
    Same as numpy.unique (without the extra parameters), but, in most cases:
    - Does not copy the data to a new internal array.
    - Leaves the input array sorted.

    If you don't care about the input array, this may be more efficient than numpy.unique.
    """
    array = np.reshape(array, -1)
    array.sort()  # in place!
    mask = np.empty(array.size, dtype=np.bool_)
    mask[:1] = True
    np.not_equal(array[1:], array[:-1], out=mask[1:])
    return array[mask]


@overload
def split_with_overlap(array: ArrayLike, length: int, overlap: int = 0,
                       partials: Literal[True] = True) -> np.ma.MaskedArray: ...
@overload
def split_with_overlap(array: ArrayLike, length: int, overlap: int, partials: Literal[False]) -> NDArray: ...
@overload
def split_with_overlap(array: ArrayLike, length: int, overlap: int,
                       partials: bool) -> NDArray | np.ma.MaskedArray: ...
def split_with_overlap(array, length, overlap=0, partials=True):
    """
    Split an array to segments of a certain length, allowing overlaps between segments.

    This is done efficiently by returning a read-only view over the array. No data is being copied.
    The result is a 2D array, where each row is a segment.

    If `partials` is True (the default), partial last segments are included. In this case, a MaskedArray
    is returned, and incomplete segments at the end of the array have their redundant values masked out.

    Parameters
    ----------
    array : array-like
        The array to split.
        If it is not 1D, items will be split as-if the array is 1D, in row-major order.

    length : int
        The length of each segment.

    overlap : int (default 0)
        The number of elements common to each pair of neighboring segments.

    partials : bool (default True)
        Should partial segments at the end of the array be included.

    Returns
    -------
    2D MaskedArray if `partials` is True, otherwise a 2D array.
    """
    array = np.ravel(array)
    skip = length - overlap
    n_chunks = int(np.ceil(array.size / skip))
    remainder = n_chunks * skip - array.size + overlap
    if not partials:
        n_chunks = max(n_chunks - int(np.ceil(remainder / skip)), 0)
    strided = np.lib.stride_tricks.as_strided(
        array,
        shape=(n_chunks, max(length, 0)),
        strides=(skip * array.itemsize, array.itemsize),
        writeable=False,
    )
    if not partials:
        return strided
    strided = strided.view(np.ma.MaskedArray)
    for row, rem in zip(range(n_chunks - 1, -1, -1), range(remainder, 0, -skip)):
        strided[row, -rem:] = np.ma.masked
    return strided


def gen_slices(n_items: int, n_slices: int | None = None, slice_size: int | None = None) -> Iterator[slice]:
    """
    Generate slices fit for slicing sequences into mutually-exclusive batches.

    If the number of slices is given, the slices generated are as even in size as possible.
    If the size of each slice is given, it determines the number of slices generated.
    Must provide exactly one of these parameters.

    Parameters
    ----------
    n_items : int
        Total number of items.

    n_slices : int, optional
        Number of slices to generate.
        Generated slices are as even in size as possible.

    slice_size : int, optional
        Size of each slice.
        Number of slices generated is determined automatically.
        The last slice may be smaller.

    Yields
    ------
    slice
    """
    if n_items < 0:
        raise ValueError(f"`n_items` must be non-negative; got {n_items}.")
    if slice_size is None:
        if n_slices is None:
            raise ValueError("Must provide either `n_slices` or `slice_size`")
        if n_slices < 0:
            raise ValueError(f"`n_slices` must be non-negative; got {n_slices}.")
        if n_slices == 0:
            return
        slice_size, surplus = divmod(n_items, n_slices)
    elif n_slices is not None:
        raise ValueError("Must provide either `n_slices` or `slice_size`, but not both.")
    elif slice_size <= 0:
        raise ValueError(f"`slice_size` must be positive; got {slice_size}.")
    else:
        surplus = 0
    start = 0
    surplus_counter = 0
    while start < n_items:
        end = start + slice_size
        if surplus_counter < surplus:
            end += 1
        yield slice(start, end)
        surplus_counter += 1
        start = end
