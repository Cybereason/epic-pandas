import numpy as np
import pandas as pd

from typing import Hashable
from numpy.typing import NDArray, ArrayLike

from scipy.stats import entropy
from scipy.ndimage import gaussian_filter1d

from .numpy import asnpbytearray, ExtendedArrayLike


def byte_histogram(data: ExtendedArrayLike) -> NDArray[np.int64]:
    """
    Compute a histogram of the byte values, with 256 bins.

    Parameters
    ----------
    data : array-like
        Input data.

    Returns
    -------
    ndarray of ints
        Length is 256.
    """
    return np.bincount(asnpbytearray(data), minlength=256)


def shannon_entropy(data: ExtendedArrayLike) -> float:
    """
    Calculate the Shannon entropy of the data.

    Parameters
    ----------
    data : array-like
        Input data.

    Returns
    -------
    float
        Shannon entropy, in the range 0.0 - 8.0.
    """
    return entropy(byte_histogram(data), base=2)


def kullback_leibler(data1: ExtendedArrayLike, data2: ExtendedArrayLike) -> float:
    """
    Compute the Kullback-Leibler divergence between two data sets.

    Parameters
    ----------
    data1, data2 : array-like
        Input data.
        Note that KL is not symmetric w.r.t. interchanging them.

    Returns
    -------
    float
        The KL divergence between the byte histograms of `data1` and `data2`.

    Notes
    -----
    We add a small constant to each bin of the histogram of `data2`, in order to avoid
    an infinite result. This introduces a small skew of the result, more pronounced for
    a smaller data set.
    """
    return entropy(byte_histogram(data1), byte_histogram(data2) + 1e-7, base=2)


def smooth_with_gaussian(data: ArrayLike, sigma: float = 0) -> NDArray[np.float64]:
    """
    Smooth the input data with a Gaussian kernel.

    Parameters
    ----------
    data : array-like
        Data to smooth.
        Will be converted to 1D if not already so.

    sigma : float, default 0
        Standard deviation for the Gaussian kernel.

    Returns
    -------
    ndarray of floats
        Smoothed data.
    """
    if sigma < 0:
        raise ValueError(f"`sigma` cannot be negative; got {sigma}")
    data = np.ravel(data).astype(np.float64, copy=False)
    return gaussian_filter1d(data, sigma) if sigma > 0 else data


def weighted_average(df: pd.DataFrame, data_col: Hashable, weights_col: Hashable) -> float:
    """
    Compute the weithed average of a column of a DataFrame, where the weights are given in another column.

    Parameters
    ----------
    df : DataFrame
        Input data.

    data_col : hashable
        Name of the column in `df` containing the data to average.

    weights_col : hashable
        Name of the column in `df` containing the weights of the data.

    Returns
    -------
    float
        Weighted average of the data.
    """
    return np.average(df[data_col], weights=df[weights_col])
