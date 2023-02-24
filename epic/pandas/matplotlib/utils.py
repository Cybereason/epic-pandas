import numpy as np

from itertools import count
from collections.abc import Iterator

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec


def grid_size(num_items: int, max_columns: int | None = None) -> tuple[int, int]:
    """
    Calculate the number of rows and columns in a balanced rectangular grid.
    Useful for creating a subplot grid given the number of plots.

    Parameters
    ----------
    num_items : int
        The total number of items in the grid.

    max_columns : int, optional
        Maximum number of columns in the grid.

    Returns
    -------
    tuple
        2-tuple of (n_rows, n_columns)
    """
    rows = np.round(np.sqrt(num_items))
    cols = int(np.ceil(num_items / rows))
    if max_columns is not None and cols > max_columns:
        cols = max_columns
        rows = np.ceil(num_items / cols)
    return int(rows), cols


def grid_axes(fig: Figure | None = None, max_columns: int | None = None, resize: bool = False) -> Iterator[Axes]:
    """
    A generator generating axes in a balanced grid, one at a time.

    Will NOT work correctly if any axes is given a colorbar, since doing so replaces the axes
    in the grid with two sub-axes, one for the original axes and one for the colorbar.

    Parameters
    ----------
    fig : Figure, optional
        Figure to populate with axes.
        Default is to use the current figure.

    max_columns : int, optional
        Maximum number of columns in the grid.

    resize : bool, default False
        If True, the figure is resized each time the grid size changes to keep an approximate 3:4 ratio.

    Yields
    ------
    Axes
    """
    if fig is None:
        fig = plt.gcf()
    else:
        fig.clf()
    rows = cols = 0
    spec = None
    for n_axes in count(1):
        if n_axes > rows * cols:
            rows, cols = grid_size(n_axes, max_columns)
            spec = GridSpec(rows, cols, figure=fig)
            for i, ax in enumerate(fig.axes):
                ax.set_subplotspec(spec[np.unravel_index(i, (rows, cols))])
            if resize:
                fig.set_size_inches(cols * 4, rows * 3)
        yield fig.add_subplot(spec[np.unravel_index(n_axes - 1, (rows, cols))])
