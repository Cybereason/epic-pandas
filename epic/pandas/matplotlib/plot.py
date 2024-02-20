import numpy as np
import pandas as pd

from typing import Literal, cast, TypeAlias
from numpy.typing import ArrayLike, NDArray
from epic.common.general import coalesce, is_iterable, to_list
from collections.abc import Callable, Mapping, Hashable, Collection, Iterable

from matplotlib.axes import Axes
from matplotlib import pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.units import registry as mpl_units_registry
from matplotlib.colors import Colormap, LogNorm, ListedColormap, BoundaryNorm

from ..utils import canonize_df_and_cols
from .colors import sequential_cmap, ColorSpec

Rotation: TypeAlias = float | Literal['vertical', 'horizontal'] | None
Ticks: TypeAlias = Literal['full', 'auto'] | int | float | Iterable[int] | None


def logxplot(series: pd.Series, **kwargs) -> Axes:
    """
    Plot a Series with a logarithmic x-axis.

    A data point of 0 is prepended to the data so that it would map to -infinity
    instead of the first actual data point.

    Parameters
    ----------
    series : Series
        Series to plot.

    **kwargs :
        Sent to Series.plot method.
        If `logx` is provided, it is ignored.

    Returns
    -------
    Axes
    """
    kwargs['logx'] = True
    return pd.concat([pd.Series(0), series]).plot(**kwargs)


def pie_chart(
        data: pd.Series,
        *,
        threshold: float | None = None,
        counterclock: bool = False,
        startangle: float = 90,
        autopct: str | Callable[[float], str] | None = '%.1f%%',
        title: str | None = None,
        title_y: float = 1.05,
        others_name: str = 'Others',
        others_color: ColorSpec = 'grey',
        cmap: str | Colormap | None = 'Set1',
        colors: Mapping[Hashable, ColorSpec] | None = None,
        ax: Axes | None = None,
        **kwargs,
) -> Axes:
    """
    A convenience function for plotting pie charts.

    Parameters
    ----------
    data : Series
        Data to plot.

    threshold : float, optional
        All values below this threshold will be grouped together to form the "others" category.
        Useful to avoid clutter with many small categories.

    counterclock : bool, default False
        Whether to arrange the categories counterclockwise.

    startangle : float, default 90
        Starting angle of the first category relative to the x-axis.

    autopct : string or callable or None, default '%.1f%%'
        String format for each value (in percentages), or a callable for coverting each value to a string.

    title : string, optional
        Title for the plot.

    title_y : float, default 1.05
        Vertical position of the title.

    others_name : string, default "Others"
        If `threshold` is given and the "others" category is created, this is its label.

    others_color : color spec, default "grey"
        If `threshold` is given and the "others" category is created, this is its color.

    cmap : string, Colormap or None, default "Set1"
        Colormap from which to draw colors for the wedges.
        Only used if `colors` is not provided.

    colors : mapping of data index to color specs, optional
        The color for each category.
        If not provided, `cmap` is used.

    ax : Axes, optional
        Axes on which to plot.

    **kwargs :
        Sent to plotting function.

    Returns
    -------
    Axes
    """
    if ax is None:
        ax = plt.subplots()[1]
    if threshold is not None:
        others = data <= threshold
        others_sum = data[others].sum()
        data = data[~others]
        if colors is None:
            colors = zip(data.index, plt.get_cmap(cmap, len(data))(range(len(data))))
        if not isinstance(colors, Mapping):
            colors = dict(colors)
        data.at[others_name] = others_sum
        colors[others_name] = others_color
    if colors is None:
        kwargs['cmap'] = cmap
    else:
        if not isinstance(colors, Mapping):
            colors = dict(colors)
        if not set(data.index) <= set(colors):
            raise ValueError("Invalid `colors` argument: there are missing colors.")
        kwargs['colors'] = pd.Series(colors)[data.index].values
    data.plot.pie(label='', counterclock=counterclock, startangle=startangle, autopct=autopct, ax=ax, **kwargs)
    if title is not None:
        ax.set_title(title, y=title_y)
    ax.axis('equal')
    return ax


def two_level_pie(
        dataframe: pd.DataFrame,
        category: Hashable,
        subcategory: Hashable,
        *,
        weight: Hashable | None = None,
        sort_categories: bool = True,
        ax: Axes | Collection[Axes] | None = None,
        threshold_pct: float | None = None,
        cmap: str | Colormap | None = 'Set1',
        startangle: float = 90,
) -> None:
    """
    Plot a pie chart for data with categories and sub-categories.
    The function computes the histogram by itself.
    The chart consists of an outer pie with categories, and an inner pie with the subdivision.

    Parameters
    ----------
    dataframe : DataFrame
        Data to plot.
        The data will be grouped by the categories and sub-categories and the number of values
        in each group counted to get the sizes of the wedges.

    category : hashable
        Column name for the categories.

    subcategory : hashable
        Column name for the sub-categories.

    weight : hashable, optional
        Column name containing weights for the different values.
        If not given, all values have the same weight.

    sort_categories: bool, default True
        Whether to sort the categories by their sizes.
        If False, a sorting scheme that "looks good" is attempted.

    ax : Axes or a collection of two Axes, optional
        Axes on which to plot.
        If a collection of Axes, on the first only the categories will be plotted, and on the
        second the full two-level pie.

    threshold_pct : float, optional
        Percentage in [0, 100] below which items will be grouped together under "Others"
        to prevent visual clutter.

    cmap : string, Colormap or None, default "Set1"
        Colormap from which to draw colors for the wedges.

    startangle : float, default 90
        Starting angle of the first category relative to the x-axis.

    Returns
    -------
    None
    """
    grp = dataframe.groupby([category, subcategory])
    if weight is None:
        df = grp.size()
    else:
        df = grp[weight].sum()
        df = df[df > 0]
    df: pd.DataFrame = df.reset_index()
    df.columns = ['category', 'subcategory', 'count']
    df['ratio'] = df['count'] / df['count'].sum()
    threshold = coalesce(threshold_pct, -1) / 100

    def _unify_items(data: pd.DataFrame) -> pd.DataFrame:
        m = data['ratio'] <= threshold
        if not m.any():
            return data
        others = data[m].sum()
        others.name = data[m].index[0]
        others.category = data.iloc[0].category
        others.subcategory = "Others"
        return data.loc[~m].append(others)

    unified = df.groupby('category').apply(_unify_items)
    if isinstance(unified.index, pd.MultiIndex):
        unified.index = unified.index.droplevel()
    grp = unified.groupby('category')['count']
    if sort_categories:
        sortby = grp.transform('sum')
    else:
        # Try to put small categories with many subcategories next to large categories with few subcategories
        score = grp.transform(lambda x: x.sum() / x.count())
        vals = set(score)

        def _zigzag_perm(length):
            r = range(length)
            z = list(zip(r, reversed(r)))
            z = sum(z[:(length + 1) // 2], ())
            if length & 1:
                z = z[:-1]
            return np.argsort(z)

        sortby = score.map(dict(zip(sorted(vals), _zigzag_perm(len(vals)))))
    grp = unified.loc[sortby.sort_values().index].groupby('category', sort=False)
    inner = grp.agg([np.sum, np.size])['count']
    inner['color'] = list(plt.cm.get_cmap(cmap, len(inner))(range(len(inner)))[::-1, :3])
    outer = grp.apply(lambda x: x.sort_values('count'))
    outer['color'] = sum((sequential_cmap(v['color'], n_colors=v['size'])[::-1] for k, v in inner.iterrows()), [])
    if ax is None:
        ax = plt.subplots()[1]
    if isinstance(ax, Collection) and len(ax) == 2:
        cat_ax, subcat_ax = ax
        inner['label'] = inner.apply(lambda x: f"{x.name} ({x['sum']})", axis=1)
        cat_ax.pie(inner['sum'], startangle=startangle, colors=inner['color'], labels=inner['label'], autopct='%.1f%%')
        cat_ax.axis('equal')
        outer['label'] = outer.apply(lambda x: f"{x['subcategory']} ({x['count']})", axis=1)
        subcat_ax.pie(outer['count'], startangle=startangle, colors=outer['color'], labels=outer['label'],
                      autopct='%.1f%%')
        subcat_ax.pie(inner['sum'], startangle=startangle, colors=inner['color'], radius=.75)
        subcat_ax.axis('equal')
    else:
        inner['label'] = grp.apply(
            lambda x: '\n'.join(
                x.sort_values('count', ascending=False).apply(
                    lambda y: '{subcategory} {ratio:.1%} ({count})'.format(**y),
                    axis=1,
                )
            )
        )
        inner['label'] = inner.apply(
            lambda x: (
                    "{:^%d}\n\n{}" % max(map(len, x['label'].split('\n')))
            ).format(
                "{} ({})".format(x.name, x['sum']), x['label'],
            ),
            axis=1,
        )
        ax.pie(outer['count'], startangle=startangle, colors=outer['color'])
        ax.pie(inner['sum'], startangle=startangle, colors=inner['color'], radius=.75, autopct='%.1f%%',
               labels=inner['label'], labeldistance=1.8, pctdistance=.7)
        ax.axis('equal')


def plot_2d_hist(
        arg, /, *args,
        bins: Literal['log'] | int | ArrayLike | None = 'log',
        figsize: tuple[float, float] | None = None,
        ax: Axes | None = None,
        cmap: str | Colormap | None = 'viridis',
        corr_title: bool = True,
        **kwargs,
) -> Axes:
    """
    Plot a 2D histogram of some data using hexagons.

    Parameters
    ----------
    arg, *args:
        Positional-only parameters.
        Either:
            dataframe : DataFrame
                Input frame.

            x_column_name, y_column_name : hashable
                Column names for x- and y-data.

        or:
            x_data, y_data : array-like
                Data to plot.

    bins : 'log', int, array-like or None, default 'log'
        Binning scheme for the color values.

    figsize : two-tuple of floats, optional
        Figure size.
        Only used if `ax` is not given.

    ax : Axes, optional
        Axes on which to plot.

    cmap : string, Colormap or None, default "viridis"
        Colormap to use.

    corr_title : bool, default True
        If True, adds a title to the plot showing the correlation coefficient of the two data series.

    **kwargs :
        Sent to plotting function.

    Returns
    -------
    Axes
    """
    df, x, y = canonize_df_and_cols(arg, *args)
    if ax is None:
        ax = plt.subplots(figsize=figsize)[1]
    df.plot.hexbin(x, y, ax=ax, bins=bins, cmap=cmap, **kwargs)
    if corr_title:
        ax.set_title(fr'$\rho = {df[[x, y]].corr().loc[x, y]:.3g}$')
    return ax


def group_hist(
        arg, /, *args,
        bins: int | ArrayLike | Literal['auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', 'sqrt'] = 'doane',
        histtype: Literal['bar', 'barstacked', 'step', 'stepfilled'] = 'stepfilled',
        alpha: float = 0.5,
        density: bool = True,
        logy: bool = False,
        total: bool = False,
        ax: Axes | None = None,
        **kwargs,
) -> Axes:
    """
    Plot histograms of some data grouped by a grouper.
    All histograms are plotted together in the same Axes.

    Parameters
    ----------
    arg, *args:
        Positional-only parameters.
        Either:
            dataframe : DataFrame
                Input frame.

            data_column_name : hashable
                Column containing the data to plot.

            by_column_name : hashable
                Column by which to group.

        or:
            data : array-like
                Data to plot.

            by : array-like
                Data by which to group.

    bins : int, array-like or string, default 'doane'
        Binning scheme for the histograms.
        Binning is consistent for all the groups, computed on all the data together.
        If a string, mush be one of the binning strategies supported by `numpy.histogram_bin_edges`.

    histtype : {'bar', 'barstacked', 'step', 'stepfilled'}, default 'stepfilled'
        The type of all histograms.

    alpha : float, default 0.5
        Transparency factor in the range [0, 1].

    density : bool, default True
        Whether to normalize each group histogram by its total count.

    logy : bool, default False
        Whether to use a logarithmic scaling for the y-axis.

    total : bool, default False
        Whether to also include a histogram of all the data, ungrouped.

    ax : Axes, optional
        Axes on which to plot.

    **kwargs :
        Additional kwargs sent to `Series.hist`.

    Returns
    -------
    Axes
    """
    df, data_col, by_col = canonize_df_and_cols(arg, *args)
    df = df[[data_col, by_col]].dropna()
    # Convert to numbers if needed, so that numpy histogram tools can process the data.
    # This happens automatically when plotting, but not when calculating using numpy without
    # plotting, like we're doing here.
    if (converter := mpl_units_registry.get_converter(df[data_col])) is not None:
        df[data_col] = converter.convert(df[data_col], None, None)
    all_bins = np.histogram_bin_edges(df[data_col], bins=bins)
    for by_value, group in df.groupby(by_col)[data_col]:
        ax = group.hist(
            bins=all_bins, histtype=histtype, alpha=alpha, density=density, label=str(by_value), ax=ax, **kwargs,
        )
    if total:
        ax = df[data_col].hist(
            bins=all_bins, histtype=histtype, alpha=alpha, density=density, label='TOTAL', ax=ax, **kwargs,
        )
    if logy:
        ax.set_yscale('log')
    ax.legend(loc='best')
    ax.set_xlabel(data_col)
    ax.set_ylabel('Frequency')
    return ax


def group_bar_hist(
        arg, /, *args,
        rot: Rotation = 0,
        sort: bool | Hashable | list[Hashable] | pd.Index | NDArray = True,
        head: int | None = None,
        logy: bool = False,
        **kwargs,
) -> Axes:
    """
    Plot bar histograms of discrete data grouped by a grouper.
    All histograms are plotted together in the same Axes.

    Parameters
    ----------
    arg, *args:
        Positional-only parameters.
        Either:
            dataframe : DataFrame
                Input frame.

            data_column_name : hashable
                Column containing the data to plot.

            by_column_name : hashable
                Column by which to group.

        or:
            data : array-like
                Data to plot.

            by : array-like
                Data by which to group.

    rot : float, {'vertical', 'horizontal'} or None, default 0
        Rotation angle of bar labels.
        None is the same as 'vertical', and the same as 90.

    sort : bool, hashable, list, Index or ndarray, default True
        Sorting scheme for the bars:
        - False: Don't sort.
        - True: Sort by the mean across all groups.
        - Hashable: This should be one of the grouping values. Sort by this group.
        - List, Index or ndarray: Explicit order of the data values.

    head : int, optional
        If given, only this many top bars (after sorting) are plotted.

    logy : bool, default False
        Whether to use a logarithmic scaling for the y-axis.

    **kwargs :
        Additional kwargs sent to `DataFrame.plot.bar`.

    Returns
    -------
    Axes
    """
    df, data_col, by_col = canonize_df_and_cols(arg, *args)
    counts = df.groupby(by_col)[data_col].value_counts(normalize=True).mul(100).unstack(level=0)
    if isinstance(sort, list | pd.Index | np.ndarray):
        counts = counts.loc[sort]
    elif sort:
        if isinstance(sort, bool):
            # sort by the mean
            counts = counts.iloc[np.argsort(counts.fillna(0).mean(axis=1))[::-1]]
        else:
            # sort by one of the categories
            counts.sort_values(sort, ascending=False, inplace=True)
    if head is not None:
        counts = counts.head(head)
    ax = counts.plot.bar(rot=rot, **kwargs)
    ax.set_ylabel('Frequency [%]')
    if logy:
        ax.set_yscale('log')
    return ax


def imshow(
        data: ArrayLike | pd.DataFrame,
        *,
        cmap: str | Colormap | None = 'viridis',
        clim: tuple[float, float] | None = None,
        colorbar: bool = True,
        rot: Rotation = 30,
        ax: Axes | None = None,
        **kwargs,
) -> Axes:
    """
    Show an image.
    This is a thin wrap around `imshow`, with interpolation set to "nearest".

    Parameters
    ----------
    data : array-like or DataFrame
        Image to show.

    cmap : string, Colormap or None, default "viridis"
        Colormap to use.

    clim : 2-tuple of floats, optional
        Color limits of the image.

    colorbar : bool, default True
        Whether to add a colorbar.

    rot : float, {'vertical', 'horizontal'} or None, default 30
        Rotation angle of the x-axis labels.
        None is the same as 'vertical', and the same as 90.

    ax : Axes, optional
        Axes on which to plot.

    **kwargs :
        Additional kwargs sent to `imshow`.

    Returns
    -------
    Axes
    """
    if not isinstance(data, pd.DataFrame):
        data = np.asarray(data)
    if ax is None:
        ax = plt.axes()
    im = ax.imshow(data, cmap=cmap, interpolation='nearest', **kwargs)
    if colorbar:
        ax.figure.colorbar(im, ax=ax)
    if clim:
        im.set_clim(*clim)
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    if isinstance(data, pd.DataFrame):
        if isinstance(rot, float | int):
            if rot < 80:
                alignment = 'right'
            elif rot > 100:
                alignment = 'left'
            else:
                alignment = 'center'
        elif rot == 'horizontal':
            alignment = 'right'
        else:
            alignment = 'center'
        ax.set_xticklabels(data.columns, rotation=rot, ha=alignment)
        ax.set_yticklabels(data.index)
        if data.columns.name is not None:
            ax.set_xlabel(data.columns.name)
        if data.index.name is not None:
            ax.set_ylabel(data.index.name)
    ax.figure.tight_layout()
    return ax


def imagesc(
        data: ArrayLike | pd.DataFrame,
        *,
        ticks: Ticks | tuple[Ticks, Ticks] = 'auto',
        aspect: Literal['equal', 'auto'] | float = 'auto',
        cmap: str | Colormap | None = None,
        colorbar: bool = True,
        clim: tuple[float, float] | None = None,
        logscale: bool = False,
        ax: Axes = None,
) -> Axes:
    """
    Show an image, with special care to the ticks and labels.

    Parameters
    ----------
    data : array-like or DataFrame
        Image to show.

    ticks : None, int, float, iterable if ints, 'full', 'auto' or a 2-tuple of these, default 'auto'
        - None     : No ticks.
        - int      : Number of evenly-spaced ticks.
        - float    : Fraction of cells to tick; must be in (0, 1].
        - iterable : Indices of specific cells to tick.
        - 'full'   : All cells are ticked.
        - 'auto'   : If `data` is a DataFrame, use 'full', otherwise use None.
        - If a tuple of length 2, specifies ticks for (rows, columns). Otherwise, the same for both.

    aspect : {'equal', 'auto'} or float, default 'auto'
        Aspect ratio of the Axes.

    cmap : string or Colormap, optional
        Colormap to use.

    colorbar : bool, default True
        Whether to add a colorbar.

    clim : 2-tuple of floats, optional
        Color limits of the image.

    logscale : bool, default False
        Whether to map image values to colors using a logarithmic scale.

    ax : Axes, optional
        Axes on which to plot.

    Returns
    -------
    Axes
    """
    if isinstance(data, pd.DataFrame):
        indices = (data.index, data.columns)
        columns_orientation = 'vertical'
        data = data.values
        if ticks == 'auto':
            ticks = 'full'
    else:
        data = np.asarray(data)
        indices = map(pd.RangeIndex, data.shape)
        columns_orientation = 'horizontal'
        if ticks == 'auto':
            ticks = None
    if ax is None:
        ax = plt.axes()
    image = ax.imshow(data, aspect=aspect, cmap=cmap, interpolation='nearest', origin='upper',
                      norm=LogNorm() if logscale else None)
    rotation = ('horizontal', columns_orientation)
    if not isinstance(ticks, tuple) or len(ticks) != 2:
        ticks = (ticks, ticks)
    for ax_name, index, rot, tck, size in zip('yx', indices, rotation, ticks, data.shape):
        multi = index.nlevels > 1
        axis = getattr(ax, ax_name + 'axis')
        if isinstance(tck, float) and 0. < tck <= 1.:
            tck = int(size * tck)
        if tck is None:
            t = []
        elif tck == 'full':
            t = np.arange(size)
            if multi:
                axis.set_ticks(np.r_[-1, np.where(np.diff(cast(pd.MultiIndex, index).codes[-1]) < 0)[0]] + 0.5)
                axis.set_ticklabels(
                    index.droplevel(-1).unique().map(lambda x: ' | '.join(map(str, to_list(x)))),
                    rotation=rot,
                )
                axis.set_tick_params(which='major', pad=40)
                axis.grid(color='k', linestyle='-', linewidth=2)
        elif isinstance(tck, int) and 0 < tck <= size:
            t = np.linspace(0, size, num=tck, endpoint=False, dtype=int)
        elif is_iterable(tck):
            t = np.sort(np.asarray(tck, dtype=int))
        else:
            raise ValueError(f"Invalid `tick`: {tck}")
        axis.set_ticks(t, minor=multi)
        axis.set_ticklabels(index.get_level_values(-1)[t], rotation=rot, minor=multi)
        lim = [-0.5, size - 0.5]
        if ax_name == 'y':
            lim = reversed(lim)
        getattr(ax, f"set_{ax_name}lim")(*lim)
        if len(t) and any(x is not None for x in index.names):
            getattr(ax, f"set_{ax_name}label")(' | '.join(map(str, index.names)))
    if clim:
        image.set_clim(*clim)
    if colorbar:
        ax.figure.colorbar(image, ax=ax)
    ax.figure.tight_layout(rect=[0, 0.03, 1, 0.95])
    return ax


def plot_two_ys(
        x1: ArrayLike, y1: ArrayLike, color1: ColorSpec, label1: str,
        x2: ArrayLike, y2: ArrayLike, color2: ColorSpec, label2: str,
        xlabel: str | None = None, ax: Axes | None = None,
) -> Axes:
    """
    Plot two data sets on the same Axes.

    Parameters
    ----------
    x1, y1 : array-like
        First data set to plot.

    color1 : color spec
        Color for first data set.

    label1 : string
        Label for first data set.

    x2, y2 : array-like
        Second data set to plot.

    color2 : color spec
        Color for second data set.

    label2 : string
        Label for second data set.

    xlabel : string, optional
        Label for the x-axis.

    ax : Axes, optional
        Axes on which to plot.

    Returns
    -------
    Axes
    """
    if ax is None:
        ax = plt.axes()
    ax.plot(x1, y1, color=color1)
    ax.set_ylabel(label1, color=color1)
    ax.tick_params(axis='y', labelcolor=color1)
    twin = ax.twinx()
    twin.plot(x2, y2, color=color2)
    twin.set_ylabel(label2, color=color2)
    twin.tick_params(axis='y', labelcolor=color2)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    return ax


def plot_colors(
        colors: Collection[ColorSpec] | NDArray[np.floating],
        sizes: Collection[float | int] | None = None,
        *,
        drawedges: bool = True,
        orientation: Literal['vertical', 'horizontal'] = 'horizontal',
        ax: Axes | None = None,
) -> ColorbarBase:
    """
    Plot some colors as a row or column of rectangles.

    Parameters
    ----------
    colors : collection of color specs or a numpy array (Nx3 or Nx4) of color values
        Colors to plot.

    sizes : collection of numbers, optional
        Relative sizes of rectangles.
        If not provided, all are the same size.

    drawedges : bool, default True
        Whether to draw the edges between rectangles.

    orientation : {'vertical', 'horizontal'}, default 'horizontal'
        Orientation of the plot.

    ax : Axes, optional
        Axes on which to plot.

    Returns
    -------
    ColorbarBase
    """
    if sizes is None:
        sizes = np.ones(len(colors), dtype=np.uint32)
    elif len(sizes) != len(colors):
        raise ValueError("If provided, `sizes` must have the same length as `colors`.")
    bounds = np.r_[0, np.cumsum(sizes)]
    if ax is None:
        figsize = (6, 0.65)
        if orientation == 'vertical':
            figsize = tuple(reversed(figsize))
        ax = plt.subplots(figsize=figsize)[1]
    return ColorbarBase(
        ax=ax,
        cmap=ListedColormap(colors),
        norm=BoundaryNorm(bounds, len(sizes)),
        boundaries=bounds,
        ticks=[],
        spacing='proportional',
        orientation=orientation,
        drawedges=drawedges,
    )
