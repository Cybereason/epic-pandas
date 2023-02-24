import numpy as np
import pandas as pd
import networkx as nx

from toolz import identity
from typing import TypeVar, Any
from numpy.typing import NDArray
from epic.common.general import indexer_dict
from collections.abc import Mapping, Iterable, Callable, Hashable

from matplotlib.axes import Axes
from matplotlib import pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.colors import Colormap, ListedColormap, to_hex

from .matplotlib.colors import ColorSpec

Node = TypeVar("Node", bound=Hashable)


def draw_graph(
        graph: nx.Graph,
        *,
        pos: Mapping[Node, tuple[float, float]] | None = None,
        cmap: str | Colormap | Iterable[ColorSpec] | NDArray[np.floating] = 'tab20',
        color_group: Mapping[Node, Hashable] | pd.Series | Iterable[Hashable] | None = None,
        edge_attr: str | None = None,
        node_name_func: Callable[[Node], Any] | None = None,
        node_attrs: bool = True,
        labels_font_family: str = 'monospace',
        labels_font_size: int = 12,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
) -> Axes:
    """
    Draw a Networkx graph with node labels, optional node and edge attributes and different node colors.

    Parameters
    ----------
    graph : Graph
        The graph to draw.

    pos : dict, optional
        Positions of the nodes. If not given, the graphviz layout is used.

    cmap : str, Colormap, an iterable of color specs or a numpy array (Nx3 or Nx4) of color values, default "tab20"
        The color map from which to draw colors.
        If an iterable of an array, lists the colors to use.

    color_group : dict, Series or iterable, optional
        A mapping from each node to a group. Each group will be assigned a color from the color map (given in `cmap`).
        If an iterable, the order is assumed to be the same as the nodes in `graph`.
        Default is a single group, meaning the same color for all nodes.

    edge_attr : str, optional
        The name of an edge attribute to draw.

    node_name_func : callable, optional
        A function from a node to a name to be displayed for it.
        Default is the identity function.

    node_attrs : bool, default True
        Whether to display all node attributes.

    labels_font_family : str, default 'monospace'
        Font family for the labels.

    labels_font_size : int, default 12
        Font size for the labels.

    ax : Axes, optional
        Axes on which to draw the graph.

    figsize : 2-tuple, optional
        Figure size.
        Only used is `ax` is not given. Then, a new figure is created.

    Returns
    -------
    Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    if pos is None:
        pos = nx.drawing.nx_agraph.graphviz_layout(graph)
    elif set(pos.keys()) != set(graph.nodes):
        raise ValueError("Invalid `pos` argument: keys must match graph nodes.")
    nodes = pd.DataFrame.from_dict(dict(graph.nodes(data=True)), orient='index')
    if color_group is None:
        color_group = pd.Series(0, index=nodes.index)
    elif isinstance(color_group, Mapping):
        color_group = pd.Series(color_group)
    elif not isinstance(color_group, pd.Series):
        color_group = pd.Series(color_group, index=nodes.index)
    ind = indexer_dict()
    color_group = color_group.astype(object).reindex(nodes.index, fill_value=object()).map(ind)
    if isinstance(cmap, str | Colormap):
        colormap = plt.get_cmap(cmap, len(ind))
    else:
        colormap = ListedColormap(cmap, N=len(ind))
    color_group.groupby(color_group).apply(
        lambda c: nx.draw_networkx_nodes(
            graph, pos, nodelist=c.index.tolist(), node_color=to_hex(colormap(c.name)), ax=ax,
        )
    )
    if node_name_func is None:
        node_name_func = identity

    def make_label(node_info: pd.Series) -> str:
        name = str(node_name_func(node_info.name))
        if node_attrs:
            attrs = node_info.dropna()
            if not attrs.empty:
                name += '\n' + attrs.to_string()
        return name

    texts = nx.draw_networkx_labels(
        graph, pos, labels=nodes.apply(make_label, axis=1).to_dict(), horizontalalignment='left',
        font_family=labels_font_family, font_size=labels_font_size, ax=ax,
    )
    # adjust horizontal positions of labels so that they're not cut off by the edge of the axes
    display2axes = ax.transAxes.inverted()
    axes2data = ax.transAxes + ax.transData.inverted()
    renderer = fig.canvas.get_renderer()
    for text in texts.values():
        # bbox in display coords
        bbox = text.get_window_extent(renderer=renderer)
        # transform to axes coords
        bbox = Bbox(display2axes.transform(bbox))
        if bbox.xmax > 1:
            # move the box
            bbox = bbox.translated(1 - bbox.xmax, 0)
            # transform to data coords
            bbox = axes2data.transform(bbox)
            # move the text
            text.set_position(bbox[0])
    nx.draw_networkx_edges(graph, pos, ax=ax)
    if edge_attr is not None and (e := {
        (u, v): attrs[edge_attr] for u, v, attrs in graph.edges(data=True) if edge_attr in attrs
    }):
        nx.draw_networkx_edge_labels(
            graph, pos, edge_labels=e, font_family=labels_font_family, font_size=labels_font_size, ax=ax,
        )
    ax.set_axis_off()
    return ax
