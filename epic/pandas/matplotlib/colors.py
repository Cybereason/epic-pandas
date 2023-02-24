import numpy as np

from matplotlib.colors import to_rgb
from colorsys import rgb_to_hls, hls_to_rgb

RgbTuple = HlsTuple = tuple[float, float, float]
RgbaTuple = tuple[float, float, float, float]
ColorSpec = RgbTuple | RgbaTuple | str


def to_hls(color: ColorSpec) -> HlsTuple:
    """
    Translate a color spec to HLS.
    """
    return rgb_to_hls(*to_rgb(color))


def lighten(color: ColorSpec, factor: float) -> RgbTuple:
    """
    Make a lighter version of a color, with the same hue and saturation.

    Parameters
    ----------
    color : color spec
        Input color.

    factor : float in [0, 1]
        How much to lighten.
        0 - Do nothing.
        1 - Completely white.

    Returns
    -------
    tuple
        3-tuple of RGB values.
    """
    if not 0 <= factor <= 1:
        raise ValueError(f"`factor` should be between 0 and 1; got {factor}")
    h, l, s = to_hls(color)
    return hls_to_rgb(h, l + (1 - l) * factor, s)


def sequential_cmap(color: ColorSpec, n_colors: int = 256, max_lightness_factor: float = 1) -> list[RgbTuple]:
    """
    Given a color, create a list of lighter and lighter colors in the same hue and saturation.

    Parameters
    ----------
    color : color spec
        Base color.
        It is the first color in the returned list.

    n_colors : int, default 256
        Number of colors to create.

    max_lightness_factor : float in [0, 1], default 1
        Maximum lightness of the last color in the sequence.
        0 - Same as input color.
        1 - White.

    Returns
    -------
    list of colors
        Length is `n_colors`.
    """
    if not 0 <= max_lightness_factor <= 1:
        raise ValueError(f"`max_lightness_factor` should be between 0 and 1; got {max_lightness_factor}")
    h, l, s = to_hls(color)
    return [hls_to_rgb(h, ll, s) for ll in np.linspace(l, l + (1 - l) * max_lightness_factor, n_colors)]
