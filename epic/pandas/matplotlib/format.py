import math

from epic.common.general import human_readable

from matplotlib.axis import Axis
from matplotlib import pyplot as plt
from matplotlib.ticker import Formatter, AutoLocator


class _BinaryAutoLocator(AutoLocator):
    @staticmethod
    def _staircase(steps):
        return AutoLocator._staircase(steps) * 1.024


class _HumanReadableFormatter(Formatter):
    MIN_PRECISION = 2

    def __init__(self, binary: bool = False):
        self.binary = binary
        self.precision = self.MIN_PRECISION

    def __call__(self, x, pos=None):
        return human_readable(x, binary=self.binary, n_digits=self.precision)

    @classmethod
    def calc_precision(cls, a, b):
        try:
            precision = abs(math.floor(math.log10(2 * (b - a) / (b + a))))
        except Exception:
            return cls.MIN_PRECISION
        return max(precision, cls.MIN_PRECISION)

    def set_locs(self, locs):
        self.precision = self.calc_precision(*locs[:2]) if len(locs) >= 2 else self.MIN_PRECISION
        super().set_locs(locs)


def human_readable_axis(axis: Axis | None = None, binary: bool = False) -> None:
    """
    Format an axis to display tick labels in human-readable format.

    Parameters
    ----------
    axis : Axis, optional
        Axis to format.
        Default is the x-axis of the current Axes.

    binary : bool, default False
        If True, the labels are suitable for binary data:
        - Units are in Kibi, Mibi, etc.
        - Ticks are placed in 1024-based intervals.

    Returns
    -------
    None
    """
    if axis is None:
        axis = plt.gca().xaxis
    if binary:
        axis.set_major_locator(_BinaryAutoLocator())
    axis.set_major_formatter(_HumanReadableFormatter(binary))
