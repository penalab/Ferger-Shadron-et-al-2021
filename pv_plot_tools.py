"""PV Figures Plot Tools
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def figure_outline(fig = None):
    if fig is None:
        fig = plt.gcf()
    with plt.xkcd(.5):
        ax0 = fig.add_axes((0.001,0.001,.998,.998), frameon=True, xticks=[], yticks=[], facecolor='none', zorder=-100)
        plt.setp(ax0.spines.values(), color='r', linestyle=':')
mpl.figure.Figure.outline = figure_outline


def figure_add_axes_inch(fig,
    left=None, width=None, right=None,
    bottom=None, height=None, top=None,
    label = None,
    **kwargs,
    ):
    """Add Axes to a figure with inch coordinates."""
    # Check number of arguments:
    n_horz_args = sum([left is not None, width is not None, right is not None])
    if not n_horz_args == 2:
        raise ValueError(f"Need exactly 2 horizontal arguments, but {n_horz_args} given.")
    n_vert_args = sum([bottom is not None, height is not None, top is not None])
    if not n_vert_args == 2:
        raise ValueError(f"Need exactly 2 horizontal arguments, but {n_vert_args} given.")
    # Unique label:
    if label is None:
        label = f"ax{len(fig.get_axes()):02}"

    # Figure dimensions:
    fig_w, fig_h = fig.get_size_inches()
    
    # Horizontal:
    if right is None:
        l = left / fig_w
        w = width / fig_w
    elif width is None:
        l = left / fig_w
        w = (fig_w - left - right) / fig_w
    else: # left is None
        w = width / fig_w
        l = (fig_w - right - width) / fig_w
    
    # Vertical:
    if top is None:
        b = bottom / fig_h
        h = height / fig_h
    elif height is None:
        b = bottom / fig_h
        h = (fig_h - bottom - top) / fig_h
    else: # bottom is None
        h = height / fig_h
        b = (fig_h - top - height) / fig_h
    
    return fig.add_axes((l, b, w, h), label = label, **kwargs)
mpl.figure.Figure.add_axes_inch = figure_add_axes_inch


def figure_add_axes_group_inch(fig, nrows = 1, ncols = 1,
        group_top = 0.2, group_left = 0.8,
        individual_width = 1.2, individual_height = 0.8,
        wspace = 0.1, hspace = 0.1,
    ):
    axs = []
    for kr in range(nrows):
        axs.append([])
        for kc in range(ncols):
            ax = fig.add_axes_inch(
                top = group_top + kr * (individual_height + hspace),
                height = individual_height,
                left = group_left + kc * (individual_width + wspace),
                width = individual_width,
            )
            axs[-1].append(ax)
    axs = np.asarray(axs)
    axg = fig.add_axes_inch(
        top = group_top,
        height = nrows * individual_height + (nrows - 1) * hspace,
        left = group_left,
        width = ncols * individual_width + (ncols - 1) * wspace
    )
    plt.setp(axg, frame_on=False, xticks=[], yticks=[], zorder=20)
    return axs, axg
mpl.figure.Figure.add_axes_group_inch = figure_add_axes_group_inch


def axes_subplot_indicator(ax, label = None, fontsize = 16, **kwargs):
    trans = ax.transAxes + \
            mpl.transforms.ScaledTranslation(
                -fontsize/2/72.,
                +0/72.,
                ax.figure.dpi_scale_trans
            )
    if label is None:
        label = ax.get_label()
    if 'ha' in kwargs:
        kwargs['horizontalalignment'] = kwargs.pop('ha')
    if 'va' in kwargs:
        kwargs['verticalalignment'] = kwargs.pop('va')
    textkwargs = dict(
        horizontalalignment = 'right',
        verticalalignment = 'bottom',
        fontsize = fontsize,
        fontweight = 'bold',
    )
    textkwargs.update(kwargs)
    ax.text(0.0, 1.0, label, transform=trans, **textkwargs)
mpl.axes.Axes.subplot_indicator = axes_subplot_indicator
