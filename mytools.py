"""Some functions to make my life easier. Not actually part of data analysis.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

bc_colors = {
    100: 'green',
    40: 'orange',
    20: 'red'
}
einstein_blue_light = "#9acae8"
einstein_blue_dark = "#18397a"

def reload(module_or_function):
    """Reload a module or the module a function was defined in
    """
    import importlib, sys
    from types import ModuleType, FunctionType
    # if it's a function or class, find the module it was defined in:
    if (hasattr(module_or_function, '__module__') and
        module_or_function.__module__ != '__main__'):
            module_or_function = sys.modules[module_or_function.__module__]
    if isinstance(module_or_function, ModuleType):
        # We have a module now that we CAN reload:
        reloaded_module = importlib.reload(module_or_function)
        print(f"Reloaded module {reloaded_module.__name__}")
        return
    raise ValueError(f"Cannot reload whatever {module_or_function!r} is."
                     " Must be a module or a function defined in one")

def make_dimensions(n_rows = 1, n_cols = 1,
                    individual_width_inch = 3.0,  wspace_inch = 0.5, left_inch = 2.0, right_inch = .5,
                    individual_height_inch = 3.0, hspace_inch = 0.5, bottom_inch = 1.0, top_inch = 0.5,
                   ):
    """Make dimensions (figsize and gridspec_kw) for a subplotted figure from absolute values
    """
    wspace = wspace_inch / individual_width_inch
    plot_area_width_inch = individual_width_inch * n_cols + (n_cols-1) * wspace_inch
    total_width_inch = plot_area_width_inch + left_inch + right_inch
    left = left_inch / total_width_inch
    right = 1 - right_inch / total_width_inch
    
    hspace = hspace_inch / individual_height_inch
    plot_area_height_inch = individual_height_inch * n_rows + (n_rows-1) * hspace_inch
    total_height_inch = plot_area_height_inch + bottom_inch + top_inch
    bottom = bottom_inch / total_height_inch
    top = 1 - top_inch / total_height_inch
    
    return (total_width_inch, total_height_inch), dict(left=left, right=right, wspace=wspace, bottom=bottom, top=top, hspace=hspace)

def make_figure(n_rows = 1, n_cols = 1,
        individual_width_inch = 3.0,  wspace_inch = 0.5, left_inch = 2.0, right_inch = .5,
        individual_height_inch = 3.0, hspace_inch = 0.5, bottom_inch = 1.0, top_inch = 0.5,
        sharex = False, sharey = False):
        
    dimensions_dict = dict(
        n_rows = n_rows, n_cols = n_cols,
        individual_width_inch = individual_width_inch,
        wspace_inch = wspace_inch,
        left_inch = left_inch,
        right_inch = right_inch,
        individual_height_inch = individual_height_inch,
        hspace_inch = hspace_inch,
        bottom_inch = bottom_inch,
        top_inch = top_inch,
    )

    figsize, gridspec_kw = make_dimensions(**dimensions_dict)

    fig, axs = plt.subplots(n_rows, n_cols, figsize = figsize,
        gridspec_kw = gridspec_kw,
        sharex = sharex, sharey = sharey,
        # make sure, axs is ALWAYS a 2D-array:
        squeeze = False,
    )
    fig.dimensions_dict = dimensions_dict
    fig.dimensions_dict['w'] = figsize[0]
    fig.dimensions_dict['h'] = figsize[1]
    # Create Axes for global axis labels
    if axs.size > 1:
        axg = fig.add_subplot(axs[0, 0].get_gridspec()[:], zorder = -100)
        plt.setp(axg, frame_on=False, xticks=[], yticks=[])
    else:
        axg = axs[0, 0]
    axg.set_zorder(20)
    return fig, axs, axg

def unit_colors(s):
    colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']
    return {
        ku: colors[kku % len(colors)]
        for kku, ku in enumerate(s.units.keys())
    }

def vmarker(ax, x, above_axis = True, **plot_kwargs):
    plot_kwargs = {**dict(
        markersize = 10,
        marker = 'v',
        mfc = 'w',
        mec = 'k',
        zorder = 10,
    ), **plot_kwargs}
    trans = (
        mpl.transforms.blended_transform_factory(
            ax.transData, ax.transAxes
        ) + 
        mpl.transforms.ScaledTranslation(
            0, (1 if above_axis else -1) * plot_kwargs['markersize']/2/72.,
            ax.figure.dpi_scale_trans
        )
    )
    return ax.plot(x, 0, transform=trans, clip_on=False, **plot_kwargs)
mpl.axes.Axes.vmarker = vmarker

def vmarker_left(ax, x, above_axis = True, **plot_kwargs):
    plot_kwargs = {**dict(
        markersize = 10,
        marker = 'v',
        mfc = 'w',
        mec = 'k',
        zorder = 10,
    ), **plot_kwargs}
    trans = (
        mpl.transforms.blended_transform_factory(
            ax.transAxes, ax.transData
        ) + 
        mpl.transforms.ScaledTranslation(
            (1 if above_axis else -1) * plot_kwargs['markersize']/2/72., 0,
            ax.figure.dpi_scale_trans
        )
    )
    return ax.plot(0, x, transform=trans, clip_on=False, **plot_kwargs)
mpl.axes.Axes.vmarker_left = vmarker_left

def make_cont_x(*values, before = 50, after = 50, n = 100):
    left = np.min([np.min(v) for v in values]) - before
    right = np.max([np.max(v) for v in values]) + after
    return np.linspace(left, right, n)

