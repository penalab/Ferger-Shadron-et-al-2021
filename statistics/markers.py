'''
Write markers (mostly asterisks) to mark significant data

@author: Roland Ferger (roland@bio2.rwth-aachen.de)
'''
import numpy as np
import matplotlib.pyplot as plt
from . import pairedtests
from . import unpairedtests
from . import pretests


def no_test(x, y):
    return None


def norm_test(x,y=None):
    if y is None:
        return pretests.normality(x)
    else:
        return pretests.normality(y-x)


def stats_marker(x, y, x_pos, y_pos, write_ns=False, testfun=None, paired=True, marker="*", only_return=False, return_p=False, **kwargs):
    if 'ax' in kwargs:
        ax = kwargs.pop('ax')
    else:
        ax = plt.gca()
    if marker == "*":
        fontsize = 12.0
    elif marker == "#":
        fontsize = 8.0
    else:
        fontsize = 12.0
    if not ('size' in kwargs or 'fontsize' in kwargs):
        kwargs.update(size=fontsize)
    if not 'color' in kwargs:
        kwargs.update(color='k')
    if not 'horizontalalignment' in kwargs or 'ha' in kwargs:
        kwargs.update(ha='center')
    if paired:
        if testfun is None:
            testfun = pairedtests.diff_test
    else:
        if testfun is None:
            testfun = unpairedtests.diff_test
    p = testfun(x, y)
    if marker == 'p':
        txt = r"$p \approx {:.2g}$".format(p)
    elif marker == 'n':
        if paired:
            txt = r"$n = {:.2g}$".format(len(x))
        else:
            txt = r"$n_x = {:.2g}, n_y = {:.2g}$".format(len(x), len(y))
    elif p < 0.0005:
        txt = marker*3
    elif p < 0.005:
        txt = marker*2
    elif p < 0.05:
        txt = marker*1
    elif write_ns:
        txt = 'n.s.'
    else:
        txt = ''
    if not only_return:
        ax.text(x_pos, y_pos, txt, **kwargs)
    if return_p:
        return p
    return txt


def stats_diff_marker(x, y, x_left, x_right, y_pos=None, paired=False, text_pos=5, **kwargs):
    if 'ax' in kwargs:
        ax = kwargs['ax']
    else:
        ax = plt.gca()
        kwargs.update(ax = ax)
    if y_pos is None:
        y_pos = np.ceil(np.max([np.max(x), np.max(y)]) +
                        0.1 * np.max([np.max(x)-np.min(x), np.max(y)-np.min(y)])
                        )
    if not ('va' in kwargs or 'ma' in kwargs or 'verticalalignment' in kwargs):
        kwargs.update(va='bottom')
    if 'arm' in kwargs:
        arm = kwargs['arm']
    elif 'size' in kwargs:
        arm = kwargs['size']
    elif 'fontsize' in kwargs:
        arm = kwargs['fontsize']
    else:
        arm = 10
    props = {'connectionstyle': 'bar,armA=%i,armB=%i,fraction=0.0,angle=0.0' % (arm,arm),
             'arrowstyle':'-', 'lw':1}
    if 'transform' in kwargs:
        trans_dict = {'xycoords': kwargs['transform']}
    else:
        trans_dict = {}
    ax.annotate('', xy=(x_left,y_pos), xytext=(x_right,y_pos), arrowprops=props, **trans_dict)
    stats_marker(x, y, x_pos = np.mean((x_left, x_right)), y_pos = y_pos+text_pos, paired=paired, **kwargs)
