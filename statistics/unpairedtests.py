'''
Test difference of two sets of INDEPENDENT data

@author: Roland Ferger (roland@bio2.rwth-aachen.de)
'''

from .pretests import normality as normality_test
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind as ttest


def diff_test(x, y, return_normality=False):
    norm_p_x = normality_test(x)
    norm_p_y = normality_test(y)
    if min(norm_p_x, norm_p_y) > 0.05:
        #print "Both x and y are normally distributed"
        #print "use (independent) T-test"
        test_result = test_normal(x, y)
    else:
        #print "One of x and y is not normally distributed"
        #print "use Mann-Whitney U-test"
        test_result = test_notnormal(x, y)
    if return_normality:
        return test_result, (norm_p_x, norm_p_y)
    else:
        return test_result


def test_notnormal(x, y):
    return mannwhitneyu(x, y)[1]


def test_normal(x, y):
    return ttest(x, y)[1]
