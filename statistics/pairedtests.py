'''
Test difference of two sets of PAIRED data

@author: Roland Ferger (roland@bio2.rwth-aachen.de)
'''

from .pretests import normality as normality_test
from scipy.stats import wilcoxon
from scipy.stats import ttest_rel as ttest

def diff_test(x, y, return_normality=False):
    norm_p = normality_test(y-x)
    if norm_p > 0.05:
        #print "y-x is normally distributed"
        #print "use T-test"
        test_result = test_normal(x, y)
    else:
        #print "y-x is not normally distributed"
        #print "use Wilcoxon signed-rank test"
        test_result = test_notnormal(x, y)
    if return_normality:
        return test_result, (norm_p,)
    else:
        return test_result


def test_notnormal(x, y):
    return wilcoxon(x-y)[1]

    
def test_normal(x, y):
    return ttest(x, y)[1]

