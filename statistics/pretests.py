'''
Pretests are mainly used to classify data as "normally distributed" or not.

@author: Roland Ferger (roland@bio2.rwth-aachen.de)
'''

from scipy.stats import shapiro

def normality(data, test = 'shapiro', alpha = None, full_return = False,
              test_arguments = {}):
    """Test if 'data' comes from a normal distribution.
    
    Returns the p-Value or the decision if a chance level is given.
    
    Parameters
    ----------
    data : numpy.array
        The data to test, should be linearized
    test : {'shapiro'}, optional, (default: 'shapiro')
        The hypothesis test to be used. Currently only shapiro is supported.
        @see scipy.stats.shapiro
    alpha : {None, float}, optional (default: None)
        If 'alpha' is set, return the decision whether data is normally
        distributed (Null-Hypothesis => True) or if this is rejected at the
        alpha chance level (False)
        If 'alpha' is not set (i.e. None), the p-Value is returned
    full_return : boolean, optional (default: False)
        In addition to the p-Value or decision, the full output of the used test
        function may be returned as second return argument
        
    Returns:
    --------
    ret_value : float or boolean
        If 'alpha' is set: return if data is normally distributed (True) or not
        If 'alpha' is None (default): return p-Value
    
    """
    if test == "shapiro":
        # Shapiro test returns two values:
        # shapiro_out[0] -- W -- Test statistic
        # shapiro_out[1] -- p-Value of the hypothesis test
        shapiro_out = shapiro(data, **test_arguments)
        result_keys = "W","p"
        result = dict(zip(result_keys, shapiro_out))
        if alpha is not None:
            ret_value = False if result['p'] < alpha else True
        else:
            ret_value = result['p']
        if full_return:
            return (ret_value, result)
        else:
            return ret_value
    else:
        raise ValueError("Unknown test type for %s.normality" % __name__)

