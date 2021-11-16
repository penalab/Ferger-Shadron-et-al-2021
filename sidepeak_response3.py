'''
Created on 27.10.2014

@author: Roland Ferger (roland@bio2.rwth-aachen.de)
'''

import numpy as np


def find_side_peaks(X, Y, max_I=None):
    """
    For X|Y-data return indexes of all relative (but not the absolute) maxima
    """
    if max_I is None:
        max_I = np.argmax(Y)
    allsidepeaks_I = []
    for k in range(1, len(Y)-1):
        if k == max_I:
            # Do not include the main peak
            continue
        if Y[k-1] >= Y[k]:
            # Value is smaller than left value
            continue
        # From here on true: Y[k-1] < Y[k]
        if Y[k] > Y[k+1]:
            allsidepeaks_I.append(k)
        elif Y[k] == Y[k+1]:
            # Possible "plateau-peak":
            for l in range(k+1, len(Y)-1):
                if Y[l] == Y[l+1]:
                    # Continuation of plateau
                    continue
                elif Y[l] > Y[l+1]:
                    # End of plateau, next smaller.
                    # Return first index of equal values
                    allsidepeaks_I.append(k)
                else:
                    # Y[l] < Y[l+1]
                    # Next value higher. No local maximum
                    break
    return np.array(allsidepeaks_I)


def find_troughs(X, Y):
    '''
    For X|Y-data return indexes of all troughs
    '''
    mins_I = find_side_peaks(X, -Y, max_I=0)
    return mins_I
    #return mins_X = X[mins_I]
    #return mins_Y = Y[mins_I]


def find_troughs_manually(X, Y):
    '''
    For X|Y-data return indexes of all troughs
    '''
    alltroughs_I = []
    for k in range(1, len(Y)-1):
        if Y[k-1] <= Y[k]:
            # Value is greater than left value
            continue
            # From here on true: Y[k-1] > Y[k]
        if Y[k] < Y[k+1]:
            alltroughs_I.append(k)
        # if Y[k] < Y[k+1] and Y[k] < Y[k-1]:
        #     alltroughs_I.append(k)
    mins_I = np.array(alltroughs_I)
    return mins_I
    #return mins_X = X[mins_I]
    #return mins_Y = Y[mins_I]

            

def side_peak_relation3(X, Y, sp_option='greatest primary', tr_option='nearest' , **kwargs):
    """From X|Y-data calculate the Y-relation of main and highest side peak
    
    Keyword arguments:
        max_I       - Force max_I, the index of max(Y), instead of searching for
                      the maximum in Y
        side_I      - Force side_I, the index of the side peak in X|Y, instead
                      of searching it in Y
        mins_I      - Force mins_I, the indices of the troughs
        min_X_diff  - minimal difference of max_X and side_X
        sp_option   - Choose between 'greatest primary'(default): uses greatest primary side peak;
                      'greatest': uses greatest side peak; 'mean': uses mean values of the two primary side peaks
                      (tr_option must be 'minimum')
        tr_option   - Choose between 'nearest' (default): uses nearest trough to SP; 'minimum': uses absolute minimum of curve
    
    Return dict with fields:
        max_Y    - max(Y)
        max_X    - Value in X corresponding to max_Y
        side_Y   - secondary peak
        side_X   - Value in X corresponding to side_Y
        min_Y    - min(Y)
        min_X    - Value in X corresponding to min_Y
        diff_Y   - dynamic range, max_Y - min_Y
        rel_Y    - relative Y of secondary peak, where min_Y = 0 and max_Y = 1
    """
    X = X.reshape((-1))
    Y = Y.reshape((-1))
    ## Main peak data
    if 'max_I' in kwargs:
        max_I = kwargs['max_I']
    else:
        max_I = np.argmax(Y)
    max_X = X[max_I]
    max_Y = Y[max_I]
    ## Side peak data
    if 'side_I' in kwargs:
        side_I = kwargs['side_I']
    else:
        sides_I = find_side_peaks(X, Y, max_I=max_I)
        if 'min_X_diff' in kwargs:
            sides_I = sides_I[np.abs(X[sides_I]-max_X) >= kwargs['min_X_diff']]
        if sp_option == 'greatest primary':
            # Use the greatest primary side peak of the derived side peaks
            primaries_I = []
            if np.any((max_I - sides_I) > 0):
                primaries_I += [np.max(sides_I[(max_I - sides_I) > 0])]
            if np.any((max_I - sides_I) < 0):
                primaries_I += [np.min(sides_I[(max_I - sides_I) < 0])]
            #primaries_I = np.array(primaries_I)
            side_I = primaries_I[np.argmax(Y[primaries_I])]
            #side_L = np.max(sides_I[(max_I - sides_I) > 0]) # closest prim. SP to MP from left side
            #side_R = np.min(sides_I[(max_I - sides_I) < 0]) # closest prim. SP to MP from rightside
            #side_I = side_L if Y[side_L] > Y[side_R] else side_R # use greatest prim. SP
            side_X = X[side_I]
            side_Y = Y[side_I]
        if sp_option == 'greatest':
            # Use greatest side peak
            try:
                side_I = sides_I[np.argmax(Y[sides_I])]
            except IndexError as e:
                print(sides_I)
                raise e
            side_X = X[side_I]
            side_Y = Y[side_I]
        if sp_option == 'mean':
            # Use mean value of the two primary side peaks
            side_L_I = max(sides_I[(max_I - sides_I) > 0]) # closest prim. SP to MP from left side
            side_R_I = min(sides_I[(max_I - sides_I) < 0]) # closest prim. SP to MP from rightside
            side_L_X = X[side_L_I]
            side_L_Y = Y[side_L_I]
            side_R_X = X[side_R_I]
            side_R_Y = Y[side_R_I]
            side_Y = np.mean([side_L_Y, side_R_Y])
    ## Minimum data (all troughs)
    if 'mins_I' in kwargs:
        mins_I = kwargs['mins_I']
    else:
        mins_I = find_troughs(X, Y)
    #mins_X = X[mins_I]
    #mins_Y = Y[mins_I]
    # Minimum data
    if tr_option == 'nearest':
        if sp_option == 'mean':
            min_I = np.argmin(Y) # otherwise no value for min_I
        else:
            if side_I < max_I:
                min_I = min(mins_I[(side_I - mins_I) < 0])
            else: # side_I can't be max_I (side_I == max_I does not occur)
                min_I = max(mins_I[(side_I - mins_I) > 0])
    if tr_option == 'minimum':
            min_I = np.argmin(Y)
    min_X = X[min_I]
    min_Y = Y[min_I]
    # Dynamic range
    diff_Y = max_Y - min_Y
    # Relative secondary peak height
    rel_Y = (side_Y - min_Y) / diff_Y
    if sp_option == 'mean':
        return dict(max_X = max_X, max_Y = max_Y, min_X = min_X, min_Y = min_Y,
                side_X = [side_L_X, side_R_X], side_Y = side_Y,
                diff_Y = diff_Y, rel_Y = rel_Y,
                )
    else:
        return dict(max_X = max_X, max_Y = max_Y, min_X = min_X, min_Y = min_Y,
                side_X = side_X, side_Y = side_Y,
                diff_Y = diff_Y, rel_Y = rel_Y,
                )

