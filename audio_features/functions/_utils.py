"""
Some util functions for internal use

Author(s): Aditya Vaidya
Last Modified: Unknown
"""
#IMPORTS
##third-party
import numpy as np


## Demean -- remove the mean from each column
demean = lambda v: v-v.mean(0)
demean.__doc__ = """Removes the mean from each column of [v]."""
dm = demean

## Z-score -- z-score each column
zscore = lambda v: (v-v.mean(0))/v.std(0)
zscore.__doc__ = """Z-scores (standardizes) each column of [v]."""
zs = zscore

## Rescale -- make each column have unit variance
rescale = lambda v: v/v.std(0)
rescale.__doc__ = """Rescales each column of [v] to have unit variance."""
rs = rescale

## Matrix corr -- find correlation between each column of c1 and the corresponding column of c2
mcorr = lambda c1,c2: (zs(c1)*zs(c2)).mean(0)
mcorr.__doc__ = """Matrix correlation. Find the correlation between each column of [c1] and the corresponding column of [c2]."""

## Cross corr -- find corr. between each row of c1 and EACH row of c2
xcorr = lambda c1,c2: np.dot(zs(c1.T).T,zs(c2.T)) / (c1.shape[1])
xcorr.__doc__ = """Cross-column correlation. Finds the correlation between each row of [c1] and each row of [c2]."""

def _zscore(mat, return_unzvals=False):
    """Z-scores the rows of [mat] by subtracting off the mean and dividing
    by the standard deviation.
    If [return_unzvals] is True, a matrix will be returned that can be used
    to return the z-scored values to their original state.

    :param mat: array like, matrix
    :return zmat: array like, z-scored matrix
    """
    zmat = np.empty(mat.shape, mat.dtype)
    unzvals = np.zeros((zmat.shape[0], 2), mat.dtype)
    for ri in range(mat.shape[0]):
        unzvals[ri,0] = np.std(mat[ri,:])
        unzvals[ri,1] = np.mean(mat[ri,:])
        zmat[ri,:] = (mat[ri,:]-unzvals[ri,1]) / (1e-10+unzvals[ri,0])
    
    if return_unzvals:
        return zmat, unzvals
    
    return zmat