"""
Align times in ascending order for a feature set

Author(s): Daniela Wiepert
Last modified: 12/04/2024
"""
#IMPORTS
##built-in
from typing import Dict
##third-party
import numpy as np 

def align_times(feats:Dict[str:np.ndarray], times:Dict[str:np.ndarray]) -> Dict[str:Dict[str:np.ndarray]]:
    """
    Sort features in ascending order based on time

    :param feats: dict of features, stimulus keys
    :param times: dict of times, stimulus keys
    :param features: dict of dicts, with each dict being a feature/time dict for a story
    """
    features = {}
    for s in list(feats.keys()):
        if s == 'weights':
            continue
        f = feats[s]
        t = times[s]
        sort_i = np.argsort(t, axis=0)[:,0]
        f = f[sort_i,:]
        t = t[sort_i,:]
        features[s] = {'features': f, 'times': t}
    return features