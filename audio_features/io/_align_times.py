"""
"""
#IMPORTS
##third-party
import numpy as np 

def align_times(feats, times):
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