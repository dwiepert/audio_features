import os
from pathlib import Path
from typing import Union

import numpy as np

def load_features(feature_dir:Union[str,Path], cci_features=None, recursive:bool=False, ignore_str:str='_times', search_str:str=None):
    """
    Load features

    TODO
    """
    feature_dir = Path(feature_dir)
    if cci_features is None:
        if recursive:
            paths = sorted(list(feature_dir.rglob('*.npz')))
        else:
            paths = sorted(list(feature_dir.glob('*.npz')))

    else:
        print('TODO: check how lsdir outputs features - is it automatically recursive?')
        all = cci_features.lsdir(feature_dir)
        paths = [Path(f) for f in all]
        paths = [f for f in all if f.parent==feature_dir]

    if ignore_str is not None:
        paths = [f for f in paths if ignore_str not in str(f)] #don't load times files
    if search_str is not None:
        paths = [f for f in paths if search_str in str(f)]
    features = {}
    for f in paths:
        if cci_features is not None:
            features[f] = cci_features.download_raw_array(f)
        else:
            l = np.load(f)
            key = list(l)[0]
            features[f] = l[key]
    
    features['path_list'] = paths
    return features



