import os
from pathlib import Path
from typing import Union

import numpy as np

def load_features(feature_paths:Union[str,Path], cci_features=None):
    """
    Load features

    TODO
    """
    feature_paths = [Path(f) for f in feature_paths]
    features = {}
    for f in feature_paths:
        if cci_features is not None:
            features[f] = cci_features.download_raw_array(f)
        else:
            features[f] = np.load(f)
    
    features['path_list'] = feature_paths
    return features



