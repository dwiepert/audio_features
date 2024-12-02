"""
Load features from .npz file

Author(s): Daniela Wiepert
Last modified: 11/28/2024
"""
#IMPORTS
##built-in
from pathlib import Path
from typing import Union

##third-party
import numpy as np

def load_features(feature_dir:Union[str,Path], cci_features=None, recursive:bool=False, ignore_str:Union[str,list]=None,search_str:Union[str,list]=None):
    """
    Load features

    :param feature dir: str/Path object, points to directory with feature dirs
    :param cci_features: cotton candy bucket
    :param recursive: bool, indicates whether to load features recursively
    :param ignore_str: str, string pattern to ignore when loading features
    :param search_str: str, string pattern to search for when loading features
    :param features: loaded feature dict
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
        if isinstance(ignore_str, list):
            new_paths = []
            for f in paths:
                include = True
                for s in ignore_str:
                    if s in str(f):
                        include = False
                if include:
                    new_paths.append(f)

            paths=new_paths
        else:
            paths = [f for f in paths if ignore_str not in str(f)] #don't load times files

    if search_str is not None:
        if isinstance(search_str, list):
            new_paths = []
            
            for f in paths:
                include = False
                for s in ignore_str:
                    if s in str(f):
                        include = True
                if include:
                    new_paths.append(f)

            paths=new_paths
        else:
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



