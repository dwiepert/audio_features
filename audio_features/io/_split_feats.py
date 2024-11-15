"""
"""
import os
from pathlib import Path 
def split_features(features:dict):

    path_list = features['path_list']
    path_to_fname = {}
    new_feat_dict = {}

    for p in path_list:
        fname = os.path.splitext(Path(p).name)
        path_to_fname[str(p)] = fname[0]
    
    for f in features:
        if f != 'path_list':
            feats = features[f]
            fname = path_to_fname[str(f)]
            new_feat_dict[fname] = feats
    
    return new_feat_dict


