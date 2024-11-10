"""
Save features

Author(s): Aditya Vaidya, Daniela Wiepert
Last modified: 11/08/2024
"""
#IMPORTS
##built-in
import os
from typing import Union
from pathlib import Path

##third-party
import numpy as np

def save_features(sample:dict, features_save_path:Union[str,Path], times_save_path:Union[str,Path], module_save_paths:dict, cci_features=None):
    """
    Save features
    :param sample: dict, audio sample containing features
    :param features_save_path: str/Path, path to save features to, BUT it should not yet contain the extension
    :param times_save_path: str/Path, path to save times to, BUT it should not yet contain the extension
    :param module_save_paths: Dict[str], dictionary with module name mapped to where you should save the file to
    :param cci_features: cotton candy interface for save bucket. Defaults to None.
    :return: None, saves features
    """
    assert '.npz' not in str(features_save_path), 'Extension should not be included in features_save_path'
    assert '.npz' not in str(times_save_path), 'Extension should not be included in times_save_path'
    mod_ft = 'module_features' in sample
    if mod_ft: 
        assert module_save_paths is not None, 'Must give module save paths if you have module_features to save'
        for m in module_save_paths:
            assert '.npz' not in str(m), 'Extension should not be included in module_save_paths'

    out_features = sample['out_features']
    if not isinstance(out_features, np.ndarray): out_features = out_features.numpy()
    times = sample['times']
    if not isinstance(times, np.ndarray): times = times.numpy()
    

    if cci_features is None:
        os.makedirs(features_save_path.parent, exist_ok=True)
        np.savez_compressed(str(features_save_path) + '.npz', features=out_features)
        np.savez_compressed(str(times_save_path) + '.npz', times=times)
    else:
        cci_features.upload_raw_array(features_save_path, out_features)
        cci_features.upload_raw_array(times_save_path, times)

    # This is the "save name" of the module (not its original name)
    if mod_ft:
        module_features = sample['module_features']
        for module_name, features in module_features.items():
            features_save_path = module_save_paths[module_name]
            if not isinstance(features, np.ndarray): features=features.numpy()
            #times_save_path = f"{features_save_path}_times"
            if cci_features is None:
                os.makedirs(os.path.dirname(features_save_path), exist_ok=True)
                np.savez_compressed(features_save_path + '.npz', features=features)
                #np.savez_compressed(times_save_path, times=times)
            else:
                #cci_features.upload_raw_array(times_save_path, times)
                cci_features.upload_raw_array(features_save_path, features)