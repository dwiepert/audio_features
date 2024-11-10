import os
from pathlib import Path
from typing import Union

import numpy as np

def load_features(features_save_path:Union[str,Path], times_save_path:Union[str,Path], module_save_paths:dict, cci_features=None):
    """
    Load features

    :param features_save_path: str/Path, path to save features to, BUT it should not yet contain the extension
    :param times_save_path: str/Path, path to save times to, BUT it should not yet contain the extension
    :param module_save_paths: Dict[str], dictionary with module name mapped to where you should save the file to
    :param cci_features: cotton candy interface for save bucket. Defaults to None.
    :return sample: dict, audio sample containing features
    """
    assert '.npz' not in str(features_save_path), 'Extension should not be included in features_save_path'
    assert '.npz' not in str(times_save_path), 'Extension should not be included in times_save_path'
    if module_save_paths is not None:
        for m in module_save_paths:
            assert '.npz' not in str(m), 'Extension should not be included in module_save_paths'
    else:
        print('Skipping module feature loading. Please confirm if this is desired for your feature type.')

    if cci_features is not None:
        #load feature from bucket
        print('TODO')
        raise NotImplementedError('Loading features from bucket not yet supported')
    else:
        assert Path(features_save_path).parent.exists()
        assert Path(times_save_path).parent.exists()

        out_features = np.load(str(features_save_path)+'.npz')
        times = np.load(str(times_save_path)+'.npz')

        if module_save_paths is not None:
            module_features = {}
            for m in module_save_paths:
                m = Path(m)
                assert m.parent.exists()
                module_name = str(m.name)
                module_features[module_name] = np.load(str(m)+'.npz')
    
    sample = {'out_features':out_features, 'times':times, 'module_features':module_features}
    return sample

