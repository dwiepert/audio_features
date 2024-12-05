"""
"""
#IMPORTS
##built-in
import json
import os
from typing import Dict, Union
from pathlib import Path
##third-party
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
##local
from ._base_model import BaseModel

class residualPCA(BaseModel):
    """
    :param iv: dict, independent variable(s), keys are stimulus names, values are array like features
    :param iv_type: str, type of feature for documentation purposes
    :param save_path: path like, path to save features to (can be cc path or local path)
    :param zscore: bool, indicate whether to zscore
    :param cci_features: cotton candy interface for saving
    :param overwrite: bool, indicate whether to overwrite values
    :param local_path: path like, path to save config to locally if save_path is not local
    """
    def __init__(self, iv:Dict[str,np.ndarray], iv_type:str, save_path:Union[str,Path], n_components:int=13,
                 cci_features=None, overwrite:bool=False, local_path:Union[str,Path]=None):
        
        self.n_components = n_components
        super().__init__(model_type='pca', iv=iv, iv_type=iv_type, dv=iv, dv_type=iv_type, 
                            config={'n_components':self.n_components}, save_path=save_path, cci_features=cci_features, overwrite=overwrite, local_path=local_path)

        #new_path = self.save_path / 'model'
        #self.result_paths['weights']= new_path /'weights'
        #self.result_paths['emawav'] = {}
        #for f in self.fnames:
        #     save = save_path / 'emawav'
        #     self.result_paths['emawav'][f] = save / f

        self._check_previous()
        self._fit()
    
    def _fit(self):
        """
        """
        if self.weights_exist and not self.overwrite:
            print('Model already fitted and should not be overwritten')
        else:
            self.scaler = StandardScaler().fit(self.iv)
            self.model = PCA(n_components=self.n_components)
            self.model.fit(self.scaler.transform(self.iv))

            self._save_model(self.model, self.scaler)
            eval = {'explained_variance_ratio':[float(f) for f in list(self.model.explained_variance_ratio_)]}
            os.makedirs(str(self.result_paths['train_eval'].parent), exist_ok=True)
            with open(str(self.result_paths['train_eval'])+'.json', 'w') as f:
                json.dump(eval, f)
    
    def score(self, feats:Dict[str,np.ndarray], feats2:Dict[str,np.ndarray], fname:str) -> tuple[np.ndarray, Dict[str,np.ndarray]]:
        """
        Extract residuals for a set of features

        :param feats: dict, feature dictionary, stimulus names as keys
        :param ref_feats: dict, feature dictionary of ground truth predicted features, stimulus names as keys
        :param fname: str, name of stimulus to extract for
        :return r: extracted residuals
        :return: Dictionary of true and predicted values 
        """
        #if self.cci_features is not None:
        #    if self.cci_features.exists_object(self.result_paths['metric'][fname]) and not self.overwrite:
        #        return 
        #else:
        #    if Path(str(self.result_paths['metric'][fname]) + '.npz').exists() and not self.overwrite:
        #        return 
        
        assert self.model is not None, 'PCA has not been run yet. Please do so.'
        
        f = feats['features']
        t = feats['times']

        pca = self.model.transform(self.scaler.transform(f))

        self._save_metrics(pca, fname, 'metric')
       

        return pca, {'true': f, 'pred':pca}