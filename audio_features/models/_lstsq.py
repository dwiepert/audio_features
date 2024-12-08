"""
Run Least squares regression

Author(s): Daniela Wiepert
Last Modified: 12/04/2024
"""
#IMPORTS
##built-in
import json
import os
from pathlib import Path
from typing import Union,Dict
##third-party
import numpy as np
from sklearn.metrics import r2_score, f1_score,  mean_squared_error
from sklearn.preprocessing import StandardScaler
##local
from ._base_model import BaseModel

class LSTSQRegression(BaseModel):
    """
    :param iv: dict, independent variable(s), keys are stimulus names, values are array like features
    :param iv_type: str, type of feature for documentation purposes
    :param dv: dict, dependent variable(s), keys are stimulus names, values are array like features
    :param dv_type: str, type of feature for documentation purposes
    :param save_path: path like, path to save features to (can be cc path or local path)
    :param zscore: bool, indicate whether to zscore
    :param cci_features: cotton candy interface for saving
    :param overwrite: bool, indicate whether to overwrite values
    :param local_path: path like, path to save config to locally if save_path is not local
    """
    def __init__(self, iv:Dict[str,np.ndarray], iv_type:str, dv:Dict[str,np.ndarray], dv_type:str, save_path:Union[str,Path],
                 cci_features=None, overwrite:bool=False, local_path:Union[str,Path]=None):
        
        super().__init__(model_type='lstsq', iv=iv, iv_type=iv_type, dv=dv, dv_type=dv_type,
                            config={}, save_path=save_path, cci_features=cci_features, overwrite=overwrite, local_path=local_path)

        new_path = self.save_path / 'model'
        self.result_paths['weights']= new_path /'weights'
        self.result_paths['emawav'] = {}
        for f in self.fnames:
             save = save_path / 'emawav'
             self.result_paths['emawav'][f] = save / f

        self._check_previous()
        self._fit()
    
    def _check_previous(self):
        """
        Check if weights already exist and load them - LSTSQ specific (override BaseModel)
        """
        self.weights_exist = False
        if self.cci_features is not None:
            if self.cci_features.exists_object(str(self.save_path)):
                if self.cci_features.exists_object(str(self.result_paths['weights'])): self.weights_exist=True
        else:
            if Path(str(self.result_paths['weights'])+'.npz').exists(): self.weights_exist=True

        if self.weights_exist and not self.overwrite: 
            if self.cci_features is not None:
                self.wt = self.cci_features.download_raw_array(str(self.result_paths['weights']))
            else:
                temp = np.load(str(self.result_paths['weights'])+'.npz')
                k = list(temp)[0]
                self.wt = temp[k]
        else:
            self.wt = None 

    
    def _fit(self):
        """
        Run least squares regression
        Saves features, sets self.wt to the learned weights
        """
        #self.scaler = StandardScaler().fit(self.iv)
        if self.weights_exist and not self.overwrite:
            assert self.wt is not None, 'Weights do not exist. Loading weights went wrong.'
            print('Weights already exist and should not be overwritten')
        else:
            print(f'DV shape: {self.dv.shape}')
            print(f'IV shape: {self.iv.shape}')
            x, residuals, rank, s = np.linalg.lstsq(self.iv, self.dv, rcond=None)
            
            self.wt = x 

        #train metrics
        pred_dv = self.iv @ self.wt
        metrics = self.eval_model(self.dv, pred_dv)


        if self.cci_features:
            self.cci_features.upload_raw_array(self.result_paths['weights'], self.wt)
            #self.cci_features.upload_raw_array(self.result_paths['train_eval'], metrics)
            #self.cci_features.upload_raw_array(self.result_paths['lstsq_residuals'], residuals)
        else:
            os.makedirs(self.result_paths['weights'].parent, exist_ok=True)
            np.savez_compressed(str(self.result_paths['weights'])+'.npz', self.wt)
            #np.savez_compressed(str(self.result_paths['train_eval'])+'.npz', metrics)
            #np.savez_compressed(str(self.result_paths['lstsq_residuals']['concat']+'.npz'), residuals)
        
        #config_name = self.local_path / f'{self.model_type}_config.json'
        os.makedirs(str(self.result_paths['train_eval'].parent), exist_ok=True)
        with open(str(self.result_paths['train_eval'])+'.json', 'w') as f:
            json.dump(metrics,f)

    def score(self, feats:Dict[str,np.ndarray], ref_feats:Dict[str,np.ndarray], fname:str) -> tuple[np.ndarray, Dict[str,np.ndarray]]:
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
        
        assert self.wt is not None, 'Regression has not been run yet. Please do so.'
        
        f = feats['features']
        t = feats['times']
        rf = ref_feats['features']
        rt = ref_feats['times']

        assert np.equal(t, rt).all(), f'Time alignment skewed across features for stimulus {fname}.'
        f = f.astype(self.wt.dtype)
        pred = f @ self.wt 

        r = np.subtract(pred, rf)

        self._save_metrics(r, fname, 'metric')
        self._save_metrics(pred, fname, 'emawav')

        per_story_metrics = self.eval_model(rf, pred)
        self._save_metrics(per_story_metrics, fname, 'eval')

        return r, {'true': rf, 'pred':pred}