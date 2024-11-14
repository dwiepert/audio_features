"""
"""
import json
from pathlib import Path
from typing import Union, List
import numpy as np
from ._utils import _zscore

class LSTSQRegression:
    """
    :param iv: array like, independent variable(s)
    :param dv: array like, dependent variable(s)
    """
    def __init__(self, iv:dict, iv_type:str, dv:dict, dv_type:str, save_path:Union[str,Path], zscore:bool=True,
                 cci_features=None, overwrite:bool=False, local_path:Union[str,Path]=None):
        self.iv = iv
        self.iv_type = iv_type
        self.dv = dv
        self.dv_type = dv_type

        self.fnames = list(self.iv.keys())

        self.iv, self.iv_rows = self._process_features(iv)
        self.dv, self.dv_rows = self._process_features(dv)

        #TODO: some kind of input to understand how to break apart the concatenated information? or concatenate in here?
        if zscore:
            self.iv = _zscore(self.iv)
            self.dv = _zscore(self.dv)

        self.save_path=Path(save_path)
        self.cci_features = cci_features
        if self.cci_features is None:
            self.local_path = self.save_path
        else:
            assert self.local_path is not None, 'Please give a local path to save config files to (json not cotton candy compatible)'
        self.fnames=self.fnames
        self.overwrite=overwrite

        self.config = {
            'iv_type': self.iv_type,'iv_shape': self.iv.shape(), 'iv_rows': self.iv_rows, 'dv_type': self.dv_type, 'dv_shape': self.dv.shape(), 'dv_rows':self.dv_rows,
            'train_fnames': self.fnames, 'overwrite': self.overwrite
        }

        with open(str(self.local_path / 'LSTSQRegression_config.json'), 'w') as f:
            json.dump(self.config)
        
        self.result_paths = {'weights': self.save_path/'weights', 'lstsq_residuals': self.save_path/'lstsq_residuals'}
        self.result_paths['residuals'] = {}
        r_path = self.save_path / 'residuals'
        for f in self.fnames:
            self.result_paths['residuals'][f] = r_path / f

        self._check_previous()
    
    def _process_features(self, feat):
        nrows = {}
        concat = None 
        for f in self.fnames:
            n = feat[f]
            if concat is None:
                concat = n 
                start_ind = 0
            else:
                concat = np.vstack((concat, n))
                start_ind = concat.shape[0]-1
            
            end_ind = start_ind + n.shape[0]
            nrows[f] = [start_ind, end_ind]
        return concat, nrows

    def _check_previous(self):
        self.weights_exist = False
        if self.cci_features is not None:
            if self.cci_features.exists_object(str(self.save_path)):
                if self.cci_features.exists_object(str(self.results_paths['weights'])): self.weights_exist=True
                if self.cci_features.exists_object(str(self.results_paths['valpha'])): self.valphas_exist=True 
            else:

                if Path(str(self.results_paths['weights'])+'.npz').exists(): self.weights_exist=True
                if Path(str(self.results_paths['valphas'])+'.npz'): self.valphas_exist = True

        if self.weights_exist and not self.overwrite: 
            if self.cci_features is not None:
                self.wt = self.cci_features.download_raw_array(str(self.results_paths['weights']))
            else:
                self.wt = np.load(str(self.results_paths['weights'])+'.npz')
        else:
            self.wt = None 
    
    def run_regression(self):
        if self.wt is not None and not self.overwrite:
            print('Weights already exist and should not be overwritten')
            return

        x, residuals, rank, s = np.linalg.lstsq(self.iv, self.dv)
        
        self.wt = x 

        if self.cci_features:
            self.cci_features.upload_raw_array(self.result_paths['weights'], self.wt)
            self.cci_features.upload_raw_array(self.result_paths['lstsq_residuals'], residuals)
        else:
            np.savez_compressed(str(self.result_paths['weights']+'.npz'), self.wt)
            np.savez_compressed(str(self.result_paths['lstsq_residuals']+'.npz'), residuals)
        
        return

    def extract_residuals(self, feats, fname):
        """
        """
        if self.cci_features is not None:
            if self.cci_features.exists_object(self.result_paths['residuals'][fname]) and not self.overwrite:
                return 
        else:
            if Path(str(self.result_paths['residuals'][fname]) + '.npz').exists() and not self.overwrite:
                return 
        
        assert self.wt is not None, 'Regression has not been run yet. Please do so.'

        feats = feats.astype(self.wt.dtype)
        pred = feats @ self.wt 
        r = np.substract(pred, feats)

        if fname not in self.results_paths['residuals']:
            r_path = self.save_path / 'residuals'
            self.result_paths['residuals'][fname] = r_path / fname
        
        if self.cci_features:
            self.cci_features.upload_raw_array(self.result_paths['residuals'][fname], residuals)
        else:
            np.savez_compressed(str(self.result_paths['residuals'][fname])+'.npz', residuals)
