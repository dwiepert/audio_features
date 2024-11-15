"""
Run Least squares regression

Author(s): Daniela Wiepert
Last Modified: 11/14/2024
"""
#IMPORTS
##built-in
import json
import os
from pathlib import Path
from typing import Union, List

##third-party
import numpy as np

##local
from ._utils import _zscore

class LSTSQRegression:
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
    def __init__(self, iv:dict, iv_type:str, dv:dict, dv_type:str, save_path:Union[str,Path], zscore:bool=True,
                 cci_features=None, overwrite:bool=False, local_path:Union[str,Path]=None):
        self.iv = iv
        self.iv_type = iv_type
        self.dv = dv
        self.dv_type = dv_type

        self.fnames = list(self.iv.keys())

        self.iv, self.iv_rows = self._process_features(self.iv)
        self.dv, self.dv_rows = self._process_features(self.dv)

        #TODO: some kind of input to understand how to break apart the concatenated information? or concatenate in here?
        if zscore:
            self.iv = _zscore(self.iv)
            self.dv = _zscore(self.dv)

        self.save_path=Path(save_path)
        if local_path is None or self.cci_features is None:
            self.local_path = self.save_path
        else:
            self.local_path = Path(local_path)
        self.cci_features = cci_features
        if self.cci_features is None:
            self.local_path = self.save_path
        else:
            assert self.local_path is not None, 'Please give a local path to save config files to (json not cotton candy compatible)'
        #self.fnames=self.fnames
        self.overwrite=overwrite

        self.config = {
            'iv_type': self.iv_type,'iv_shape': self.iv.shape, 'iv_rows': self.iv_rows, 'dv_type': self.dv_type, 'dv_shape': self.dv.shape, 'dv_rows':self.dv_rows,
            'train_fnames': self.fnames, 'overwrite': self.overwrite
        }

        os.makedirs(self.local_path, exist_ok=True)
        with open(str(self.local_path / 'LSTSQRegression_config.json'), 'w') as f:
            json.dump(self.config,f)
        
        rr_path = self.save_path /'lstsq_residuals'
        self.result_paths = {'weights': self.save_path/'weights'}
        self.result_paths['residuals'] = {}
        #self.result_paths['lstsq_residuals'] = {}
        r_path = self.save_path / 'residuals'
        for f in self.fnames:
            self.result_paths['residuals'][f] = r_path / f
            #self.result_paths['lstsq_residuals'][f] = rr_path /f
        #self.result_paths['lstsq_residuals']['concat'] = rr_path
        self._check_previous()
    
    def _process_features(self, feat:dict):
        """
        Concatenate features from separate files into one and maintain information to undo concatenation
        
        :param feat: dict, feature dictionary, stimulus as keys, values are arrays
        :return concat: np.ndarray, concatenated array
        :return nrows: dict, start/end indices for each stimulus
        """
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
    
    def _unprocess_features(self, concat, nrows):
        """
        Undo concatenation process
        """
        feats = {}
        for f in nrows:
            inds = nrows[f]
            feats[f] = concat[inds[0]:inds[1],:]
        return feats
    
    def _check_previous(self):
        """
        Check if weights already exist and load them
        """
        self.weights_exist = False
        if self.cci_features is not None:
            if self.cci_features.exists_object(str(self.save_path)):
                if self.cci_features.exists_object(str(self.results_paths['weights'])): self.weights_exist=True
            else:
                if Path(str(self.results_paths['weights'])+'.npz').exists(): self.weights_exist=True

        if self.weights_exist and not self.overwrite: 
            if self.cci_features is not None:
                self.wt = self.cci_features.download_raw_array(str(self.results_paths['weights']))
            else:
                self.wt = np.load(str(self.results_paths['weights'])+'.npz')
        else:
            self.wt = None 
    
    def run_regression(self):
        """
        Run least squares regression
        Saves features, sets self.wt to the learned weights
        """
        if self.wt is not None and not self.overwrite:
            print('Weights already exist and should not be overwritten')
            return
        print('CHANGE BACK')
        x, residuals, rank, s = np.linalg.lstsq(self.iv, self.dv)
        
        self.wt = x 

        if self.cci_features:
            self.cci_features.upload_raw_array(self.result_paths['weights'], self.wt)
            #self.cci_features.upload_raw_array(self.result_paths['lstsq_residuals'], residuals)
        else:
            os.makedirs(self.save_path, exist_ok=True)
            np.savez_compressed(str(self.result_paths['weights']+'.npz'), self.wt)
            #np.savez_compressed(str(self.result_paths['lstsq_residuals']['concat']+'.npz'), residuals)

        return

    def extract_residuals(self, feats, fname):
        """
        Extract residuals for a set of features
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
            os.makedirs(r_path, exist_ok=True)
            self.result_paths['residuals'][fname] = r_path / fname
        
        if self.cci_features:
            self.cci_features.upload_raw_array(self.result_paths['residuals'][fname], r)
        else:
            os.makedirs(self.result_paths['residuals'][fname].parent, exist_ok=True)
            np.savez_compressed(str(self.result_paths['residuals'][fname])+'.npz', r)
