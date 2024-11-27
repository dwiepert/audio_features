"""
Run Ridge Regression

Author(s): Rachel Yam
Last Modified: 11/23/2024
"""
#IMPORTS
##built-in
import json
import os
from pathlib import Path
from typing import Union, List

##third-party
import numpy as np
#ridge regression
#from sklearn.linear_model import ridge_regression
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedKFold
import pickle

##local
from ._utils import _zscore

class RRegression:
    """
    :param iv: dict, independent variable(s), keys are stimulus names, values are array like features
    :param iv_type: str, type of feature for documentation purposes
    :param dv: dict, dependent variable(s), keys are stimulus names, values are array like features
    :param dv_type: str, type of feature for documentation purposes
    :param save_path: path like, path to save features to (can be cc path or local path)
    :param cci_features: cotton candy interface for saving
    :param overwrite: bool, indicate whether to overwrite values
    :param local_path: path like, path to save config to locally if save_path is not local
    """
    def __init__(self, iv:dict, iv_type:str, dv:dict, dv_type:str, save_path:Union[str,Path], zscore:bool=True,
                 cci_features=None, overwrite:bool=False, local_path:Union[str,Path]=None):
         self.iv_type = iv_type
         self.dv_type = dv_type
    
         self.fnames = list(iv.keys())
    
         self.iv, self.iv_rows, self.iv_times = self._process_features(iv)
         self.dv, self.dv_rows, self.dv_times = self._process_features(dv)
    
         self._check_rows()
    
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
        
         self.overwrite=overwrite
    
         self.config = {
             'iv_type': self.iv_type,'iv_shape': self.iv.shape, 'iv_rows': self.iv_rows, 
             'dv_type': self.dv_type, 'dv_shape': self.dv.shape, 'dv_rows':self.dv_rows,
             'train_fnames': self.fnames, 'overwrite': self.overwrite
         }
    
         os.makedirs(self.local_path, exist_ok=True)
         with open(str(self.local_path / 'ridgeRegression_config.json'), 'w') as f:
             json.dump(self.config,f)
         
         self.result_paths = {'pcorr': self.save_path/'pcorr'}
         self.result_paths['corr'] = {}
        
         for f in self.fnames:
             self.result_paths['corr'][f] = self.save_path / f
          
         #self._check_previous()
         self.pearson_coeff = None
         self.trained_model = None
    
    
    def run_regression(self):
        # cross-validation
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

        # try to find optimal alpha
        ridge_reg = RidgeCV(alphas=np.arange(0, 1, 0.01), cv=cv, scoring='neg_mean_absolute_error')

        # Fit model with best alpha
        ridge_reg.fit(self.iv, self.dv)
        self.trained_model = ridge_reg

        # predict
        ridge_predict = ridge_reg.predict(self.iv)

        # pearson coefficients
        correlations = []
        for i in range(self.dv.shape[1]):
            corr = np.corrcoef(self.dv[:, i], ridge_predict[:, i])[0, 1]
            correlations.append(corr)
        
        # ridge regression solution
        coefficients = ridge_reg.coef_
        
     
        # pearson correlations
        self.pearson_coeff = np.array(correlations)

        # Save the model to a file using pickle
        with open(self.save_path / 'ridge_regression_model.pkl', 'wb') as file:
            pickle.dump(ridge_reg, file)
            
        if self.cci_features:
            self.cci_features.upload_raw_array(self.result_paths['pcorr'], self.pearson_coeff)
                                           
        else:
            os.makedirs(self.save_path, exist_ok=True)
            np.savez_compressed(str(self.result_paths['pcorr'])+'.npz', self.pearson_coeff)
          


        
    def _process_features(self, feat:dict):
        """
        Concatenate features from separate files into one and maintain information to undo concatenation
        
        :param feat: dict, feature dictionary, stimulus names as keys
        :return concat: np.ndarray, concatenated array
        :return nrows: dict, start/end indices for each stimulus
        :return concat_times: np.ndarray, concatenated tiems array
        """
        nrows = {}
        concat = None 
        concat_times = None
        for f in self.fnames:
            temp = feat[f]
            n = temp['features']
            t = temp['times']
            if concat is None:
                concat = n 
                concat_times = t
                start_ind = 0
            else:
                concat = np.vstack((concat, n))
                concat_times = np.vstack((concat_times,t))
                start_ind = concat.shape[0]-1
            
            end_ind = start_ind + n.shape[0]
            nrows[f] = [start_ind, end_ind]
        return concat, nrows, concat_times
    
    
    def _unprocess_features(self, concat, nrows, concat_times):
        """
        Undo concatenation process
        """
        feats = {}
        for f in nrows:
            inds = nrows[f]
            n = concat[inds[0]:inds[1],:]
            t = concat_times[inds[0]:inds[1],:]
            feats[f] = {'features':n, 'times':t}
        return feats
    
    def _check_rows(self):
        for s in list(self.iv_rows.keys()):
            assert all(np.equal(self.iv_rows[s], self.dv_rows[s])), f'Stimulus {s} has inconsistent sizes. Please check features.'
    
    
    def extract_residuals(self, feats, ref_feats, fname):
        """
        Extract residuals for a set of features
        """
        #with open(self.save_path / 'ridge_regression_model.pkl', 'rb') as file:
         #   self.trained_model = pickle.load(file)
        if self.cci_features is not None:
            if self.cci_features.exists_object(self.result_paths['corr'][fname]) and not self.overwrite:
                return 
        else:
            if Path(str(self.result_paths['corr'][fname]) + '.npz').exists() and not self.overwrite:
                return 
        
        #assert self.pearson_coeff is not None, 'Regression has not been run yet. Please do so.'
        f = feats['features']
        t = feats['times']
        rf = ref_feats['features']
        rt = ref_feats['times']
        
        assert np.equal(t, rt).all(), f'Time alignment skewed across features for stimulus {fname}.'
        f = f.astype(self.trained_model.coef_.dtype)
        pred = self.trained_model.predict(f)
        
        correlations = []
        for i in range(rf.shape[1]):
            corr = np.corrcoef(rf[:, i], pred[:, i])[0, 1]
            correlations.append(corr)
        r2 = np.mean(correlations)
        


        if fname not in self.result_paths['corr']:
            self.result_paths['corr'][fname] = self.save_path / fname
        
        if self.cci_features:
            print('features')
            self.cci_features.upload_raw_array(self.result_paths['corr'][fname], r2)
            print(self.result_paths['residuals'][fname])
        else:
            print('not features')
            os.makedirs(self.result_paths['corr'][fname].parent, exist_ok=True)
            np.savez_compressed(str(self.result_paths['corr'][fname])+'.npz', r2)