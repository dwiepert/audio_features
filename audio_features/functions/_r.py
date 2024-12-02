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
from database_utils.functions import zscore as _zscore

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
    def __init__(self, iv:dict, iv_type:str, dv:dict, dv_type:str, save_path:Union[str,Path],alphas:np.ndarray=np.logspace(-5,2,num=10), n_splits:int=10, n_repeats:int=20, zscore:bool=True,
                 scoring:str='neg_mean_absolute_error',cci_features=None, overwrite:bool=False, local_path:Union[str,Path]=None):
         self.iv_type = iv_type
         self.dv_type = dv_type
         self.n_splits = n_splits
         self.n_repeats = n_repeats
         self.fnames = list(iv.keys())
         self.alphas = alphas
         self.scoring = scoring
         self.zscore = zscore
         self.iv, self.iv_rows, self.iv_times = self._process_features(iv)
         self.dv, self.dv_rows, self.dv_times = self._process_features(dv)
    
         self._check_rows()
    
         if zscore:
            self.iv, self.iv_unz = _zscore(self.iv, return_unzvals=True)
            self.dv, self.dv_unz = _zscore(self.dv,return_unzvals=True)
    
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
             'alphas': self.alphas.tolist(), 'scoring':self.scoring, 'n_splits': self.n_splits,
             'n_repeats': self.n_repeats, 'train_fnames': self.fnames, 'overwrite': self.overwrite
         }
    
         os.makedirs(self.local_path, exist_ok=True)
         with open(str(self.local_path / 'ridgeRegression_config.json'), 'w') as f:
             json.dump(self.config,f)
        
         if self.cci_features is None:
            self.result_paths = {'model': self.save_path/'model'}
         else:
            self.result_paths = {'model': self.local_path/'model'}
         #self.result_paths = {'pcorr': self.save_path/'pcorr'}
         self.result_paths['corr'] = {}
        
         for f in self.fnames:
             self.result_paths['corr'][f] = self.save_path / f
          
         self._check_previous()
         self._fit()
         #self.pearson_coeff = None
         #self.ridge_reg = None
    
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
            #if self.zscore:
            #    n = _zscore(n)
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

        :param concat: np.ndarray, concatenated array
        :param nrows: dict, start/end indices for each stimulus
        :param concat_times: np.ndarray, concatenated tiems array
        :return feats: dict, feature dictionary, stimulus names as keys
        """
        feats = {}
        for f in nrows:
            inds = nrows[f]
            n = concat[inds[0]:inds[1],:]
            t = concat_times[inds[0]:inds[1],:]
            feats[f] = {'features':n, 'times':t}
        return feats
    
    def _check_rows(self):
        """
        Check that all rows match
        """
        for s in list(self.iv_rows.keys()):
            assert all(np.equal(self.iv_rows[s], self.dv_rows[s])), f'Stimulus {s} has inconsistent sizes. Please check features.'
    
    def _check_previous(self):
        """
        """
        self.weights_exist = False
        if Path(str(self.result_paths['model'])+'.pkl').exists(): self.weights_exist=True

        if self.weights_exist and not self.overwrite:
            with open(str(self.result_paths['model'])+'.pkl', 'rb') as f:
                self.model = pickle.load(f)
        else:
            self.model=None

    def _fit(self):
        """
        Run regression
        """
        if self.weights_exist and not self.overwrite:
            print('Model already fitted and should not be overwritten')
            return
        # cross-validation
        cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=1)

        # try to find optimal alpha
        self.model = RidgeCV(alphas=self.alphas, cv=cv, scoring=self.scoring)

        # Fit model with best alpha
        self.model.fit(self.iv, self.dv)
        #self.trained_model = ridge_reg

        # predict
        #ridge_predict = self.ridge_reg.predict(self.iv)

        # pearson coefficients
        #correlations = []
        #for i in range(self.dv.shape[1]):
        #    corr = np.corrcoef(self.dv[:, i], ridge_predict[:, i])[0, 1]
        #    correlations.append(corr)
        
        # ridge regression solution
        #self.coefficients = self.ridge_reg.coef_
        
        # pearson correlations
        #self.pearson_coeff = np.array(correlations)

        if self.cci_features:
            print('Model cannot be saved to cci_features. Saving to local path instead')
        
            #self.cci_features.upload_raw_array(self.result_paths['models'], self.ridge)

        # Save the model to a file using pickle
        os.makedirs(self.result_paths['model'], exist_ok=True)
        with open(str(self.result_paths['model'])+'.pkl', 'wb') as file:
            pickle.dump(self.model, file)
            
        #if self.cci_features:
        #    self.cci_features.upload_raw_array(self.result_paths['pcorr'], self.pearson_coeff)
                                           
       #else:
        #    os.makedirs(self.save_path, exist_ok=True)
         #   np.savez_compressed(str(self.result_paths['pcorr'])+'.npz', self.pearson_coeff)
          
    
    def calculate_correlations(self, feats, ref_feats, fname):
        """
        Calculate average correlation

        :param feats: dict, feature dictionary, stimulus names as keys
        :param ref_feats: dict, feature dictionary of ground truth predicted features, stimulus names as keys
        :param fname: str, name of stimulus to extract for
        :param r2: averaged correlation across features
        """
        #with open(self.save_path / 'ridge_regression_model.pkl', 'rb') as file:
         #   self.trained_model = pickle.load(file)
        if fname in self.result_paths['corr']:
            if self.cci_features is not None:
                if self.cci_features.exists_object(self.result_paths['corr'][fname]) and not self.overwrite:
                    return 
            else:
                if Path(str(self.result_paths['corr'][fname]) + '.npz').exists() and not self.overwrite:
                    return 
        
        #assert self.pearson_coeff is not None, 'Regression has not been run yet. Please do so.'
        assert self.model is not None, 'Regression has not been run yet. Please do so.'

        f, f_unz = _zscore(feats['features'], return_unzvals=True)
        t = feats['times']
        rf = ref_feats['features']
        rt = ref_feats['times']
        
        assert np.equal(t, rt).all(), f'Time alignment skewed across features for stimulus {fname}.'
        f = f.astype(self.model.coef_.dtype)
        pred = self.model.predict(f)
        
        correlations = []
        for i in range(rf.shape[0]):
            corr = np.corrcoef(pred[i,:], rf[i,:])[0, 1]
            correlations.append(corr)
        r2 = np.array(correlations)
        #r2 = np.mean(correlations)
        
        if fname not in self.result_paths['corr']:
            self.result_paths['corr'][fname] = self.save_path / fname
        
        if self.cci_features:
            #print('features')
            self.cci_features.upload_raw_array(self.result_paths['corr'][fname], r2)
            #print(self.result_paths['residuals'][fname])
        else:
            #print('not features')
            os.makedirs(self.result_paths['corr'][fname].parent, exist_ok=True)
            np.savez_compressed(str(self.result_paths['corr'][fname])+'.npz', r2)
        
        return r2