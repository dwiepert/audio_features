"""
Run linear classification

Author(s): Daniela Wiepert
Last modified: 11/28/2024
"""
#IMPORTS
##built-in
import json
import os
from pathlib import Path
import pickle
from typing import Union, List

##third-party
import numpy as np
from sklearn.linear_model import RidgeClassifierCV, LogisticRegression
from sklearn import preprocessing
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import RepeatedKFold

##local
from database_utils.functions import zscore as _zscore

class LinearClassification:
    """
    Types of scoring https://scikit-learn.org/1.5/modules/model_evaluation.html#scoring-parameter
    """
    def __init__(self, iv:dict, iv_type:str, dv:dict, dv_type:str, metric_type:str, save_path:Union[str,Path], classification_type:str='multiclass_clf', zscore:bool=True,
                       cci_features=None, overwrite:bool=False, local_path:Union[str,Path]=None):
        self.iv_type = iv_type
        self.dv_type = dv_type
        self.overwrite = overwrite
        self.metric_type= metric_type
        self.classification_type=classification_type

        self.fnames = list(iv.keys())
        self.zscore = False
        self.iv, self.iv_rows, self.iv_times = self._process_features(iv)
        self.dv, self.dv_rows, self.dv_times = self._process_features(dv)

        if self.zscore:
            self.iv = _zscore(self.iv)

        self._check_rows()

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

        self.config = {
            'iv_type': self.iv_type,'iv_shape': self.iv.shape, 'iv_rows': self.iv_rows, 
            'dv_type': self.dv_type, 'dv_shape': self.dv.shape, 'dv_rows':self.dv_rows,
            'classification_type': self.classification_type,
            'metric_type':self.metric_type, 'train_fnames': self.fnames, 'overwrite': self.overwrite
        }

        os.makedirs(self.local_path, exist_ok=True)
        with open(str(self.local_path / 'LinearClassification_config.json'), 'w') as f:
            json.dump(self.config,f)
        
        if self.cci_features is None:
            self.result_paths = {'model': self.save_path/'model'}
        else:
            self.result_paths = {'model': self.local_path/'model'}
        
        self.result_paths[self.metric_type] = {}

        for f in self.fnames:
            self.result_paths[self.metric_type][f] = self.save_path / f

        self._check_previous()
        self._fit()

    def _process_targets(self, feat:dict):
        """
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
    

    def _process_features(self, feat:dict):
        """
        Concatenate features from separate files into one and maintain information to undo concatenation
        
        :param feat: dict, feature dictionary, stimulus names as keys
        :param zscore: bool, indicate whether to zscore
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
        Check if all rows are equal
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
        """
        if self.weights_exist and not self.overwrite:
            assert self.model is not None, 'Weights do not exist. Loading weights went wrong.'
            print('Weights already exist and should not be overwritten')
            return
        
        self.scaler = preprocessing.StandardScaler().fit(self.iv)

        self.model = LogisticRegression(random_state=0, max_iter=1000)

        if self.classification_type=='multilabel_clf':
            self.model = MultiOutputClassifier(self.model)
        

        #cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

        #self.model = RidgeClassifierCV(alphas=self.alphas, cv=cv,scoring=self.scoring)
        if self.dv.ndim == 2:
            self.dv = np.argmax(self.dv, axis=1)
        
        self.model.fit(self.scaler.transform(self.iv), self.dv)

        if self.cci_features:
            print('Model cannot be saved to cci_features. Saving to local path instead')
        
            #self.cci_features.upload_raw_array(self.result_paths['models'], self.ridge)

        # Save the model to a file using pickle
        os.makedirs(self.result_paths['model'], exist_ok=True)
        with open(str(self.result_paths['model'])+'.pkl', 'wb') as file:
            pickle.dump(self.model, file)

    def score(self, feats, ref_feats, fname):
        if self.cci_features is not None:
            if self.cci_features.exists_object(self.result_paths[self.metric_type][fname]) and not self.overwrite:
                return 
        else:
            if Path(str(self.result_paths[self.metric_type][fname]) + '.npz').exists() and not self.overwrite:
                return 
        
        #assert self.pearson_coeff is not None, 'Regression has not been run yet. Please do so.'
        assert self.model is not None, 'Regression has not been run yet. Please do so.'

        f = feats['features']
        if self.zscore:
            f = _zscore(f)
        t = feats['times']

        rf = ref_feats['features']
        if rf.ndim == 2:
            rf = np.argmax(rf, axis=1)
        rt = ref_feats['times']
        
        assert np.equal(t, rt).all(), f'Time alignment skewed across features for stimulus {fname}.'
        #TODO:
        f = f.astype(self.model.coef_.dtype)
        pred = self.model.predict(self.scaler.transform(f))

        #TODO: SCORING
        if self.metric_type == 'accuracy':
            metric = np.array([accuracy_score(rf, pred)])
        else:
            raise NotImplementedError()
        

        if fname not in self.result_paths[self.metric_type]:
            self.result_paths[self.metric_type][fname] = self.save_path / fname
        
        if self.cci_features:
            #print('features')
            self.cci_features.upload_raw_array(self.result_paths[self.metric_type][fname], metric)
            #print(self.result_paths['residuals'][fname])
        else:
            #print('not features')
            os.makedirs(self.result_paths[self.metric_type][fname].parent, exist_ok=True)
            np.savez_compressed(str(self.result_paths[self.metric_type][fname])+'.npz', metric)

        return metric