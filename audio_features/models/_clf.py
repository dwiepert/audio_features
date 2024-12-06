"""
Run linear classification

Author(s): Daniela Wiepert
Last modified: 12/04/2024
"""
#IMPORTS
##built-in
import json
import os
from pathlib import Path
import time
from typing import Union
##third-party
import numpy as np
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
##local
from database_utils.functions import *
from ._base_model import BaseModel

class LinearClassification(BaseModel):
    """
    Types of scoring https://scikit-learn.org/1.5/modules/model_evaluation.html#scoring-parameter
    """
    def __init__(self, iv:dict, iv_type:str, dv:dict, dv_type:str, save_path:Union[str,Path], classification_type:str='multiclass_clf',
                        n_splits:int=5, n_repeats:int=3,
                       cci_features=None, overwrite:bool=False, local_path:Union[str,Path]=None):
        self.classification_type=classification_type
        self.n_splits = n_splits
        self.n_repeats = n_repeats

        add_config = {
            'classification_type': self.classification_type, 'n_splits': self.n_splits, 'n_repeats': self.n_repeats
        }

        super().__init__(model_type='clf', iv=iv, iv_type=iv_type, dv=dv, dv_type=dv_type, config=add_config, save_path=save_path, cci_features=cci_features,
                          overwrite=overwrite, local_path=local_path)
        
        self._check_previous()
        self._fit()

    def _process_targets(self, targets):
        if targets.ndim == 2 and self.classification_type=='multiclass_clf':
            if targets.shape[1] == 1:
                targets = np.squeeze(targets)
            else:
                targets = np.argmax(targets, axis=1)
        print(f'Targets shape: {targets.shape}')
        return targets
  
    def _fit(self):
        """
        """
        if self.weights_exist and not self.overwrite:
            assert self.model is not None, 'Weights do not exist. Loading weights went wrong.'
            print('Weights already exist and should not be overwritten')
            #return
        else:
            st = time.time()
            print('Fitting model...')

            self.scaler = StandardScaler().fit(self.iv)
            cv = RepeatedStratifiedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=1)

            #self.model = LogisticRegression(max_iter=1000)
            self.model = LogisticRegressionCV(cv=cv,random_state=1, max_iter=1000)

            #if self.classification_type=='multilabel_clf':
            #    self.model = MultiOutputClassifier(self.model)
            
            #cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

            #self.model = RidgeClassifierCV(alphas=self.alphas, cv=cv,scoring=self.scoring)
            self.dv = self._process_targets(self.dv)
            
            self.model.fit(self.scaler.transform(self.iv), self.dv)
            et = time.time()
            total = (et-st)/60
            print(f'Model fit in {total} minutes.')
            self._save_model(self.model, self.scaler)

        preds = self.model.predict(self.scaler.transform(self.iv))
        eval = self.eval_model(self.dv, preds)  
        os.makedirs(str(self.result_paths['train_eval'].parent), exist_ok=True)
        with open(str(self.result_paths['train_eval'])+'.json', 'w') as f:
            json.dump(eval,f)

    def eval_model(self, true, pred):
        acc = accuracy(true, pred)
        balanced_acc, recall, precision, f1 = clf_metrics(true, pred)

        metrics = {'accuracy': acc,'balanced_accuracy':balanced_acc, 'precision': precision, 'recall': recall, 'f1': f1}

        return metrics


    def score(self, feats, ref_feats, fname):
        """
        """
        #if fname in self.result_paths['metric']:
        #    if self.cci_features is not None:
        #        if self.cci_features.exists_object(self.result_paths['metric'][fname]) and not self.overwrite:
        #            return 
        #    else:
        #        if Path(str(self.result_paths['metric'][fname]) + '.npz').exists() and not self.overwrite:
        #            return 
        
        #assert self.pearson_coeff is not None, 'Regression has not been run yet. Please do so.'
        assert self.model is not None, 'Regression has not been run yet. Please do so.'

        f = feats['features']
        t = feats['times']

        rf = ref_feats['features']
        rt = ref_feats['times']

        rf = self._process_targets(rf)
        
        assert np.equal(t, rt).all(), f'Time alignment skewed across features for stimulus {fname}.'
        #TODO:
        f = f.astype(self.model.coef_.dtype)
        pred = self.model.predict(self.scaler.transform(f))

        per_story_metrics = self.eval_model(rf, pred)
        self._save_metrics(per_story_metrics, fname, 'eval')

       
        return per_story_metrics, {'true': rf, 'pred':pred}