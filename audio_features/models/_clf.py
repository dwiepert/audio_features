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
from typing import Union, Dict
##third-party
import numpy as np
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold, StratifiedKFold
##local
from database_utils.functions import *
from ._base_model import BaseModel

class LinearClassification(BaseModel):
    """
    :param iv: dict, independent variable(s), keys are stimulus names, values are array like features
    :param iv_type: str, type of feature for documentation purposes
    :param dv: dict, dependent variable(s), keys are stimulus names, values are array like features
    :param dv_type: str, type of feature for documentation purposes
    :param save_path: path like, path to save features to (can be cc path or local path)
    :param classification_type: str, multiclass_clf or multilabel_clf
    :param n_splits: int, number of cross validation splits (default = 5)
    :param n_repeats: int, number of cross validation repeats (default = 3)
    :param cci_features: cotton candy interface for saving
    :param overwrite: bool, indicate whether to overwrite values
    :param local_path: path like, path to save config to locally if save_path is not local
    """
    def __init__(self, iv:Dict[str,Dict[str, np.ndarray]], iv_type:str, dv:dict, dv_type:str, save_path:Union[str,Path], classification_type:str='multiclass_clf',
                        n_splits:int=5, n_repeats:int=3, cci_features=None, overwrite:bool=False, local_path:Union[str,Path]=None):
        self.classification_type=classification_type
        assert self.classification_type in ['multiclass_clf', 'multilabel_clf'], 'Invalid classification_type.'
        self.n_splits = n_splits #not currently used
        self.n_repeats = n_repeats #not currently used
        self.keep = None
        self.remove = None

        add_config = {
            'classification_type': self.classification_type, 'n_splits': self.n_splits, 'n_repeats': self.n_repeats
        }

        super().__init__(model_type='clf', iv=iv, iv_type=iv_type, dv=dv, dv_type=dv_type, config=add_config, save_path=save_path, cci_features=cci_features,
                          overwrite=overwrite, local_path=local_path)
        
        self._check_previous()
        self._fit()

    def _process_targets(self, targets:np.ndarray) -> np.ndarray:
        """
        Process targets to be compatible with sklearn LogisticRegression 

        :param targets: numpy array of target labels
        :return targets: numpy array of modified target labels (dimension/changes)
        """
        if targets.ndim == 2 and self.classification_type=='multiclass_clf':
            if targets.shape[1] == 1:
                targets = np.squeeze(targets)
            else:
                targets = np.argmax(targets, axis=1)

        elif targets.ndim ==2 and self.classification_type == 'multilabel_clf':
            if self.keep is None:
                self.keep = []
                self.remove = []
                for i in range(targets.shape[1]):
                    count = np.sum(targets[:,i])
                    if count > 0 :
                        self.keep.append(i)
                    else:
                        self.remove.append(i)

            targets = targets[:,self.keep]

        print(f'Targets shape: {targets.shape}')
        return targets
  
    def _fit(self):
        """
        Fit a classification model
        """
        if self.weights_exist and not self.overwrite:
            assert self.model is not None, 'Weights do not exist. Loading weights went wrong.'
            print('Weights already exist and should not be overwritten')
            krpath = self.result_paths['train_eval'].parent / 'keepremove.json'
            if self.classification_type=='multilabel_clf':
                assert krpath.exists()
                with open(str(krpath), 'r') as f:
                    kr = json.load(f)
                self.keep = kr['keep']
                self.remove = kr['remove']
            self.dv = self._process_targets(self.dv)
            #return
        else:
            st = time.time()
            print('Fitting model...')

            self.scaler = StandardScaler().fit(self.iv)
            #cv = StratifiedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=1)

            scaled_iv = self.scaler.transform(self.iv)
            self.dv = self._process_targets(self.dv)

            self.model = LogisticRegression(max_iter=1000)
            #self.model = LogisticRegressionCV(cv=self.n_splits,random_state=1, max_iter=1000)

            if self.classification_type=='multilabel_clf':
                self.model = MultiOutputClassifier(self.model)
            
            #cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

            #self.model = RidgeClassifierCV(alphas=self.alphas, cv=cv,scoring=self.scoring)
            
            self.model.fit(scaled_iv, self.dv)
            et = time.time()
            total = (et-st)/60
            print(f'Model fit in {total} minutes.')
            self._save_model(self.model, self.scaler)

            if self.keep is not None:
                out = {'keep': self.keep, 'remove': self.remove}
                os.makedirs(str(self.result_paths['train_eval'].parent), exist_ok=True)
                with open(str(self.result_paths['train_eval'].parent / 'keepremove.json'), 'w') as f:
                    json.dump(out,f)

        preds = self.model.predict(self.scaler.transform(self.iv))
        eval = self.eval_model(self.dv, preds)  
        os.makedirs(str(self.result_paths['train_eval'].parent), exist_ok=True)
        with open(str(self.result_paths['train_eval'])+'.json', 'w') as f:
            json.dump(eval,f)

    def eval_model(self, true:np.ndarray, pred:np.ndarray) -> Dict[str, np.ndarray]:
        """
        Override base model eval_model function for use with classification metrics.

        :param true: np.ndarray, true values
        :param pred: np.ndarray, predicted values
        :return metrics: dictionary of values
        """
        acc = accuracy(true, pred)
        balanced_acc, recall, precision, f1 = clf_metrics(true, pred)

        if self.remove is not None:
            for r in self.remove:
                acc.insert(r, np.nan)
                balanced_acc.insert(r, np.nan)
                recall.insert(r, np.nan)
                precision.insert(r, np.nan)
                f1.insert(r, np.nan)

        metrics = {'accuracy': acc,'balanced_accuracy':balanced_acc, 'precision': precision, 'recall': recall, 'f1': f1}

        return metrics


    def score(self, feats: Dict[str, Dict[str, np.ndarray]], ref_feats:Dict[str, Dict[str, np.ndarray]], fname:str) -> tuple[Dict[str,np.ndarray], Dict[str,np.ndarray]]:
        """
        Score classification model

        :param feats: dict, feature dictionary, stimulus names as keys
        :param ref_feats: dict, feature dictionary of ground truth predicted features, stimulus names as keys
        :param fname: str, name of stimulus to extract for
        :return per_story_metrics: dictionary of model eval metrics
        :return: Dictionary of true and predicted values 
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
        if not isinstance(self.model, MultiOutputClassifier):
            f = f.astype(self.model.coef_.dtype)
        pred = self.model.predict(self.scaler.transform(f))

        per_story_metrics = self.eval_model(rf, pred)
        self._save_metrics(per_story_metrics, fname, 'eval')

       
        return per_story_metrics, {'true': rf, 'pred':pred}