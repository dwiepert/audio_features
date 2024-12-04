"""
Run Ridge Regression

Author(s): Rachel Yam
Last Modified: 12/04/2024
"""
#IMPORTS
##built-in
from pathlib import Path
import time
from typing import Union, Dict
##third-party
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold
##local
from ._base_model import BaseModel

class RRegression(BaseModel):
    """
    :param iv: dict, independent variable(s), keys are stimulus names, values are array like features
    :param iv_type: str, type of feature for documentation purposes
    :param dv: dict, dependent variable(s), keys are stimulus names, values are array like features
    :param dv_type: str, type of feature for documentation purposes
    :param save_path: path like, path to save features to (can be cc path or local path)
    :param alphas: np.ndarray of alphas to consider for ridge regression
    :param n_splits: int, number of cross validation splits (default = 5)
    :param n_repeats: int, number of cross validation repeats (default = 3)
    :param corr_type: str, specify whether to calculate correlation across features (feature) or times (time)
    :param cci_features: cotton candy interface for saving
    :param overwrite: bool, indicate whether to overwrite values
    :param local_path: path like, path to save config to locally if save_path is not local
    """
    def __init__(self, iv:Dict[str:np.ndarray], iv_type:str, dv:Dict[str:np.ndarray], dv_type:str, save_path:Union[str,Path],
                 alphas:np.ndarray=np.logspace(-5,2,num=10), n_splits:int=5, n_repeats:int=3, corr_type:str="feature",
                 cci_features=None, overwrite:bool=False, local_path:Union[str,Path]=None):
         
         #ridge specific
         self.n_splits = n_splits
         self.n_repeats = n_repeats
         self.alphas = alphas
         self.corr_type = corr_type

         add_config = {'alphas': self.alphas.tolist(), 'n_splits': self.n_splits,
             'correlation_type':self.corr_type, 'n_repeats': self.n_repeats
         }
         
         super().__init__(model_type='ridge', iv=iv, iv_type=iv_type, dv=dv, dv_type=dv_type, config=add_config, save_path=save_path, cci_features=cci_features,
                          overwrite=overwrite, local_path=local_path)
         
         self._check_previous()
         self._fit()

    def _fit(self):
        """
        Run regression
        """
        if self.weights_exist and not self.overwrite:
            print('Model already fitted and should not be overwritten')
            return
        # cross-validation
        cv = RepeatedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=1)

        # try to find optimal alpha
        self.model = RidgeCV(alphas=self.alphas, cv=cv)

        st = time.time()
        print('Fitting model...')
        # Fit model with best alpha
        self.scaler = StandardScaler().fit(self.iv)
        self.model.fit(self.scaler.transform(self.iv), self.dv)
        et = time.time()

        pred = self.model.predict(self.scaler.transform(self.iv))
        eval = self.eval_model(self.dv, pred)

        total = (et-st)/60
        print(f'Model fit in {total} minutes.')

        self._save_model(self.model, self.scaler, eval)     
       
          
    def score(self, feats: Dict[str:np.ndarray], ref_feats: Dict[str:np.ndarray], fname: str)-> tuple[np.ndarray, Dict[str:np.ndarray]]:
        """
        Score ridge regression

        :param feats: dict, feature dictionary, stimulus names as keys
        :param ref_feats: dict, feature dictionary of ground truth predicted features, stimulus names as keys
        :param fname: str, name of stimulus to extract for
        :return r: np.ndarray, correlations
        :return: Dictionary of true and predicted values 
        """
        #with open(self.save_path / 'ridge_regression_model.pkl', 'rb') as file:
         #   self.trained_model = pickle.load(file)
        if fname in self.result_paths['metric']:
            if self.cci_features is not None:
                if self.cci_features.exists_object(self.result_paths['metric'][fname]) and not self.overwrite:
                    return 
            else:
                if Path(str(self.result_paths['metric'][fname]) + '.npz').exists() and not self.overwrite:
                    return 
        
        #assert self.pearson_coeff is not None, 'Regression has not been run yet. Please do so.'
        assert self.model is not None, 'Regression has not been run yet. Please do so.'

        f = feats['features']
        rf = ref_feats['features']
        t = feats['times']
        rt = ref_feats['times']
        
        assert np.equal(t, rt).all(), f'Time alignment skewed across features for stimulus {fname}.'
        f = f.astype(self.model.coef_.dtype)
        pred = self.model.predict(self.scaler.transform(f))

        per_story_metrics = self.eval_model(rf, pred)
        
        correlations = []
        if self.corr_type=='time':
            for i in range(rf.shape[0]):
                corr = np.corrcoef(pred[i,:], rf[i,:])[0, 1]
                correlations.append(corr)
        else:
            for i in range(rf.shape[1]):
                corr = np.corrcoef(pred[:,i], rf[:,i])[0, 1]
                correlations.append(corr)

        r2 = np.array(correlations)
        
        self._save_metrics(r2, fname)
        self._save_metrics(per_story_metrics, fname, 'eval')

        return r2, {'true': rf, 'pred':pred}
