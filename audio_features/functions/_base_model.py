import numpy as np
import pickle
import os
import json
from typing import Union
from pathlib import Path

from sklearn.metrics import r2_score, f1_score,  mean_squared_error

class BaseModel:
    def __init__(self, model_type:str, iv:dict, iv_type:str, dv:dict, dv_type:str, config:dict, save_path:Union[str,Path],
                 cci_features=None, overwrite:bool=False, local_path:Union[str,Path]=None):
        self.model_type = model_type
        self.iv_type = iv_type
        self.dv_type = dv_type
        self.fnames = list(iv.keys())
        self.overwrite=overwrite

        self.iv, self.iv_rows, self.iv_times = self._process_features(iv)
        self.dv, self.dv_rows, self.dv_times = self._process_features(dv)
        self._check_rows()

        ## local path
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

        self.config = self.config = {
             'iv_type': self.iv_type,'iv_shape': self.iv.shape, 'iv_rows': self.iv_rows, 
             'dv_type': self.dv_type, 'dv_shape': self.dv.shape, 'dv_rows':self.dv_rows, 
             'train_fnames': self.fnames, 'overwrite': self.overwrite
         }
        self.config.update(config)

        config_name = self.local_path / f'{self.model_type}_config.json'
        os.makedirs(self.local_path, exist_ok=True)
        with open(str(config_name), 'w') as f:
            json.dump(self.config,f)
        
        if self.cci_features is None:
            new_path = self.save_path/'model'
        else:
            new_path = self.local_path/'model'

        config_name = new_path/ f'{self.model_type}_config.json'
        os.makedirs(new_path, exist_ok=True)
        with open(str(config_name), 'w') as f:
            json.dump(self.config,f)

        self.result_paths = {'model': new_path/'model', 'scaler': new_path/'scaler'}
        
        self.result_paths['train_eval'] = new_path / 'train_eval'
        self.result_paths['test_eval'] = new_path / 'test_eval'
        self.result_paths['metric'] = {}
        self.result_paths['eval'] = {}
        for f in self.fnames:
             self.result_paths['metric'][f] = self.save_path/f
             self.result_paths['eval'][f] = new_path/f"{f}_eval"
        
        #self._check_previous()
    
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
        if Path(str(self.result_paths['model'])+'.pkl').exists() and Path(str(self.result_paths['scaler'])+'.pkl').exists(): self.weights_exist=True

        if self.weights_exist and not self.overwrite:
            with open(str(self.result_paths['model'])+'.pkl', 'rb') as f:
                self.model = pickle.load(f)
            with open(str(self.result_paths['scaler'])+'.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
        else:
            self.model=None
            self.scaler=None

    def _save_model(self, model, scaler, eval):
        if self.cci_features:
            print('Model cannot be saved to cci_features. Saving to local path instead')
        

        # Save the model to a file using pickle
        os.makedirs(self.save_path, exist_ok=True)
        with open(str(self.result_paths['model'])+'.pkl', 'wb') as file:
            pickle.dump(model, file)
        with open(str(self.result_paths['scaler'])+'.pkl', 'wb') as file:
            pickle.dump(scaler, file)
        
        np.savez_compressed(str(self.result_paths['train_eval'])+'.npz', eval)
        

    def _save_metrics(self, metric, fname, name='metric'):
        if fname not in self.result_paths[name]:
            self.result_paths[name][fname] = self.save_path / fname

        if name == 'eval':
            os.makedirs(str(self.result_paths['eval'][fname].parent), exist_ok=True)
            with open(str(self.result_paths['eval'][fname])+'.json', 'w') as f:
                json.dump(metric,f)
            return 
        
        if self.cci_features:
            #print('features')
            self.cci_features.upload_raw_array(self.result_paths[name][fname], metric)
            #print(self.result_paths['residuals'][fname])
        else:
            #print('not features')
            os.makedirs(self.result_paths[name][fname].parent, exist_ok=True)
            np.savez_compressed(str(self.result_paths[name][fname])+'.npz', metric)
    
    def eval_model(self, true, pred):
        r2 = r2_score(true, pred)
        rmse = np.sqrt(mean_squared_error(true, pred))
        #f1 = f1_score(true, pred)
        metrics = {'r2': float(r2), 'rmse':float(rmse)}
        return metrics
        
    