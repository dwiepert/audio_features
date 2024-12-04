"""
openSMILE feature extractor

Author(s): Daniela Wiepert, Lasya Yakkala, Rachel Yamamoto
Last Modified: 11/20/2024
"""
#IMPORTS 
##built-in
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Union

##third-party
import numpy as np
import opensmile
import torch

##local
from ._base_extraction import BaseExtractor

class opensmileExtractor(BaseExtractor):
    """
    openSMILE feature extractor 

    :param save_path: local path to save extractor configuration information to
    :param feature_set: str, opensmile feature set to use (Default = 'eGeMAPSv02', can also take 'ComParE_2016')
    :param feature_level: str, opensmile feature level to use (Default = 'lld', can also take 'func')
    :param defaut_extractor: bool, specify whether to use default open smile extractor build
    :param frame_length: float, frame length in ms for extraction. Default=25.
    :param frame_shift: float, frame shift in ms for extraction. Default=20.
    :param target_sample_rate: int, target sample rate for model
    :param min_length_samples: int, minimum length a sample can be to be fed into the model
    :param return_numpy: bool, true if returning numpy
    :param num_select_frames: int, default=1. This can safely be set to 1 for chunk size of 100 and non-whisper models. 
                              This specifies how many frames to select features from. 
    :param frame_skip: int, default=5. This goes with num_select_frames. For most HF models the window is 20ms, 
                       so in order to take 1 feature per batched waveform with chunksz = 100ms, you set 5 to say you take num_select_frames (1) every frame_skip. 
    :param keep_all: bool, true if you want to keep all outputs from each batch
    """
    def __init__(self, save_path:Union[str,Path], feature_set:str='eGeMAPSv02', feature_level:str='lld', default_extractor:bool=False,
                 frame_length:float=25., frame_shift:float=20., target_sample_rate:int=16000, min_length_samples:int=0, 
                 return_numpy:bool=True, num_select_frames:int=1, frame_skip:int=5, keep_all:bool=False):
        #INHERITED VALUES
        super().__init__(target_sample_rate=target_sample_rate, min_length_samples=min_length_samples, 
                         return_numpy=return_numpy, num_select_frames=num_select_frames, frame_skip=frame_skip)
        
        self.feature_set=feature_set
        self.feature_level=feature_level
        self.frame_length = self._str_mng(f"{(frame_length/1000):.3g}")
        self.frame_shift = self._str_mng(f"{(frame_shift/1000):.3g}")
        self.default_extractor = default_extractor
        if self.feature_level=='func':
            self.default_extractor = True
            print('Extracting functional features. Using default extractor.')
        
        self._set_extractor()

        self.feature_names = self.smile.feature_names
        self.keep_all=keep_all
        self.config = {'feature_type':'opensmile', 'feature_set': self.feature_set, 'feature_level': self.feature_level, 'feature_names':self.feature_names,
                       'default_extractor':self.default_extractor, 'frame_length': self.frame_length, 'frame_shift': self.frame_shift,
                       'target_sample_rate': self.target_sample_rate, 'min_length_samples':self.min_length_samples,
                       'return_numpy': self.return_numpy, 'num_select_frames':self.num_select_frames, 'frame_skip': self.frame_skip, 'keep_all': self.keep_all}
        
        #saving things
        self.save_path = Path(save_path)
        os.makedirs(self.save_path, exist_ok=True)
        with open(str(save_path /'opensmileExtractor_config.json'), 'w') as f:
            json.dump(self.config,f)

        self.modules = None

    def _str_mng(self, st:str) ->str:
        if len(st) != 5:
            temp = st
            for i in range(5-len(st)):
                temp += "0"
            return temp
        else:
            return st
    
    def _set_extractor(self):
        """
        select proper feature set and level
        """
        if self.default_extractor:
            if self.feature_set == 'eGeMAPSv02':
                self.selected_features = opensmile.FeatureSet.eGeMAPSv02
                self.frame_length="0.020"
                self.frame_shift="0.010"
            elif self.feature_set == 'ComParE_2016':
                self.selected_features = opensmile.FeatureSet.ComParE_2016
                self.frame_length="0.060"
                self.frame_shift="0.010"
            else:
                raise NotImplemented(f'{self.feature_set} is not an implemented feature set for opensmile feature extraction')
            
            if self.feature_level == 'func':
                self.selected_level = opensmile.FeatureLevel.Functionals
            elif self.feature_level == 'lld':
                self.selected_level = opensmile.FeatureLevel.LowLevelDescriptors
            else:
                raise NotImplemented(f'{self.feature_level} is not an implemented feature level for opensmile feature extraction')
            
            self.smile = opensmile.Smile(feature_set=self.selected_features, feature_level=self.selected_level)
        
        else:
            self._edit_config()
    
    def _edit_config(self):
        """
        Use custom frame length and shifts. This will check if the right .conf files exist yet and if not, create them
        """
        if self.feature_set == 'eGeMAPSv02':
            dir_name = Path(f'./audio_features/configs/opensmile/egemaps_len{self.frame_length}_shift{self.frame_shift}')
            if not dir_name.exists():
                already_exists = False
                shutil.copytree('./audio_features/configs/opensmile/egemaps', str(dir_name))
                base_path = dir_name/'GeMAPSv01b_core.lld.conf.inc'
            else:
                already_exists = True
            main_path = dir_name/'eGeMAPSv02.conf'

        elif self.feature_set == 'ComParE_2016':
            dir_name = Path(f'./audio_features/configs/opensmile/compare_len{self.frame_length}_shift{self.frame_shift}')
            if not dir_name.exists():
                already_exists = False
                shutil.copytree('./audio_features/configs/opensmile/compare', str(dir_name))
                base_path = dir_name/'ComParE_2016_core.lld.conf.inc'
            else:
                already_exists = True
            main_path = dir_name/'ComParE_2016.conf'
        else:
            raise NotImplemented(f'{self.feature_set} is not an implemented feature set for opensmile feature extraction')
        
        if not already_exists:
            with open(base_path, 'r') as f:
                to_edit = f.read()
            
            #check if there are two of them!!!!
            to_edit = re.sub("frameSize = [0-9].[0-9]{3}", f"frameSize = {self.frame_length}", to_edit)
            to_edit = re.sub("frameStep = [0-9].[0-9]{3}", f"frameStep = {self.frame_shift}", to_edit)
            with open(base_path, 'w') as f:
                f.write(to_edit)
        
        self.smile = opensmile.Smile(
            feature_set=str(main_path),
            feature_level=self.feature_level
        )

    def __call__(self, sample:dict) -> dict:
        """
        Run feature extraction on a snippet of an audio sample

        :param sample: dict, audio sample with metadata
        :return sample: sample with outputs
        """
        keys = ['waveform', 'snippet_starts', 'snippet_ends', 'sample_rate']
        for k in keys: assert k in sample, f'{k} not in sample. Check that audio is processed correctly.'
        assert sample['waveform'].ndim == 2, 'Batched waveform must be of size (batch size x waveform size) with no extra channels'
        assert sample['sample_rate'] == self.target_sample_rate , f'Sample rate of the audio should be {self.target_sampling_rate}. Please check audio processing.'
        
        # for fbank extraction, our batch size needed to be 1, so assert that the first dim of waveform shape is 1
        s = sample['waveform'].shape[0]
        assert sample['waveform'].shape[0] == 1, f'Only batch size of 1 is supported with MFCC extraction but audio has batch size of {s}'

        out_features = [] #empty list that contains one representative feature. For example, it might be the final hidden state output of an SSL model, or the EMA features rather than pitch/loudness information that is also extracted from a SPARC model
        times = [] # times are shared across all features. The indices are aligned between all of these previous variables, such that a module_features[k][i] was extracted from times[i]

        #take out snippet starts and ends
        snippet_starts = sample['snippet_starts']
        snippet_ends = sample['snippet_ends']

        wav = sample['waveform']
        #desired number of frames: WE ROUND DOWN FOR OTHER FEATURES
        frames = int(math.floor((wav.shape[1]-(float(self.frame_length)-float(self.frame_shift))*self.target_sample_rate)/(float(self.frame_shift)*self.target_sample_rate)))
        
        features = self.smile.process_signal(wav, sampling_rate=self.target_sample_rate)
        features = features.to_numpy()

        if frames != features.shape[0]: #handle difference because of rounding
            difference = np.abs(features.shape[0]-frames)*-1
            features = features[:difference]

        if self.keep_all: 
            output_inds = list(range(features.shape[0]))
        else: 
            output_inds = self.output_inds

        for out_idx, output_offset in enumerate(output_inds):
            # TODO: avoid re-stacking the times. may require tracking snippet
            # idxs and indexing into `snippet_times`
            times.append(torch.stack([snippet_starts, snippet_ends], dim=1))
            output_representation = features[output_offset, :] # TODO shape: (batchsz, hidden_size)
            output_representation = np.expand_dims(output_representation, axis=0)
            if not self.return_numpy: output_representation = output_representation = torch.from_numpy(output_representation)
            out_features.append(output_representation)
        
        out_features = np.concatenate(out_features, axis=0) if self.return_numpy else torch.cat(out_features, dim=0) # shape: (timesteps, features)
        times = torch.cat(times, dim=0) / self.target_sample_rate # convert samples --> seconds. shape: (timesteps,)
        if self.return_numpy: times = times.numpy()

        sample['out_features'] = out_features
        sample['times'] = times
        return sample

def set_up_opensmile_extractor(save_path:Union[str,Path], feature_set:str='eGeMAPSv02', feature_level:str='lld', default_extractor:bool=False,
                 frame_length:float=25., frame_shift:float=20., target_sample_rate:int=16000, min_length_samples:int=0, 
                 return_numpy:bool=True, num_select_frames:int=1, frame_skip:int=5, keep_all:bool=False) -> opensmileExtractor:
    """
    Set up opensmileExtractor

    :param save_path: local path to save extractor configuration information to
    :param feature_set: str, opensmile feature set to use (Default = 'eGeMAPSv02', can also take 'ComParE_2016')
    :param feature_level: str, opensmile feature level to use (Default = 'lld', can also take 'func')
    :param defaut_extractor: bool, specify whether to use default open smile extractor build
    :param frame_length: float, frame length in ms for extraction. Default=25.
    :param frame_shift: float, frame shift in ms for extraction. Default=20.
    :param target_sample_rate: int, target sample rate for model
    :param min_length_samples: int, minimum length a sample can be to be fed into the model
    :param return_numpy: bool, true if returning numpy
    :param num_select_frames: int, default=1. This can safely be set to 1 for chunk size of 100 and non-whisper models. 
                              This specifies how many frames to select features from. 
    :param frame_skip: int, default=5. This goes with num_select_frames. For most HF models the window is 20ms, 
                       so in order to take 1 feature per batched waveform with chunksz = 100ms, you set 5 to say you take num_select_frames (1) every frame_skip. 
    :param keep_all: bool, true if you want to keep all outputs from each batch

    :return: opensmileExtractor
    """
    
    return opensmileExtractor(save_path=save_path, feature_set=feature_set, feature_level=feature_level, default_extractor=default_extractor, frame_length=frame_length, frame_shift=frame_shift, 
                          target_sample_rate=target_sample_rate, min_length_samples=min_length_samples, return_numpy=return_numpy,
                          num_select_frames=num_select_frames, frame_skip=frame_skip, keep_all=keep_all)