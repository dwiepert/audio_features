"""
Extract EMA features using SPARC

Author(s): Daniela Wiepert, Aditya Vaidya
Last Modified: 11/14/2024
"""

#IMPORTS
##built-in
import json
import os
from pathlib import Path
from typing import List, Union

##third-party
import numpy as np
import torch

##local
from ._base_extraction import BaseExtractor
from sparc import load_model, SPARC, SpeechWave

class SPARCExtractor(BaseExtractor):
    """
    Extract EMA and other related features with SPARC

    :param model: pretrained SPARC model
    :param model_type: str, type of SPARC model
    :param save_path: local path to save extractor configuration information to
    :param target_sample_rate: int, target sample rate for model
    :param min_length_samples: int, minimum length a sample can be to be fed into the model
    :param return_numpy: bool, true if returning numpy
    :param num_select_frames: int, default=1. This can safely be set to 1 for chunk size of 100 and non-whisper models. 
                              This specifies how many frames to select features from. 
    :param frame_skip: int, default=5. This goes with num_select_frames. For most HF models the window is 20ms, 
                       so in order to take 1 feature per batched waveform with chunksz = 100ms, you set 5 to say you take num_select_frames (1) every frame_skip
    :param frame_len_sec: int, information on how long a frame is in the model in seconds
    :param keep_all: bool, true if you want to keep all outputs from each batch
    """
    def __init__(self, model:SPARC, model_type:str,  save_path: Union[str, Path], target_sample_rate:int=16000, return_numpy:bool=True, 
                frame_length_sec:float=None, keep_all:bool=False):
        
        super().__init__(target_sample_rate=target_sample_rate, return_numpy=return_numpy)
        self.model = model
        self.model_type = model_type
        self.frame_length_sec = frame_length_sec
        self.keep_all = keep_all #THIS IS FOR FEATURES THAT OUTPUT MORE THAN ONE FEATURE PER BATCH
    
        #self.output_inds = np.array([-1 - self.frame_skip*i for i in reversed(range(self.num_select_frames))])
        #TODO: figure out output inds for SPARC
        assert len(self.output_inds) == 1, "Only one output per evaluation is "\
            "supported  (because they don't provide the downsampling rate)"
        
        self.config = {'feature_type': 'sparc', 'model_type': self.model_type, 'target_sample_rate': self.target_sample_rate, 
                       'return_numpy': self.return_numpy, 'keep_all':self.keep_all}
        
        #saving things
        self.save_path = Path(save_path)
        os.makedirs(self.save_path, exist_ok=True)
        with open(str(save_path /'SPARCExtractor_config.json'), 'w') as f:
            json.dump(self.config, f)

        self.modules= None
    
    def _temp_sparc_processing(self, wavfiles: List[np.ndarray]) -> SpeechWave:
        """
        Expects that wavfiles is a list of np.ndarrays
        """
        assert isinstance(wavfiles, np.ndarray)
        wavs = [wavfiles]
        wavs = [torch.from_numpy(wav).float() for wav in wavs]
        input_lens = np.array([len(wav) for wav in wavs])
        wavs = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True, padding_value=0.0)
        wavs = SpeechWave(input_values=wavs, input_lens=input_lens)
        wavs = wavs.to(self.model.device)
        return wavs

    def __call__(self,sample:dict) -> dict:
        """
        Run feature extraction on a snippet of a sample 

        SPARC: output has 50 Hz sampling rate which matches the sampling rate of the SSL features? 
        butterworth filter removes high-frequency noise
        #ADD VOCAL SOURCE FEATURES TO OUTPUT SO IT IS 14 dim
        Because it is same as WavLM, use the same features of output stuff as WavLM

        :param sample: dict, audio sample with metadata
        :return sample: sample with outputs
        """
        ## ASSERTIONS to check sample is correctly processed
        keys = ['waveform', 'snippet_starts', 'snippet_ends', 'sample_rate']
        for k in keys: assert k in sample, f'{k} not in sample. Check that audio is processed correctly.'
        assert sample['waveform'].ndim == 2, 'Batched waveform must be of size (batch size x waveform size) with no extra channels'
        assert sample['sample_rate'] == self.target_sample_rate , f'Sample rate of the audio should be {self.target_sampling_rate}. Please check audio processing.'

        #initialize common output variables 
        out_features = []
        times = [] # times are shared across all layers

        #get start and ends of snippets 
        snippet_starts = sample['snippet_starts']
        snippet_ends = sample['snippet_ends']

        #prep wavs for SPARC model.squee
        wavs = sample['waveform']
        #wav = sample['waveform'].squeeze_().numpy()
        input = [wavs[i,:].squeeze_().numpy() for i in range(wavs.shape[0])]
        #input = self._temp_sparc_processing(wav)
            
        #sample['waveform'].squeeze_().numpy()]
        chunk_features = self.model.encode(input)

        if self.keep_all: 
            output_inds = list(range(chunk_features['ema'].shape[0])) 
        else: 
            output_inds = self.output_inds

        for out_idx, output_offset in enumerate(output_inds):
            # TODO: avoid re-stacking the times. may require tracking snippet
            # idxs and indexing into `snippet_times`
            times.append(torch.stack([snippet_starts, snippet_ends], dim=1))

            if not isinstance(chunk_features, list): chunk_features = [chunk_features]
            for c in chunk_features:
                #get 'ema' features
                ema = c['ema'][output_offset,:]
                #print('Currently loudness and pitch do NOT output at the same frame rate - need to explore SPARC more for that bug...')
                ema = np.append(ema, c['loudness'][output_offset,:]) 
                temp = c['pitch']
                if temp.shape[0] != c['ema'].shape[0]: 
                    difference = abs(temp.shape[0] - c['ema'].shape[0])*-1
                    c['pitch'] = temp[:difference,:] #TODO: unsure about this
                    #print(f'Prev shape: {temp.shape[0]}, Curr shape: {temp[:-1,:].shape[0]}')
                ema = np.append(ema, c['pitch'][output_offset,:])
                ema = np.expand_dims(ema, axis=0) if self.return_numpy else torch.unsqueeze(torch.from_numpy(ema),0)
                out_features.append(ema)                       
        
        features = np.concatenate(out_features, axis=0) if self.return_numpy else torch.cat(out_features, dim=0) # shape: (timesteps, features)
        times = torch.cat(times, dim=0) / self.target_sample_rate # convert samples --> seconds. shape: (timesteps,)
        if self.return_numpy: times = times.numpy()

        del chunk_features # TODO: trying to fix memory leak. remove if unneeded
           
        sample['out_features'] = features
        sample['times'] = times
        return sample
            
def set_up_sparc_extractor(save_path:Union[str, Path], model_name:str="en", config:Union[str,Path]=None, ckpt:str=None, use_penn:bool=False,
                           target_sample_rate:int=16000, return_numpy:bool=True, min_length_samples:float=None, keep_all:bool=False) -> SPARCExtractor:
    """
    Set up sparc extractor
    :param save_path: local path to save extractor configuration information to
    :param model_name: str, sparc model name, from [en, multi, en+, feature_extraction]
    :param config: default None, see if configs exist in the Speech articulatory coding github
    :param ckpt: default None, can load a model checkpoint if desired
    :param use_penn:bool, specify whether to use pitch tracker
    :param target_sample_rate: int, target sample rate for model
    :param min_length_samples: int, minimum length a sample can be to be fed into the model
    :param return_numpy: bool, true if returning numpy
    :param keep_all: bool, true if you want to keep all outputs from each batch
    :return: initialized SPARCExtractor
    """
    assert model_name, 'Must give model name for sparc models'

    use_cuda = torch.cuda.is_available()
    print(f"use cuda: {use_cuda}")
    if use_cuda: 
        device = "cuda"
    else:
        device = "cpu"
    model = load_model(model_name=model_name, config=config, ckpt=ckpt, device=device, use_penn=use_penn)
    #TODO: figure out frame_length_sec

    return SPARCExtractor(model=model, model_type=model_name, save_path=save_path, target_sample_rate=target_sample_rate, return_numpy=return_numpy, keep_all=keep_all)

