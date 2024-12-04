"""
FBANK feature extractor

Author(s): Daniela Wiepert, Lasya Yakkala, Rachel Yamamoto
Last Modified: 11/20/2024
"""
#IMPORTS 
##built-in
import json
import os
from pathlib import Path
from typing import Union

##third-party
import numpy as np
import torch
import torchaudio

##local
from ._base_extraction import BaseExtractor

class FBANKExtractor(BaseExtractor):
    """
    FBANK feature extractor 

    :param save_path: local path to save extractor configuration information to
    :param num_mel_bins: int, number of mel bins for fbank extraction. Default=23
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
    def __init__(self, save_path:Union[str,Path], num_mel_bins:int=23, frame_length:float=25., frame_shift:float=20.,target_sample_rate:int=16000, min_length_samples:int=0, 
                 return_numpy:bool=True, num_select_frames:int=1, frame_skip:int=5, keep_all:bool=False):
        #INHERITED VALUES
        super().__init__(target_sample_rate=target_sample_rate, min_length_samples=min_length_samples, 
                         return_numpy=return_numpy, num_select_frames=num_select_frames, frame_skip=frame_skip)
        
        self.num_mel_bins = num_mel_bins
        self.frame_length= frame_length
        self.frame_shift = frame_shift
        self.keep_all=keep_all
        self.config = {'feature_type':'fbank', 'num_mel_bins': self.num_mel_bins, 'frame_length':self.frame_length, 'frame_shift':self.frame_shift,
                       'target_sample_rate': self.target_sample_rate, 'min_length_samples':self.min_length_samples,
                       'return_numpy': self.return_numpy, 'num_select_frames':self.num_select_frames, 'frame_skip': self.frame_skip, 'keep_all': self.keep_all}
        
        #saving things
        self.save_path = Path(save_path)
        os.makedirs(self.save_path, exist_ok=True)
        with open(str(save_path /'FBANKExtractor_config.json'), 'w') as f:
            json.dump(self.config,f)

        self.modules = None
    
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

        fbank = torchaudio.compliance.kaldi.fbank(sample['waveform'], num_mel_bins=self.num_mel_bins, sample_frequency=float(self.target_sample_rate), frame_length=self.frame_length, frame_shift=self.frame_shift)

        if self.keep_all: 
            output_inds = list(range(fbank.shape[0]))
        else: 
            output_inds = self.output_inds

        for out_idx, output_offset in enumerate(output_inds):
            # TODO: avoid re-stacking the times. may require tracking snippet
            # idxs and indexing into `snippet_times`
            times.append(torch.stack([snippet_starts, snippet_ends], dim=1))
            output_representation = fbank[output_offset, :] # TODO: shape: (batchsz, hidden_size)
            if self.return_numpy: output_representation = output_representation.numpy()

            output_representation = np.expand_dims(output_representation, axis=0) if self.return_numpy else torch.unsqueeze(torch.from_numpy(output_representation),0)
            out_features.append(output_representation)
        
        out_features = np.concatenate(out_features, axis=0) if self.return_numpy else torch.cat(out_features, dim=0) # shape: (timesteps, features)
        times = torch.cat(times, dim=0) / self.target_sample_rate # convert samples --> seconds. shape: (timesteps,)
        if self.return_numpy: times = times.numpy()

        sample['out_features'] = out_features
        sample['times'] = times
        return sample

def set_up_fbank_extractor(save_path:Union[str,Path], num_mel_bins:int=23, frame_length:float=25., frame_shift:float=20.,target_sample_rate:int=16000, min_length_samples:int=0, 
                 return_numpy:bool=True, num_select_frames:int=1, frame_skip:int=5, keep_all:bool=False) -> FBANKExtractor:
    """
    Set up FBANKExtractor

    :param save_path: local path to save extractor configuration information to
    :param num_mel_bins: int, number of mel bins for fbank extraction. Default=23
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
    :return: FBANKExtractor
    """
    
    return FBANKExtractor(save_path=save_path, num_mel_bins=num_mel_bins, frame_length=frame_length, frame_shift=frame_shift, 
                          target_sample_rate=target_sample_rate, min_length_samples=min_length_samples, return_numpy=return_numpy,
                          num_select_frames=num_select_frames, frame_skip=frame_skip, keep_all=keep_all)