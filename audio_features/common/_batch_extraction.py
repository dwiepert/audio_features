"""
Extract features in batched snippets

Authors: Aditya Vaidya, Daniela Wiepert
Last Modified: 11/08/2024
"""
#IMPORTS
##built-in
import collections
import torch
import torchvision 
from typing import Dict

##third-party
import numpy as np
from tqdm import tqdm

##local
from audio_preprocessing.transforms import Path2Wave, ResampleAudio, Window, PadSilence

class BatchExtractor:
    """
    Run common batching strategy for all feature extractors

    Expects chunksz and contextsz as seconds
    
    :param extractor: feature extractor for feature type
    :param batchsz: int, batch size for feature. Default=1
    :param chunksz: float, chunk size in seconds. Default=0.1
    :param contextsz: float, context size in seconds. Default=8
    :param require_full_context: bool, true if full context should be included in each window
    :param min_length_samples: float, minimum length of samples in seconds
    :param return_numpy: bool, true if returning numpy
    :param pad_silence: bool, true if padding silence
    """
    def __init__(self, extractor, batchsz:int=1, chunksz:float=0.1, contextsz:float=8., require_full_context:bool=True, 
                 min_length_samples:float=0, return_numpy:bool=True, pad_silence:bool=False):
        self.extractor = extractor
        print('Assert extractor type once an extractor is working')
        self.batchsz = batchsz
        self.sampling_rate = self.extractor.target_sample_rate
        self.chunksz=chunksz
        self.chunksz_samples = int(self.chunksz*self.sampling_rate)
        self.contextsz=contextsz
        self.contextsz_samples = int(self.contextsz*self.sampling_rate)
        self.require_full_context=require_full_context
        self.min_length_samples = self.extractor.min_length_samples
        if self.min_length_samples is None:
            self.min_length_samples = min_length_samples
        initial_transforms = [Path2Wave(), ResampleAudio(resample_rate=self.sampling_rate),
                           Window(chunksz=self.chunksz, contextsz=self.contextsz, batchsz=self.batchsz, sampling_rate=self.sampling_rate, require_full_context=self.require_full_context, min_length_samples=self.min_length_samples)]
        self.pad_silence = pad_silence
        if self.pad_silence:
            self.padding = PadSilence(context_sz=self.contextsz)
            initial_transforms.append(self.padding)
        self.transforms = torchvision.transforms.Compose(initial_transforms)
        self.return_numpy = return_numpy      


    def __call__(self, sample: Dict):
        """
        Take in a sample and perform batched feature extraction on the sample. The sample should be preprocessed and also be windowed using audio_processing.transforms.Window.
        :param sample: dict, audio sample with metadata
        :return sample: sample post feature extraction
        """
        #expects samples to come in as a path that has ALREADY BEEN PREPROCESSED, then run path to wave and window. Window will check that sample rate is target sample rate
        sample = self.transforms(sample)
        wav = torch.squeeze(sample['waveform'])

        snippet_iter = sample['snippet_iter']
        batched_features = []
        times = []
        module_features = collections.defaultdict(list)

        for batch_idx, (snippet_starts, snippet_ends) in enumerate(tqdm(snippet_iter)):
            sample2 = sample
            if ((snippet_ends - snippet_starts) < (self.contextsz_samples + self.chunksz_samples)).any() and self.require_full_context:
                raise ValueError("This shouldn't happen with require_full_context")

            # If we don't have enough samples, skip this chunk.
            if (snippet_ends - snippet_starts < self.min_length_samples).any():
                print('If this is true for any, then you might be losing more snippets than just the offending (too short) snippet. Consider increasing the input (chunk or context) to the model.')
                assert False

            # Construct the input waveforms for the batch
            #wav_in = wav[snippet_starts : snippet_ends]
            # This can maybe be optimized...
            batched_wav_in_list = []
            for batch_snippet_idx, (snippet_start, snippet_end) in enumerate(zip(snippet_starts, snippet_ends)):
                # Stacking might be inefficient, so populate a pre-allocated array.
                #batched_wav_in[batch_snippet_idx, :] = wav[snippet_start:snippet_end]
                # But stacking makes variable batch size easier!
                batched_wav_in_list.append(wav[snippet_start:snippet_end])
            batched_wav_in = torch.stack(batched_wav_in_list, dim=0)

            # The final batch may be incomplete if batchsz doesn't evenly divide
            # the number of snippets.
            if (snippet_starts.shape[0] != batched_wav_in.shape[0]) and (snippet_starts.shape[0] != self.batchsz):
                batched_wav_in = batched_wav_in[:snippet_starts.shape[0]]

        
            sample2['waveform'] = batched_wav_in
            sample2['snippet_starts'] = snippet_starts
            sample2['snippet_ends'] = snippet_ends

            output = self.extractor(sample2)
            if output is None:
                continue
            
            batched_features.append(output['out_features'])
            times.append(output['times'])
            mod_ft = 'module_features' in output
            
            if mod_ft:
                for k in output['module_features']:
                    if k in module_features:
                        temp1 = module_features[k]
                        temp1.append(output['module_features'][k])
                        module_features[k] = temp1 
                    else:
                        module_features[k] = [output['module_features'][k]]

        if not self.return_numpy and isinstance(out_features, np.ndarray):
            out_features = torch.from_numpy(out_features)
            if mod_ft: module_features = {name: (torch.from_numpy(features))for name, features in module_features.items()}

        out_features = np.concatenate(batched_features, axis=0) if self.return_numpy else torch.cat(batched_features, dim=0) # shape: (timesteps, features)
        
        if mod_ft:
            module_features = {name: (np.concatenate(features, axis=0) if self.return_numpy else torch.cat(features, dim=0))\
                            for name, features in module_features.items()}
            assert all(features.shape[0] == out_features.shape[0] for features in module_features.values()),\
                "Missing timesteps in the module activations!! (possible PyTorch bug)"
        times = np.concatenate(times, axis=0) if self.return_numpy else torch.cat(times, dim=0) / self.sampling_rate # shape: (timesteps, features)
        #times = np.concatenate(timestorch.cat(times, dim=0) / self.sampling_rate # convert samples --> seconds. shape: (timesteps,)

        sample['final_outputs'] = out_features
        if mod_ft: sample['module_features'] = module_features  
        sample['times']= times
        if self.pad_silence:
            sample = self.padding.remove_padding(sample)

        return sample
    