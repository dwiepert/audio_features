"""
Extract features in batched snippets

Authors: Aditya Vaidya, Daniela Wiepert
Last Modified: 11/04/2024
"""
#IMPORTS
##built-in
import collections
import copy
import json
import os
from pathlib import Path
from typing import Dict, Union, List

##third-party
import numpy as np
from tqdm import tqdm
import torch
import torchvision 

##local
from audio_preprocessing.transforms import Path2Wave, ResampleAudio, Window, PadSilence
from audio_features.io import save_features

class BatchExtractor:
    """
    Run common batching strategy for all feature extractors

    Expects chunksz and contextsz as seconds
    
    :param extractor: feature extractor for feature type
    :param save_path: location to save features to
    :param cci_features: cotton candy interface for saving features
    :param fnames: list of file names we are extracting features for
    :param batchsz: int, batch size for feature. Default=1
    :param chunksz: float, chunk size in seconds. Default=0.1
    :param contextsz: float, context size in seconds. Default=8
    :param require_full_context: bool, true if full context should be included in each window
    :param min_length_samples: float, minimum length of samples in seconds
    :param return_numpy: bool, true if returning numpy
    :param pad_silence: bool, true if padding silence
    :param local_path: str/Path, local directory to save config file to if using self.cci_features. 
    """
    def __init__(self, extractor, save_path:Union[str, Path],  cci_features=None, fnames:List=[], overwrite:bool=False,
                 batchsz:int=1, chunksz:float=0.1, contextsz:float=8., require_full_context:bool=True, 
                 min_length_samples:float=0, return_numpy:bool=True, pad_silence:bool=False, local_path:Union[str,Path] = None):
        self.extractor = extractor
        self.cci_features=cci_features 
        self.save_path=Path(save_path)
        self.fnames=fnames
        self.overwrite = overwrite
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

        self.config = {'batchsz': self.batchsz, 'sampling_rate': self.sampling_rate, 'chunksz':self.chunksz, 'contextsz':self.contextsz, 'fnames':self.fnames,
                       'require_full_context':self.require_full_context, 'min_length_samples': self.min_length_samples, 'pad_silence':self.pad_silence,
                       'return_numpy': self.return_numpy}
        
        self.local_path = local_path
        if self.local_path is None or self.cci_features is None:
            self.local_path = self.save_path
            os.makedirs(self.local_path, exist_ok=True)
        else:
            self.local_path = Path(self.local_path)
        with open(str(self.local_path /'BatchExtractor_config.json'), 'w') as f:
            json.dump(self.config,f)
        
        self._get_result_paths()

    def _get_result_paths(self):
        """
        Set up result paths
        """
        self.result_paths = {}
        self.result_paths['features'] = {}
        self.result_paths['times'] = {}

        self.modules = self.extractor.modules
        if self.modules is not None:
            self.result_paths['modules'] = {}

        for f in self.fnames:
            self.result_paths['features'][f] = self.save_path / f
            self.result_paths['times'][f] = self.save_path / (f+'_times')

            if self.modules is not None:
                for m in self.modules:
                    #print(m)
                    #print(self.modules[m])
                    n = self.save_path/self.modules[m]
                    os.makedirs(n, exist_ok=True)
                    if f not in self.result_paths['modules']:
                        self.result_paths['modules'][f] = {self.modules[m]:n/f}
                    else:
                        temp =  self.result_paths['modules'][f]
                        temp[self.modules[m]] = n/f
                        self.result_paths['modules'][f] = temp
            #print(self.result_paths['modules'])
    
    def _save(self, sample:dict, fname:str):
        """
        Save features

        :param sample: sample dict with features
        :param fname: str, stimulus name
        :return: None, saving features
        """
        feats = self.result_paths['features']
        if fname not in feats:
            new_name = self.save_path / fname
            self.result_paths['features'][fname] = new_name
        
        t = self.result_paths['times']
        if fname not in t:
            new_name = self.save_path / fname+"_times"
            self.result_paths['times'][fname] = new_name

        
        if sample['module_features'] is not None:
            if sample['module_features'] is not {}:
                m = self.result_paths['modules']
                if fname not in m:
                    new_name = self.save_path / fname
                    for m in self.modules:
                        n = self.save_path/self.modules[m]

                        if fname not in self.result_paths['modules']:
                           self.result_paths['modules'][fname] = {self.modules[m]:n/fname}
                        else:
                            temp =  self.result_paths['modules'][fname]
                            temp[self.modules[m]] = n/fname
                            self.result_paths['modules'][fname] = temp                   
                module_path = m[fname]
            else:
                module_path = None
        else:
            module_path=None
        
        save_features(sample, self.result_paths['features'][fname], self.result_paths['times'][fname], module_path, self.cci_features)


    def __call__(self, sample: Dict):
        """
        Take in a sample and perform batched feature extraction on the sample. The sample should be preprocessed and also be windowed using audio_processing.transforms.Window.
        :param sample: dict, audio sample with metadata
        :return sample: sample post feature extraction
        """
        fname = sample['fname']
        if not self.overwrite:
            if self.cci_features is None:
                if Path(str(self.result_paths['features'][fname]) + '.npz').exists():
                    print(f"Skipping {fname}.")
                    return
            else: 
                if self.cci_features.exists_object(self.result_paths['features'][fname]):
                    print(f"Skipping {fname}.")
                    return
        #expects samples to come in as a path that has ALREADY BEEN PREPROCESSED, then run path to wave and window. Window will check that sample rate is target sample rate
        sample = self.transforms(sample)
        
        wav = torch.squeeze(sample['waveform'])

        snippet_iter = sample['snippet_iter']
        batched_features = []
        times = []
        module_features = collections.defaultdict(list)

        for batch_idx, (snippet_starts, snippet_ends) in enumerate(tqdm(snippet_iter)):
            sample2 = copy.deepcopy(sample)
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
            del sample2 
            del output

        if not self.return_numpy and isinstance(out_features, np.ndarray):
            out_features = torch.from_numpy(out_features)
            if mod_ft: module_features = {name: (torch.from_numpy(features))for name, features in module_features.items()}

        out_features = np.concatenate(batched_features, axis=0) if self.return_numpy else torch.cat(batched_features, dim=0) # shape: (timesteps, features)
        
        if mod_ft:
            module_features = {name: (np.concatenate(features, axis=0) if self.return_numpy else torch.cat(features, dim=0))\
                            for name, features in module_features.items()}
            assert all(features.shape[0] == out_features.shape[0] for features in module_features.values()),\
                "Missing timesteps in the module activations!! (possible PyTorch bug)"
        else: 
            module_features = None
        times = np.concatenate(times, axis=0) if self.return_numpy else torch.cat(times, dim=0) / self.sampling_rate # shape: (timesteps, features)
        #times = np.concatenate(timestorch.cat(times, dim=0) / self.sampling_rate # convert samples --> seconds. shape: (timesteps,)
    
        sample['out_features'] = out_features
        sample['times']= times
        sample['module_features'] = module_features
        
        if self.pad_silence:
            sample = self.padding.remove_padding(sample)
        #print('Why remove silence')

        self._save(sample, fname)

        del sample #sth weird w memory
        return
    