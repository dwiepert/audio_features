"""
Extract wavLM like EMA efeatures using emaae

Author(s): Daniela Wiepert
Last modified: 02/14/2025
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
from emaae.models import  CNNAutoEncoder

class EMAAEExtractor(BaseExtractor):
    """
    :param model: pretrained EMAAE model
    :param save_path: local path to save extractor configuration information to
    :param target_sample_rate: int, target sample rate for model
    :param return_numpy: bool, true if returning numpy
    """
    def __init__(self, model:CNNAutoEncoder, save_path:Union[str, Path], target_sample_rate:int=16000, return_numpy:bool=True):
        super().__init__(target_sample_rate=target_sample_rate, return_numpy=return_numpy,)
        self.model = model 
        self.model_type = model.get_type()

        assert len(self.output_inds) == 1, "Only one output per evaluation is "\
            "supported  (because they don't provide the downsampling rate)"
        
        self.config = {'feature_type': 'emaae', 'model_type': self.model_type,
                       'return_numpy': self.return_numpy, 'keep_all':self.keep_all}

        #saving things
        self.save_path = Path(save_path)
        os.makedirs(self.save_path, exist_ok=True)
        with open(str(save_path /'EMAAEExtractor_config.json'), 'w') as f:
            json.dump(self.config, f)

        self.modules= None
        
    def __call__(self, sample:dict) -> dict:
        """
        Run feature extraction on a snippet of a sample 

        EMAAE output has 50 Hz sampling rate which matches the sampling rate of the SSL features? 

        :param sample: dict, audio sample with metadata
        :return sample: sample with outputs
        """

        keys = ['ema']
        for k in keys: assert k in sample, f'{k} not in sample. Check that audio is processed correctly.'

        ema = sample['ema']

        features = torch.squeeze(self.model.encode(ema))

        if self.return_numpy:
            features = features.numpy()
            add = sample['fname'] + '.npz'
            new_path = self.save_path / add
            np.save(new_path, features)
        else:
            add = sample['fname'] + '.pt'
            new_path = self.save_path / add
            torch.save(features, str(new_path))

        sample['out_features'] = features

        return sample

def set_up_emaae_extractor(save_path:Union[str,Path],ckpt:str, config:Union[str,Path], return_numpy:bool=True) -> EMAAEExtractor:
    """
    Set up emaae extractor

    :param save_path: local path to save extractor configuration information to
    :param ckpt: model checkpoint
    :param config: model config
    :param return_numpy: bool, true if returning numpy
    :param keep_all: bool, true if you want to keep all outputs from each batch
    :return: initialized EMAAEExtractor
    """

    use_cuda = torch.cuda.is_available()
    print(f"use cuda: {use_cuda}")
    if use_cuda: 
        device = "cuda"
    else:
        device = "cpu"
    
    assert Path(config).exists(), 'Must give model config file for EMAAE'
    assert Path(checkpoint).exists(), 'Must give a model checkpoint for EMAAE.'
    with open(str(config), "rb") as f:
        model_config = json.load(f)

    model = CNNAutoEncoder(input_dim=model_config['input_dim'], n_encoder=model_config['n_encoder'], n_decoder=model_config['n_decoder'], inner_size=model_config['inner_size'])
    checkpoint = torch.load(ckpt, map_location='cpu')
    model.load_state_dict(checkpoint)
    model = model.to(device)

    return EMAAEExtractor(model=model, save_path=save_path, return_numpy=return_numpy)