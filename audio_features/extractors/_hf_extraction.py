"""
Extract features using hugging face

Author(s): Aditya Vaidya, Daniela Wiepert
Last Modified: 11/14/2024
"""

#IMPORTS
##built-in
import collections
import copy
import json
import os
from pathlib import Path
from typing import Optional, List, Union

##third-party
import numpy as np
import torch
from transformers import AutoModel, AutoModelForPreTraining, PreTrainedModel,\
                         AutoFeatureExtractor, WhisperModel

##local
from ._base_extraction import BaseExtractor

def set_up_hf_extractor(model_name:str, save_path:Union[str, Path], use_featext:bool, sel_layers: Optional[List[int]],
                        target_sample_rate:int=16000, model_config_path:Union[str,Path]='audio_features/configs/hf_model_configs.json', 
                        return_numpy:bool=True, num_select_frames:int=1, frame_skip:int=5):
    """
    Function for setting up an hf feature extractor (loading model in, setting seeds, freezing extractor, etc.)

    :param model_name: str, hugging face model name (key in model_configs)
    :param save_path: local path to save extractor configuration information to
    :param use_featext: bool, true if model has a separate feature extractor
    :param sel_layers: List[int], list of layers to generate features for. Optional
    :param target_sample_rate: int, target sampling rate (default=16000 hz)
    :param model_config_path: str or Path, path to model configs json for hugging face
    :param return_numpy: bool, true if returning numpy
    :param num_select_frames: int, default=1. This can safely be set to 1 for chunk size of 100 and non-whisper models. 
                              This specifies how many frames to select features from. 
    :param frame_skip: int, default=5. This goes with num_select_frames. For most HF models the window is 20ms, 
                       so in order to take 1 feature per batched waveform with chunksz = 100ms, you set 5 to say you take num_select_frames (1) every frame_skip
    :return: initialized extractor
    """
    assert model_name is not None, 'Must give model name for hugging face models'
    assert model_config_path is not None, 'Must give model config for hugging face models'
    #Load model configuration
    with open(str(model_config_path), 'r') as f:
        model_config = json.load(f)[model_name]
        model_hf_path = model_config['huggingface_hub']

    #Read in data from model_config
    if 'min_input_length' in model_config:
        # this is stored originally in **samples**!!!
        min_length_samples = model_config['min_input_length']
    elif 'win_ms' in model_config:
        min_length_samples = model.config['win_ms'] / 1000. * target_sample_rate
    else:
        min_length_samples = 0

    if 'stride' in model_config:
        frame_len_sec = model_config['stride'] / target_sample_rate
    else:
        frame_len_sec = None

    #Load feature extractor
    feature_extractor = None
    if use_featext:
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_hf_path)
    if feature_extractor is not None:
        target_sample_rate = feature_extractor.sampling_rate
    
    #Load model
    use_cuda = torch.cuda.is_available() #USED IN CASE YOU TEST ON COMPUTER WITHOUT GPU
    print('Loading model', model_name, 'from the Hugging Face Hub...')
    model = AutoModel.from_pretrained(model_hf_path, output_hidden_states=True, trust_remote_code=True)
    if use_cuda:
        model = model.cuda()

   
    # Re-initialize the weights, if requested (using the a specific seed, if
    # specified)
    if ('random_weights' in model_config) and model_config['random_weights']:
        print("Re-initializing model weights...")
        if 'random_seed' in model_config:
            seed = model_config['random_seed']
            # Re-seed all RNGs because some models *might* use non-pytorch RNGs
            torch.manual_seed(seed)
            np.random.seed(seed)
            import random
            random.seed(seed)
        else:
            print("User did not specify a random seed")

        #Freeze the feature extractor
        freeze_extractor = model_config.get('freeze_extractor', False)
        if freeze_extractor:
            print("Randomizing weights but NOT for the feature extractor")
            ext_state_dict = copy.deepcopy(model.feature_extractor.state_dict())

        model.apply(model._init_weights)

        if freeze_extractor:
            model.feature_extractor.load_state_dict(ext_state_dict)
            del ext_state_dict # try to save some memory

    return hfExtractor(model=model, model_type=model_name, save_path=save_path, feature_extractor=feature_extractor, target_sample_rate=target_sample_rate, 
                       min_length_samples=min_length_samples, sel_layers=sel_layers, return_numpy=return_numpy,
                       num_select_frames=num_select_frames,frame_skip=frame_skip, frame_len_sec=frame_len_sec)
    
class hfExtractor(BaseExtractor):
    """
    Feature extractor based on hugging face models

    :param model: pretrained HF model
    :param model_type: str, type of hugging face model
    :param save_path: local path to save extractor configuration information to
    :param feature_extractor: loaded feature extractor (default = None)
    :param target_sample_rate: int, target sample rate for model
    :param min_length_samples: int, minimum length a sample can be to be fed into the model
    :param sel_layers: List[int], list of layers to generate features for. Optional
    :param return_numpy: bool, true if returning numpy
    :param num_select_frames: int, default=1. This can safely be set to 1 for chunk size of 100 and non-whisper models. 
                              This specifies how many frames to select features from. 
    :param frame_skip: int, default=5. This goes with num_select_frames. For most HF models the window is 20ms, 
                       so in order to take 1 feature per batched waveform with chunksz = 100ms, you set 5 to say you take num_select_frames (1) every frame_skip
    :param frame_len_sec: int, information on how long a frame is in the model in seconds
    """
    def __init__(self, model: PreTrainedModel, model_type:str, save_path:Union[str,Path], target_sample_rate:int=16000, min_length_samples:int=0,
                 feature_extractor=None, sel_layers: Optional[List[int]]=None, return_numpy:bool=True,
                 num_select_frames:int=1, frame_skip:int=5, frame_len_sec:float=None):
        
        #INHERITED VALUES
        super().__init__(target_sample_rate=target_sample_rate, min_length_samples=min_length_samples, 
                         return_numpy=return_numpy, num_select_frames=num_select_frames, frame_skip=frame_skip)
        

        #Hugging Face specific values
        self.model = model #pretrained HF model
        self.model_type = model_type
        torch.set_grad_enabled(False) # VERY important! (for memory)
        self.model.eval()
        self.is_whisper_model = isinstance(self.model, WhisperModel)

        self.feature_extractor = feature_extractor # model specific feature extractor
        
        self.frame_len_sec = frame_len_sec #unnecessary variable, useful for understanding the model you are extracting from 

        self.move_to_cpu = False #for hugging face, if you want to return numpy you need to move it to the cpu
        if self.return_numpy:
            self.move_to_cpu = True
        
        self.sel_layers=sel_layers #select which layers you want to output (besides the final hidden state)

        assert len(self.output_inds) == 1, "Only one output per evaluation is "\
            "supported for Hugging Face (because they don't provide the downsampling rate)"
        
        self.config = {'feature_type':'hf', 'model_type': self.model_type, 'target_sample_rate': self.target_sample_rate, 'min_length_samples':self.min_length_samples,
                       'sel_layers': self.sel_layers, 'return_numpy': self.return_numpy, 'num_select_frames':self.num_select_frames, 'frame_skip': self.frame_skip,
                       'frame_len_sec': self.frame_len_sec}
        
        #saving things
        self.save_path = Path(save_path)
        os.makedirs(self.save_path, exist_ok=True)
        with open(str(save_path /'hfExtractor_config.json'), 'w') as f:
            json.dump(self.config, f)

        self.modules = {}
        if self.sel_layers is not None:
            for s in self.sel_layers:
                if self.is_whisper_model:
                    # Leave the option open for using decoder layers in the
                    # future
                    module_name = f"encoder.{s}"
                else:
                    module_name = f"layer.{s}"
                    
                if s not in self.modules: self.modules[s] = module_name
    
    def __call__(self, sample:dict):
        """
        Run feature extraction on a snippet of a sample for hugging face models

        :param sample: dict, audio sample with metadata
        :return sample: sample with outputs
        """
        ## ASSERTIONS to check sample is correctly processed
        keys = ['waveform', 'snippet_starts', 'snippet_ends', 'sample_rate']
        for k in keys: assert k in sample, f'{k} not in sample. Check that audio is processed correctly.'
        assert sample['waveform'].ndim == 2, 'Batched waveform must be of size (batch size x waveform size) with no extra channels'
        assert sample['sample_rate'] == self.target_sample_rate , f'Sample rate of the audio should be {self.target_sampling_rate}. Please check audio processing.'
        #assert sample['sample_rate'] == self.target_sample_rate, f'Sample rate of the audio should be {self.target_sample_rate} but was {sample['sample_rate']}. Please check audio processing.'
        
        #Hugging face model specific assertions
        assert not self.model.training, 'Model must be in eval mode'
        assert not torch.is_grad_enabled(), 'Gradients must be disabled'

        #initialize common output variables 
        module_features = collections.defaultdict(list)
        out_features = []
        times = [] # times are shared across all layers

        #get start and ends of snippets 
        snippet_starts = sample['snippet_starts']
        snippet_ends = sample['snippet_ends']

        # Use a pre-processor if given (e.g. to normalize the waveform), and
        # then feed into the model.
        if self.feature_extractor is not None:

            feature_extractor_kwargs = {}
            if self.is_whisper_model: 
                # Because Whisper auto-pads all inputs to 30 sec., we'll use
                # the attention mask to figure out when the "last" relevant
                # input was.
                features_key = 'input_features'
                feature_extractor_kwargs['return_attention_mask'] = True
            else:
                features_key = 'input_values'
            
            preprocessed_snippets = self.feature_extractor(list(sample['waveform'].cpu().numpy()),
                                                           return_tensors='pt',
                                                           sampling_rate=self.target_sample_rate,
                                                           **feature_extractor_kwargs)
            
            if self.is_whisper_model: #WHISPER HAS SOME WEIRD STUFF THAT HAPPENS, SKIP TO ELSE STATEMENT IF NOT USING WHISPER
                chunk_features = self.model.encoder(preprocessed_snippets[features_key].to(self.model.device))

                # Now we need to figure out which output index to use, since 2
                # conv layers downsample the inputs before passing them into
                # the encoder's Transformer layers. We can redo the encoder's
                # 1-D conv's on the attention mask to find the final output that
                # was influenced by the snippet.
                contributing_outs = preprocessed_snippets.attention_mask # 1 if part of waveform, 0 otherwise. shape: (batchsz, 3000)
                # Taking [0] works because all snippets have the same length.
                # Add the dimension back for `conv1d` to work
                # TODO: assert that all clips are the same length?
                contributing_outs = contributing_outs[0].unsqueeze(0)

                contributing_outs = torch.nn.functional.conv1d(contributing_outs,
                                                               torch.ones((1,1)+self.model.encoder.conv1.kernel_size).to(contributing_outs),
                                                               stride=self.model.encoder.conv1.stride,
                                                               padding=self.model.encoder.conv1.padding,
                                                               dilation=self.model.encoder.conv1.dilation,
                                                               groups=self.model.encoder.conv1.groups)
                # shape: (batchsz, 1500)
                contributing_outs = torch.nn.functional.conv1d(contributing_outs,
                                                               torch.ones((1,1)+self.model.encoder.conv2.kernel_size).to(contributing_outs),
                                                               stride=self.model.encoder.conv2.stride,
                                                               padding=self.model.encoder.conv2.padding,
                                                               dilation=self.model.encoder.conv2.dilation,
                                                               groups=self.model.encoder.conv1.groups)

                final_output = contributing_outs[0].nonzero().squeeze(-1).max()
            else:
                #ipdb.set_trace()
                # sampling rates must match if not using a pre-processor
                chunk_features = self.model(preprocessed_snippets[features_key].to(self.model.device))
        else:
            chunk_features = self.model(sample['waveform'].to(self.model.device))


        # Make sure we have enough outputs
        # TODO: looking at hidden state shapes sounds dangerous b/c the
        # order of dims might be different across models. Can we look at
        # 'last_hidden_state' instead?
        if(chunk_features['last_hidden_state'].shape[1] < (self.num_select_frames-1) * self.frame_skip - 1):
            # TODO: instead of skipping it entirely, we should probably
            # just make sure we have at least 1 output, and skip the rest
            # of the selected output frames.
            print("Skipping - only had", chunk_features['last_hidden_state'].shape[1],
                    "outputs, whereas", (self.num_select_frames-1) * self.frame_skip - 1, "were needed.")
            return None

        # If whisper model, the output inds functions a little differently. Address accourdingly
        if self.is_whisper_model:
            self.output_inds = [final_output]
        
        for out_idx, output_offset in enumerate(self.output_inds):
            # TODO: avoid re-stacking the times. may require tracking snippet
            # idxs and indexing into `snippet_times`
            times.append(torch.stack([snippet_starts, snippet_ends], dim=1))

            output_representation = chunk_features['last_hidden_state'][:, output_offset, :] # shape: (batchsz, hidden_size)
            if self.move_to_cpu: output_representation = output_representation.cpu()
            if self.return_numpy: output_representation = output_representation.numpy()
            out_features.append(output_representation)

            # Collect features from individual layers
            # NOTE: outs['hidden_states'] might have an extra element at
            # the beginning for the feature extractor.
            # e.g. 25 "layers" --> CNN output + 24 transformer layers' output
            for layer_idx, layer_activations in enumerate(chunk_features['hidden_states']):
                # TODO: can we get the layer names programatically from
                # their API? (in s3prl, we could use
                # outs['hidden_layer_info'], or something.)

                # Only save layers that the user wants (if specified)
                if self.sel_layers:
                    if layer_idx not in self.sel_layers: continue

                    layer_representation = layer_activations[:, output_offset, :] # shape: (batchsz, hidden_size)
                    if self.move_to_cpu: layer_representation = layer_representation.cpu()
                    if self.return_numpy: layer_representation = layer_representation.numpy() # TODO: convert to numpy at the end
                    
                    module_name = self.modules[layer_idx]

                    module_features[module_name].append(layer_representation)
        
        out_features = np.concatenate(out_features, axis=0) if self.return_numpy else torch.cat(out_features, dim=0) # shape: (timesteps, features)
        module_features = {name: (np.concatenate(features, axis=0) if self.return_numpy else torch.cat(features, dim=0))\
                        for name, features in module_features.items()}
        assert all(features.shape[0] == out_features.shape[0] for features in module_features.values()),\
            "Missing timesteps in the module activations!! (possible PyTorch bug)"
        times = torch.cat(times, dim=0) / self.target_sample_rate # convert samples --> seconds. shape: (timesteps,)
        if self.return_numpy: times = times.numpy()

        del chunk_features # TODO: trying to fix memory leak. remove if unneeded
            
        sample['out_features'] = out_features
        sample['module_features'] = module_features
        sample['times'] = times
        return sample
        