"""
Example MFCC feature extractor

Author: Daniela Wiepert
Last Modified: 11/08/2024
"""
#IMPORTS
##built-in
import collections

##third-party
import librosa
import numpy as np
import torch

##local
from ._base_extraction import BaseExtractor

"""
SOME KEY THINGS ABOUT BUILDING EXTRACTORS WITHIN THIS PACKAGE:
* Make sure to save the name of python script to _EXTRACTIONTYPE_extraction.py (with the underscore)
* In order to run the code and access it outside of this directory, you will need to do the following:
1) go to audio_features/extractors/__init__.py and add the following:
    * from ._EXTRACTIONTYPE_extraction import *
    * in the __all__ variable, add elements 'EXTRACTORCLASSNAME', 'set_up_EXTRACTORTYPE_extractor'. 
    you can see how this is done for other extractors in that init file
2) If there are any new pip packages you needed to install to get this to work, go to setup.py and in the install_requires list, add the
    the package name and package version you installed (e.g. numpy==1.26.4)
3) Go to stimulus_features.py and add the following:
    * parser.add_argument() for any new arguments that you use to set up the extractor and aren't already included. You can specifically add
      a stimulus group as done for the cc_args (arg_group = parser.add_argument_group('name','What it does'), and add any 
      thing new to that arg_group. You can access with the regular args.VARIABLENAME still but it keeps things cleaner)
    * go to the TODO section (TODO: FEATURE EXTRACTION TYPES). add an elif branch, decide a name for the feature, and initialize the extractor
    * Don't mess with the default arguments, so if you can create some kind of debugging json (like the launch.json for VSCODE) you can set them there. 
      You can however set your own defaults for the new arguments. 
      Some things that must be toggled for every run:
        --full_context (IF YOU DON'T USE THIS IT WON'T RUN PROPERLY!!! I learned the hard way)
        --stimulus_dir=LOCAL_PATH_TO_STIMULUS (once I send)
        --feature_type=your new feature type
        --output_dir=LOCAL_PATH_TO_SAVE_TO (gotta save those features somehow)
4) Run and profit
"""

def set_up_mfcc_extractor(target_sample_rate:int=16000, min_length_samples:int=0, 
                 return_numpy:bool=True, num_select_frames:int=1, frame_skip:int=5):
    """
    We don't need this function for a base extractor, but if you have special parameters to deal with that require you to
    load models or read config files, I recommend using this kind of function to cleanly deal with setting up your extractor

    :param target_sample_rate: int, target sample rate for model
    :param min_length_samples: int, minimum length a sample can be to be fed into the model
    :param return_numpy: bool, true if returning numpy
    :param num_select_frames: int, default=1. This can safely be set to 1 for chunk size of 100 and non-whisper models. 
                              This specifies how many frames to select features from. 
    :param frame_skip: int, default=5. This goes with num_select_frames. For most HF models the window is 20ms, 
                       so in order to take 1 feature per batched waveform with chunksz = 100ms, you set 5 to say you take num_select_frames (1) every frame_skip. 
    :return: BaseExtractor
    """
    #TODO: Any extractor specific setup
    return MFCCExtractor(target_sample_rate=target_sample_rate, min_length_samples=min_length_samples, return_numpy=return_numpy,
                         num_select_frames=num_select_frames, frame_skip=frame_skip)


# Note how this inherits from BaseExtractor
class MFCCExtractor(BaseExtractor):
    """
    MFCC extractor class. 
    :param n_mfcc: int, number of mfccs to extract
    :param target_sample_rate: int, target sample rate for model
    :param min_length_samples: int, minimum length a sample can be to be fed into the model
    :param return_numpy: bool, true if returning numpy
    :param num_select_frames: int, default=1. This can safely be set to 1 for chunk size of 100 and non-whisper models. 
                              This specifies how many frames to select features from. 
    :param frame_skip: int, default=5. This goes with num_select_frames. For most HF models the window is 20ms, 
                       so in order to take 1 feature per batched waveform with chunksz = 100ms, you set 5 to say you take num_select_frames (1) every frame_skip. 
    """
    def __init__(self, n_mfcc:int=20, target_sample_rate:int=16000, min_length_samples:int=0, 
                 return_numpy:bool=True, num_select_frames:int=1, frame_skip:int=5):
        #INHERITED VALUES
        super().__init__(target_sample_rate=target_sample_rate, min_length_samples=min_length_samples, 
                         return_numpy=return_numpy, num_select_frames=num_select_frames, frame_skip=frame_skip)
        #if creating an actual extractor, you put function parameters that you may want to change in the extraction step 
        if n_mfcc:
          self.n_mfcc = n_mfcc
        else:
          self.n_mfcc = 20
        """
          Big note: for features like mfcc you can completely ignore num_select_frames and frame_skip such that the super function is actually
          super().__init__(target_sample_rate=target_sample_rate, min_length_samples=min_length_samples,return_numpy=return_numpy)
           
          This is because for MFCC you get a single feature for an entire window whereas for SSL you get a feature per frame, so a .1 second window would output 5 feature representations. 
          I only leave these in as an example for other feature extraction techniques. Talk to me about it if you're not sure whether to skip or not.
        """
      
    def __call__(self, sample:dict):
        """
        EXAMPLE EXTRACTOR: MFCCS

        Every extractor takes in a sample which is a dictionary containing information about an audio signal.
        For extraction a sample should have as keys: 'waveform', 'snippet_starts', 'snippet_ends', 'sample_rate'
        
        :param sample: dict, contains audio information
        :return sample: dict, contains audio information + added feature information
        """
        # STEP 1: CHECK INPUT
        #assertions for checking the sample is correctly processed
        keys = ['waveform', 'snippet_starts', 'snippet_ends', 'sample_rate']
        for k in keys: assert k in sample, f'{k} not in sample. Check that audio is processed correctly.'
        assert sample['waveform'].ndim == 2, 'Batched waveform must be of size (batch size x waveform size) with no extra channels'
        assert sample['sample_rate'] == self.target_sample_rate , f'Sample rate of the audio should be {self.target_sampling_rate}. Please check audio processing.'
        
        ##TODO: INCLUDE MODEL SPECIFC ASSERTIONS IF NECESSARY
        # for mfcc extraction with librosa, our batch size needed to be 1, so assert that the first dim of waveform shape is 1
        s = sample['waveform'].shape[0]
        assert sample['waveform'].shape[0] == 1, f'Only batch size of 1 is supported with MFCC extraction but audio has batch size of {s}'

        # STEP 2: INITIALIZE OUTPUT VARIABLES
        #set variables that we will be filling with features. All of these values should be added to the sample before outputting the results
        module_features = collections.defaultdict(list) #module_features is a dictionary where each key specifies what kind of features the values are. For example, if extracting multiple layers from an SSL model, a key-value pair in this dictionary would be 'layer.0':[FEATURES]. See hfExtractor and SPARCExtractor for examples of how this output should look.
        out_features = [] #empty list that contains one representative feature. For example, it might be the final hidden state output of an SSL model, or the EMA features rather than pitch/loudness information that is also extracted from a SPARC model
        times = [] # times are shared across all features. The indices are aligned between all of these previous variables, such that a module_features[k][i] was extracted from times[i]

        #take out snippet starts and ends
        snippet_starts = sample['snippet_starts']
        snippet_ends = sample['snippet_ends']

        # STEP 3: RUN FEATURE EXTRACTION
        ##TODO: run feature extraction. As an example, here is mfcc extraction. There should be a separate mfcc extractor, though. The BaseExtractor should never be used directly.
        ## EXTRACTOR SPECIFIC CHANGES TO WAVEFORM
        # librosa.feature.mfcc requires numpy and will expect a waveform of (samples, channels), which in our case should be channels = 1 as everything is monochannel
        waveform_np = sample['waveform'].numpy()
        waveform_np = np.squeeze(waveform_np)
        S = librosa.feature.melspectrogram(y=waveform_np, sr=self.target_sample_rate, n_mels=128,fmax=8000)
        chunk_features = librosa.feature.mfcc(S=librosa.power_to_db(S),n_mfcc=self.n_mfcc) 

        # STEP 4: format output
        """
        Now we will be putting the information into the proper output variables. 
        
        To do this, we use output_inds. For our purposes just use the default output_inds always.
        
        If you have a window with multiple frames of output (e.g. the feature extraction outputs features in a sliding window like SSL representations which output a feature for every 20ms sliding window), 
        the default output_inds will only save out the very final frame's representation. Basically, we should have 1 output for every window.
        As a result, we expect the output_features to be of size (batch size, feature dim).

        If we ever want to save out more for some reason or the features have some shape (feat dim 1, feat dim 2, ...), then we
        will need to manually change things so the output dimensions are still leading with batch size (batch_size, feat dim 1, feat dim 2, ...)
        """
        
        for out_idx, output_offset in enumerate(self.output_inds): 
            """
            Big note: for features like mfcc you can completely ignore this as you get a single feature for an entire window
            whereas for SSL you get a feature per frame, so a .1 second window would output 5 feature representations. 
            As of now, we only want to take the last one so we use output inds which tells us the index to take (-1)
            
            If your feature does not have frames, you will not need to use self.output_inds, self.num_select_frames, or self.frame_skip ever 
            (so these can also be ignored in the super.__init__() inheritance line (just don't give it as input)) and you can remove the for loop
            Talk to me about it if you're not sure whether to skip or not.
            """
            # TODO: avoid re-stacking the times. may require tracking snippet
            # idxs and indexing into `snippet_times`
            
            times.append(torch.stack([snippet_starts, snippet_ends], dim=1)) #KEEP AS IS

            #TODO: append the correct feature to out_features
            out_representation = chunk_features
            ##oops this is an example I was talking about...the output is not based on frames but rather gives an entire features vector for each MFCC so need to add a dim for batch size
            out_representation = np.expand_dims(out_representation, axis=0)

            #need to check whether we want to return as numpy or torch tensor. 
            #This will change depending on how your features are output
            #For this example, mfccs are output as a numpy, so we need to check if they should be converted to a tensor
            if not self.return_numpy: torch.from_numpy(out_representation)
            #Then append to output
            out_features.append(out_representation) #

            #TODO: handle module_features
            #while some features may only have one output, If we had layers, we would want to append each desired layer representation to module_features,
            # you would create some kind of code equivalent to the following line, or do an iterator to append each layer (see hfExtractor for example)
            #module_features['mfcc'].append(out_representation)
        
        # STEP 5: concatenate features and times - INCLUDE AS IS
        # since we want to be flexible with the batch size for many feature extractors, and because our output is currently a list of arrays, we concatenate to get the proper format
        # This can be included as is as long as you make sure you have already converted the represenations to the right format (numpy.ndarray, torch tensor)
        # We handled the conversion in line 101, so nothing needs to be changed here

        out_features = np.concatenate(out_features, axis=0) if self.return_numpy else torch.cat(out_features, dim=0) # shape: (timesteps, features)
        
        #### IF YOU WERE WORKING WITH MODULE_FEATURES, UNCOMMENT
        #module_features = {name: (np.concatenate(features, axis=0) if self.return_numpy else torch.cat(features, dim=0))\
        #                for name, features in module_features.items()}
        #assert all(features.shape[0] == out_features.shape[0] for features in module_features.values()),\
        #    "Missing timesteps in the module features!!" #this shouldn't be possible with our current example, but could be a bug in some pytorch models with multiple layers, so keep this assertion
    
        times = torch.cat(times, dim=0) / self.target_sample_rate # convert samples --> seconds. shape: (timesteps,)
        if self.return_numpy: times = times.numpy()

        # STEP 6: clean up and prep output
        del chunk_features # TODO: trying to fix memory leak. remove if unneeded

        sample['out_features'] = out_features
        #IF YOU WERE WORKING WITH MODULE FEATURES, UNCOMMENT
        #sample['module_features'] = module_features
        sample['times'] = times
        return sample