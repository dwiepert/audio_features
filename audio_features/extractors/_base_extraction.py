"""
Base feature extractor

Author: Daniela Wiepert
Last Modified: 11/08/2024
"""
#IMPORTS
##third-party
import numpy as np

class BaseExtractor:
    """
    Base extractor class. Initializes common variables
    :param target_sample_rate: int, target sample rate for model
    :param min_length_samples: int, minimum length a sample can be to be fed into the model
    :param return_numpy: bool, true if returning numpy
    :param num_select_frames: int, default=1. This can safely be set to 1 for chunk size of 100 and non-whisper models. 
                              This specifies how many frames to select features from. 
    :param frame_skip: int, default=5. This goes with num_select_frames. For most HF models the window is 20ms, 
                       so in order to take 1 feature per batched waveform with chunksz = 100ms, you set 5 to say you take num_select_frames (1) every frame_skip. 
    """
    def __init__(self, target_sample_rate:int=16000, min_length_samples:int=0, 
                 return_numpy:bool=True, num_select_frames:int=1, frame_skip:int=5):
        self.target_sample_rate = target_sample_rate
        self.min_length_samples = min_length_samples
        self.return_numpy = return_numpy
        self.num_select_frames=num_select_frames
        self.frame_skip=frame_skip
        # Output inds determines how many features to save per chunk. With defaults it should be 1 per chunk, specifically the last one (-1)
        self.output_inds = np.array([int(-1 - self.frame_skip*i) for i in reversed(range(self.num_select_frames))])
        self.modules = None