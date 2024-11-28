"""
"""
from typing import List
import numpy as np
from audio_features.models import SemanticModel
from database_utils.functions import get_story_wordseqs, make_semantic_model_v1, lanczosinterp2D

class wordIdentity:
    """
    """
    def __init__(self, fnames:List[str], pretrained_path:str='./audio_features/data/english1000sm.hf5'):
        self.pretrained_path = pretrained_path
        self.fnames = fnames
        self.model = SemanticModel.load(self.pretrained_path)
        
        self.wordseqs = get_story_wordseqs(self.fnames)
        self.wordds = {}
        vectors = {}
        #TODO: get TIMES
        for story in self.fnames:
            sm = make_semantic_model_v1(self.wordseqs[story], self.model)
            self.wordds[story] = sm
            vectors[story] = sm.data
        self.downsampled_vectors = self._downsample_word_vectors(self.fnames, vectors)


    def _downsample_word_vectors(self, allstories, word_vectors):
        """Get Lanczos downsampled word_vectors for specified stories.

        Args:
            allstories: List of stories to obtain vectors for.
            word_vectors: Dictionary of {story: <float32>[num_story_words, vector_size]}

        Returns:
            Dictionary of {story: downsampled vectors}
        """
        downsampled_semanticseqs = dict()
        for story in allstories:
            downsampled_semanticseqs[story] = lanczosinterp2D(
                word_vectors[story], self.wordseqs[story].data_times,
                self.wordseqs[story].tr_times, window=3)
        return downsampled_semanticseqs

    def get_seqs(self):
        return self.wordseqs 
    
    def get_features(self):
        return self.downsampled_vectors