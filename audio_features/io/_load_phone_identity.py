"""
Load phoneme features

Author(s): Aditya Vaidya, Daniela Wiepert, Others unknown
Last modified: 11/28/2024
"""
#IMPORTS
##built-in
from typing import List
##local
from database_utils.functions import *

_articulates = ['bilabial','postalveolar','alveolar','dental','labiodental',
			   'velar','glottal','palatal', 'plosive','affricative','fricative',
			   'nasal','lateral','approximant','voiced','unvoiced','low', 'mid',
			   'high','front','central','back']

class phoneIdentity:
    """
    Phoneme Identity features

    :param fnames: List of stimulus names
    """
    def __init__(self, fnames:List[str]):
        self.fnames = fnames

        ###From encoding-model-large-features
        self.artdict = cc.get_interface('subcortical', verbose=False).download_json('artdict')
        self.phonseqs = get_story_phonseqs(self.fnames) #(phonemes, phoneme_times, tr_times)
        
        self.downsampled_arthistseqs = {}
        for story in self.fnames:
            olddata = np.array(
                [ph.upper().strip("0123456789") for ph in self.phonseqs[story].data])
            ph_2_art = self._ph_to_articulate(olddata, self.artdict)
            arthistseq = self._histogram_articulates(ph_2_art, self.phonseqs[story])
            self.downsampled_arthistseqs[story] = lanczosinterp2D(
			arthistseq[0], arthistseq[2], arthistseq[3])
    
    def _ph_to_articulate(ds: DataSequence, ph_2_art:dict):
        """ 
        Following make_phoneme_ds converts the phoneme DataSequence object to an
        articulate Datasequence for each grid.
        From encoding-model-large-features
        :param ds: a DataSequence
        :param ph_2_art: dictionary mapping phonemes to articulatory features
        :return articulate_ds: DataSequence with articulatory features
        """
        articulate_ds = []
        for ph in ds:
            try:
                articulate_ds.append(ph_2_art[ph])
            except:
                articulate_ds.append([''])
	
        return articulate_ds
    
    def _histogram_articulates(ds:DataSequence, data,  articulateset:List[str]=_articulates):
        """
        Histograms the articulates in the DataSequence [ds].
        From encoding-models-large-features

        :param ds: a DataSequence
        :param data: original data
        :param articulateset: list of articulatory features
        :return final_data: articulatory features as np.ndarray as 0/1
        :return data.split_inds, data.data_times, data.tr_times: original values from data
        """
        final_data = []
        for art in ds:
            final_data.append(np.isin(articulateset, art))
        final_data = np.array(final_data)
        return (final_data, data.split_inds, data.data_times, data.tr_times)

    def get_features(self):
        """
        :return downsampled_arthistseqs: return downsampled articulatory features
        """
        return self.downsampled_arthistseqs
    
    def get_seqs(self):
        """
        :return phonseqs: return phoneme sequences
        """
        return self.phonseqs