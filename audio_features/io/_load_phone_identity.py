from database_utils.functions import *

_articulates = ['bilabial','postalveolar','alveolar','dental','labiodental',
			   'velar','glottal','palatal', 'plosive','affricative','fricative',
			   'nasal','lateral','approximant','voiced','unvoiced','low', 'mid',
			   'high','front','central','back']

class phoneIdentity:
    """
    """
    def __init__(self, fnames):
        self.fnames = fnames

        ###From encoding-model-large-features
        self.artdict = cc.get_interface('subcortical', verbose=False).download_json('artdict')
        self.phonseqs = get_story_phonseqs(self.fnames) #(phonemes, phoneme_times, tr_times)
        
        downsampled_arthistseqs = {}
        for story in self.fnames:
            olddata = np.array(
                [ph.upper().strip("0123456789") for ph in self.phonseqs[story].data])
            ph_2_art = self._ph_to_articulate(olddata, self.artdict)
            arthistseq = self._histogram_articulates(ph_2_art, self.phonseqs[story])
            self.downsampled_arthistseqs[story] = lanczosinterp2D(
			arthistseq[0], arthistseq[2], arthistseq[3])
    
    def _ph_to_articulate(ds, ph_2_art):
        """ Following make_phoneme_ds converts the phoneme DataSequence object to an
        articulate Datasequence for each grid.
        From encoding-model-large-features
        """
        articulate_ds = []
        for ph in ds:
            try:
                articulate_ds.append(ph_2_art[ph])
            except:
                articulate_ds.append([''])
	
        return articulate_ds
    
    def _histogram_articulates(ds, data,  articulateset=_articulates):
        """Histograms the articulates in the DataSequence [ds].
        From encoding-models-large-features"""
        final_data = []
        for art in ds:
            final_data.append(np.isin(articulateset, art))
        final_data = np.array(final_data)
        return (final_data, data.split_inds, data.data_times, data.tr_times)

    def get_features(self):
        return self.downsampled_arthistseqs
    
    def get_seqs(self):
        return self.phonseqs