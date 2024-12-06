"""
Load phone or word features

Author(s): Aditya Vaidya, Daniela Wiepert, Others unknown
Last modified: 12/04/2024
"""

#IMPORTS
##built-in
import os
import json
import time
from typing import List, Optional
from pathlib import Path
##third-party
import numpy as np
##local
from database_utils.functions import *
from audio_features.io import load_features
from audio_features.models import SemanticModel

_articulates = ['bilabial','postalveolar','alveolar','dental','labiodental',
			   'velar','glottal','palatal', 'plosive','affricative','fricative',
			   'nasal','lateral','approximant','voiced','unvoiced','low', 'mid',
			   'high','front','central','back']

_bad_words = ['{IG}', '{CG}','{NS}', 'SP\x7f', 'BR ', 'BR', 'LS', 'NS', 'SP', 'BRLS', 'AH0V', '', ' ']

class Identity:
    """
    Identity features
    """

    def __init__(self, identity_type:str, features:dict, identity_dir:str, align_dir:str, pretrained_path:Optional[str], cci_features=None, recursive:bool=False, overwrite:bool=False):
        self.identity_type=identity_type
        assert self.identity_type in ['word','phone']
        self.pretrained_path = pretrained_path
        if self.identity_type=='word':
            assert self.pretrained_path is not None, 'Give pretrained path'
        self.features = features 
        self.fnames = list(features.keys())
        self.identity_dir =Path(identity_dir)
        self.align_dir = Path(align_dir)
        self.cci_features = cci_features
        self.recursive=recursive
        self.identity = {}
        self.overwrite = overwrite

        self._load_aligned_feats()

        if self.aligned_feats is None:
            self._load_id_feats()

            if not bool(self.identity):
                self._generate_id_feats()
            
            self._identity_to_ind()
            self._generate_aligned_feats()

    def _load_aligned_feats(self):
        """
        """
        self.aligned_feats=None
        feats = load_features(self.align_dir, self.identity_type, self.cci_features, self.recursive, ignore_str=['_idtargets','_regtargets', '_times'])
        id_targets = load_features(self.align_dir, self.identity_type,  self.cci_features, self.recursive, search_str='_idtargets')
        reg_targets = load_features(self.align_dir, self.identity_type,  self.cci_features, self.recursive, search_str='_regtargets')
        times = load_features(self.align_dir, self.identity_dir,  self.cci_features, self.recursive, search_str='_times')
        #all equivalent
        if bool(feats) and bool(id_targets) and bool(reg_targets) and bool(times):
            fstories = set(list(feats.keys()))
            istories = set(list(id_targets.keys()))
            rstories = set(list(reg_targets.keys()))
            tstories = set(list(times.keys()))

            if fstories == set(self.fnames) and istories == set(self.fnames) and rstories == set(self.fnames) and tstories == set(self.fnames):
                self.aligned_feats={'features': feats, 'identity_targets':id_targets, 'reg_targets':reg_targets, 'times':times}
        
    def _load_id_feats(self):
        """
        """
        olddata = load_features(self.identity_dir, self.identity_type, self.cci_features, self.recursive, search_str='_identity')
        newdata = load_features(self.identity_dir, self.identity_type, self.cci_features, self.recursive, ignore_str=['_identity', '_times'])
        times = load_features(self.identity_dir, self.identity_type, self.cci_features, self.recursive, search_str='_times')

        if bool(olddata) and bool(newdata) and bool(times):
            ostories = set(list(olddata.keys()))
            nstories = set(list(newdata.keys()))
            tstories = set(list(times.keys()))

            if ostories == set(self.fnames) and nstories == set(self.fnames) and tstories == set(self.fnames):
                for story in self.fnames:
                    self.identity[story] = {'original_data':olddata[story], 'feature_data':newdata[story], 'times': times[story]}

    def _generate_id_feats(self):
        if self.identity_type == 'phone':
            self._generate_phone_feats()
        else:
            self._generate_word_feats()
    
    def _generate_phone_feats(self):
        self.artdict = cc.get_interface('subcortical', verbose=False).download_json('artdict')
        self.phonseqs = get_story_phonseqs(self.fnames) #(phonemes, phoneme_times, tr_times)

        for story in self.fnames:
            olddata = [ph.upper().strip("0123456789") for ph in self.phonseqs[story].data]
            olddata = np.array(
                [ph.strip(' ') for ph in olddata])
            ph_2_art = self._ph_to_articulate(ds=olddata, ph_2_art=self.artdict)
            arthistseq = self._histogram_articulates(ds=ph_2_art, data=self.phonseqs[story])
    
            ### CHECK  feature data doesn't need to be as type int???
            self._save_identity(story, {'original_data': olddata, 'feature_data': arthistseq[0], 'times': arthistseq[2]})
    
    def _save_identity(self, story, data_dict):
        self.identity[story] = data_dict
        if self.cci_features is not None:
            self.cci_features.upload_raw_array(str(self.identity_dir/story) + '_identity', data_dict['original_data'])
            self.cci_features.upload_raw_array(str(self.identity_dir/story), data_dict['feature_data'])
            self.cci_features.upload_raw_array(str(self.identity_dir/story) + '_times', data_dict['times'])
        else:
            os.makedirs(self.identity_dir, exist_ok=True)
            np.savez_compressed(str(self.identity_dir /story)+'_identity.npz', data_dict['original_data'])
            np.savez_compressed(str(self.identity_dir /story)+'.npz', data_dict['feature_data'])
            np.savez_compressed(str(self.identity_dir /story)+'_times.npz',data_dict['times'])
            
    def _ph_to_articulate(self, ds:DataSequence, ph_2_art:dict):
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
    
    def _histogram_articulates(self, ds:DataSequence, data,  articulateset:List[str]=_articulates):
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

    def _generate_word_feats(self):
        self.model = SemanticModel.load(self.pretrained_path)
        self.wordseqs = get_story_wordseqs(self.fnames)

        for story in self.fnames:
            sm = make_semantic_model_v1(self.wordseqs[story], self.model)
            olddata = np.array(self.wordseqs[story].data)
            newdata = sm.data
            times =  self.wordseqs[story].data_times

            self._save_identity(story,{'original_data': olddata, 'feature_data': newdata, 'times':times})
    
    def _identity_to_ind(self):
        vdir = self.identity_dir / 'vocab.json'
        if vdir.exists():
            with open(str(vdir), 'r') as f:
                self.vocab = json.load(f)
        else:
            self.vocab = {}
            total = 0
            i = 0
            for s in self.fnames:
                # print('remove later')
                # if s in self.word_identity:
                pi = self.identity[s]['original_data']
                for p in pi:
                    #print(p)
                    p = p.strip(" ")
                    if p not in self.vocab:
                        if p not in _bad_words:
                            self.vocab[p] = i
                            i += 1
            print(f'total after filtering: {i}')
            
            os.makedirs(self.identity_dir, exist_ok=True)
            with open(str(vdir), 'w') as f:
                json.dump(self.vocab, f)
    
    def _generate_aligned_feats(self):
        new_feats = {}
        id_targets = {}
        reg_targets = {}
        tdict = {}
        for story in self.fnames:
            print(f'Extracting {story}')
            stime = time.time()
            feat = self.features[story]['features']
            times = self.features[story]['times']

            pi = self.identity[story]['original_data']
            pf = self.identity[story]['feature_data']
            pt = self.identity[story]['times']

            id_target = []
            reg_target = []
            pooled_feats = []
            tms = []
            for i in range(pt.shape[0]):
                if i % 1000 == 0:
                    print(f"{i}/{pt.shape[0]} completed.")
                p = pi[i].strip(" ")
                if p not in self.vocab:
                    continue
                #target.append(self.vocab[pi[i]])
                start_t = pt[i,0]
                end_t = pt[i, 1]
                pool = []
                for j in range(times.shape[0]):
                    max1 = times[j,1]
                    min1 = max1 - (25/1000)
                    #option 2 = min = times[j,0]
                    if np.max([start_t, min1]) <= np.min([end_t, max1]):
                        pool.append(feat[j,:])

                if pool != []:
                    id_target.append(self.vocab[p])
                    reg_target.append(pf[i])
                    pooled_feats.append(np.mean(np.array(pool), axis=0))
                    tms.append(np.array([start_t, end_t]))

            p1 = np.row_stack(pooled_feats)
            t1 = np.array(id_target)
            at1 = np.row_stack(reg_target)
            tm1 = np.row_stack(tms)

            assert p1.shape[0] == t1.shape[0] and t1.shape[0] == at1.shape[0] and at1.shape[0] == tm1.shape[0], f"Mismatch size error in {story}"


            new_feats[story] = p1
            id_targets[story] = t1
            reg_targets[story] = at1
            tdict[story] = tm1
            e_time = time.time()
            tm = (e_time-stime)/60
            print(f"{story} took {tm} seconds to complete.")

            if self.cci_features is not None:
                self.cci_features.upload_raw_array(str(self.align_dir/story), p1)
                self.cci_features.upload_raw_array(str(self.align_dir/story) + '_idtargets', t1)
                self.cci_features.upload_raw_array(str(self.align_dir/story) + '_regtargets', at1)
                self.cci_features.upload_raw_array(str(self.align_dir/story) + '_times', tm1)

            else:
                os.makedirs(self.align_dir, exist_ok=True)
                np.savez_compressed(str(self.align_dir /story)+'.npz', p1)
                np.savez_compressed(str(self.align_dir /story)+'_idtargets.npz', t1)
                np.savez_compressed(str(self.align_dir /story)+'_regtargets.npz', at1)
                np.savez_compressed(str(self.align_dir /story)+'_times.npz', tm1)

        self.aligned_feats =  {'features':new_feats, 'identity_targets': id_targets, 'reg_targets':reg_targets, 'times':tdict}
    
    def get_aligned_feats(self):
        return self.aligned_feats
