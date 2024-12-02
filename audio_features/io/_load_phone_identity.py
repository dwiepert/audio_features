"""
Load phoneme features

Author(s): Aditya Vaidya, Daniela Wiepert, Others unknown
Last modified: 11/28/2024
"""
#IMPORTS
##built-in
import os
import json
import time
from typing import List
from pathlib import Path
##third-party
import numpy as np
##local
from database_utils.functions import *
from audio_features.io import load_features

_articulates = ['bilabial','postalveolar','alveolar','dental','labiodental',
			   'velar','glottal','palatal', 'plosive','affricative','fricative',
			   'nasal','lateral','approximant','voiced','unvoiced','low', 'mid',
			   'high','front','central','back']

_bad_words = ['{IG}', '{CG}','{NS}', 'SP\x7f', 'BR ', 'BR', 'LS', 'NS', 'SP', 'BRLS', 'AH0V', '']

class phoneIdentity:
    """
    Phoneme Identity features

    :param fnames: List of stimulus names
    """
    def __init__(self, fnames:List[str], phone_dir:str, cci_features=None, recursive:bool=False, overwrite:bool=False):
        self.fnames = fnames
        self.phone_dir =Path(phone_dir)
        self.cci_features = cci_features
        self.recursive=recursive
        self.phone_identity = {}
        self.overwrite = overwrite
        
        if self.cci_features is not None:
            self._load_from_bucket()
        elif self.phone_dir.exists():
            self._load(self.fnames)
        else:
            self._full_load(self.fnames)

        self._phone_to_ind()
    
    def _load_from_bucket(self):
        ###TODO CHECK IF CCI FEATURES PATH EXISTS
            new_fnames = []
            to_load = []
            for story in self.fnames:
                #e = []
                if self.cci_features.exists_object(str(self.phone_dir/story)+'_phones') and self.cci_features.exists_object(str(self.phone_dir/story)) and self.cci_features.exists_object(str(self.phone_dir/story)+'_times'):
                    if not self.overwrite:
                        to_load.append(story)
                    else:
                        new_fnames.append(story)
                else:
                    new_fnames.append(story)
            
            if to_load != []:
                self._load(to_load)

            if new_fnames != []:
                self._full_load(new_fnames)

    def _full_load(self, fnames):
        self.artdict = cc.get_interface('subcortical', verbose=False).download_json('artdict')
        self.phonseqs = get_story_phonseqs(fnames) #(phonemes, phoneme_times, tr_times)

        for story in fnames:
            olddata = [ph.upper().strip("0123456789") for ph in self.phonseqs[story].data]
            olddata = np.array(
                [ph.strip(' ') for ph in olddata])
            ph_2_art = self._ph_to_articulate(ds=olddata, ph_2_art=self.artdict)
            arthistseq = self._histogram_articulates(ds=ph_2_art, data=self.phonseqs[story])
            self.phone_identity[story] = {'original_data': olddata, 'feature_data': arthistseq[0], 'times': arthistseq[2]}

            if self.cci_features is not None:
                self.cci_features.upload_raw_array(str(self.phone_dir/story) + '_phones', olddata)
                self.cci_features.upload_raw_array(str(self.phone_dir/story), arthistseq[0].astype(int))
                self.cci_features.upload_raw_array(str(self.phone_dir/story) + '_times', arthistseq[2])
            else:
                os.makedirs(self.phone_dir, exist_ok=True)
                np.savez_compressed(str(self.phone_dir /story)+'_phones.npz', olddata)
                np.savez_compressed(str(self.phone_dir /story)+'.npz', arthistseq[0].astype(int))
                np.savez_compressed(str(self.phone_dir /story)+'_times.npz',arthistseq[2])
            
            #self.downsampled_arthistseqs[story] = {'original': arthistseq[0], 'original_times': arthistseq[2], 'downsampled':lanczosinterp2D(
			#arthistseq[0], arthistseq[2], arthistseq[3])}

    def _load(self, fnames):
        olddata = load_features(self.phone_dir, 'phone', self.cci_features, self.recursive, search_str='_phones')
        newdata = load_features(self.phone_dir, 'phone', self.cci_features, self.recursive, ignore_str=['_phones', '_times'])
        times = load_features(self.phone_dir, 'phone', self.cci_features, self.recursive, search_str='_times')
        for story in fnames:
            #print('REMOVE LATER')
            #if story in olddata and story in newdata and story in times:
            self.phone_identity[story] = {'original_data':olddata[story], 'feature_data':newdata[story], 'times': times[story]}
    
    def _phone_to_ind(self):
        vdir = self.phone_dir / 'vocab.json'
        if vdir.exists():
            with open(str(vdir), 'r') as f:
                self.vocab = json.load(f)
        else:
            self.vocab = {}
            i = 0
            for s in self.fnames:
                # print('remove later')
                # if s in self.phone_identity:
                pi = self.phone_identity[s]['original_data']
                for p in pi:
                    #print(p)
                    p = p.strip(" ")
                    if p not in self.vocab:
                        if p not in _bad_words:
                            self.vocab[p] = i
                            i += 1
            print(f'total after filtering: {i}')

            # for p in self.vocab:
            #     temp = self.vocab[p]
            #     one_hot = np.zeros((i))
            #     one_hot[temp] = 1
            #     self.vocab[p] = one_hot
            
            os.makedirs(self.phone_dir, exist_ok=True)
            with open(str(vdir), 'w') as f:
                json.dump(self.vocab, f)
    

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
    
    def _load_aligned(self, save_dir):
        p1 = load_features(save_dir, 'phone', self.cci_features, self.recursive, search_str='_downsampled1')
        t1 = load_features(save_dir, 'phone',  self.cci_features, self.recursive, search_str='_targets1')
        at1 = load_features(save_dir, 'phone',  self.cci_features, self.recursive, search_str='_arttargets1')
        tm1 = load_features(save_dir, 'phone',  self.cci_features, self.recursive, search_str='_times1')
        p2 = load_features(save_dir, 'phone',  self.cci_features, self.recursive, search_str='_downsampled2')
        t2 = load_features(save_dir, 'phone',  self.cci_features, self.recursive, search_str='_targets2')
        at2 = load_features(save_dir, 'phone',  self.cci_features, self.recursive, search_str='_arttargets2')
        tm2 = load_features(save_dir, 'phone',  self.cci_features, self.recursive, search_str='_times2')

        out1 = {'features':p1, 'identity_targets': t1, 'reg_targets':at1, 'times':tm1}
        out2 = {'features':p2, 'identity_targets': t2, 'reg_targets':at2, 'times':tm2}

        return out1, out2


    def align_features(self, features, save_dir):
        save_dir = Path(save_dir)

        if not self.overwrite and save_dir.exists():
            out1, out2 = self._load_aligned(save_dir)

            skip = True
            for o in out1:
                if not bool(o):
                    skip = False
            for o in out2:
                if not bool(o):
                    skip = False

            if skip:
                print('Skipping')
                return out1, out2
            
        new_feats1 = {}
        new_feats2 = {}
        targets1 = {}
        targets2 = {}
        art_targets1 = {}
        art_targets2 = {}
        tdict1 = {}
        tdict2 = {}
        for story in list(features.keys()):
            print(f'Extracting {story}')
            stime = time.time()
            feat = features[story]['features']
            times = features[story]['times']

            pi = self.phone_identity[story]['original_data']
            pf = self.phone_identity[story]['feature_data']
            pt = self.phone_identity[story]['times']

            target1 = []
            art_target1 = []
            target2 = []
            art_target2 = []
            pooled_features1 = []
            pooled_features2 = []
            times1 = []
            times2 = []
            for i in range(pt.shape[0]):
                if i % 1000 == 0:
                    print(f"{i}/{pt.shape[0]} completed.")
                p = pi[i].strip(" ")
                if p not in self.vocab:
                    continue
                #target.append(self.vocab[pi[i]])
                start_t = pt[i,0]
                end_t = pt[i, 1]
                pool1 = []
                pool2 = []
                for j in range(times.shape[0]):
                    max1 = times[j,1]
                    min1 = max1 - (25/1000)
                    min2 = times[j,0]
                    #option 2 = min = times[j,0]
                    if np.max([start_t, min1]) <= np.min([end_t, max1]):
                        pool1.append(feat[j,:])
                    if np.max([start_t, min2]) <= np.min([end_t, max1]):
                        pool2.append(feat[j,:])
                if pool1 != []:
                    target1.append(self.vocab[p])
                    art_target1.append(pf[i])
                    pooled_features1.append(np.mean(np.array(pool1), axis=0))
                    times1.append(np.array([start_t, end_t]))
                if pool2 != []:
                    art_target2.append(pf[i])
                    target2.append(self.vocab[p])
                    pooled_features2.append(np.mean(np.array(pool2), axis=0))
                    times2.append(np.array([start_t, end_t]))

            p1 = np.row_stack(pooled_features1)
            t1 = np.row_stack(target1)
            at1 = np.row_stack(art_target1)
            tm1 = np.row_stack(times1)
            p2 = np.row_stack(pooled_features2)
            t2 = np.row_stack(target2)
            at2 = np.row_stack(art_target2)
            tm2 = np.row_stack(times2)
            

            new_feats1[story] = p1
            targets1[story] = t1
            art_targets1[story] = at1
            tdict1[story] = tm1
            new_feats2[story] = p2
            targets2[story] = t2
            art_targets2[story] = at2
            tdict2[story] = tm2
            e_time = time.time()
            tm = (e_time-stime)/60
            print(f"{story} took {tm} seconds to complete.")

            if self.cci_features is not None:
                self.cci_features.upload_raw_array(str(save_dir/story) + '_downsampled1', p1)
                self.cci_features.upload_raw_array(str(save_dir/story) + '_targets1', t1)
                self.cci_features.upload_raw_array(str(save_dir/story) + '_arttargets1', at1)
                self.cci_features.upload_raw_array(str(save_dir/story) + '_times1', tm1)
                self.cci_features.upload_raw_array(str(save_dir/story) + '_downsampled2', p2)
                self.cci_features.upload_raw_array(str(save_dir/story) + '_targets2', t2)
                self.cci_features.upload_raw_array(str(save_dir/story) + '_arttargets2', at2)
                self.cci_features.upload_raw_array(str(save_dir/story) + '_times2', tm2)

            else:
                os.makedirs(save_dir, exist_ok=True)
                np.savez_compressed(str(save_dir /story)+'_downsampled1.npz', p1)
                np.savez_compressed(str(save_dir /story)+'_targets1.npz', t1)
                np.savez_compressed(str(save_dir /story)+'_arttargets1.npz', at1)
                np.savez_compressed(str(save_dir /story)+'_times1.npz', tm1)
                np.savez_compressed(str(save_dir /story)+'_downsampled2.npz', p2)
                np.savez_compressed(str(save_dir /story)+'_targets2.npz', t2)
                np.savez_compressed(str(save_dir /story)+'_arttargets2.npz', at2)
                np.savez_compressed(str(save_dir /story)+'_times2.npz', tm2)
            
        return {'features':new_feats1, 'identity_targets': targets1, 'reg_targets':art_targets1, 'times':tdict1}, {'features':new_feats2, 'identity_targets': targets2, 'reg_targets':art_targets2, 'times':tdict2}
