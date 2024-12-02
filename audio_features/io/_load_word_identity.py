"""
Load word identity features

Author(s): Alexander Huth, Aditya Vaidya, Daniela Wiepert, others Unknown
Last modified: 11/28/2024
"""
#IMPORTS
##built-in
import os
import time
from typing import List
from pathlib import Path
##third-party
import numpy as np
##local
from audio_features.models import SemanticModel
from audio_features.io import load_features, split_features
from database_utils.functions import get_story_wordseqs, make_semantic_model_v1, lanczosinterp2D

_bad_words = ["", " "]
class wordIdentity:
    """
    Word Identity features

    :param fnames: list,List of stimulus names
    :param pretrained_path: str, path to pretrained SemanticModel
    """
    def __init__(self, fnames:List[str], dir:str, cci_features=None,pretrained_path:str='./audio_features/data/english1000sm.hf5', recursive:bool=False, overwrite:bool=False):
        self.pretrained_path = pretrained_path
        self.fnames = fnames
        self.dir = Path(dir)
        self.cci_features = cci_features
        self.recursive=recursive
        self.overwrite=overwrite
        self.word_identity = {}
        
        if self.cci_features is not None:
            self._load_from_bucket()
        elif self.dir.exists():
            self._load(self.fnames)
        else:
            self._full_load(self.fnames)
        #if self.cci_features is not None or not self.dir.exists():
        self._word_to_onehot()

    def _load_from_bucket(self):
        new_fnames = []
        to_load = []
        for story in self.fnames:
            #e = []
            if self.cci_features.exists_object(str(self.dir/story)+'_words') and self.cci_features.exists_object(str(self.dir/story)) and self.cci_features.exists_object(str(self.dir/story)+'_times'):
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
        self.model = SemanticModel.load(self.pretrained_path)
        self.wordseqs = get_story_wordseqs(fnames)


        for story in fnames:
            sm = make_semantic_model_v1(self.wordseqs[story], self.model)
            olddata = np.array(self.wordseqs[story].data)
            newdata = sm.data
            times =  self.wordseqs[story].data_times
            self.word_identity[story] = {'original_data': olddata, 'feature_data': newdata, 'times':times}

            if self.cci_features is not None:
                self.cci_features.upload_raw_array(str(self.dir/story) + '_words', )
                self.cci_features.upload_raw_array(str(self.dir/story), newdata)
                self.cci_features.upload_raw_array(str(self.dir/story) + '_times', times)
            else:
                os.makedirs(self.dir, exist_ok=True)
                np.savez_compressed(str(self.dir /story)+'_words.npz', olddata)
                np.savez_compressed(str(self.dir /story)+'.npz', newdata)
                np.savez_compressed(str(self.dir /story)+'_times.npz',times)
            
    
    def _load(self, fnames):
        olddata = load_features(self.dir, self.cci_features, self.recursive, search_str='_words')
        newdata = load_features(self.dir, self.cci_features, self.recursive, ignore_str=['_words', '_times'])
        times = load_features(self.dir, self.cci_features, self.recursive, search_str='_times')
        olddata = split_features(olddata)
        newdata = split_features(newdata)
        times = split_features(times)
        for story in fnames:
            self.word_identity[story] = {'original_data':olddata[story], 'feature_data':newdata[story], 'times': times[story]}

    def _word_to_onehot(self):
        self.vocab = {}
        i = 0
        for s in self.fnames:
            pi = self.word_identity[s]['original_data']
            for p in pi:
                #print(p)
                p = p.strip(" ")
                if p not in self.vocab and p not in _bad_words:
                    self.vocab[p] = i
                    i += 1
        for p in self.vocab:
            temp = self.vocab[p]
            one_hot = np.zeros((len(self.vocab)))
            one_hot[temp] = 1
            self.vocab[p] = one_hot

    def _load_aligned(self):
        p1 = load_features(self.dir, self.cci_features, self.recursive, search_str='_downsampled1')
        t1 = load_features(self.dir, self.cci_features, self.recursive, search_str='_targets1')
        at1 = load_features(self.dir, self.cci_features, self.recursive, search_str='_embedtargets1')
        p2 = load_features(self.dir, self.cci_features, self.recursive, search_str='_downsampled2')
        t2 = load_features(self.dir, self.cci_features, self.recursive, search_str='_targets2')
        at2 = load_features(self.dir, self.cci_features, self.recursive, search_str='_embedtargets2')

        p1 = split_features(p1)
        t1 = split_features(t1)
        at1 = split_features(at1)
        p2 = split_features(p2)
        t2 = split_features(t2)
        at2 = split_features(at2)

        out1 = {'features':p1, 'identity_targets': t1, 'embedding_targets':at1}
        out2 = {'features':p2, 'identity_targets': t2, 'embedding_targets':at2}

        return out1, out2

    def align_features(self, features):
        self.dir = self.dir / 'aligned_features'

        out1, out2 = self._load_aligned()

        skip = True
        for o in out1:
            if bool(o):
                skip = False
        for o in out2:
            if bool(o):
                skip = False

        if skip:
            return out1, out2
            
        new_feats1 = {}
        new_feats2 = {}
        targets1 = {}
        targets2 = {}
        art_targets1 = {}
        art_targets2 = {}
        for story in list(features.keys()):


            print(f'Extracting {story}')
            stime = time.time()
            feat = features[story]['features']
            times = features[story]['times']

            pi = self.word_identity[story]['original_data']
            pf = self.word_identity[story]['feature_data']
            pt = self.word_identity[story]['times']

            target1 = []
            art_target1 = []
            target2 = []
            art_target2 = []
            pooled_features1 = []
            pooled_features2 = []
            for i in range(pt.shape[0]):
                if i % 100 == 0:
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
                    max = times[j,1]
                    min1 = max - (25/1000)
                    min2 = times[j,0]
                    #option 2 = min = times[j,0]
                    if np.max([start_t, min1]) <= np.min([end_t, max]):
                        pool1.append(feat[j,:])
                    if np.max([start_t, min2]) <= np.min([end_t, max]):
                        pool2.append(feat[j,:])
                if pool1 != []:
                    target1.append(self.vocab[p])
                    art_target1.append(pf[i])
                    pooled_features1.append(np.mean(np.array(pool1), axis=0))
                if pool2 != []:
                    art_target2.append(pf[i])
                    target2.append(self.vocab[p])
                    pooled_features2.append(np.mean(np.array(pool2), axis=0))

            p1 = np.row_stack(pooled_features1)
            t1 = np.row_stack(target1)
            at1 = np.row_stack(art_target1)
            p2 = np.row_stack(pooled_features2)
            t2 = np.row_stack(target2)
            at2 = np.row_stack(art_target2)

            new_feats1[story] = p1
            targets1[story] = t1
            art_targets1[story] = at1
            new_feats2[story] = p2
            targets2[story] = t2
            art_targets2[story] = at2
            e_time = time.time()
            tm = (e_time-stime)/60
            print(f"{story} took {tm} seconds to complete.")

            if self.cci_features is not None:
                self.cci_features.upload_raw_array(str(self.dir/story) + '_downsampled1', p1)
                self.cci_features.upload_raw_array(str(self.dir/story) + '_targets1', t1)
                self.cci_features.upload_raw_array(str(self.dir/story) + '_embedtargets1', at1)
                self.cci_features.upload_raw_array(str(self.dir/story) + '_downsampled2', p2)
                self.cci_features.upload_raw_array(str(self.dir/story) + '_targets2', t2)
                self.cci_features.upload_raw_array(str(self.dir/story) + '_embedtargets2', at2)

            else:
                os.makedirs(self.dir, exist_ok=True)
                np.savez_compressed(str(self.dir /story)+'_downsampled1.npz', p1)
                np.savez_compressed(str(self.dir /story)+'_targets1.npz', t1)
                np.savez_compressed(str(self.dir /story)+'_embedtargets1.npz', at1)
                np.savez_compressed(str(self.dir /story)+'_downsampled2.npz', p2)
                np.savez_compressed(str(self.dir /story)+'_targets2.npz', t2)
                np.savez_compressed(str(self.dir /story)+'_embedtargets2.npz', at2)
            
        return {'features':new_feats1, 'identity_targets': targets1, 'embedding_targets':art_targets1}, {'features':new_feats2, 'identity_targets': targets2, 'embedding_targets':art_targets2}
