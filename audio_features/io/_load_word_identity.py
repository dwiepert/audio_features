"""
Load word identity features

Author(s): Alexander Huth, Aditya Vaidya, Daniela Wiepert, others Unknown
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
from audio_features.models import SemanticModel
from audio_features.io import load_features
from database_utils.functions import get_story_wordseqs, make_semantic_model_v1, lanczosinterp2D

_bad_words = ["", " "]
class wordIdentity:
    """
    Word Identity features

    :param fnames: list,List of stimulus names
    :param pretrained_path: str, path to pretrained SemanticModel
    """
    def __init__(self, fnames:List[str], word_dir:str, cci_features=None,pretrained_path:str='./audio_features/data/english1000sm.hf5', recursive:bool=False, overwrite:bool=False):
        self.pretrained_path = pretrained_path
        self.fnames = fnames
        self.word_dir = Path(word_dir)
        self.cci_features = cci_features
        self.recursive=recursive
        self.overwrite=overwrite
        self.word_identity = {}
        
        if self.cci_features is not None:
            self._load_from_bucket()
        elif self.word_dir.exists():
            self._load(self.fnames)
        else:
            self._full_load(self.fnames)
        #if self.cci_features is not None or not self.word_dir.exists():
        self._word_to_ind()

    def _load_from_bucket(self):
        new_fnames = []
        to_load = []
        for story in self.fnames:
            #e = []
            if self.cci_features.exists_object(str(self.word_dir/story)+'_words') and self.cci_features.exists_object(str(self.word_dir/story)) and self.cci_features.exists_object(str(self.word_dir/story)+'_times'):
                to_load.append(story)
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
                self.cci_features.upload_raw_array(str(self.word_dir/story) + '_words', )
                self.cci_features.upload_raw_array(str(self.word_dir/story), newdata)
                self.cci_features.upload_raw_array(str(self.word_dir/story) + '_times', times)
            else:
                os.makedirs(self.word_dir, exist_ok=True)
                np.savez_compressed(str(self.word_dir /story)+'_words.npz', olddata)
                np.savez_compressed(str(self.word_dir /story)+'.npz', newdata)
                np.savez_compressed(str(self.word_dir /story)+'_times.npz',times)
            
    
    def _load(self, fnames):
        olddata = load_features(self.word_dir, 'word', self.cci_features, self.recursive, search_str='_words')
        newdata = load_features(self.word_dir, 'word',self.cci_features, self.recursive, ignore_str=['_words', '_times'])
        times = load_features(self.word_dir,'word', self.cci_features, self.recursive, search_str='_times')
        for story in fnames:
            self.word_identity[story] = {'original_data':olddata[story], 'feature_data':newdata[story], 'times': times[story]}

    def _word_to_ind(self):
        vdir = self.word_dir / 'vocab.json'
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
                pi = self.word_identity[s]['original_data']
                for p in pi:
                    #print(p)
                    p = p.strip(" ")
                    if p not in self.vocab:
                        total += 1
                        if p not in _bad_words:
                            self.vocab[p] = i
                            i += 1
            print(f'total before filtering: {total}')
            print(f'total after filtering: {i}')

            # for p in self.vocab:
            #     temp = self.vocab[p]
            #     one_hot = np.zeros((i))
            #     one_hot[temp] = 1
            #     self.vocab[p] = one_hot
            
            os.makedirs(self.word_dir, exist_ok=True)
            with open(str(vdir), 'w') as f:
                json.dump(self.vocab, f)

    def _load_aligned(self, save_dir):
        p1 = load_features(save_dir, 'word',self.cci_features, self.recursive, search_str='_downsampled1')
        t1 = load_features(save_dir, 'word', self.cci_features, self.recursive, search_str='_targets1')
        at1 = load_features(save_dir,  'word',self.cci_features, self.recursive, search_str='_embedtargets1')
        tm1 = load_features(save_dir,  'word',self.cci_features, self.recursive, search_str='_times1')
        p2 = load_features(save_dir,  'word',self.cci_features, self.recursive, search_str='_downsampled2')
        t2 = load_features(save_dir, 'word', self.cci_features, self.recursive, search_str='_targets2')
        at2 = load_features(save_dir, 'word', self.cci_features, self.recursive, search_str='_embedtargets2')
        tm2 = load_features(save_dir,  'word',self.cci_features, self.recursive, search_str='_times2')

        out1 = {'features':p1, 'identity_targets': t1, 'reg_targets':at1, 'times':tm1}
        out2 = {'features':p2, 'identity_targets': t2, 'reg_targets':at2, 'times':tm2}

        return out1, out2

    def align_features(self, features, save_dir):
        save_dir = Path(save_dir)
        if not self.overwrite:
            out1, out2 = self._load_aligned(save_dir)

            skip = True
            for o in out1:
                if not bool(o):
                    skip = False
            for o in out2:
                if not bool(o):
                    skip = False

            if skip:
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

            pi = self.word_identity[story]['original_data']
            pf = self.word_identity[story]['feature_data']
            pt = self.word_identity[story]['times']

            target1 = []
            art_target1 = []
            target2 = []
            art_target2 = []
            pooled_features1 = []
            pooled_features2 = []
            times1 = []
            times2 = []
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
                self.cci_features.upload_raw_array(str(save_dir/story) + '_embedtargets1', at1)
                self.cci_features.upload_raw_array(str(save_dir/story) + '_times1', tm1)
                self.cci_features.upload_raw_array(str(save_dir/story) + '_downsampled2', p2)
                self.cci_features.upload_raw_array(str(save_dir/story) + '_targets2', t2)
                self.cci_features.upload_raw_array(str(save_dir/story) + '_embedtargets2', at2)
                self.cci_features.upload_raw_array(str(save_dir/story) + '_times2', tm2)

            else:
                os.makedirs(save_dir, exist_ok=True)
                np.savez_compressed(str(save_dir /story)+'_downsampled1.npz', p1)
                np.savez_compressed(str(save_dir /story)+'_targets1.npz', t1)
                np.savez_compressed(str(save_dir /story)+'_embedtargets1.npz', at1)
                np.savez_compressed(str(save_dir /story)+'_times1.npz', tm1)
                np.savez_compressed(str(save_dir /story)+'_downsampled2.npz', p2)
                np.savez_compressed(str(save_dir /story)+'_targets2.npz', t2)
                np.savez_compressed(str(save_dir /story)+'_embedtargets2.npz', at2)
                np.savez_compressed(str(save_dir /story)+'_times2.npz', tm2)
            
        return {'features':new_feats1, 'identity_targets': targets1, 'reg_targets':art_targets1, 'times':tdict1}, {'features':new_feats2, 'identity_targets': targets2, 'reg_targets':art_targets2, 'times':tdict2}
