"""
Perform functions that take in two pairs of features and run functions on them. 

Author(s): Daniela Wiepert, Lasya Yakkala, Rachel Yamamoto
Last modified: 11/27/2024
"""
#%%
#IMPORTS
#built-in
import argparse
import collections
import os
from pathlib import Path

#third-party
import cottoncandy as cc
import numpy as np
from tqdm import tqdm

#local
from audio_features.io import DatasetSplitter, load_features, split_features, phoneIdentity, wordIdentity, downsample_features
from audio_features.functions import LSTSQRegression, RRegression, LinearClassification
from audio_preprocessing.io import select_stimuli


def process_ema(ema_feats:dict):
    """
    """
    mask = np.ones(14, dtype=bool)
    mask[[12]] = False
    for f in ema_feats:
        temp = ema_feats[f]
        temp = temp[:,mask]
        ema_feats[f] = temp

    return ema_feats

def align_times(feats, times):
    features = {}
    for s in list(feats.keys()):
        f = feats[s]
        t = times[s]
        sort_i = np.argsort(t, axis=0)[:,0]
        f = f[sort_i,:]
        t = t[sort_i,:]
        features[s] = {'features': f, 'times': t}
    return features

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #Common arguments
    parser.add_argument('--feat_dir1', type=str, required=True,
                        help='Specify the path to the first set of features (Always assumes this will be the features to use for the first feature argument in a function)')
    parser.add_argument('--feat1_type', type=str, required=True,
                        help='Specify the type of feature you are using for first argument')
    parser.add_argument('--feat1_times', type=str, default=None)
    parser.add_argument('--feat_dir2', type=str,
                        help='Specify the path to the second set of features (Always assumes this will be the features to use for the second feature argument in a function)')
    parser.add_argument('--feat2_type', type=str,
                        help='Specify the type of feature you are using for second argument')
    parser.add_argument('--feat2_times', type=str, default=None)
    parser.add_argument('--out_dir', type=str, required=True,
                        help="Specify a local directory to save configuration files to. If not saving features to corral, this also specifies local directory to save files to.")
    parser.add_argument("--recursive", action="store_true", help='Recursively find .wav,.flac,.npz files in the feature and stimulus dirs')
    parser.add_argument("--function", required=True,type=str, help="specify what function to use as a string")
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing features (default behavior is to skip)')
    #cotton candy related args
    cc_args = parser.add_argument_group('cc', 'cottoncandy related arguments (loading/saving to corral)')
    cc_args.add_argument('--stimulus_dir', type=str, required=True,
                        help='Specify a local directory with the audio stimulus you want to extract features from. Audio should be already processed.')
    cc_args.add_argument('--stim_bucket', type=str, default=None, 
                         help="bucket to select stimulus from. If set, it will override stimulus_dir. Would use stimulidb")
    cc_args.add_argument('--out_bucket', type=str, default=None,
                         help="Bucket to save extracted features to. If blank, save to local filesystem.")
    cc_args.add_argument('--sessions', nargs='+', type=str,
                                   help="Only process stories presented in these sessions."
                                   "Can be used in conjuction with --subject to take an intersection.")
    cc_args.add_argument('--stories', '--stimuli', nargs='+', type=str,
                                   help="Only process the given stories."
                                   "Overrides --sessions and --subjects.")
    #Model specific
    model_args = parser.add_argument_group('model', 'model related arguments')
    model_args.add_argument("--zscore", action="store_true",
                            help="specify if you want to zscore before running regression.")
    model_args.add_argument("--scoring", type=str)
    model_args.add_argument("--metric_type", type=str)
    model_args.add_argument("--identity_type", type=str, help='Specify whether classifying phones or word sequences')
    data_args = parser.add_argument_group('data', 'data splitting related args')
    data_args.add_argument("--split", action='store_true')
    data_args.add_argument("--split_path", type=str)
    data_args.add_argument("--n_splits", type=int, default=5)
    data_args.add_argument("--train_ratio", type=float, default=.8)
    data_args.add_argument("--val_ratio",type=float, default=0.)

    args = parser.parse_args()
    
    args.out_dir = Path(args.out_dir)

    if args.function != 'clf':
        assert args.feat_dir2 is not None, 'Must give two features if not doing classification.'

    if (args.out_bucket is not None) and (args.out_bucket != ''):
        cci_features = cc.get_interface(args.out_bucket, verbose=False)
        print("Loading features from bucket", cci_features.bucket_name)
        raise NotImplementedError("Can't load from bucket just yet")
    else:
        cci_features = None
        print('Loading features from local filesystem.')

    stimulus_paths = select_stimuli(stim_dir=args.stimulus_dir, stim_bucket=args.stim_bucket, sessions=args.sessions, stories=args.stories, recursive=args.recursive)
    assert len(stimulus_paths) > 0, "no stimuli to process!"

    stimulus_paths = collections.OrderedDict(sorted(stimulus_paths.items(), key=lambda x: x[0]))
    stimulus_names = list(stimulus_paths.keys())
    
    #print(cci_features)
    feats1 = load_features(args.feat_dir1, cci_features, args.recursive, ignore_str='times')
    if args.feat1_times is None:
        args.feat1_times = args.feat_dir1
    feat1_times = load_features(args.feat1_times, cci_features, args.recursive, search_str='times')
    #print(feats1)
    feats1 = split_features(feats1)
    #print(feats1)
    feat1_times = split_features(feat1_times)
    if args.feat1_type == 'ema':
        feats1 = process_ema(feats1)
    aligned_feats1 = align_times(feats1, feat1_times)

    if args.function != 'clf':
        feats2 = load_features(args.feat_dir2, cci_features, args.recursive, ignore_str='times')
        if args.feat2_times is None:
            args.feat2_times = args.feat_dir2
        feat2_times = load_features(args.feat2_times, cci_features, args.recursive, search_str='times')
        feats2 = split_features(feats2)
        feat2_times = split_features(feat2_times)
        if args.feat2_type == 'ema':
            feats2 = process_ema(feats2)
        aligned_feats2 = align_times(feats2, feat2_times)
    
    ################# DEBUG #############################################
    
    save_path = Path(f'{args.function}_{args.feat1_type}_to_{args.feat2_type}')
    if args.zscore:
            save_path = save_path / 'zscore' 
    if cci_features is None:
        save_path = args.out_dir / save_path
        local_path = None
    else:
        local_path = args.out_dir / save_path
    print('Saving results to:', save_path)

    if args.split:
        if args.split_path is not None: 
            args.split_path = Path(args.split_path)
        else:
            args.split_path = save_path /'splits'

        splitter = DatasetSplitter(stories=stimulus_names, output_dir=args.split_path, num_splits=args.n_splits, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
        splits = splitter.load_splits()
        
    else:
        splits = [None]    
    #print('localpath', local_path) # DEBUG
    ###########################################################################
    
    for i in range(len(splits)):
        s = splits[i]
        if s is None:
            train_feats1 = aligned_feats1
            train_feats2 = aligned_feats2
            test_feats1 = aligned_feats1
            test_feats2 = aligned_feats2
        else:
            new_path = save_path/f'split{i}'
            train_feats1, val_feats1, test_feats1 = splitter.split_features(aligned_feats1, s)
            train_feats2, val_feats2, test_feats2 = splitter.split_features(aligned_feats2, s)

        if args.function == 'lstsq':          
            print('Saving regression results to:', save_path)
            print('localpath', local_path) # DEBUG
            regressor = LSTSQRegression(iv=train_feats1, iv_type=args.feat1_type, dv=train_feats2, dv_type=args.feat2_type,
                                        save_path=new_path, zscore=args.zscore, cci_features=cci_features, overwrite=args.overwrite,
                                        local_path=local_path)
            

            for k in tqdm(list(test_feats1.keys())):
                regressor.extract_residuals(test_feats1[k], test_feats2[k], s)
        
        # define what to do for ridge regression here
        elif args.function == 'ridge':
            print('Ridge Regression')
        
            regressor = RRegression(iv=train_feats1, 
                                    iv_type=args.feat1_type,
                                    dv=train_feats2,
                                    dv_type=args.feat2_type,
                                    zscore=args.zscore,
                                    scoring=args.scoring,
                                    cci_features=cci_features,
                                    overwrite=args.overwrite,
                                    local_path=local_path,
                                    save_path = new_path)
            
            for k in tqdm(list(test_feats1.keys())):
                regressor.calculate_correlations(test_feats1[k], test_feats2[k], k)
    
        elif args.function == 'clf':
            assert args.identity_type in ['word', 'phone']
            print(f'{args.identity_type} Classification')
            
            if args.identity_type == 'phone':
                identity = phoneIdentity(stimulus_names)
            else:
                identity = wordIdentity(fnames=stimulus_names, pretrained_path='./audio_features/data/english1000sm.hf5')

            seqs = identity.get_seqs()
            v = identity.get_features()

            raise NotImplementedError('Classifier not implemented completely. Need to sort out train/val/test split information')
            downsampled_feats = downsample_features(aligned_feats1, seqs)

            classifier = LinearClassification(iv=downsampled_feats,
                                            iv_type=args.feat1_type,
                                            dv=v,
                                            dv_type=args.identity_type,
                                            metric_type=args.metric_type,
                                            save_path=new_path,
                                            zscore=args.zscore,
                                            scoring=args.scoring,
                                            cci_features=cci_features,
                                            overwrite=args.overwrite,
                                            local_path=local_path)

            for s in tqdm(stimulus_names):
                classifier.score(downsampled_feats[s], v[s], s)

        #print('pause')
      


    else:
        raise NotImplementedError(f'{args.function} is not implemented')



    


    

