"""
Perform functions that take in two pairs of features and run functions on them. 

Author(s): Daniela Wiepert, Lasya Yakkala, Rachel Yamamoto
Last modified: 02/15/2024
"""
#%%
#IMPORTS
#built-in
import argparse
import collections
import glob
import json
import os
from pathlib import Path
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import cottoncandy as cc

#third-party
import cottoncandy as cc
import numpy as np
from tqdm import tqdm

#local
from audio_features.io import DatasetSplitter, load_features, align_times, Identity, copy_times
from audio_features.models import LSTSQRegression, RRegression, LinearClassification, residualPCA
from audio_preprocessing.io import select_stimuli
from audio_features.extractors import set_up_emaae_extractor

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
    model_args.add_argument("--cv_splits", type=int, default=5)
    model_args.add_argument("--n_boots", type=int, default=3)
    model_args.add_argument("--corr_type", type=str, default='feature', help="feature or time")
    model_args.add_argument('--model_checkpoint', type=str, default='.',
                        help="Specify a path to a saved model checkpoint if required. For EMAAE specifically. Will need local files")
    model_args.add_argument('--model_config_path', type=str, default='./audio_features/configs/hf_model_configs.json',
                        help="Specify a path to a json with model information if required. For hugging face, use: audio_features/configs/hf_model_configs.json")
    data_args = parser.add_argument_group('data', 'data splitting related args')
    data_args.add_argument("--split", action='store_true')
    data_args.add_argument("--split_path", type=str)
    data_args.add_argument("--n_splits", type=int, default=5)
    data_args.add_argument("--train_ratio", type=float, default=.8)
    
  
    args = parser.parse_args()
    
    args.out_dir = Path(args.out_dir)

    if 'clf' not in args.function or args.function not in ['pca','emaae']:
        assert args.feat_dir2 is not None, 'Must give two features if not doing classification.'

    if (args.out_bucket is not None) and (args.out_bucket != ''):
        cci_features = cc.get_interface(args.out_bucket, verbose=False)
        print("Loading features from bucket", cci_features.bucket_name)
        raise NotImplementedError("Can't load from bucket just yet")
    else:
        cci_features = None
        print('Loading features from local filesystem.')

    try: 
        stimulus_paths = select_stimuli(stim_dir=args.stimulus_dir, stim_bucket=args.stim_bucket, sessions=args.sessions, stories=args.stories, recursive=args.recursive)
        assert len(stimulus_paths) > 0, "no stimuli to process!"

        stimulus_paths = collections.OrderedDict(sorted(stimulus_paths.items(), key=lambda x: x[0]))
        stimulus_names = list(stimulus_paths.keys())
    except:
        stimulus_names=None

    
    ## LOAD IN FEATURES
    #print(cci_features)
    feats1 = load_features(args.feat_dir1, args.feat1_type, cci_features, args.recursive, ignore_str='times')
    if args.feat1_times is None:
        try:
            assert not any(char.isdigit() for char in args.feat1_type), 'Must give feat1_times dir if working with wavlm layers.'
        except:
            print('May not be times.npz files in your feat1 directory.')
        args.feat1_times = args.feat_dir1
    feat1_times = load_features(args.feat1_times, 'times', cci_features, args.recursive, search_str='times')

    if stimulus_names is None:
        stimulus_names = list(feats1.keys())

    aligned_feats1 = align_times(feats1, feat1_times)

    if args.function == 'pca' or args.function=='emaae':
        aligned_feats2 = aligned_feats1
    elif args.feat2_type not in ['word', 'phone'] and args.function != 'pca' and args.function != 'emaae':
        feats2 = load_features(args.feat_dir2, args.feat2_type, cci_features, args.recursive, ignore_str='times')
        feat2_times = load_features(args.feat_dir2,'times', cci_features, args.recursive, search_str='times')
        aligned_feats2 = align_times(feats2, feat2_times)
    else:
        if args.feat2_type == 'word':
            pretrained_path = './audio_features/data/english1000sm.hf5'
        else:
            pretrained_path = None
        
        save_dir = Path(args.feat_dir2) / f'aligned_{args.feat1_type}'
        print(save_dir)
        identity = Identity(identity_type=args.feat2_type, features=aligned_feats1, identity_dir=args.feat_dir2,
                            align_dir = save_dir, pretrained_path=pretrained_path, cci_features=cci_features,
                            recursive=args.recursive, overwrite=args.overwrite)
        
        temp = identity.get_aligned_feats()

        if args.function == 'multiclass_clf':
            aligned_feats1 = align_times(temp['features'], temp['times'])
            aligned_feats2 = align_times(temp['identity_targets'], temp['times'])
        else:
            aligned_feats1 = align_times(temp['features'], temp['times'])
            aligned_feats2 = align_times(temp['reg_targets'], temp['times'])
    
    for t in aligned_feats1:
        f = aligned_feats1[t]['features']
        t = aligned_feats2[t]['features']
        assert t.shape[0] == f.shape[0], f"Shape not aligned for {t}"

    ## SAVING
    save_path = Path(f'{args.function}_{args.feat1_type}_to_{args.feat2_type}')
    if cci_features is None:
        save_path = args.out_dir / save_path
        local_path = None
    else:
        local_path = args.out_dir / save_path
    print('Saving results to:', save_path)

    ## GENERATE SPLITS
    if args.split:
        if args.split_path is not None: 
            args.split_path = Path(args.split_path)
        else:
            args.split_path = save_path /'splits'

        splitter = DatasetSplitter(stories=stimulus_names, output_dir=args.split_path, num_splits=args.n_splits, train_ratio=args.train_ratio, val_ratio=0.)
        splits = splitter.load_splits()
        
    else:
        splits = [None]    
    #print('localpath', local_path) # DEBUG
    ###########################################################################
        
    for i in range(len(splits)):
        ## SPLIT FEATURES
        s = splits[i]
        if s is None:
            train_feats1 = aligned_feats1
            test_feats1 = aligned_feats1

            train_feats2 = aligned_feats2
            test_feats2 = aligned_feats2
            new_path = save_path
        else:
            new_path = save_path/f'split{i}'
            if cci_features:
                local_path = local_path/f'split{i}'
            train_feats1, val_feats1, test_feats1 = splitter.split_features(aligned_feats1, s)
            train_feats2, val_feats2, test_feats2 = splitter.split_features(aligned_feats2, s)

        print('Saving regression results to:', new_path)
        
        key_filter = list(train_feats1.keys()) + list(test_feats1.keys())

        if args.function == 'lstsq': 
            ## EXTRACT RESIDUALS
            print('LSTSQ Regression')
            model = LSTSQRegression(iv=train_feats1,
                                    iv_type=args.feat1_type,
                                    dv=train_feats2,
                                    dv_type=args.feat2_type,
                                    save_path=new_path, 
                                    cci_features=cci_features,
                                    overwrite=args.overwrite,
                                    local_path=local_path)
            copy_times(args.feat1_times, new_path, key_filter)
        elif args.function == 'ridge':
            print('Ridge Regression')

            model = RRegression(iv=train_feats1,
                                iv_type=args.feat1_type,
                                dv=train_feats2,
                                dv_type=args.feat2_type,
                                save_path=new_path,
                                n_splits=args.cv_splits,
                                n_repeats=args.n_boots,
                                corr_type=args.corr_type,
                                cci_features=cci_features,
                                overwrite=args.overwrite,
                                local_path=local_path)
            
    
        elif 'clf' in args.function:
            print(f'Classification: {args.function}')
            model = LinearClassification(iv=train_feats1,
                                         iv_type=args.feat1_type,
                                         dv=train_feats2,
                                         dv_type=args.feat2_type,
                                         save_path=new_path,
                                         classification_type=args.function,
                                         n_splits=args.cv_splits,
                                         n_repeats=args.n_boots,
                                         cci_features=cci_features,
                                         overwrite=args.overwrite,
                                         local_path=local_path)

        elif args.function == 'pca':
            print('PCA')
            model = residualPCA(iv=train_feats1,
                                iv_type=args.feat1_type,
                                save_path=new_path,
                                cci_features=cci_features,
                                overwrite=args.overwrite,
                                local_path=local_path
                                )
            
        elif args.function == "emaae":
            extractor = set_up_emaae_extractor(save_path=new_path, ckpt=args.model_checkpoint, config=args.model_config_path, return_numpy=args.return_numpy)
            for idx, story in enumerate(tqdm(train_feats1.keys())):
                sample = {'fname':story, 'ema':train_feats1[story]}
                new_sample = extractor(sample)     

            copy_times(args.feat1_times, new_path, key_filter)                              
            print('Extraction completed')

        elif args.function == 'extract':
            print('Extraction completed')
        else:
            raise NotImplementedError(f'{args.function} is not implemented')

        if args.function not in ['emaae', 'extract']:
            print(f'Scoring split {i}')
            true = None
            pred = None
            for k in tqdm(list(test_feats1.keys())):
                out, tpd = model.score(test_feats1[k], test_feats2[k], k)

                if true is None:
                    true = tpd['true']
                    pred = tpd['pred']
                else:
                    true = np.concatenate((true, tpd['true']), axis=0)
                    pred = np.concatenate((pred, tpd['pred']), axis=0)

            if args.function != 'pca':
                metrics = model.eval_model(true, pred)

                os.makedirs(model.result_paths['test_eval'].parent, exist_ok=True)
                with open(str(model.result_paths['test_eval'])+'.json', 'w') as f:
                    json.dump(metrics,f)           


