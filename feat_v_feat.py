"""
Perform functions that take in two pairs of features and run functions on them. 

Author(s): Daniela Wiepert, Lasya Yakkala, Rachel Yamamoto
Last modified: 11/09/2024
"""
#IMPORTS
#built-in
import argparse
import collections
from pathlib import Path
from typing import Union

#third-party
import cottoncandy as cc
import numpy as np

#local
from audio_features.io import load_features
from audio_features.functions import bootstrap_ridge
from audio_preprocessing.io import select_stimuli

def split_features(features:dict, stimulus_names:list, parent_dir:Union[str,Path]):
    new_features = {}
    feature_paths = features['path_list']

    for s in stimulus_names:
        s_dict = {}
        module_dict = {}
        for f in feature_paths:
            f = f
            if s in str(f):
                if 'times' in str(f):
                    s_dict['times'] = features[f]
                elif str(f.parent) != str(parent_dir):
                    module = f.parent.name
                    module_dict[module] = features[f]
                else:
                    s_dict['out_features'] = features[f]
        s_dict['module_features'] = module_dict
        new_features[s] = s_dict
    
    return new_features

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #Common arguments
    parser.add_argument('--reference_dir', type=str, required=True,
                        help='')
    parser.add_argument('--module', type=str, default="layer.9")
    parser.add_argument('--predict_dir', type=str, required=True,
                        help='')
    parser.add_argument("--recursive", action="store_true", help='Recursively find .wav,.flac,.npz files in the feature and stimulus dirs')
    parser.add_argument("--function", require=True, help="specify what function to use.")
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
    cc_args.add_argument('--recursive', action='store_true',
                                   help='Recursively find .wav and .flac in the stimulus_dir.')
    # TODO: str --> pathlib.Path
    cc_args.add_argument('--overwrite', action='store_true',
                                   help='Overwrite existing features (default behavior is to skip)')
    args = parser.parse_args()
    
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
    
    #TODO: how to load from bucket?
    if args.recursive:
        stim_feats = sorted(list(args.reference_dir.rglob('*.npz')))
        resp_feats = sorted(list(args.predict_dir.rglob('*.npz')))
    else:
        stim_feats = sorted(list(args.reference_dir.glob('*.npz')))
        resp_feats = sorted(list(args.predict_dir.glob('*.npz')))

    args.reference_dir = Path(args.reference_dir)
    args.predict_dir = Path(args.predict_dir)

    stim_feats = load_features(stim_feats, cci_features)
    stim_feats = split_features(stim_feats, stimulus_names, args.reference_dir)
    resp_feats = load_features(resp_feats, cci_features)
    resp_feats = split_features(resp_feats, stimulus_names, args.predict_dir)

    for s in stimulus_names:
        Rstim = stim_feats[s]
        Rresp = resp_feats[s]

        if args.function == "ridge":
            alphas = None
            nboots=None
            chunklen=None
            nchunks=None
            singcutoff=None
            single_alpha=None
            use_corr=None
            wt, r = bootstrap_ridge(Rstim=Rstim, Rresp=Rresp, alphas=alphas, nboots=nboots,
                                    chunklen=chunklen, nchunks=nchunks, singcutoff=singcutoff, single_alpha=single_alpha,
                                    use_corr=use_corr, solver_dtype=np.float32)
        else:
            raise NotImplementedError(f"{args.function} not implemented.")
        
    




    


    

