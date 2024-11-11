"""
Perform functions that take in two pairs of features and run functions on them. 

Author(s): Daniela Wiepert, Lasya Yakkala, Rachel Yamamoto
Last modified: 11/09/2024
"""
import argparse
import os
from pathlib import Path
from audio_features.io import load_features

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #Common arguments
    parser.add_argument('--reference_dir', type=str, required=True,
                        help='')
    parser.add_argument('--module', type=str, default="layer.9")
    parser.add_argument('--predict_dir', type=str, required=True,
                        help='')
    parser.add_argument("--recursive", action="store_true", help="")
    args = parser.parse_args()

    
    #TODO: how to load from bucket?
    if args.recursive:
        reference_feats = sorted(list(args.reference_dir.rglob('*.npz')))
        predict_feats = sorted(list(args.predict_dir.rglob('*.npz')))
    else:
        reference_feats = sorted(list(args.reference_dir.glob('*.npz')))
        predict_feats = sorted(list(args.predict_dir.glob('*.npz')))

    args.reference_dir = Path(args.reference_dir)
    args.predict_dir = Path(args.predict_dir)
    ### separate into feature groups
    features = {'ref': reference_feats, 'pred':predict_feats}
    
    
    for k in features:
        flist = features[k]
        stim_names = set([Path(f).name for f in flist])

        for s in stim_names:
            paths = [f for f in flist if s in f]
            fpath = None
            tpath = None
            mpaths = {}
            for p in paths:
                if '.npz' not in str(f):
                    continue
                f = Path(os.path.splitext(f)[0])

                if f.parent != args.reference_dir and f.parent != args.predict_dir:
                    mpaths[f.parent.name] = f
                elif 'times' in f.name:
                    tpath = f
                else:
                    fpath = f

            sample = load_features(features_save_path=fpath,times_save_path=tpath, module_save_paths=mpaths)




    


    

