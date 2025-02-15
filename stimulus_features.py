"""
Extract features from stimulus

Author(s): Daniela Wiepert, Lasya Yakkala, Rachel Yamamoto
Last modified: 11/14/2024
"""
#IMPORTS
##built-in
import argparse
import collections
from pathlib import Path
import warnings

##third-party
import cottoncandy as cc
from tqdm import tqdm
import torchaudio
import torch

##local
from audio_features.extractors import *
from audio_preprocessing.io import select_stimuli

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #Common arguments
    parser.add_argument('--stimulus_dir', type=str, required=True,
                        help='Specify a local directory with the audio stimulus you want to extract features from. Audio should be already processed.')
    parser.add_argument('--out_dir', type=str, required=True,
                        help="Specify a local directory to save configuration files to. If not saving features to corral, this also specifies local directory to save files to.")
    parser.add_argument('--feature_type', type=str, required=True,
                        help="Specify what feature type to extract")
    parser.add_argument('--return_numpy', action='store_true',
                        help='Toggle if you want the features to be output as numpy arrays')
    parser.add_argument('--batchsz', type=int, default=1,
                        help='Number of audio clips to evaluate at once. (Only uses one GPU.)')
    parser.add_argument('--chunksz', type=float, default=100,
                        help="Divide the stimulus waveform into chunks of this many *milliseconds*.")
    parser.add_argument('--contextsz', type=float, default=8000,
                        help="Use these many milliseconds as context for each chunk.")
    parser.add_argument('--full_context', action='store_true',
                        help="Only extract the representation for a stimulus if it is as long as the feature extractor's specified context (context_sz)")
    parser.add_argument('--stride', type=float,
                        help='Extract features every <n> seconds. If using --custom_stimuli, consider changing this argument. Don\'t use this for extracting story features to train encoding models (use --chunksz instead). 0.5 is a good value.')
    parser.add_argument('--min_length_samples', type=float, default=1, 
                        help='specify minimum length of samples in seconds. Defaults to model config min_length_samples if it exists')
    parser.add_argument('--pad_silence', action='store_true',
                        help='Pad short clips (less than context_sz+chunk_sz) with silence at the beginning')
    parser.add_argument('--num_select_frames', type=int, default=1, 
                        help='This can safely be set to 1 for chunk size of 100 and non-whisper models.This specifies how many frames to select features from. ')
    parser.add_argument('--frame_skip', type=int, default=5, 
                        help='This goes with num_select_frames. For most HF models the window is 20ms, so in order to take 1 feature per batched waveform with chunksz = 100ms, you set 5 to say you take num_select_frames (1) every frame_skip')
    parser.add_argument('--target_sample_rate', type=int, default=16000, 
                        help='Most models use a target sample rate of 16000.')
    parser.add_argument('--overwrite', action='store_true',
                                   help='Overwrite existing features (default behavior is to skip)')
    parser.add_argument('--keep_all', action='store_true',
                                   help='keep all features rather than following frame_skip, num_select parameters')
    parser.add_argument("--frame_length", type=float, default=25.,
                            help="Specify frame length for extraction in MS for methods without set length. We set default to 25ms to match WavLM frames.")
    parser.add_argument("--frame_shift", type=float, default=20.,
                            help="Specify frame shift for extraction in MS for methods without set shift. We set default to 20ms to match WavLM frames.")
    parser.add_argument('--recursive', action='store_true',
                                   help='Recursively find .wav and .flac in the stimulus_dir.')
    parser.add_argument('--skip_window',action='store_true',
                            help='Skip windowing for feature extraction.')
    #Cotton candy related args (stimulus selection + save to bucket)
    #Cotton candy related args (stimulus selection + save to bucket)
    cc_args = parser.add_argument_group('cc', 'cottoncandy related arguments (loading/saving to corral)')
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
    #Model loading related arguments (hugging face/SPARC)
    model_args = parser.add_argument_group('models', 'Args for loading models')
    model_args.add_argument('--model_name', type=str,
                        help="Specify model name. Will be dependent on which extactor you are using")
    model_args.add_argument('--model_config_path', type=str, default='./audio_features/configs/hf_model_configs.json',
                        help="Specify a path to a json with model information if required. For hugging face, use: audio_features/configs/hf_model_configs.json")
    model_args.add_argument('--use_featext', action='store_true',
                        help="Specify whether to use a pretrained feature extractor (Hugging face specific)")
    model_args.add_argument('--layers', nargs='+', type=int, default = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],
                        help="Only save the features from these layers. Usually doesn't speed up execution "
                        "time, but may speed up upload time and reduce total disk usage. "
                        "NOTE: only works with numbered layers (currently).")
    #mfcc args
    mfcc_args = parser.add_argument_group('mfcc', 'args for mfcc extraction')
    mfcc_args.add_argument('--n_mfcc', type=int, default=20,
                           help="Specify number of mfccs to extract. If none given, defaults to 20")
    #fbank args
    fbank_args = parser.add_argument_group('fbank', 'args for fbank extraction')
    fbank_args.add_argument("--num_mel_bins", type=int, default=23,
                            help="specify number of mel bins for fbank extraction. If none given, defaults to 23, which is also the default of the original function.")
    #opensmile args
    os_args = parser.add_argument_group('opensmile', 'args for opensmile extraction')
    os_args.add_argument("--feature_set", type=str, default="eGeMAPSv02", help="specify opensmile feature set")
    os_args.add_argument("--feature_level", type=str, default="lld", help="Specify feature level for opensmile. (lld or func)")
    os_args.add_argument("--default_extractor", action="store_true", help="Specify always using default feature extractor with default frame length/frame shift")
    args = parser.parse_args()

    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # STEP 1: some initial variable things
    args.stimulus_dir = Path(args.stimulus_dir)
    args.out_dir = Path(args.out_dir)
    print(args.model_name)
    print(args.model_config_path)
    if args.model_config_path:
        args.model_config_path = Path(args.model_config_path)

    if (args.out_bucket is not None) and (args.out_bucket != ''):
        cci_features = cc.get_interface(args.out_bucket, verbose=False)
        print("Saving features to bucket", cci_features.bucket_name)
        raise NotImplementedError("Can't save to bucket just yet")
    else:
        cci_features = None
        print('Saving features to local filesystem.')
        print('NOTE: You can use ./upload_features_to_corral.sh to upload them later if you wish')

    #Convert chunks from ms to seconds
    contextsz_sec = args.contextsz/1000.
    chunksz_sec = args.chunksz/1000.
    if args.stride:
        contextsz_sec = chunksz_sec+contextsz_sec - args.stride
        chuncksz_sec = args.stride

    #set save path name
    if args.skip_window: 
        model_save_path=Path(f'{args.feature_type}')
    else:
        model_save_path = Path(f"features_cnk{chunksz_sec:0.1f}_ctx{contextsz_sec:0.1f}_pick{args.num_select_frames}_skip{args.frame_skip}/{args.feature_type}")
    if args.model_name:
        model_save_path = model_save_path / args.model_name
    if args.feature_type=='opensmile':
        to_add = f"{args.feature_set}_{args.feature_level}"
        if not args.default_extractor:
            to_add += f"_len{args.frame_length}_shift{args.frame_shift}"
        model_save_path = model_save_path /to_add
    if args.stride and not args.skip_window:
        # If using a custom stride length (e.g. for snippets), store in a
        # separate directory.
        model_save_path = model_save_path / f"stride_{args.stride}"
    
    if cci_features is None:
        model_save_path = args.out_dir / model_save_path
        local_path = None
    else:
        local_path = args.out_dir / model_save_path
        
    print('Saving features to:', model_save_path)
    
    # STEP 2: STIMULUS SELECTION
    stimulus_paths = select_stimuli(stim_dir=args.stimulus_dir, stim_bucket=args.stim_bucket, sessions=args.sessions, stories=args.stories, recursive=args.recursive)
    assert len(stimulus_paths) > 0, "no stimuli to process!"

    stimulus_paths = collections.OrderedDict(sorted(stimulus_paths.items(), key=lambda x: x[0]))

    
    # STEP 3: TODO: FEATURE EXTRACTION TYPES
    if 'hf' in args.feature_type: #there are multiple hugging face features, so only check that hf is in the type
        extractor = set_up_hf_extractor(model_name=args.model_name, save_path=model_save_path, use_featext=args.use_featext, sel_layers=args.layers, target_sample_rate=args.target_sample_rate, model_config_path=args.model_config_path, return_numpy=args.return_numpy, num_select_frames=args.num_select_frames, frame_skip=args.frame_skip, keep_all=args.keep_all)
    elif args.feature_type=='sparc':
        extractor = set_up_sparc_extractor(model_name=args.model_name, save_path=model_save_path, keep_all=args.keep_all)
    elif 'mfcc' in args.feature_type:
        extractor = set_up_mfcc_extractor(save_path=model_save_path, n_mfcc=args.n_mfcc, target_sample_rate=args.target_sample_rate, min_length_samples=args.min_length_samples, return_numpy= args.return_numpy, num_select_frames=args.num_select_frames, frame_skip=args.frame_skip)
    elif 'fbank' in args.feature_type:
        extractor = set_up_fbank_extractor(save_path=model_save_path, num_mel_bins=args.num_mel_bins, frame_length=args.frame_length, frame_shift=args.frame_shift,target_sample_rate=args.target_sample_rate, min_length_samples=args.min_length_samples, return_numpy= args.return_numpy, num_select_frames=args.num_select_frames, frame_skip=args.frame_skip, keep_all=args.keep_all)
    elif 'opensmile' in args.feature_type:
        extractor = set_up_opensmile_extractor(save_path=model_save_path, feature_set=args.feature_set, feature_level=args.feature_level, default_extractor=args.default_extractor, frame_length=args.frame_length, frame_shift=args.frame_shift,target_sample_rate=args.target_sample_rate, min_length_samples=args.min_length_samples, return_numpy= args.return_numpy, num_select_frames=args.num_select_frames, frame_skip=args.frame_skip, keep_all=args.keep_all )
    elif 'emaae' in args.feature_type:
        extractor = set_up_emaae_extractor(save_path=model_save_path, ckpt=args.checkpoint, config=args.modle_config_path, return_numpy=args.return_numpy, keep_all=args.keep_all)
    else:
        raise NotImplementedError(f'{args.feature_type} not supported.')
    
    # STEP 4: Set up batch extrator object
    batching = BatchExtractor(extractor=extractor, save_path=model_save_path, fnames=list(stimulus_paths.keys()), overwrite=args.overwrite, batchsz=args.batchsz, chunksz=chunksz_sec, contextsz=contextsz_sec, require_full_context=args.full_context, min_length_samples=args.min_length_samples, return_numpy=args.return_numpy, pad_silence=args.pad_silence, local_path=local_path, skip_window=args.skip_window)
    
    # STEP 5: RUN BATCHING FOR EACH STIMULUS
   
    # Make sure that all preprocessed stimuli exist and are readable.
    if not args.skip_window:
        num = enumerate(stimulus_paths.items())
    else:
        num = enumerate(tqdm(stimulus_paths.items()))

    for idx, (stimulus_name, stimulus_local_path) in num:
        try:
            wav, sample_rate = torchaudio.load(stimulus_local_path)
        except:
            f'{stimulus_name} does not exist at {stimulus_local_path}. Skipping stimulus.'
            continue
        
        sample = {'path':str(stimulus_local_path), 'fname':stimulus_name}
        output_sample = batching(sample)

   