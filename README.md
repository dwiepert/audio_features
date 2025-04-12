# Audio Features
Package for extracting various features from audio stimuli

## Setup
In order to install all components easily, it is best to have a conda environment (some packages installed with conda and not setup.py)

To install, use

```
$ git clone https://github.com/dwiepert/audio_features.git
$ cd audio_features
$ pip install . 
```

This will be the first stage of setting up this package to run. You will also need to install:
* [preprocessing repo](https://github.com/dwiepert/audio_preprocessing.git)
* [database_utils repo](https://github.com/dwiepert/database_utils.git)
* [sparc repo](https://github.com/dwiepert/sparc.git)
* ffmpeg=6.1.1, you can do this with `conda install conda-forge::ffmpeg=6.1.1`
* [emaae repo](https://github.com/dwiepert/emaae.git)
* pytables, install with `conda install pytables`
Please note that we are installing a fork of sparc to deal with a bug that hasn't been addressed in the official implementation

The other git packages are installed with a similar process (git clone, cd into the new repo, pip install), though you can check out their READMEs for the full process.

## Extracting features
Extracting features directly from the audio is done with [stimulus_features.py](https://github.com/dwiepert/audio_features/stimulus_features.py). The following arguments are required/recommended to run extraction:
* `--stimulus_dir=PATH_TO_STIMULUS_FILES`
* `--out_dir=PATH_TO_SAVE_OUTPUTS`
* `--feature_type=FEATURETYPE`
* `--return_numpy`
* `--full_context`

This code can extract the following features:
* hugging face based models as specified in [hf_model_configs.json](https://github.com/dwiepert/audio_features/audio_features/configs/hf_model_configs.json). To specify this give the following arguments(s): `--feature_type=hf --model_name=HFMODELNAME`, where HFMODELNAME is from the configs json. Optional arguments for hugging face based extraction are:
    - `--use_featext` to use the feature extractor associated with a model
    - `--layers 0 1 .... n_layers` to select features from layers beyond the final one. List the number associated with each layer you want to extract.
    - `--model_config_path=CONFIGPATH` if using a config file different than the default hf_model_configs.json.
* Speech articulatory coding ([SPARC](https://github.com/Berkeley-Speech-Group/Speech-Articulatory-Coding/tree/main)) framework. To specify this give the following argument(s): `--feature_type=sparc --model_name=SPARCMODELNAME` where SPARCMODELNAME is "en", "multi", or "en+" (specifies models trained on different data). We used "en" for our experiments. 
* FBANK features. To specify this give the following argument(s): `--feature_type=fbank`. Some optional arguents are:
    - `--num_mel_bins=NUM`, number of mel bins to use
    - `--frame_length=FLOAT`, frame length for extraction in ms
    - `--frame_shift=FLOAT`, frame shift for extraction in ms
* openSMILE based features. ComParE2016 is currently implemented but could easily be extended to other features available from openSMILE ([opensmile with python](https://audeering.github.io/opensmile-python/)). To specify this give the following argument(s): `--feature_type=opensmile`. You can optionally set:
    - `--feature_set=OPENSMILEFEATURESETNAME`, the opensmile feature set you want to use
    - `--feature_level=OPENSMILEFEATURELEVEL`, the feature level you want to use (either lld or func)
    - `--default_extractor`, use the default extractor rather than a custom extractor (boolean)
    - `--frame_length=FLOAT`, frame length for extraction in m (only for custom extractor)
    - `--frame_shift=FLOAT`, frame shift for extraction in ms (only for custom extractor)

Other optional features of interest:
* `--batchsz=INT`: batch size
* `--chunksz=FLOAT`, `--contextsz=FLOAT`: window size parameters
* `--min_length_samples=FLOAT`: minimum length any sample can be in seconds
* `--pad_silence`, specify whether to pad short clips with silence
* `--overwrite`, overwrite existing features
* `--keep_all`, turn off downsampling of features
* `--recursive`, recursively load audio files
* `--stim_bucket=BUCKETNAME, --out_bucket= BUCKETNAME, --sessions 1 2 ... num_sessions, --stories name1 ... namen` for working directly with stimulus bucket. If so, use sessions 1 2 3 4 5.

## Extracting features from features
A handful of features are extracted based on other features (residuals, ema-wav, pca, emaae) or require alignment with features (word/phone identity). These are extracted with [feat_v_feat.py](https://github.com/dwiepert/audio_features/feat_v_feat.py). The following arguments are required/recommended to run extraction: 
* `--feat_dir1=PATH, --feat_dir2=PATH`: give full file path to feature directories. `feat_dir1` specifies the from feature while `feat_dir2` is the feature to extraction. For word/phone identiy, feat_dir2 is the name of the directory you would like to save the word/phone features to. 
* `--feat1_type=FEATNAME, --feat2_type=FEATNAME`: specify the feature name. You can choose whatever name you want for features unless you are attempting to extract word/phone identity in which case it must be either 'word' or 'phone'.
* `--feat1_times=PATH : give full file path to directory with times associated with the features. This is ONLY REQUIRED for using features that have modules like wavLM layer features!
* `--out_dir=PATH`: path to save files to
* `--function=FUNCTIONNAME`: choose function depending on which feature is being extracted (see below)

The following features can be extracted using this script:
* residuals and ema-wav features (predicted wavLM features from regression trained to map EMA to WAV). To extract these features, set the following arguments:
    - `--feat_dir1=PATH_TO_WAVLM_FEATS --feat1_type=wavlm-large.LAYER`. Note that you can change `feat1_type` as you desire. If an error pops up for missing times, you can use [save_times.py](https://github.com/dwiepert/audio_features/helpers/save_times.py) to manually copy the the times from the larger directory.
    - `--feat_dir2=PATH_TO_SPARC/EMA_FEATS --feat2_type=ema`. The feat2_type MUST be 'ema' for purposes of processing ema features correctly.
    - `--function=lstsq`: specifies we're extracting with least squares regression. 
* pca features (based on residuals from lstsq regression). To extract these features, set the following arguments:
    - `--feat_dir1=PATH_TO_RESIDUALS --feat1_type=lstsq`. Note that you can change `feat1_type` as you desire. 
    - `--function=pca`
* word/phone identity features. These need to be extracted for EACH of the main feature sets (wavlm, ema, emawav, residuals, pca-residuals). To extract these features, set the following arguments:
    - `--feat_dir1=PATH_TO_FEATURES --feat1_type=FEATURENAME`. Note that you can change `feat1_type` as you desire. 
    - `--feat_dir2=PATH_TO_WORD/PHONE_IDENTITY_DIR --feat2_type=word/phone`. Set either word or phone for `feat2_type` and decide output name for `feat_dir2`.
    - `--function=extract`
* EMAAE features (wavLM like features from SPARC EMA features). To extract these features, set the following arguments:
    - `--feat_dir1=PATH_TO_SPARC_FEATS --feat1_type=ema`. `feat1_times` is not needed.
    - `--feat_dir2` and other values are not needed.
    - `--function=emaae`
    - `--model_checkpoint=PATH_TO_LOCAL_CHECKPONT`. Should be a local .pth model weights file.
    - `--model_config_path=PATH_TO_MODEL_CONFIG`. Should point to an EMAAE model config json file

Other optional features of interest:
* `--overwrite`, overwrite existing features
* `--recursive`, recursively load audio files
* `--skip_window`, extract features from entire signal without following the windowing procedure.
* `--stim_bucket=BUCKETNAME, --out_bucket= BUCKETNAME, --sessions 1 2 ... num_sessions, --stories name1 ... namen` for working directly with stimulus bucket. If so, use sessions 1 2 3 4 5.

## Linear probing
Linear probing is also done with [feat_v_feat.py](https://github.com/dwiepert/audio_features/feat_v_feat.py). You can fit either ridge regressions or logistic regressions (classification) with this code. Feature 1 will always be the independent variable, so set `--feat_dir1 --feat1_type --feat1_times` accordingly. Feature 2 will be either the dependent vaiable or the targets so set  `--feat_dir2 --feat2_type --feat2_times` accordingly. The following arguments are required/recommended to run probing: 
* `--feat_dir1=PATH, --feat_dir2=PATH`: give full file path to feature directories. `feat_dir1` specifies the from feature while `feat_dir2` is the feature needed for extraction. For word/phone identiy, feat_dir2 is the name of the directory you would like to save the word/phone features to. 
* `--feat1_type=FEATNAME, --feat2_type=FEATNAME`: specify the feature name. You can choose whatever name you want for features unless you are attempting to extract word/phone identity in which case it must be either 'word' or 'phone'.
* `--feat1_times=PATH, --feat2_times=PATH`: give full file path to directory with times associated with the features. This is only required if a feature does not save times in the same directory as the features. 
* `--out_dir=PATH`: path to save files to
* `--function=FUNCTIONNAME`: choose function depending on which kind of probe you want to run.

The following details the two functions and when to use them:
* `--function=ridge` specifies Ridge regression. This is used for fbank/opensmile/word embedding probes. Some additional arguments for this function include:
    - `--cv_split=INT`: number of cross validation splits to use (default=5) 
    - `--n_boots=INT`: number of repeats of cross validation to use (default=3)
    - `--corr_type=STRTYPE`: either 'feature' or 'time'. Designates how to average correlations. 'time' is only used for word embedding probe. (default=feature)
* `--function=multiclass_clf`: specifies logistic regression for identity probes. Targets should be int values associated with a unique word/phone. No additional arguments are needed.
* `--function=multilabel_clf`: used for categorical articulation features when there are multiple labels to fit regressions for. 

You can aso create dataset splits using the following arguments:
* `--split`: flag for generating splits
* `--split_path=PATH`: path to split directory. Should be given to maintain consistency across probes.
* `--n_splits=INT`: number of splits to make (default=5)
* `--train_ratio=FLOAT`: train ratio out of 1 (default=.8)

Other optional features of interest:
* `--overwrite`, overwrite existing features
* `--recursive`, recursively load audio files
* `--stim_bucket=BUCKETNAME, --out_bucket= BUCKETNAME, --sessions 1 2 ... num_sessions, --stories name1 ... namen` for working directly with stimulus bucket. If so, use sessions 1 2 3 4 5.

## Extractor example
See audio_features/extractors/_mfcc_extraction.py for a walk through of how to code an extractor. Copy and fill in TODOs as you see fit for whatever feature type you decide to extract. The other examples can give you an idea for more complex types

## Extracting orthogonal features
To do this, we use np.linalg.lstsq - we are using a modified version of the SPARC EMA features (with loudness removed (i.e. we are removing index 12 from the ema array) as loudness is purely acoustic) to predict layer 9 WavLM features. This is run in feat_v_feat.py. 

## RESULT ANALYSIS
Plots and analyis done in [plots.Rmd](https://github.com/dwiepert/audio_featurs/plots.Rmd)
# TRACK CHANGES
SPARC bug? - kernel size 400 (25ms) while stride is 320 (20ms)? output wasn't right dimensions before Amplitude thing - talk to Alex
Changed line 369 in generator.py of SPARC to torch.nn.utils.parametrizations.weight_norm(m) bc of utils.weight_norm becoming deprecated
Ignoring loudness - acoustic

Common issues:
WINDOWS PATHS : make sure / not \ (which is how it shows up in a terminal)

TODO:
1. Handle overwriting - allow for adding new stories without overwriting everything that already existed
2. Debug cci

