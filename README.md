# Audio Features
Package for extracting various features from audio stimuli

## Setup
Before installing: if you have access to conda it is good practice to create a new environment for use with this package. Make sure the environment is active before installing. 

To install, use

```
$ git clone https://github.com/dwiepert/audio_features.git
$ cd audio_features
$ pip install . 
```

This will be the first stage of setting up this package to run. You will also need to install:
* https://github.com/dwiepert/audio_preprocessing
* https://github.com/dwiepert/database_utils
* ffmpeg=6.1.1, you can do this with `conda install conda-forge::ffmpeg=6.1.1`
* https://github.com/cheoljun95/Speech-Articulatory-Coding.git

The other git packages are installed with a similar process (git clone, cd into the new repo, pip install), though you can check out their READMEs for the full process.

## Editing code
Since we're using a shared private github, when you edit it - use the following git commands

```
cd audio_features #must be in this directory for anything to work
git remote # it should list origin if you are in fact connected to the shared github
git fetch origin # I'll try to let you know if this needs to be run because I changed some higher level code
git branch BRANCHNAME #create a local branch PRIOR TO CODING ANYTHING
git push origin BRANCHNAME
```

when you want to merge it it's probably easiest to do that on the website instead, but if you know how to merge and resolve conflicts in a terminal go ahead. The main thing that we need to be careful of is editing stimulus_features.py as this will be potentially edited by everyone. Once you've edited and debugged your implementation of the extractor and added it to stimulus_features/made sure it runs, feel free to merge it with the master branch and let us know to pull it. 

## Extractor example
See audio_features/extractors/_mfcc_extraction.py for a walk through of how to code an extractor. Copy and fill in TODOs as you see fit for whatever feature type you decide to extract. 

## Probes: TODO
Nothing has been implemented for this yet.

## Example arguments for different things
If you can see the .vscode/launch.json file, that has debugging environments with the arguments you would need specified.

Mainly, you should always have --require_full_context toggled on and probably --return_numpy and you should specify --stimulus_dir and --out_dir. You can mess with --batchsz but don't touch any of contextsz, chunksz, etc. 



# TRACK CHANGES
SPARC bug? - kernel size 400 (25ms) while stride is 320 (20ms)? output wasn't right dimensions before Amplitude thing - talk to Alex
Also talk to alex about pitch
do we want to just ignore voicing? 