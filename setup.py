from setuptools import setup, find_packages
from audio_features._version import __version__

setup(
    name = 'audio_features.py',
    packages = find_packages(),
    author = 'HuthLab',
    python_requires='>=3.8',
    install_requires=[
        'numpy==1.26.4',
        'librosa==0.10.2.post1',
        'transformers==4.46.1',
        'torchaudio==2.2.0',
        'torchvision==0.17.2',
        'torch==2.2.2',
        'cottoncandy==0.2.0',
        'opensmile==2.5.0',
        'h5py==3.12.1'
    ],
    include_package_data=False,  
    version = __version__,
)