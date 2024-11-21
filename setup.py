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
        'torchvision==0.20.0',
        'torch==2.5.0',
        'cottoncandy==0.2.0',
        'torchaudio==2.5.0',
        'opensmile==2.5.0'
    ],
    include_package_data=False,  
    version = __version__,
)