"""
Copy times from the original feature directory to the new feature directory

Used primarily for feat_v_feat.py which creates new features given existing features and therefore the times are exactly the same across them. 

Author(s): Daniela Wiepert
Last modified: 02/15/2025
"""
from typing import Union,List
from pathlib import Path
import shutil

def copy_times(original_times_dir:Union[str,Path], new_dir:Union[str,Path], key_filter:List[str]):
    """
    :param original_times_dir: path like object pointing to original dir
    :param new_dir: path like object pointing to new destination
    :param key_filter: list of keys to include in the copy
    """
    times_paths = Path(original_times_dir).glob('*_times.npz')
    print(times_paths)
    times_str = [str(s) for s in times_paths]
    times_keys = [s.name for s in times_paths]
    print(times_keys)
    key_filter = [i for i in range(len(times_str)) if keys[i] in key_filter]
    times_str = times_str[key_filter]
    new_dir = Path(new_dir)
    new_dir.mkdir(exist_ok=True)
    for t in times_str:
        shutil.copy(t, new_dir)

