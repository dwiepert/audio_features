"""
Train/Validation/Test Splits

Author(s): Lasya Yakkala
Last modified: 11/29/2024
"""
#IMPORTS
##built-in
import os
import json
import random
from typing import List, Dict, Tuple, Union
from pathlib import Path
##third-party
import numpy as np

class DatasetSplitter:
    """
    Split into train/val/test splits based on story

    :param stories: List of str stories
    :param output_dir: str/path object, path to save splits to
    :param num_splits: int, number of splits to generate (default = 5)
    :param train_ratio: float (default = 0.8)
    :param val_ratio: float (default = 0.1)
    """
    def __init__(self, stories: List[str], output_dir:Union[Path,str], num_splits:int = 5, train_ratio:float = 0.8, val_ratio:float = 0.1):
        self.stories = stories
        self.num_splits = num_splits
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def split_stories(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Split stories based on train and val ratio

        :return train_stories: list of train stories
        :return val_stories: list of validation stories
        :return test_stories: list of test stories
        """
        random.shuffle(self.stories)
        num_train = int(len(self.stories) * self.train_ratio)
        num_val = int(len(self.stories) * self.val_ratio)
        
        train_stories = self.stories[:num_train]
        val_stories = self.stories[num_train:num_train + num_val]
        test_stories = self.stories[num_train + num_val:]
        
        return train_stories, val_stories, test_stories

    def _generate_splits(self) -> List[Dict[str,List[str]]]:
        """
        Generate all splits and save

        :return splits: list of splits
        """
        splits = []
        for _ in range(self.num_splits):
            train, val, test = self.split_stories()
            splits.append({'train': train, 'val': val, 'test': test})
        
        #saves splits to file
        splits_file = os.path.join(self.output_dir, 'splits.json')
        with open(splits_file, 'w') as f:
            json.dump(splits, f, indent=4)
        print(f"Saved splits to {splits_file}")
        return splits

    def load_splits(self) -> List[Dict[str, List[str]]]:
        """
        Load splits from an existing json file

        :return splits: list of splits
        """
        splits_file = os.path.join(self.output_dir, 'splits.json')
        if not os.path.exists(splits_file):
            return self._generate_splits()
            #raise FileNotFoundError(f"No splits file found at {splits_file}. Run generate_splits first.")
        with open(splits_file, 'r') as f:
            splits = json.load(f)
        return splits

    def split_features(self, features: Dict[str,Dict[str,np.ndarray]], split: Dict[str, List[str]]) -> Tuple[Dict[str,Dict[str,np.ndarray]],  Dict[str,Dict[str,np.ndarray]],  Dict[str,Dict[str,np.ndarray]]]:
        """
        Split a feature dict based on splits

        :param features: dictionary of features
        :param split: split dictionary
        :return train_features: dictionary of train features
        :return val_features: dictionary of validation features
        :return test_features: dictionary of test features
        """
        train_features = {k: v for k, v in features.items() if k in split['train']}
        val_features = {k: v for k, v in features.items() if k in split['val']}
        test_features = {k: v for k, v in features.items() if k in split['test']}
        return train_features, val_features, test_features

#here's an example of how you would run it:
if __name__ == "__main__":
    stories = [f"Story_{i+1}" for i in range(27)]  #example list of 27 stories
    features = {f"Story_{i+1}": {"feature": i} for i in range(27)}  #example feature dictionary

    splitter = DatasetSplitter(stories, output_dir="splits", num_splits=5)
    
    splitter.generate_splits()
    splits = splitter.load_splits()
    
    #example: use the first split to split features
    train_features, val_features, test_features = splitter.split_features(features, splits[0])
    print("Train Features:", train_features)
    print("Validation Features:", val_features)
    print("Test Features:", test_features)