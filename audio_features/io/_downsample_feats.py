
import numpy as np
from database_utils.functions import lanczosinterp2D
def downsample_features(features:dict, seqs):

    downsampled_feats = {}
    for story in list(features.keys()):
        f = features[story]['features']
        t = features[story]['times'][:,1] #USING FINAL TIMES BECAUSE THIS IS WHERE A FEATURE IS EXTRACTED FROM
        #avg_times = np.sum(t, axis=1)/2.0

        downsampled_feats[story] = {'features':lanczosinterp2D(
					f, t, seqs[story].tr_times), 'times':features[story]['times']}

    return downsampled_feats