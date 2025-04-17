### RUN THIS PRIOR TO PLOTS.RMD

import numpy as np
import json
import glob
import os
import numpy as np

folder = './data/new_results' # WHERE RESULTS WERE SAVED TO
jsons = glob.glob(os.path.join(folder,'*/*/*/*.json'),recursive=True)
jsons = [j for j in jsons if 'multilabel' in j]
#jsons = [j for j in jsons if 'test_eval' in j]
#jsons = [j for j in jsons if ('multilabel' in j and '_new' not in j)]
#jsons = [j for j in jsons if 'eval' not in j]
jsons = [j for j in jsons if 'config' not in j]
jsons = [j for j in jsons if 'multilabel' in j]
jsons = [j for j in jsons if '_new' not in j]

for j in jsons:
    with open(j, 'rb') as f:
        temp = json.load(f)
    new_dict = {}
    for t in temp:
        outs = temp[t]
        sub_dict = {}
        new_outs = []
        for i in range(len(outs)):
            if not np.isnan(outs[i]):
                new_outs.append(float(outs[i]))
                #name = "".join([t, str(i)])
                #new_dict[name] = float(outs[i])
        new_dict[t] = new_outs
    new_j = os.path.splitext(j)[0]+'_new.json'
    with open(new_j, 'w') as f:
        json.dump(new_dict, f)
print('pause')