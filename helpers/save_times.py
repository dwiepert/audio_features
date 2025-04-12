import shutil
from pathlib import Path
root_dir = Path('./data/features/features_cnk0.1_ctx8.0_pick1_skip5/hf/wavlm-large')
mv_dir = Path('./data/dwiepert/data/features/features_cnk0.1_ctx8.0_pick1_skip5/hf/wavlm-large/layer.8')

files = root_dir.glob("*times.npz")
for f in files:
    shutil.copy(str(f), str(mv_dir))
