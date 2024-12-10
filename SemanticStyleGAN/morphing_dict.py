import subprocess
from glob import glob
import os


MAX_RETRIES = 300
for attempt in range(MAX_RETRIES):
    try:
        cmd = f'PYTHONPATH=.:$PYTHONPATH python visualize/simplex.py --ckpt pretrained/pti.pt --outdir results/interpolation/ --step 100 --axis_name AN --axis mat/axis.mat --points mat/points_all.mat --points_info mat/points_info_all.json --fit_points mat/fit_points.mat --dim 6 --partition 10 --candidate_number 300 --eps 1 --sup 1.2 --inf -0.2'
        subprocess.run(cmd, shell=True, check=True)
        print(f'{attempt} time complete!')
        break
    except subprocess.CalledProcessError as e:
        pass
else:
    print("ERROR MAX")
