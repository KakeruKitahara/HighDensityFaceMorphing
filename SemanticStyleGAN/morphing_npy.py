import subprocess
from glob import glob
import os


original_model_path = 'pretrained/CelebAMask-HQ-512x512.pt'
latent_path = 'results/inversion/latent/*npy'

npy_list = glob(latent_path)

for npy1 in npy_list :
  for npy2 in npy_list :
    if npy2 == npy1 :
      continue
    print(f'Morphing {npy1}, {npy2}')
    cmd1 = f'PYTHONPATH=.:$PYTHONPATH python visualize/generate_morph.py {original_model_path}  --outdir results/interpolation --latent_start {npy1} --latent_end {npy2} --step 100'
    subprocess.run(cmd1, shell=True, check=True)