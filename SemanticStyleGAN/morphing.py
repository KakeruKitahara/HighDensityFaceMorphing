import subprocess
from glob import glob
import os


original_model_path = 'pretrained/CelebAMask-HQ-512x512.pt'
latent_path = 'results/inversion/latent/'

path = 'pretrained/*.pt'
pt_list = glob(path)

for pt in pt_list :
  if original_model_path == pt :
    continue
  pair_name = os.path.splitext(os.path.basename(pt))[0]
  npt_list = glob(f'results/inversion/latent/{pair_name}/*.npy')
  print(f'Morphing {pair_name}')
  cmd1 = f'PYTHONPATH=.:$PYTHONPATH python visualize/generate_morph.py {pt}  --outdir results/interpolation --latent_start {npt_list[0]} --latent_end {npt_list[1]} --step 100'
  subprocess.run(cmd1, shell=True, check=True)