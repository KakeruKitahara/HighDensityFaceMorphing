import subprocess
from glob import glob
import os

path ='results/inversion/latent/*.npy'
pt = 'pretrained/pti.pt'
npy_list = glob(path)

for npy_path in npy_list :
  name = os.path.splitext(os.path.basename(npy_path))[0]
  out_path = os.path.join('results/components/', name)
  os.makedirs(out_path, exist_ok=True)
  print(f'Display {name}...')
  cmd = f'PYTHONPATH=.:$PYTHONPATH python visualize/generate_components.py {pt} --outdir {out_path} --latent {npy_path}'
  subprocess.run(cmd, shell=True, check=True)

