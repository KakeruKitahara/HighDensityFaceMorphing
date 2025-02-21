import subprocess
from glob import glob
import os

PT = 'pretrained/pti.pt'
LATENT_PATH = 'results/inversion/latent'

if __name__ == "__main__":
  print(PT)

  path = 'images/*.png'
  img_list = sorted(glob(path))

  for st_path in img_list:
    for ed_path in img_list:
      if st_path < ed_path :
        st_name = os.path.splitext(os.path.basename(st_path))[0]
        ed_name = os.path.splitext(os.path.basename(ed_path))[0]
        cmd = f'PYTHONPATH=.:$PYTHONPATH python visualize/generate_morph.py --ckpt {PT}  --outdir results/interpolation/ --latent_start {LATENT_PATH}/{st_name}.npy --latent_end {LATENT_PATH}/{ed_name}.npy --step 100'
        subprocess.run(cmd, shell=True, check=True)