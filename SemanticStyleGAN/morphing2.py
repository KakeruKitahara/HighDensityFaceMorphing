import subprocess
from glob import glob
import os


pt = 'pretrained/pti.pt'
latent_path = 'results/inversion/latent/'

path = 'images/*.png'
img_list = glob(path)

for image_name in img_list:
  img = os.path.splitext(os.path.basename(image_name))[0]
  npt_list = f'results/inversion/latent/{img}.npy'
  cmd1 = f'PYTHONPATH=.:$PYTHONPATH python visualize/generate_morph.py {pt}  --outdir results/interpolation --latent_start results/inversion/latent/AN.npy --latent_end {npt_list} --step 100'
  subprocess.run(cmd1, shell=True, check=True)

"""
PYTHONPATH=.:$PYTHONPATH python visualize/generate_morph.py pretrained/pti.pt  --outdir results/interpolation --latent_start results/inversion/latent/SU.npy --latent_end results/inversion/latent/AN.npy --step 100

  """