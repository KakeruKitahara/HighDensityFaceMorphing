import subprocess
from glob import glob
import os

img_size = 512
model = 'CelebAMask-HQ-512x512.pt'
finetune_step = 300

print(f'Inverting from {model} ...')
cmd = f'PYTHONPATH=.:$PYTHONPATH python visualize/invert.py --ckpt pretrained/{model} --imgdir images --outdir results/inversion --size {img_size} --finetune_step {finetune_step}'
subprocess.run(cmd, shell=True, check=True)

path ="results/inversion/weights/*.pt"
pt_list = glob(path)

for pt_path in pt_list :
  pt_name = os.path.basename(pt_path)
  print(f'Inverting from {pt_name} ...')
  cmd = f'PYTHONPATH=.:$PYTHONPATH python visualize/invert.py --ckpt {pt_path} --imgdir images --outdir results/inversion --size {img_size} --finetune_step {finetune_step}'
  subprocess.run(cmd, shell=True, check=True)
