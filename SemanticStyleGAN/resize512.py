import argparse
import os
from glob import glob
import cv2

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--indir', help='input dir')
  args = parser.parse_args()

  if args.indir is not None : 
    path = f'{args.indir}/*.png'
    img_list = glob(path)
    for ol_img in img_list :
      img = cv2.imread(ol_img)

      dst = cv2.resize(img, dsize=(256, 256))
      filename = os.path.basename(ol_img)
      print(f"{filename} : {img.shape} -> {dst.shape}")

      cv2.imwrite(ol_img, dst)

