import argparse
import os
from glob import glob
import cv2

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--indir', help='input directory')
  args = parser.parse_args()

  if args.indir is not None : 
    path = f'{args.indir}/*.png'
    img_list = glob(path)
    for ol_img in img_list :
      img = cv2.imread(ol_img)
      img = cv2.resize(img, (512, 512))
      if img.shape[2] == 4 :
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

      print(img.shape)
      cv2.imwrite(ol_img, img)