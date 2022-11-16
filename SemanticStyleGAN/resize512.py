import argparse
import os
import cv2
from IPython.display import Image, display


parser = argparse.ArgumentParser()
parser.add_argument('--input', help='input file')
args = parser.parse_args()

img = cv2.imread(args.input)

dst = cv2.resize(img, dsize=(512, 512))
filename = os.path.basename(args.input)
print(f"{filename} : {img.shape} -> {dst.shape}")

cv2.imwrite(args.input, dst)

