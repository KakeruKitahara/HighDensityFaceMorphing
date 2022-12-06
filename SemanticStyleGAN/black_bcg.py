import argparse
import os
from glob import glob
import cv2
from operator import xor

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--indir', help='input dir')
  parser.add_argument('--input', help='input file')
  args = parser.parse_args()


  input_file = args.input
  input_dir = args.indir
  img_list = []
  paths = []
  black = 3

  if not xor(input_file is not None, input_dir is not None) : 
    assert AssertionError('Error!!')

  if input_dir is not None : 
    path = f'{args.indir}/*.png'
    paths = glob(path)
    for p in paths : 
      img_list.append(cv2.imread(p))

  if input_file is not None : 
    paths.append(input_file)
    img_list.append(cv2.imread(paths[0]))
  
  for img, path in zip(img_list, paths) :
    print(path)
    bcg_ys = []
    eps = 0
    while 1 : 
      print('eps=', eps)
      for y in range(0, len(img)) : 
        cnt = 0
        for x in range(0, len(img[y])) : 
          for c in range(0, len(img[y][x])) :
            if abs(img[y][x][c] - black) <= eps :
              cnt += 1
        if cnt == len(img[y]) * len(img[y][x]) :
          bcg_ys.append(y)

      black_idx = -1
      for y_i in range(0, len(bcg_ys) - 1) : 
        if 100 < bcg_ys[y_i + 1] - bcg_ys[y_i] :
          black_idx = y_i
          break
      if black_idx != -1 : 
        for y in range(0, bcg_ys[black_idx] + 1) : 
          for x in range(0, len(img[y])) : 
            for c in range(0, len(img[y][x])) : 
              img[y][x][c] = black

        cv2.imwrite(path, img)
        break
      eps += 5
  