import subprocess
from glob import glob
import os


PT = 'pretrained/pti.pt'
MAX_RETRIES = 3 # Number of attempts
SUP = 1.0
INF = 0
AXIS = 3 # Number of axis
CANDIDATE_NUMBERS = [700, 500, 300] # Range of nearest neighbor to search
AXIS_LIST = ["AN", "DI", "HA"] # Expresion

def create_morphing(faicial , axis, is_plus) :
    axis_file = f'axis_{faicial}.mat'
    for c in CANDIDATE_NUMBERS :
        for attempt in range(MAX_RETRIES):
            try:
                cmd = f'PYTHONPATH=.:$PYTHONPATH python visualize/simplex.py --ckpt {PT} --outdir results/interpolation/ --step 100 --axis_name {faicial} --axis mat/{axis_file} --axis_number {axis} --axis_plus {is_plus} --points mat/points.mat --points_info mat/points_info.json --fit_points mat/thresholds.mat --dim 6 --partition 10 --candidate_number {c} --eps 0.01 --sup {SUP} --inf {INF}'
                subprocess.run(cmd, shell=True, check=True)
                print(f'{attempt} time complete!')
                break
            except subprocess.CalledProcessError as e:
                pass
        else:
            print("ERROR MAX")
            break
        print("-----------------------")
    
    

if __name__ == "__main__":
    for faicial in AXIS_LIST :
        for axis in range(AXIS) :
            for is_plus in range(2):
                create_morphing(faicial, axis, is_plus)