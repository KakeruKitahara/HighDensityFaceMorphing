import os
import sys
import bz2
import requests
import shutil
from tqdm import tqdm
import argparse
from face_alignment import image_align
from landmarks_detector import LandmarksDetector

LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
CACHE_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'pytorch')
MODEL_PATH = os.path.join(CACHE_DIR, 'shape_predictor_68_face_landmarks.dat')

def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    with open(dest_path, 'wb') as f, tqdm(
        desc=dest_path,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

def extract_bz2(src_path, dest_path):
    with bz2.open(src_path, 'rb') as f_in:
        with open(dest_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', help='input directory')
    args = parser.parse_args()
  
    """
    Extracts and aligns all faces from images using DLib and a function from original FFHQ dataset preparation step
    python align_images.py /raw_images /aligned_images
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    bz2_path = f'{MODEL_PATH}.bz2'
    download_file(LANDMARKS_MODEL_URL, bz2_path)
    extract_bz2(bz2_path, MODEL_PATH)
    os.remove(bz2_path)
    RAW_IMAGES_DIR = args.indir 
    ALIGNED_IMAGES_DIR = args.indir

    landmarks_detector = LandmarksDetector(MODEL_PATH)
    for img_name in os.listdir(RAW_IMAGES_DIR):
        raw_img_path = os.path.join(RAW_IMAGES_DIR, img_name)
        for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
            face_img_name = '%s.png' % (os.path.splitext(img_name)[0])
            aligned_face_path = os.path.join(ALIGNED_IMAGES_DIR, face_img_name)

            image_align(raw_img_path, aligned_face_path, face_landmarks)