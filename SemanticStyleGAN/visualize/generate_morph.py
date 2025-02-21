# Copyright (C) 2022 ByteDance Inc.
# All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

# The software is made available under Creative Commons BY-NC-SA 4.0 license
# by ByteDance Inc. You can use, redistribute, and adapt it
# for non-commercial purposes, as long as you (a) give appropriate credit
# by citing our paper, (b) indicate any changes that you've made,
# and (c) distribute any derivative works under the same license.

# THE AUTHORS DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING ALL
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE.
# IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL
# DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING
# OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

import os
import argparse
import numpy as np
import imageio
import torch
from tqdm import tqdm
from models import make_model
from visualize.utils import mask_generate

latent_dict_celeba = {
    2:  "bcg_1",
    3:  "bcg_2",
    4:  "face_shape",
    5:  "face_texture",
    6:  "eye_shape",
    7:  "eye_texture",
    8:  "eyebrow_shape",
    9:  "eyebrow_texture",
    10: "mouth_shape",
    11: "mouth_texture",
    12: "nose_shape",
    13: "nose_texture",
    14: "ear_shape",
    15: "ear_texture",
    16: "hair_shape",
    17: "hair_texture",
    18: "neck_shape",
    19: "neck_texture",
    20: "cloth_shape",
    21: "cloth_texture",
    22: "glass",
    24: "hat",
    26: "earing",
    0:  "coarse_1",
    1:  "coarse_2",
}



if __name__ == '__main__':
    device0 = 'cuda:0'
    device1 = 'cuda:1'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, help="path to the model checkpoint")
    parser.add_argument('--latent_start', type=str, default=None,
                        help="path to the starting latent numpy")
    parser.add_argument('--latent_end', type=str, default=None,
                        help="path to the ending latent numpy")
    parser.add_argument('--outdir', type=str, default='./results/interpolation/',
                        help="path to the output directory")
    parser.add_argument('--batch', type=int, default=8,
                        help="batch size for inference")
    parser.add_argument("--step", type=int, default=100,
                        help="number of latent steps for interpolation")
    parser.add_argument("--fps", type=int, default=30,
                        help="number of fps for morphing")
    parser.add_argument("--dataset_name", type=str, default="celeba",
                        help="used for finding mapping between latent indices and names")

    args = parser.parse_args()

    print("Loading model ...")
    ckpt = torch.load(args.ckpt)
    model = make_model(ckpt['args'], device0=device0, device1=device1)
    model.eval()
    model.load_state_dict(ckpt['g_ema'])

    print("Generating original image ...")
    with torch.no_grad():
        if args.latent_start is None:
            raise Exception('Error!')
        else:
            styles_start = torch.tensor(
                np.load(args.latent_start), device=device0)
            styles_end = torch.tensor(
                np.load(args.latent_end), device=device0)
        if styles_start.ndim == 2 and styles_end.ndim == 2:
            assert styles_start.size(1) == model.style_dim
            assert styles_end.size(1) == model.style_dim
            styles_start = styles_start.unsqueeze(
                1).repeat(1, model.n_latent, 1)
            styles_end = styles_end.unsqueeze(1).repeat(1, model.n_latent, 1)

    filename_start = os.path.splitext(os.path.basename(args.latent_start))[0]
    filename_end = os.path.splitext(os.path.basename(args.latent_end))[0]
    movie_path = f'{args.outdir}/movie/{filename_start}_{filename_end}'
    flip_path = f'{args.outdir}/flip/{filename_start}_{filename_end}'
    os.makedirs(movie_path, exist_ok=True)
    os.makedirs(flip_path, exist_ok=True)
    
    print(f'Generating morphing {filename_start} -> {filename_end}')
    if args.dataset_name == "celeba":
        latent_dict = latent_dict_celeba
    else:
        raise ValueError("Unknown dataset name: f{args.dataset_name}")

    with torch.no_grad():
        itr = args.step
        stnum = styles_start.detach().cpu().numpy()
        _, b, c = stnum.shape
        styles_new = torch.empty((itr, b, c), device=device0)
        composition_mask = torch.zeros(1, model.n_local, device=device0)
        composition_mask[:,0:6] = 1
        for i in tqdm(range(itr)):
            alpha = (1/(itr-1))*i
            for latent_index, _ in latent_dict.items():
                tmp = (1-alpha) * styles_start[:, latent_index] + alpha * styles_end[:, latent_index]
                styles_new[i, latent_index] = tmp
            style_image = torch.unsqueeze(styles_new[i], dim=0)
            image, _ = mask_generate(model, style_image, randomize_noise=False, composition_mask=composition_mask)

            imageio.imwrite(f'{flip_path}/{i + 1:03}.png', image[0])
        images, segs = mask_generate(model, styles_new, randomize_noise=False, batch_size=args.batch, composition_mask=composition_mask)
        frames = [np.concatenate((img, seg), 1)
                  for (img, seg) in zip(images, segs)]

        fps = args.fps
        imageio.mimwrite(
            f'{movie_path}/{filename_start}_{filename_end}.mp4', images, fps=fps)
        imageio.mimwrite(
            f'{movie_path}/{filename_start}_{filename_end}.gif', images, fps=fps)
        imageio.mimwrite(
            f'{movie_path}/{filename_start}_{filename_end}_mask.mp4', frames, fps=fps)
