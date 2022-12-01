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
import shutil
import numpy as np
import imageio
import torch
from visualize.utils import tensor2image, tensor2seg

from models import make_model
from visualize.utils import generate, cubic_spline_interpolate

latent_dict_celeba = {
    2:  "bcg_1", # 背景
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
    0:  "coarse_1", # 顔の輪郭，位置関係
    1:  "coarse_2",
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('ckpt', type=str, help="path to the model checkpoint")
    parser.add_argument('--latent_start', type=str, default=None,
                        help="path to the starting latent numpy")
    parser.add_argument('--latent_end', type=str, default=None,
                        help="path to the ending latent numpy")
    parser.add_argument('--outdir', type=str, default='./results/interpolation/',
                        help="path to the output directory")
    parser.add_argument('--batch', type=int, default=8,
                        help="batch size for inference")
    parser.add_argument("--sample", type=int, default=10,
                        help="number of latent samples to be interpolated")
    parser.add_argument("--steps", type=int, default=160,
                        help="number of latent steps for interpolation")
    parser.add_argument("--truncation", type=float,
                        default=0.7, help="truncation ratio")
    parser.add_argument("--truncation_mean", type=int, default=10000,
                        help="number of vectors to calculate mean for the truncation")
    parser.add_argument("--dataset_name", type=str, default="celeba",
                        help="used for finding mapping between latent indices and names")
    parser.add_argument('--device', type=str, default="cuda",
                        help="running device for inference")
    parser.add_argument('--enc', type=str, default="mp4",
                        help="output exstension, [mp4], [gif] or [png]", choices=['mp4', 'gif', 'png'])
    args = parser.parse_args()

    if os.path.exists(args.outdir):
        shutil.rmtree(args.outdir)
    os.makedirs(args.outdir)

    print("Loading model ...")
    ckpt = torch.load(args.ckpt)
    model = make_model(ckpt['args'])
    model.to(args.device)
    model.eval()
    model.load_state_dict(ckpt['g_ema'])
    mean_latent = model.style(torch.randn(
        args.truncation_mean, model.style_dim, device=args.device)).mean(0)

    print("Generating original image ...")
    with torch.no_grad():
        if args.latent_start is None:
            raise Exception('Error!')
        else:
            styles_start = torch.tensor(
                np.load(args.latent_start), device=args.device)
            styles_end = torch.tensor(
                np.load(args.latent_end), device=args.device)
        if styles_start.ndim == 2 and styles_end.ndim == 2:
            assert styles_start.size(1) == model.style_dim
            assert styles_end.size(1) == model.style_dim
            styles_start = styles_start.unsqueeze(
                1).repeat(1, model.n_latent, 1)
            styles_end = styles_end.unsqueeze(1).repeat(1, model.n_latent, 1)
        images_start, segs_start = generate(
            model, styles_start, randomize_noise=False)
        images_end, segs_end = generate(
            model, styles_end, randomize_noise=False)
        imageio.imwrite(f'{args.outdir}/image_start.jpeg', images_start[0])
        imageio.imwrite(f'{args.outdir}/seg_start.jpeg', segs_start[0])
        imageio.imwrite(f'{args.outdir}/image_end.jpeg', images_end[0])
        imageio.imwrite(f'{args.outdir}/seg_end.jpeg', segs_end[0])

    print("Generating videos ...")
    if args.dataset_name == "celeba":
        latent_dict = latent_dict_celeba
    else:
        raise ValueError("Unknown dataset name: f{args.dataset_name}")

    with torch.no_grad():
        np.set_printoptions(threshold=30)
        itr = 30
        stnum = styles_start.detach().cpu().numpy()
        _, b, c = stnum.shape
        styles_new = torch.empty((itr, b, c), device=args.device)
        for i in range(itr):
            alpha = (1/(itr-1))*i
            print(alpha, i)
            for latent_index, latent_name in latent_dict.items():
                if latent_index <= 15 : 
                    tmp = alpha*styles_start[:, latent_index]+(1-alpha)*styles_end[:, latent_index]
                    styles_new[i, latent_index] = tmp
                else :
                    styles_new[i, latent_index] = styles_start[:, latent_index]
                
            style_image = torch.unsqueeze(styles_new[i], dim=0)
            image, _ = generate(model, style_image, randomize_noise=False)
        
            imageio.imwrite(f'{args.outdir}/{i}.png', image[0])
        images, segs = generate(model, styles_new, mean_latent=mean_latent,
                                randomize_noise=False, batch_size=args.batch)
        frames = [np.concatenate((img, seg), 1)
                    for (img, seg) in zip(images, segs)]
        imageio.mimwrite(
            f'{args.outdir}/morph.mp4', frames, fps=20)
        print(f"{args.outdir}/morph.mp4")
