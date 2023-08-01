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
import sys
import shutil
import math
import argparse
from tqdm import tqdm

import numpy as np
from PIL import Image
from imageio import imwrite, mimwrite
import torch
from torch import optim
import torch.nn.functional as F
from torchvision import transforms

from criteria.lpips import lpips
from models import make_model
from visualize.utils import tensor2image, tensor2seg

def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp

def get_transformation(args):
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
            ])
    return transform

def calc_lpips_loss(im1, im2):
    img_gen_resize = F.adaptive_avg_pool2d(im1, (256,256))
    target_img_tensor_resize = F.adaptive_avg_pool2d(im2, (256,256))
    p_loss = percept(img_gen_resize, target_img_tensor_resize).mean()
    return p_loss

def optimize_latent(args, g_ema, target_img_tensor, name):

    noises = g_ema.render_net.get_noise(noise=None, randomize_noise=False)
    for noise in noises:
        noise.requires_grad = True

    # initialization
    with torch.no_grad():
        noise_sample = torch.randn(10000, 512, device=device)
        latent_mean = g_ema.style(noise_sample).mean(0)
        latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(args.batch_size, 1)
        if args.w_plus:
            latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)
    latent_in.requires_grad = True

    if args.no_noises:
        optimizer = optim.Adam([latent_in], lr=args.lr)
    else:
        optimizer = optim.Adam([latent_in] + noises, lr=args.lr)

    latent_path = [latent_in.detach().clone()]
    pbar = tqdm(range(args.step))

    for i in pbar:
        optimizer.param_groups[0]['lr'] = get_lr(float(i)/args.step, args.lr)
        composition_mask = torch.zeros(1, g_ema.n_local, device=device)
        composition_mask[:,:6] = 1
        img_gen, _ = g_ema([latent_in], input_is_latent=True, randomize_noise=False, noise=noises, composition_mask=composition_mask)

        p_loss = calc_lpips_loss(img_gen, target_img_tensor)
        mse_loss = F.mse_loss(img_gen, target_img_tensor)
        n_loss = torch.mean(torch.stack([noise.pow(2).mean() for noise in noises]))

        if args.w_plus == True:
            latent_mean_loss = F.mse_loss(latent_in, latent_mean.unsqueeze(0).repeat(latent_in.size(0), g_ema.n_latent, 1))
        else:
            latent_mean_loss = F.mse_loss(latent_in, latent_mean.repeat(latent_in.size(0), 1))

        # main loss function 10, 0.1, 1, 0.1
        loss = (n_loss * args.noise_regularize +
                p_loss * args.lambda_lpips +
                mse_loss * args.lambda_mse +
                latent_mean_loss *args.lambda_mean)

        pbar.set_description(f'loss: {loss.item():.6f} perc: {p_loss.item():.6f}  mse: {mse_loss.item():.6f}  latent: {latent_mean_loss.item():.6f}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # noise_normalize_(noises)
        latent_path.append(latent_in.detach().clone())


    return latent_path, noises



if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()
    parse_boolean = lambda x: not x in ["False","false","0"]
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)

    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=1)

    parser.add_argument('--no_noises', type=parse_boolean, default=True)
    parser.add_argument('--w_plus', type=parse_boolean, default=True, help='optimize in w+ space, otherwise w space')

    parser.add_argument('--save_steps', type=parse_boolean, default=False, help='if to save intermediate optimization results')

    parser.add_argument('--truncation', type=float, default=1, help='truncation tricky, trade-off between quality and diversity')

    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lr_g', type=float, default=1e-4)
    parser.add_argument('--step', type=int, default=400, help='latent optimization steps')
    parser.add_argument('--noise_regularize', type=float, default=10)
    parser.add_argument('--lambda_mse', type=float, default=0.1)
    parser.add_argument('--lambda_lpips', type=float, default=1.0)
    parser.add_argument('--lambda_mean', type=float, default=0.1)

    args = parser.parse_args()
    print(args)

    print("Loading model ...")
    ckpt = torch.load(args.ckpt)
    g_ema = make_model(ckpt['args'])
    g_ema.to(device)
    g_ema.eval()
    g_ema.load_state_dict(ckpt['g_ema'])

    percept = lpips.LPIPS(net_type='vgg').to(device)
    


    os.makedirs(os.path.join(args.outdir, 'recon'), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, 'latent'), exist_ok=True)


    transform = get_transformation(args)
    pt_name = os.path.splitext(os.path.basename(args.ckpt))[0]
    faces = pt_name.split('-')

    for fc in faces :
        image_basename = f'images/{fc}'
        image_name = os.path.splitext(fc)[0]


        img_path =f'images/{fc}.png'

        # load target image
        target_pil = Image.open(img_path).resize((args.size,args.size), resample=Image.LANCZOS)
        target_img_tensor = transform(target_pil).unsqueeze(0).to(device)

        latent_path, noises = optimize_latent(args, g_ema, target_img_tensor, image_basename) 

        # save results
        with torch.no_grad():
            composition_mask = torch.zeros(1, g_ema.n_local, device=device)

            composition_mask[:,:6] = 1 # 髪や首は推論しない．
            
            img_gen, _ = g_ema([latent_path[-1]], input_is_latent=True, randomize_noise=False, noise=noises, composition_mask=composition_mask)
            img_gen = tensor2image(img_gen).squeeze()
            tpl = os.path.join(args.outdir, 'recon/', pt_name)
            os.makedirs(tpl, exist_ok=True)
            imwrite( f'{tpl}/{image_name }.png',  img_gen)
            
            # Latents
            image_basename = os.path.splitext(image_name)[0]
            latent_np = latent_path[-1].detach().cpu().numpy()
            tpl = os.path.join(args.outdir, 'latent/',pt_name)
            os.makedirs(tpl, exist_ok=True)
            np.save(f'{tpl}/{image_name }.npy', latent_np)