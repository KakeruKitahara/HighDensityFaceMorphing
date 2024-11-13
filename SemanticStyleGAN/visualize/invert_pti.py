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
import copy

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
    im1= F.adaptive_avg_pool2d(im1, (256,256))
    im2 = F.adaptive_avg_pool2d(im2, (256,256))
    p_loss = percept(im1.to(device0), im2.to(device0)).mean()
    return p_loss

def calc_mse_loss(im1, im2):
    # im1 = F.adaptive_avg_pool2d(im1, (256,256))
    # im2 = F.adaptive_avg_pool2d(im2, (256,256))
    mse_loss = F.mse_loss(im1.to(device0), im2.to(device0))
    return mse_loss

def optimize_latent(args, g_ema, target_img_tensor):
    noises = g_ema.render_net.get_noise(noise=None, randomize_noise=False)
    for noise in noises:
        noise.requires_grad = True

    # initialization
    with torch.no_grad():
        noise_sample = torch.randn(10000, 512, device=device0)
        latent_mean = g_ema.style(noise_sample).mean(0)
        latent_in = latent_mean.detach().clone().unsqueeze(0)
        if args.w_plus:
            latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)
    latent_in.requires_grad = True
    
    if args.no_noises:
        optimizer = optim.AdamW([latent_in], lr=args.lr)
    else:
        optimizer = optim.AdamW([latent_in] + noises, lr=args.lr)

    latent_path = [latent_in.detach().clone()]

    composition_mask = torch.zeros(1, g_ema.n_local, device=device0)
    composition_mask[:,:6] = 1
    
    pbar = tqdm(range(args.step))
    for i in pbar:
        optimizer.param_groups[0]['lr'] = get_lr(float(i)/args.step, args.lr)
        img_gen, _ = g_ema([latent_in], input_is_latent=True, randomize_noise=False, noise=noises, composition_mask=composition_mask)

        p_loss = calc_lpips_loss(img_gen, target_img_tensor)
        mse_loss = calc_mse_loss(img_gen, target_img_tensor)
        n_loss = torch.mean(torch.stack([noise.pow(2).mean() for noise in noises]))

        if args.w_plus == True:
            latent_mean_loss = F.mse_loss(latent_in, latent_mean.unsqueeze(0).repeat(latent_in.size(0), g_ema.n_latent, 1))
        else:
            latent_mean_loss = F.mse_loss(latent_in, latent_mean.repeat(latent_in.size(0), 1))

        # main loss function
        loss = (n_loss.to(device0) * args.noise_regularize + 
                p_loss.to(device0) * args.lambda_lpips + 
                mse_loss * args.lambda_mse + 
                latent_mean_loss * args.lambda_mean)

        pbar.set_description(f'perc: {p_loss.item():.6f} noise: {n_loss.item():.6f} mse: {mse_loss.item():.6f}  latent: {latent_mean_loss.item():.6f}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        latent_path.append(latent_in.detach().clone())

    return latent_path, noises

def optimize_pti(args, g_ema, target_img_tensor_array, latent_in_array, noises_array=None):
    g_ema_old = copy.deepcopy(g_ema)

    for p in g_ema.parameters():
        p.requires_grad = True
    optimizer = optim.Adam(g_ema.parameters(), lr=args.lr_g)

    composition_mask = torch.zeros(1, g_ema.n_local, device=device0)
    composition_mask[:,:6] = 1
    g_ema_old = copy.deepcopy(g_ema)

    pbar = tqdm(range(args.finetune_step))
    for i in pbar :

        indices = torch.randperm(len(latent_in_array)).tolist()
        latent_in_array_shuffled = [latent_in_array[i] for i in indices]
        noises_array_shuffled = [noises_array[i] for i in indices]
        target_img_tensor_array_shuffled = [target_img_tensor_array[i] for i in indices]

        loss_pt = 0
        for idx, (target_img_tensor, latent_in, noises) in enumerate(zip(target_img_tensor_array_shuffled, latent_in_array_shuffled, noises_array_shuffled)) :
            if idx > args.pt_size :
                break

            img_gen, _ = g_ema([latent_in], input_is_latent=True, randomize_noise=False, noise=noises, composition_mask=composition_mask)

            p_loss = calc_lpips_loss(img_gen, target_img_tensor)
            mse_loss = calc_mse_loss(img_gen, target_img_tensor)

            # main loss function
            loss_pt += (p_loss * args.lambda_lpips_pti +
                    mse_loss * args.lambda_mse_pti
            )
        
        loss_r = 0
        indices = torch.randperm(len(latent_in_array)).tolist()
        latent_in_array_shuffled = [latent_in_array[i] for i in indices]
        noises_array_shuffled = [noises_array[i] for i in indices]

        for idx, (latent_in, noises) in enumerate(zip(latent_in_array_shuffled, noises_array_shuffled)) : 
            if idx > args.r_size :
                break

            with torch.no_grad():
                noise_sample = torch.randn(10000, 512, device=device0)
                latent_mean = g_ema.style(noise_sample.to(device0)).mean(0)
                latent_rand = latent_mean.detach().clone().unsqueeze(0)
                if args.w_plus:
                    latent_rand = latent_rand.unsqueeze(1).repeat(1, g_ema.n_latent, 1)
                latent_interp = latent_in + args.lambda_alpha * F.normalize(latent_rand - latent_in)
                
            img_gen_old, _ = g_ema_old([latent_interp], input_is_latent=True, randomize_noise=False, noise=noises, composition_mask=composition_mask)
            img_gen, _ = g_ema([latent_interp], input_is_latent=True, randomize_noise=False, noise=noises, composition_mask=composition_mask)

            p_loss = calc_lpips_loss(img_gen, img_gen_old)
            mse_loss = calc_mse_loss(img_gen, target_img_tensor)
            loss_r += (p_loss * args.lambda_lpips_pti2 +
                    mse_loss * args.lambda_mse_pti2
                    )

        loss_pt /= min(args.pt_size, len(latent_in_array))
        loss_r /= min(args.r_size, len(latent_in_array))
        loss = (loss_pt * args.lambda_pt +
                loss_r * args.lambda_r
                )
                
        pbar.set_description(f'loss: {loss.item():.6f} loss_pt: {loss_pt.item():.6f} loss_r: {loss_r.item():.6f}')

        g_ema_old = copy.deepcopy(g_ema)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (not i % 100) and args.save_pt_step :
            with torch.no_grad():
                image_basename = os.path.splitext(image_name)[0]
                ckpt_new = {"g_ema": g_ema.state_dict(), "args": ckpt["args"]}
                torch.save(ckpt_new, os.path.join(args.outdir, 'weights/', f'pti_{i}.pt'))

    return g_ema


if __name__ == '__main__':
    device0 = 'cuda:0'
    device1 = 'cuda:1'

    parser = argparse.ArgumentParser()
    parse_boolean = lambda x: not x in ["False","false","0"]
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--imgdir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)

    parser.add_argument('--size', type=int, default=512)

    parser.add_argument('--no_noises', type=parse_boolean, default=True)
    parser.add_argument('--w_plus', type=parse_boolean, default=True, help='optimize in w+ space, otherwise w space')

    parser.add_argument('--save_steps', type=parse_boolean, default=False, help='if to save intermediate optimization results')

    parser.add_argument('--truncation', type=float, default=1, help='truncation tricky, trade-off between quality and diversity')

    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lr_g', type=float, default=1e-4)
    parser.add_argument('--step', type=int, default=400, help='latent optimization steps')
    parser.add_argument('--finetune_step', type=int, default=0, help='pivotal tuning inversion (PTI) steps (200-400 should give good result)')
    parser.add_argument('--noise_regularize', type=float, default=10)
    parser.add_argument('--lambda_mse', type=float, default=0.3)
    parser.add_argument('--lambda_lpips', type=float, default=1.0)
    parser.add_argument('--lambda_mean', type=float, default=0.8)

    parser.add_argument('--lambda_mse_pti', type=float, default=0.3)
    parser.add_argument('--lambda_lpips_pti', type=float, default=1.0)
    parser.add_argument('--lambda_mse_pti2', type=float, default=0.3)
    parser.add_argument('--lambda_lpips_pti2', type=float, default=1.0)
    parser.add_argument('--lambda_alpha', type=float, default=30)
    parser.add_argument('--lambda_pt', type=float, default=1.0)
    parser.add_argument('--lambda_r', type=float, default=0.1)

    parser.add_argument('--pt_size', type=int, default=999)
    parser.add_argument('--r_size', type=int, default=3)

    parser.add_argument('--exist_latent', type=parse_boolean, default=False)
    parser.add_argument('--save_pt_step', type=parse_boolean, default=False)

    args = parser.parse_args()
    print(args)
    print("Loading model ...")
    ckpt = torch.load(args.ckpt)
    g_ema = make_model(ckpt['args'], device0=device0, device1=device1)
    g_ema.eval()
    g_ema.load_state_dict(ckpt['g_ema'])

    percept = lpips.LPIPS(net_type='vgg').to(device0)

    img_list = sorted(os.listdir(args.imgdir))
    img_list = [string for string in img_list if '.gitkeep' not in string]

    # make output folder
    os.makedirs(os.path.join(args.outdir, 'latent'), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, 'pth'), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, 'recon'), exist_ok=True)
    if args.finetune_step > 0:
        os.makedirs(os.path.join(args.outdir, 'recon_finetune'), exist_ok=True)
    if args.save_steps:
        os.makedirs(os.path.join(args.outdir, 'steps'), exist_ok=True)
    if not args.no_noises:
        os.makedirs(os.path.join(args.outdir, 'noise'), exist_ok=True)
    if args.finetune_step > 0:
        os.makedirs(os.path.join(args.outdir, 'weights'), exist_ok=True)

    transform = get_transformation(args)
    latent_path_array = list()
    noises_array = list()
    target_img_tensor_array = list()

    for image_name in img_list:
        image_basename = os.path.splitext(image_name)[0]
        ckpt_basename = os.path.splitext(os.path.basename(args.ckpt))[0]
        
        print(f'model: {ckpt_basename}, image: {image_basename}')
        img_path = os.path.join(args.imgdir, f'{image_basename}.png')

        # reload the model
        if args.finetune_step > 0:
            g_ema.load_state_dict(ckpt['g_ema'], strict=True)
            g_ema.eval()

        # load target image
        target_pil = Image.open(img_path).resize((args.size,args.size), resample=Image.LANCZOS)
        target_img_tensor = transform(target_pil).unsqueeze(0).to(device0) # image tensol
        target_img_tensor_array.append(target_img_tensor)

        # skip if pth exist
        if args.exist_latent  and os.path.exists(os.path.join(args.outdir, 'pth/', f'{image_basename}.pth')):
            noises = g_ema.render_net.get_noise(noise=None, randomize_noise=False)
            latent_path = torch.load(os.path.join(args.outdir, 'pth/', f'{image_basename}.pth'))
        else : 
            latent_path, noises = optimize_latent(args, g_ema, target_img_tensor)
            
            # save results
            with torch.no_grad():
                img_gen, _ = g_ema([latent_path[-1]], input_is_latent=True, randomize_noise=False, noise=noises)
                img_gen = tensor2image(img_gen).squeeze()
                imwrite(os.path.join(args.outdir, 'recon/', image_name), img_gen)
                
                # Latents
                torch.save(latent_path, os.path.join(args.outdir, 'pth/', f'{image_basename}.pth'))
                latent_np = latent_path[-1].detach().cpu().numpy()
                np.save(os.path.join(args.outdir, 'latent/', f'{image_basename}.npy'), latent_np)
                if not args.no_noises:
                    noises_np = torch.stack(noises, dim=1).detach().cpu().numpy()
                    np.save(os.path.join(args.outdir, 'noise/', f'{image_basename}.npy'), noises_np)

        latent_path_array.append(latent_path)
        noises_array.append(noises)

    # save results pti
    if args.finetune_step > 0:

        print("pti finetune")
        latent_pti_paths = [latent_path[-1] for latent_path in latent_path_array]

        del latent_path_array
        torch.cuda.empty_cache()

        g_ema = optimize_pti(args, g_ema, target_img_tensor_array, latent_pti_paths , noises_array)

        # Weights
        with torch.no_grad():
            image_basename = os.path.splitext(image_name)[0]
            ckpt_new = {"g_ema": g_ema.state_dict(), "args": ckpt["args"]}
            torch.save(ckpt_new, os.path.join(args.outdir, 'weights/', f'pti.pt'))

            for image_name, latent_path_back in zip(img_list, latent_pti_paths):

                composition_mask = torch.zeros(1, g_ema.n_local, device=device0)
                composition_mask[:,:6] = 1

                img_gen, _ = g_ema([latent_path_back], input_is_latent=True, randomize_noise=False, noise=noises, composition_mask=composition_mask)
                img_gen = tensor2image(img_gen).squeeze()
                imwrite(os.path.join(args.outdir, 'recon_finetune/', image_name), img_gen)
