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

import numpy as np
import os
import torch
from scipy.interpolate import CubicSpline
import itertools
import plotly.graph_objects as go


color_map = {
    0: [0, 0, 0],
    1: [239, 234, 90],
    2: [44, 105, 154],
    3: [4, 139, 168],
    4: [13, 179, 158],
    5: [131, 227, 119],
    6: [185, 231, 105],
    7: [107, 137, 198],
    8: [241, 196, 83],
    9: [242, 158, 76],
    10: [234, 114, 71],
    11: [215, 95, 155],
    12: [207, 113, 192],
    13: [159, 89, 165],
    14: [142, 82, 172],
    15: [158, 115, 200], 
    16: [116, 95, 159],
}

def generate_img(model, styles, mean_latent=None, truncation=1.0, batch_size=16, *args, **kwargs):
    images = []
    for head in range(0, styles.size(0), batch_size):
        images_, _ = model([styles[head:head+batch_size]], input_is_latent=True,
                                    truncation=truncation, truncation_latent=mean_latent, *args, **kwargs)
        images.append(images_)
    images = torch.cat(images,0)
    return tensor2image(images)

def generate(model, styles, mean_latent=None, truncation=1.0, batch_size=16, *args, **kwargs):
    images, segs = [], []
    for head in range(0, styles.size(0), batch_size):
        images_, segs_ = model([styles[head:head+batch_size]], input_is_latent=True,
                                    truncation=truncation, truncation_latent=mean_latent, *args, **kwargs)
        images.append(images_.detach().cpu())
        segs.append(segs_.detach().cpu())
    images, segs = torch.cat(images,0), torch.cat(segs,0)
    return tensor2image(images), tensor2seg(segs)

    
def mask_generate(model, styles, composition_mask, mean_latent=None, truncation=1.0, batch_size=16, *args, **kwargs):
    images, segs = [], []
    for head in range(0, styles.size(0), batch_size):
        images_, segs_ = model([styles[head:head+batch_size]], input_is_latent=True, composition_mask=composition_mask, 
                                    truncation=truncation, truncation_latent=mean_latent, *args, **kwargs)
        images.append(images_.detach().cpu())
        segs.append(segs_.detach().cpu())
    images, segs = torch.cat(images,0), torch.cat(segs,0)
    return tensor2image(images), tensor2seg(segs)

def tensor2image(tensor):
    images = tensor.cpu().clamp(-1,1).permute(0,2,3,1).numpy()
    images = images * 127.5 + 127.5
    images = images.astype(np.uint8)
    return images
    
def tensor2seg(sample_seg):
    seg_dim = sample_seg.size(1)
    sample_seg = torch.argmax(sample_seg, dim=1).detach().cpu().numpy()
    sample_mask = np.zeros((sample_seg.shape[0], sample_seg.shape[1], sample_seg.shape[2], 3), dtype=np.uint8)
    for key in range(seg_dim):
        sample_mask[sample_seg==key] = color_map[key]
    return sample_mask

def cubic_spline_interpolate(styles, step):
    device = styles.device
    styles = styles.detach().cpu().numpy()
    N, K, D = styles.shape
    x = np.linspace(0.0, 1.0, N)
    y = styles.reshape(N,K*D)
    spl = CubicSpline(x, y)
    x_out = np.linspace(0.0, 1.0, step)
    results = spl(x_out) # Step x KD
    results = results.reshape(step,K,D)
    return torch.tensor(results, device=device).float()


def Scatter3d(fig, point, d1, d2, d3, colors, label, symbols) :
        fig.add_trace(go.Scatter3d(
        x=point[:, d1],
        y=point[:, d2],
        z=point[:, d3],
        mode='markers',
        marker=dict(
            size=4,
            color=colors,
            symbol=symbols,
        ), 
        hovertext=label,
        hoverinfo='text+x+y+z'
    ))



def plot3d(points, x, x_indices, start, end, dim, mode, step, args, f) :
    if f == 0 :
        d1, d2, d3 = 0,1,2
    else :
        d1, d2, d3 = 3,4,5
    st_ed = np.concatenate([start.reshape(1, dim), end.reshape(1,dim)])
    label_p = list(range(points.shape[0]))
    label_se = list(range(points.shape[0], points.shape[0] + 2))
    
    fig = go.Figure()
    
    Scatter3d(fig, points, d1, d2, d3, 'blue', label_p, 'circle')
    
    fig.add_trace(go.Scatter3d(
        x=[start[d1], end[d1]],
        y=[start[d2], end[d2]],
        z=[start[d3], end[d3]],
        mode='lines',
        line=dict(color='red', width=7),
    ))
    

    Scatter3d(fig, x,  d1, d2, d3, 'green', x_indices, 'circle')
    Scatter3d(fig, st_ed,  d1, d2, d3, 'red', label_se, ['circle', 'diamond'])
    
    plotm=np.concatenate([x, start.reshape(1, dim)])
    combinations = list(itertools.combinations(range(7), 3))
    i, j, k = zip(*combinations)
    
    fig.add_trace(go.Mesh3d(
        x=plotm[:, d1],
        y=plotm[:, d2],
        z=plotm[:, d3],
        i =i, 
        j =j, 
        k = k, 
        opacity=0.1,
        color='cyan'
    ))
    
    if mode == 1:
        name = args.axis_name
        ax_num = args.axis_number
        str_plus = '+' if args.axis_plus else '-'
        fname = f'{name}_{ax_num}_{str_plus}_{d1}{d2}{d3}'
        graph_path = f'{args.outdir}/graph/{fname}'
        os.makedirs(graph_path, exist_ok=True)
        os.makedirs(graph_path, exist_ok=True)
        fig.write_html(f'{graph_path}/{step}.html')
        
        return 
    elif mode == 2:
        resel_idx_str = input("Select index...").split()

        if len(resel_idx_str) == dim and all(s.isnumeric() for s in resel_idx_str) :
            x_indices = [int(s) for s in resel_idx_str]
            x  = points[x_indices]

        return x, x_indices
    else :
        fig.show()
        input("Key input...")
        return 
