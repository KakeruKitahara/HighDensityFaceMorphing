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


def arrowhead(start, end) :
    arrow_vec = np.array(end - start)
    arrow_length = np.linalg.norm(arrow_vec)
    arrow_unit = arrow_vec /arrow_length
    cone_height = 0.5 * arrow_length

    cone_base = end - arrow_unit * cone_height
    cone_x = cone_base[0]
    cone_y = cone_base[1] 
    cone_z = cone_base[2]
    
    return arrow_unit, (cone_x, cone_y, cone_z)



def plot3d(points, x, x_indices, start, end, dim, mode, step, args) :
    plotp = np.concatenate([points, start.reshape(1, dim), end.reshape(1,dim)])
    colors = ['blue'] * (points.shape[0] + 2)
    for idx in x_indices :
        colors[idx] = 'green' 
    colors[-2] = 'red'
    colors[-1] = 'orange'
    label = list(range(points.shape[0] + 2))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter3d(
        x=plotp[:, 0],
        y=plotp[:, 1],
        z=plotp[:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color=colors,
        ), 
        hovertext=label,
        hoverinfo='text+x+y+z'
    ))
    
    unit, cone = arrowhead(start[:3], end[:3])
    fig.add_trace(go.Scatter3d(
        x=[start[0], end[0]],
        y=[start[1], end[1]],
        z=[start[2], end[2]],
        mode='lines',
        line=dict(color='red', width=7),
    ))
    fig.add_trace(go.Cone(
        x=[cone[0]],
        y=[cone[1]],
        z=[cone[2]],
        u=[unit[0]],
        v=[unit[1]],
        w=[unit[2]],
        sizemode="absolute",
        colorscale=[[0, 'red'], [1, 'red']], 
        sizeref=0.1,
        showscale=False
    ))
    
    plotm=np.concatenate([x, start.reshape(1, dim)])
    combinations = list(itertools.combinations(range(7), 3))
    i, j, k = zip(*combinations)
    
    fig.add_trace(go.Mesh3d(
        x=plotm[:, 0],
        y=plotm[:, 1],
        z=plotm[:, 2],
        i =i, 
        j =j, 
        k = k, 
        opacity=0.1,
        color='cyan'
    ))
    
    
    if mode == 1:
        name = args.axis_name
        graph_path = f'{args.outdir}/graph/{name}'
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

