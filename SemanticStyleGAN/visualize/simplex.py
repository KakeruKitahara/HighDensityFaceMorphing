import argparse
import os
from tqdm import tqdm
import numpy as np
import imageio
import torch
from models import make_model
from visualize.utils import mask_generate
import h5py
import json
from sklearn.decomposition import PCA
import plotly.graph_objects as go
imageio.plugins.ffmpeg.download()

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

def pca(points, fit_points, dim):
    pca = PCA(n_components=dim)
    pca.fit(fit_points)
    pca_points = pca.transform(points)
    
    return pca_points

def search_nearpnts(points, e_ed) :
    dis = np.linalg.norm(points - e_ed, axis=1)
    x_indices = np.argsort(dis)[1:dim + 1]
    x = points[x_indices]

    return x, x_indices

def is_simplex(x, e_st, dim) : 
    x_from_st = np.array([p - e_st for p in x])
    eps = 10e-15
    
    rank = np.linalg.matrix_rank(x_from_st)
    if rank != dim :
        raise Exception(f"""Rank is {rank}.
                        {x}""")
    
    x_from_st_normal = np.array([p / np.linalg.norm(p, ord=2)  for p in x_from_st])
    det = np.linalg.det(x_from_st_normal)

        
    print(f'det : {det}')
    return x_from_st

def plot3d(points, x, x_indices, p_st, p_ed, dim) :
    plotp = np.concatenate([points, p_st.reshape(1, dim), p_ed.reshape(1,dim)])
    
    colors = ['blue'] * (pca_pnts.shape[0] + 2)
    for idx in x_indices :
        colors[idx] = 'green' 
    colors[-2] = 'red'
    colors[-1] = 'orange'
    label = list(range(pca_pnts.shape[0] + 2))
    

    fig = go.Figure(data=[go.Scatter3d(
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
    )])
    fig.show()
    
    resel_idx_str = input("Select index...").split()
    
    if len(resel_idx_str) == dim and all(s.isnumeric() for s in resel_idx_str) :
        x_indices = [int(s) for s in resel_idx_str]
        x  = points[x_indices]
    
    return x, x_indices



def clac_simplex(e, x, dim) :
    gamma = np.dot(e, np.linalg.inv(x))
    print(f'gamma : {gamma}')
    print(f'sum : {sum(gamma)}')

    return gamma


if __name__ == '__main__':
    device0 = 'cuda:0'
    device1 = 'cuda:1'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, help="path to the model checkpoint")
    parser.add_argument('--axis', type=str, default=None, help="direction vector in a low level space") # 研究の都合上matファイルで読み込む． ./mat/axis.mat
    parser.add_argument('--axis_name', type=str, default=None, help="name of direction vector")
    parser.add_argument('--points', type=str, default=None, help="points of a low level space") # 研究の都合上matファイルで読み込む．./mat/points.mat
    parser.add_argument('--points_info', type=str, default=None, help="points of a low level space")
    parser.add_argument('--fit_points', type=str, default=None, help="points of a low level space") # pcaの基準，研究の都合上matファイルで読み込む．./mat/thresholds_all.mat
    parser.add_argument('--latent_indir', type=str, default='./results/inversion/latent/',
                        help="path to the latent numpy directory")
    parser.add_argument('--morph_indir', type=str, default='./results/interpolation/flip/',
                        help="path to the output directory")
    parser.add_argument('--outdir', type=str, default='./results/interpolation/',
                        help="path to the output directory")
    parser.add_argument('--batch', type=int, default=8,
                        help="batch size for inference")
    parser.add_argument("--dim", type=int, default=10,
                        help="number of dimention for pca")
    parser.add_argument("--step", type=int, default=100,
                        help="number of latent steps for interpolation") # equal input morph step
    parser.add_argument("--partition", type=int, default=10,
                        help="number of division for simplex method") # step % partition == 0
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

    name = args.axis_name
    movie_path = f'{args.outdir}/movie_simplex/{name}'
    flip_path = f'{args.outdir}/flip_simplex/{name}'
    os.makedirs(movie_path, exist_ok=True)
    os.makedirs(flip_path, exist_ok=True)
    
    if args.dataset_name == "celeba":
        latent_dict = latent_dict_celeba
    else:
        raise ValueError("Unknown dataset name: f{args.dataset_name}")
    
    
    dim = args.dim
    
    with h5py.File(args.points, 'r') as f:
        pnts = f['thresholds_all'][:]
        pnts = np.transpose(np.array(pnts))

    with open(args.points_info, 'r') as f:
        json_data = json.load(f)
        idx2path_tr = [s.replace('\\', '/') for s in json_data['fullPathsUnique']]
        face2idx_tr = {k : v for k, v in zip(json_data['facialMapKeys'], json_data['facialMapValues'])}
    
    
    with h5py.File(args.fit_points, 'r') as f:
        fit_pnts  = f['thresholds_all'][:] 
        fit_pnts = np.transpose(np.array(fit_pnts))
    
    with h5py.File(args.axis, 'r') as f:
        v, d = f['v'][:], f['d'][:]
        v = np.array(v)
        d = np.array(d)
        
        d = d[1, 1] # TODO
        v = -v[1] #sample matlabでは列ベクトル TODO
    
    pca_pnts = pca(pnts, fit_pnts, dim)
    partition = args.partition
    step = args.step
    

    ax_len = np.sqrt(1 / d)
    e = v  * ax_len / partition * 2
    st = pca_pnts[face2idx_tr[name]]
    styles_st = torch.tensor(np.load(os.path.join(args.latent_indir, f'{name}.npy')), device=device0)
    
    b = 0
    c = 0
    styles_simplex_list=[styles_st]
    print(d, v)
    
    print(f'Serching for {dim}-simplex')
    for p in range(partition) :
        print(f'length : {np.linalg.norm(e, ord=2)}')
        e_st = st + e * p
        e_ed = st + e * (p + 1)
        
        x, x_inds = search_nearpnts(pca_pnts, e_ed)
        x_from_st = is_simplex(x, e_st, dim)
        gamma = clac_simplex(e, x_from_st, dim)
        
        # x, x_inds = plot3d(pca_pnts, x, x_inds, e_st, e_ed, dim)
        
        styles_x_list=[]
        for x_idx in x_inds :
            x_path = idx2path_tr[x_idx]
            x_path_list= x_path.split(os.sep)
            print(x_path_list[-2], x_path_list[-1], sep='/', end=' , ') if x_idx != x_inds[-1] else print(x_path_list[-2], x_path_list[-1], sep='/')
            latent_st_name, latent_ed_name = x_path_list[-2].split('_')
            latent_st_path =  os.path.join(args.latent_indir, f'{latent_st_name}.npy')
            latent_ed_path =  os.path.join(args.latent_indir, f'{latent_ed_name}.npy')
            
            styles_start = torch.tensor(
                np.load(latent_st_path), device=device0)
            styles_end = torch.tensor(
                np.load(latent_ed_path), device=device0)
            
            if styles_start.ndim == 2 and styles_end.ndim == 2:
                assert styles_start.size(1) == model.style_dim
                assert styles_end.size(1) == model.style_dim
                styles_start = styles_start.unsqueeze(
                    1).repeat(1, model.n_latent, 1)
                styles_end = styles_end.unsqueeze(1).repeat(1, model.n_latent, 1)
            
            stnum = styles_start.detach().cpu().numpy()
            _, b, c = stnum.shape
            styles_x = torch.empty((1, b, c), device=device0)
            
            composition_mask = torch.zeros(1, model.n_local, device=device0)
            composition_mask[:,0:6] = 1
            
            beta = (1 / (step - 1)) * (int(os.path.splitext(x_path_list[-1])[0]) - 1) #XXX sum = 1にしないといけない．
            
            for latent_index, _ in latent_dict.items():
                tmp = (1-beta) * styles_start[:, latent_index] + beta * styles_end[:, latent_index]
                styles_x[0, latent_index] = tmp
            
            styles_x_list.append(styles_x)
        
        styles_simplex = torch.empty((1, b, c), device=device0)
        
        for latent_index, _ in latent_dict.items():
            tmp = 0
            for g, styles_x in zip(gamma, styles_x_list) :
                tmp += g * styles_x[:, latent_index]
            tmp += (1 - sum(gamma)) * styles_st[:, latent_index]
            styles_simplex[0, latent_index] = tmp
        
        styles_st = styles_simplex
        styles_simplex_list.append(styles_simplex)
        
    print(f'Create Morphing {name}')
    itr = int(step / partition)
    with tqdm(total=itr * partition) as pbar: 
        styles_new = torch.empty((step, b, c), device=device0)
        for p in range(partition) :
            styles_start = styles_simplex_list[p]
            styles_end = styles_simplex_list[p + 1]
            for i in range(itr):
                alpha = (1/(itr-1))*i
                for latent_index, _ in latent_dict.items():
                    tmp = (1-alpha) * styles_start[:, latent_index] + alpha * styles_end[:, latent_index]
                    styles_new[i + p * itr, latent_index] = tmp
                style_image = torch.unsqueeze(styles_new[i + p * itr], dim=0)
                image, _ = mask_generate(model, style_image, randomize_noise=False, composition_mask=composition_mask)
                pbar.update()

                imageio.imwrite(f'{flip_path}/{(i + 1) + p * itr:03}.png', image[0])

    images, segs = mask_generate(model, styles_new, randomize_noise=False, batch_size=args.batch, composition_mask=composition_mask)
    frames = [np.concatenate((img, seg), 1)
                for (img, seg) in zip(images, segs)]
    
    fps = args.fps
    imageio.mimwrite(
        f'{movie_path}/{name}.mp4', images, fps=fps)
    imageio.mimwrite(
        f'{movie_path}/{name}.gif', images, fps=fps)
    imageio.mimwrite(
        f'{movie_path}/{name}_mask.mp4', frames, fps=fps)
