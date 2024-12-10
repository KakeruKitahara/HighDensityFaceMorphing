import argparse
import os
from tqdm import tqdm
import numpy as np
import imageio
import torch
from models import make_model
from visualize.utils import mask_generate, plot3d
import h5py
import json
from sklearn.decomposition import PCA
import random

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


def is_simplex(x, start, dim, args) : 
    eps = args.eps
    V = np.array([p - start for p in x]).T

    rank = np.linalg.matrix_rank(V)
    if rank != dim :
        return False, V, 0

    V_normal = V / np.linalg.norm(V, axis=0, ord=2)
    det = np.linalg.det(V_normal)
    if abs(det) < eps :
        return False, V, det

    return True, V, det

def clac_simplex(e, V, dim, args) :
    sup = args.sup
    inf = args.inf
    gamma = np.dot(np.linalg.inv(V), e)

    if not all(inf < g < sup for g in gamma) or not inf < 1 - sum(gamma) < sup:
        return False, gamma

    return True, gamma

def search_nearpnts(points, start, end, e, dim, args, G) :
    cand_num = args.candidate_number
    dis = np.linalg.norm(points - end, axis=1)
    x_inds_cand = list(np.argsort(dis)[1:cand_num + 1])
    att = 0
    lim = int(1e7)
    while att < lim :
        att += 1
        x_inds = random.sample(x_inds_cand, dim)
        x  = points[x_inds]

        bool_is, V , det = is_simplex(x, start, dim, args)
        if not bool_is :
            continue

        bool_clac, gamma = clac_simplex(e, V, dim, args)
        if not bool_clac :
            continue
        print(f'det : {det}')
        print(f'gamma : {gamma}, {1 - sum(gamma)}')
        print(f"dis : {d}")
        print(f'att : {att}')
        break
    else:
        raise Exception("Timeout serching near points.")

    return x, x_inds, gamma



if __name__ == '__main__':
    device0 = 'cuda:0'
    device1 = 'cuda:1'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str,
                        help="path to the model checkpoint")
    parser.add_argument('--axis', type=str, default=None,
                        help="direction vector in a low level space")
    parser.add_argument('--axis_name', type=str, default=None,
                        help="facial name of direction vector")
    parser.add_argument('--points', type=str, default=None,
                        help="matrix of facial images")
    parser.add_argument('--points_info', type=str, default=None,
                        help="json to connect facial matrix and id")
    parser.add_argument('--fit_points', type=str, default=None,
                        help="matrix of dimension-reduced facial images")
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
    parser.add_argument('--axis_number', type=int, default=1,
                        help="axis number to make the expression follow")
    parser.add_argument("--step", type=int, default=100,
                        help="number of latent steps for interpolation") #Equal input morph step
    parser.add_argument("--partition", type=int, default=10,
                        help="number of division for simplex method") #Dividing step by partition, the remainder should be zero.
    parser.add_argument("--fps", type=int, default=30,
                        help="number of fps for morphing")
    parser.add_argument("--dataset_name", type=str, default="celeba",
                        help="used for finding mapping between latent indices and names")
    parser.add_argument("--candidate_number", type=int, default=600,
                        help="number of candidate point")
    parser.add_argument("--eps", type=float, default=0.1,
                        help="threshold for well or ill-conditioned")
    parser.add_argument("--sup", type=float, default=1.2,
                        help="upper limit of simplex")
    parser.add_argument("--inf", type=float, default=-0.2,
                        help="lower limit of simplex")
    args = parser.parse_args()
    
    print(args)
    print("Loading model ...")
    ckpt = torch.load(args.ckpt)
    model = make_model(ckpt['args'], device0=device0, device1=device1)
    model.eval()
    model.load_state_dict(ckpt['g_ema'])
    
    dim = args.dim
    name = args.axis_name
    ax_num = args.axis_number
    
    fname = f'{name}_{ax_num}'
    movie_path = f'{args.outdir}/movie_simplex/{fname}'
    flip_path = f'{args.outdir}/flip_simplex/{fname}'
    os.makedirs(movie_path, exist_ok=True)
    os.makedirs(flip_path, exist_ok=True)
    
    if args.dataset_name == "celeba":
        latent_dict = latent_dict_celeba
    else:
        raise ValueError("Unknown dataset name: f{args.dataset_name}")
    
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
        G = f['G_tmp'][:]
        
    d, v = np.linalg.eig(G)
    d = d[ax_num]
    v = v[ax_num]
    
    pca_pnts = pca(pnts, fit_pnts, dim)
    partition = args.partition
    step = args.step
    

    ax_len = np.sqrt(1 / d)
    ax_step = v * ax_len / partition # 1step : 1/100*ax_len
    base_point = pca_pnts[face2idx_tr[name]]
    styles_st = torch.tensor(np.load(os.path.join(args.latent_indir, f'{name}.npy')), device=device0)

    b = 0
    c = 0
    styles_simplex_list = [styles_st]
    print(d, v)

    print(f'Serching for {dim}-simplex')
    for p in range(partition) :
        st = base_point + ax_step * p
        ed = base_point + ax_step * (p + 1)

        print(p + 1)
        x, x_inds, gamma = search_nearpnts(pca_pnts, st, ed, ax_step, dim, args, G)
        plot3d(pca_pnts, x, x_inds, st, ed, dim, 1, p, args)

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
            
            beta = (1 / (step - 1)) * (int(os.path.splitext(x_path_list[-1])[0]) - 1)
            
            for latent_index, _ in latent_dict.items():
                tmp = (1 - beta) * styles_start[:, latent_index] + beta * styles_end[:, latent_index]
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
        f'{movie_path}/{fname}.mp4', images, fps=fps)
    imageio.mimwrite(
        f'{movie_path}/{fname}.gif', images, fps=fps)
    imageio.mimwrite(
        f'{movie_path}/{fname}_mask.mp4', frames, fps=fps)
