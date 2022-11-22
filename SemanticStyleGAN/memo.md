```
def cubic_spline_interpolate(styles, step):
    device = styles.device
    styles = styles.detach().cpu().numpy()
    N, K, D = styles.shape # (10, 28, 512)
    x = np.linspace(0.0, 1.0, N) # N等分等差数列
    y = styles.reshape(N,K*D) # (10, 28, 512) -> (10, 28 x 512)
    spl = CubicSpline(x, y) # 3Dスプライン補完．返り値は多項式
    x_out = np.linspace(0.0, 1.0, step) # --stepで事前に決める->FPS
    results = spl(x_out) # Step x KD
    results = results.reshape(step,K,D)
    return torch.tensor(results, device=device).float()
```
10は等分数 --sampleで指定
28は表情パーツ
512は画素数

n -> stepへスプライン補完

```
   if os.path.exists(args.outdir):
        shutil.rmtree(args.outdir)
    os.makedirs(args.outdir)

    print("Loading model ...")
    ckpt = torch.load(args.ckpt) # pt呼び出し
    model = make_model(ckpt['args'])
    model.to(args.device)
    model.eval()
    model.load_state_dict(ckpt['g_ema'])
    mean_latent = model.style(torch.randn(args.truncation_mean, model.style_dim, device=args.device)).mean(0) # 10000回のmeanをとる512

    print("Generating original image ...")
    with torch.no_grad(): # 最初のマスク
        if args.latent is None:
            styles = model.style(torch.randn(1, model.style_dim, device=args.device))
            styles = args.truncation * styles + (1-args.truncation) * mean_latent.unsqueeze(0)
        else:
            styles = torch.tensor(np.load(args.latent), device=args.device) # npy呼び出し (1, 28, 512)
        if styles.ndim == 2:
            assert styles.size(1) == model.style_dim
            styles = styles.unsqueeze(1).repeat(1, model.n_latent, 1)
        images, segs = generate(model, styles, mean_latent=mean_latent, randomize_noise=False)
        imageio.imwrite(f'{args.outdir}/image.jpeg', images[0])
        imageio.imwrite(f'{args.outdir}/seg.jpeg', segs[0])

    print("Generating videos ...")
    if args.dataset_name == "celeba":
        latent_dict = latent_dict_celeba
    else:
        raise ValueError("Unknown dataset name: f{args.dataset_name}")

    with torch.no_grad():
        for latent_index, latent_name in latent_dict.items():
            styles_new = styles.repeat(args.sample, 1, 1) # 潜在変数から(10, 28 x 1, 512 x 1)を作成
            mix_styles = model.style(torch.randn(args.sample, 512, device=args.device)) # 乱数からモデルを通して生成 (10, 512)
            mix_styles[-1] = mix_styles[0] # 先頭を後ろに代入ループ再生するため
            mix_styles = args.truncation * mix_styles + (1-args.truncation) * mean_latent.unsqueeze(0)  # 7 : 3で平滑化
            mix_styles = mix_styles.unsqueeze(1).repeat(1,model.n_latent,1)
            styles_new[:,latent_index] = mix_styles[:,latent_index] # 各々の10要素にlatent_indexの512の画素数を代入
            styles_new = cubic_spline_interpolate(styles_new, step=args.steps)
            images, segs = generate(model, styles_new, mean_latent=mean_latent, 
                        randomize_noise=False, batch_size=args.batch) # image : fps x moprh画素値(行列), segs : fps x mask画素値(行列)
            frames = [np.concatenate((img,seg),1) for (img,seg) in zip(images,segs)] # zip : 要素を同時取得
            imageio.mimwrite(f'{args.outdir}/{latent_index:02d}_{latent_name}.mp4', images, fps=20) # つなげてmp4
            print(f"{args.outdir}/{latent_index:02d}_{latent_name}.mp4")
```


```
def generate(model, styles, mean_latent=None, truncation=1.0, batch_size=16, *args, **kwargs):
    images, segs = [], []
    for head in range(0, styles.size(0), batch_size): # styles.size(0) : --step (160, 28, 512)
        images_, segs_ = model([styles[head:head+batch_size]], input_is_latent=True, 
        # [styles[head:head+batch_size]] : [styles[head] ~ styles[head + batch_size]]
                                    truncation=truncation, truncation_latent=mean_latent, *args, **kwargs)
        images.append(images_.detach().cpu())
        segs.append(segs_.detach().cpu())
    images, segs = torch.cat(images,0), torch.cat(segs,0)
    return tensor2image(images), tensor2seg(segs)

def tensor2image(tensor):
    images = tensor.cpu().clamp(-1,1).permute(0,2,3,1).numpy() # 軸(0, 1, 2, 3) -> (0, 2, 3, 1)に入れ替えて-1 ~ 1
    images = images * 127.5 + 127.5 # -1 ~ 1 -> 0 ~ 256
    images = images.astype(np.uint8)
    return images
```