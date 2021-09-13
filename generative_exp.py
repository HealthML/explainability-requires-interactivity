import os
from os.path import join
from functools import partial
from glob import glob
import pickle
import click

from tqdm import tqdm

import numpy as np
from PIL import Image, ImageDraw

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms, models
from torch.utils.data import TensorDataset, DataLoader

import pytorch_lightning as pl

from projector import project

FF_FN = 'res34_fair_align_multi_7_20190809.pt'

@click.command()
@click.option('--phase', help='Explanation phase; in [1, 2, 3]', default=1, type=int)
@click.option('--psi', help='Phase 1 only; truncation psi in StyleGAN2', default=1, type=float)
@click.option(
        '--attrs',
        help='All phases; comma-separated list of CelebA attributes for latent space training, e.g. "Smiling,Heavy_Makeup,ff-skin-color"',
        default='Smiling',
        )
@click.option('--n_train', help='Phase 1 only; training set size', default=20000, type=int)
@click.option('--n_valid', help='Phase 1 only; valid set size', default=5000, type=int)
@click.option('--base_dir', help='All phases; where to save and load model and data files', default='.')
@click.option('--epochs', help='Number of epochs to train the latent model', default=50, type=int)
@click.option('--generator_path', help='Phase 1 only; path to stylegan2 pkl', default=None)
@click.option('--output_dir', help='Phase 3 only; where to save output images', default='.')
@click.option('--input_img_dir', help='Phase 3 only; dir of images for which to generate explanations')
@click.option('--eval_attr', help='Phase 3 only; final evaluation model to be explained (not latent space direction', default='Smiling')
@click.option('--ampl', help='Phase 3 only; how far to move in each direction in latent space', default=0.09)
@click.option('--reconstruction_steps', help='Phase 3 only; number of steps for reconstruction', default=1000)
def main(
        phase,
        attrs: str,
        psi: float,
        base_dir: str,
        epochs: int,
        eval_attr: str,
        generator_path: str,
        n_train: int,
        n_valid: int,
        output_dir: str,
        input_img_dir: str,
        ampl: float,
        reconstruction_steps: int,
        ):
    if not os.path.isdir(base_dir):
        os.mkdir(base_dir)
    attrs = attrs.split(',')
    phase = int(phase)
    if phase == 1:
        print('phase 1 -- ', attrs)
        for attr in attrs:
            create_latent_attr(
                    generator_path=generator_path,
                    n_train=n_train,
                    n_valid=n_valid,
                    batch_size=100,
                    psi=psi,
                    attr=attr,
                    base_dir=base_dir,
                    )

    elif phase == 2:
        print('phase 2')
        for attr in attrs:
            if attr in ['ff-skin-color']:
                train_and_export_latent_fairface(base_dir=base_dir, epochs=epochs)
            else:
                train_and_export_latent_celeb(base_dir=base_dir, epochs=epochs, attr=attr)

    elif phase == 3:
        print('phase 3')
        explain_attr_celeb(
                base_dir=base_dir,
                generator_path=generator_path,
                cattr=eval_attr,
                input_img_dir=input_img_dir,
                output_dir=output_dir,
                attrs=attrs,
                num_steps=reconstruction_steps,
                ampl=ampl,
                )
    else:
        raise NotImplementedError(f'phase = {phase}')




def explain_attr_celeb(
        base_dir,
        generator_path,
        input_img_dir,
        cattr='Smiling',
        output_dir='.',
        attrs=['Male', 'Young', 'Heavy_Makeup', 'Smiling', 'ff-skin-color', 'Pale_Skin'],
        dev='cuda:0',
        num_steps=1000,
        ampl=0.15,
        ):
    is_age = cattr == 'age'
    pths = glob(join(input_img_dir, '*'))
    eval_model, eval_model_tfms = load_celeb(attr=cattr, base_dir=base_dir, dev=dev)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for attr in attrs:
        print(f'starting {attr}')
        for p in pths:
            print(f'img {p}')
            out_dir = join(output_dir, f'move_{attr}')
            if not os.path.isdir(out_dir):
                os.mkdir(out_dir)
            fn = f'canvas_{p.split("/")[-1].split(".")[0]}.jpg'
            reconstruct_and_move_celeb(
                    p,
                    base_dir=base_dir,
                    generator_path=generator_path,
                    gen_model=None,
                    eval_model=eval_model,
                    eval_model_tfms=eval_model_tfms,
                    num_steps=num_steps,
                    attr=attr,
                    dev=dev,
                    lo=-ampl,
                    hi=ampl,
                    out_dir=out_dir,
                    fn=fn,
                    is_age=is_age,
                    )

def reconstruct_and_move_celeb(
        img_fn,
        generator_path,
        base_dir='.',
        gen_model=None,
        eval_model=None,
        eval_model_tfms=None,
        num_steps=250,
        attr='Smiling',
        dev='cuda:0',
        steps=11,
        lo=-1,
        hi=1,
        out_dir='',
        fn='',
        is_age=False,
        ):
    img = transforms.ToTensor()(Image.open(img_fn))
    if gen_model is None:
        gen_model = load_generator(generator_path, dev)
    if img.shape[-2:] != (gen_model.img_resolution, gen_model.img_resolution):
        img = F.interpolate(img.view(1, *img.shape), size=gen_model.img_resolution)[0]
    latent = project(gen_model, 255*img, num_steps=num_steps, verbose=True, device=dev)[-1, :1, :]
    latent_model = load_latent_celeb(attr=attr, base_dir=base_dir, dev=dev)
    alpha = latent_model.weight.data
    alpha = latent.norm() * alpha / alpha.norm()
    grid = torch.linspace(lo, hi, steps).view(-1, 1).to(dev)
    print(grid.shape, latent.shape)
    latents = latent.to(dev) + grid * alpha
    with torch.no_grad():
        imgs = [
                gen_model.synthesis(
                    x[None].repeat(1, gen_model.mapping.num_ws, 1),
                    noise_mode='const',
                    ).cpu()[0]
                for x in latents]
    imgs = [transforms.ToPILImage()((0.5 * I + 0.5).clamp(0, 1)) for I in imgs]
    canvas = to_canvas(
            imgs,
            grid.flatten(),
            hw=256,
            eval_model=eval_model,
            eval_model_tfms=eval_model_tfms,
            is_age=is_age,
            )
    if out_dir and not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    if out_dir:
        canvas.save(join(out_dir, fn))
    return imgs, canvas

@torch.no_grad()
def to_canvas(
        imgs,
        grid,
        hw=None,
        eval_model=None,
        eval_model_tfms=None,
        mult=0.25,
        is_age=False,
        ):
    if hw is None:
        hw = imgs[0].size[0]
        resize = False
    else:
        resize = True

    buffer1 = 0 if eval_model is None else 20
    buffer2 = 0 if eval_model is None else 20
    buffer = buffer1 + buffer2

    num_imgs = len(imgs)
    canvas = Image.new('RGB', (hw * num_imgs, buffer + hw))

    for i, (g, img) in enumerate(zip(grid.flatten(), imgs)):
        if eval_model is not None:
            score = eval_model(eval_model_tfms(img)[None].cuda())
            if not is_age:
                score = torch.tanh(mult * score)
            score = score.item()
        if resize:
            img = img.resize((hw, hw))
        canvas.paste(img, (i*hw, buffer))
        if eval_model is not None:
            if is_age:
                max_age = 70
                color = (int(255 * score / max_age), 0, int(75 + 200 * score / max_age))
            else:
                color = (0, int(255*score), 0) if score > 0 else (int(-255*score), 0, 0)
            ImageDraw.Draw(canvas).rectangle(
                    [i*hw, 0, (i+1)*hw, buffer1],
                    fill=(255, 255, 255),
                    )
            ImageDraw.Draw(canvas).rectangle(
                    [i*hw, buffer1, (i+1)*hw, buffer],
                    fill=color,
                    )
            ImageDraw.Draw(canvas).text(
                    (i*hw + int(0.4*hw), 5),
                    f'{score:.2f}',
                    fill=color,
                    )

    return canvas



def load_celeb(attr, base_dir, dev='cuda:0'):
    if attr == 'ff-skin-color':
        attr = 'skin-color'

    if attr in ['skin-color', 'gender', 'age']:
        print('loading fairface model')
        model_path = join(base_dir, FF_FN)
        return setup_fairface(model_path, tp=attr)
    else:
        print('loading celeb model')
        sub_dir = f'celeb-{attr}'
        model_path = join(base_dir, sub_dir, 'discriminator.pt')
        return load_evaluator(model_path, dev)

def load_evaluator(evaluator_path, dev='cuda:0'):
    model = models.resnet18()
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        ])
    model = load_model(evaluator_path, sign='pos')
    model.eval().to(dev)
    return model, tfms

def load_model(
        evaluator_path,
        sign='pos',
        base_model=models.resnet18,
        output_units=1,
        ):
    model = base_model()
    model.fc = torch.nn.Linear(model.fc.in_features, output_units)
    if '.pt' in evaluator_path:
        ptm = evaluator_path
    else:
        ptm = join(evaluator_path, 'discriminator.pt')
    model.load_state_dict(torch.load(ptm, map_location='cpu'))
    if sign == 'neg':
        model.fc.weight.data = -model.fc.weight.data
        model.fc.bias.data = -model.fc.bias.data
    for x in model.modules():
        if hasattr(x, 'inplace'):
            x.inplace = False
    model.eval()
    return model

def load_generator(path_to_sg, dev):
    if path_to_sg is None: return None
    with open(path_to_sg, 'rb') as f:
        stylegan_model = pickle.load(f)['G_ema'].to(dev)
    return stylegan_model

def custom_fairface_head(fc, tp='gender'):
    of = fc.out_features
    with torch.no_grad():
        if tp == 'gender':
            last_layer = nn.Linear(of, 1, bias=False)
            last_layer.weight[:] = 0.
            last_layer.weight[0, 7] = 1
        if tp == 'age':
            L1 = nn.Linear(of, 9, bias=False)
            L1.weight[:] = 0.
            L1.weight[:, 9:18] = torch.eye(9)
            L2 = nn.Linear(9, 1, bias=False)
            L2.weight[0, :] = torch.tensor(
                    [1., 6., 15., 25., 35., 45., 55., 65., 75.]
                    )
            last_layer = nn.Sequential(L1, nn.Softmax(dim=-1), L2)
        if tp == 'skin-color':
            last_layer = nn.Linear(of, 1, bias=False)
            last_layer.weight[:] = 0.
            last_layer.weight[0, 1] = 1.
            last_layer.weight[0, 0] = -1.

    head = nn.Sequential(
            fc,
            last_layer,
            )
    return head
def setup_fairface(evaluator_path, tp='skin-color'):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tfms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        ])
    model = load_model(evaluator_path, sign='pos', output_units=18, base_model=models.resnet34)
    if tp is not None:
        model.fc = custom_fairface_head(model.fc, tp=tp)
    model = model.eval().cuda()
    return model, tfms

def train_and_export_latent_fairface(base_dir='.', epochs=50):
    dset_fn = join(base_dir, f'latent_dl_attr-ff-skin-color.pt')
    path = join(base_dir, f'trained_latent_attr-ff-skin-color.pt')
    M, tl, vl = train_latent_model(
            epochs=epochs,
            dset_fn=dset_fn,
            )
    eval_model(M, vl)
    M = M.model.cpu().eval()
    torch.save(M.state_dict(), path)

def train_and_export_latent_celeb(base_dir='.', epochs=50, attr='Smiling'):
    dset_fn = join(base_dir, f'latent_dl_attr-{attr}.pt')
    path = join(base_dir, f'trained_latent_attr-{attr}.pt')

    M, tl, vl = train_latent_model(
            epochs=epochs,
            dset_fn=dset_fn,
            )
    eval_model(M, vl)
    M = M.model.cpu().eval()
    torch.save(M.state_dict(), path)

def load_latent_celeb(attr='Smiling', base_dir='.', dev='cuda:0'):
    if attr == 'smile': attr = 'Smiling'
    pth = f'trained_latent_attr-{attr}.pt'
    M = nn.Linear(512, 1)
    M.load_state_dict(torch.load(join(base_dir, pth), map_location='cpu'))
    M = M.eval().to(dev)
    return M

def train_latent_model(
        epochs=10,
        dset_fn='dset.pt',
        eval_model=None,
        gen_model=None,
        tfms=None,
        n_train=None,
        n_valid=None,
        dev='cuda:0',
        ):
    '''
    provide either dset_fn or all the params required to create dset

    '''
    M = LatentModel().to(dev)
    tl, vl = create_latent(
            eval_model,
            gen_model,
            tfms=tfms,
            n_train=n_train,
            n_valid=n_valid,
            dump_file=dset_fn,
            reload=False,
            )
    trainer = pl.Trainer(
            gpus=1,
            max_epochs=epochs,
            )
    trainer.fit(M, tl, vl)
    return M, tl, vl
def eval_model(M, vl):
    ys = []
    ps = []
    for x, y in vl:
        p = M(x)
        ys.append(y)
        ps.append(p)
    ys = torch.cat(ys)
    ps = torch.cat(ps)
    bce_baseline = nn.BCEWithLogitsLoss()(ys.mean() * torch.ones_like(ys), ys).item()
    bce = nn.BCEWithLogitsLoss()(ps.flatten(), ys.flatten()).item()
    acc = (ps.sigmoid().round() == ys.round()).float().mean().item()
    acc_baseline = max(ys.round().mean(), 1 - ys.round().mean()).item()
    print(f'bce: {bce:.3f} (baseline: {bce_baseline:.3f}) -- acc: {acc:.3f} (baseline: {acc_baseline:.3f})')
    return ys, ps

def create_latent_attr(
        generator_path,
        n_train=20000,
        n_valid=5000,
        batch_size=100,
        dev='cuda:0',
        psi=0.8,
        attr='Smiling',
        base_dir='.',
        ):
    gen_model = load_generator(generator_path, dev)
    torch.cuda.empty_cache()
    print(f'starting: {attr}')
    eval_model, tfms = load_celeb(attr=attr, base_dir=base_dir, dev=dev)
    fn = join(base_dir, f'latent_dl_attr-{attr}.pt')
    create_latent(
            eval_model,
            gen_model,
            tfms=tfms,
            psi=psi,
            dev=dev,
            n_train=n_train,
            n_valid=n_valid,
            batch_size=batch_size,
            dump_file=fn,
            reload=False,
            gen_bs=100,
            )

def create_latent(
        eval_model,
        gen_model,
        tfms=None,
        psi=0.8,
        dev='cuda:0',
        n_train=1000,
        n_valid=1000,
        batch_size=100,
        dump_file='',
        reload=False,
        gen_bs=100,
        ):
    if not reload and os.path.isfile(dump_file):
        tl, vl = torch.load(dump_file)
        return tl, vl
    train = create_dset(n=n_train, bs=gen_bs, eval_model=eval_model, gen_model=gen_model, tfms=tfms, psi=psi, dev=dev)
    valid = create_dset(n=n_valid, bs=gen_bs, eval_model=eval_model, gen_model=gen_model, tfms=tfms, psi=psi, dev=dev)
    tl = DataLoader(train, batch_size=batch_size, shuffle=True)
    vl = DataLoader(valid, batch_size=batch_size, shuffle=False)
    if dump_file:
        torch.save([tl, vl], dump_file)
    return tl, vl

def create_dset(n, bs, eval_model, gen_model, tfms=None, psi=0.8, dev='cuda:0'):
    latents = []
    ys = []
    for i in tqdm(range(1 + n//bs)):
        w, y = create_batch(eval_model, gen_model, tfms=tfms, bs=bs, psi=psi, dev=dev)
        latents.append(w)
        ys.append(y)
    latents = torch.cat(latents)
    ys = torch.cat(ys).sigmoid()
    dset = TensorDataset(latents, ys)
    return dset

@torch.no_grad()
def create_batch(eval_model, gen_model, tfms=None, bs=12, psi=0.8, dev='cuda:0'):
    z = torch.randn(bs, gen_model.z_dim).to(dev)
    w = gen_model.mapping(z, None)
    w = gen_model.mapping.w_avg + psi * (w + gen_model.mapping.w_avg)
    imgs = torch.cat([gen_model.synthesis(s[None], noise_mode='const') for s in w])
    imgs = (0.5 * imgs + 0.5).clamp(0, 1)
    if tfms is not None:
        imgs = torch.stack([tfms(transforms.ToPILImage()(I)).to(dev) for I in imgs])
    labels = eval_model(imgs)
    return w[:, 0].cpu(), labels.cpu()


class LatentModel(pl.LightningModule):
    def __init__(self, lr=3e-4):
        super().__init__()
        self.lr = lr

        self.model = nn.Linear(512, 1)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.BCEWithLogitsLoss()(y_hat.flatten(), y.flatten())
        return loss

    def validation_step(self, batch, idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.BCEWithLogitsLoss()(y_hat.flatten(), y.flatten())
        p = (y_hat > 0).float()
        acc = (p.flatten() == y.round().flatten()).float().mean()
        self.log('valid_loss', loss, on_epoch=True, prog_bar=True)
        self.log('valid_acc', acc, on_epoch=True, prog_bar=True)
        return loss

if __name__ == '__main__':
    main()
