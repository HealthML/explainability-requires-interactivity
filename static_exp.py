import os
from os.path import join
import numpy as np
from functools import partial
from glob import glob
import click

from PIL import Image
from matplotlib import pyplot as plt

import torch
from torchvision import transforms
from torch import nn

from captum import attr
from captum.attr import visualization as viz

import resnet_norepeat


@click.command()
@click.option('--model_path', help='path to model *.pt file (from train.py)', required=True)
@click.option('--fig_dir', help='Where to save explanations', required=True)
@click.option('--img_path', help='path to images; if none, searches for "images" in same dir as "model_path"', default=None)
@click.option('--model_name', default='celeb', help='Which model to use ("fairface" or "celeb")')
@click.option('--attr', default='age', help='For fairface, either "gender" or "age"; ignored if celeb')
@click.option('--figsize', default=(15, 5), help='Figure size', type=tuple)
def main(
        model_path: str,
        model_name: str,
        img_path: str,
        attr: None,
        fig_dir: str,
        figsize=(15, 5),
        ):
    assert model_name in ['fairface', 'celeb'], 'Only implemented for "fairface" and "celeb"'
    if img_path is None:
        img_path = join(os.sep.join(os.path.normpath(model_path).split(os.sep)[:-1]), 'images')
    if not os.path.isdir(fig_dir):
        os.mkdir(fig_dir)

    if model_name == 'fairface':
        assert attr in ['gender', 'age'], 'FairFace model attributes only implemented for "gender" and "age"'
        M, imgs, tensors = setup_fairface(model_path, img_path, tp=attr)
    else:
        M, imgs, tensors = setup_model(evaluator_path=model_path, img_path=img_path)

    A, methods = explain_all(M, tensors, cls=0)
    plot_all(A, imgs, method_names=methods, figsize=figsize, interp='bicubic', d=fig_dir)

def load_model(
        evaluator_path,
        base_model=resnet_norepeat.resnet18,
        output_units=1,
        ):
    model = base_model()
    model.fc = nn.Linear(model.fc.in_features, output_units)
    if '.pt' in evaluator_path:
        ptm = evaluator_path
    else:
        ptm = join(evaluator_path, 'discriminator.pt')
    model.load_state_dict(torch.load(ptm, map_location='cpu'))
    for x in model.modules():
        if hasattr(x, 'inplace'):
            x.inplace = False
    model.eval()
    return model

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
        
    head = nn.Sequential(
            fc,
            last_layer,
            )
    return head

def setup_fairface(evaluator_path, img_path, tp='gender'):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tfms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        ])
    model = load_model(evaluator_path, output_units=18, base_model=resnet_norepeat.resnet34)
    if tp is not None:
        model.fc = custom_fairface_head(model.fc, tp=tp)
    model.eval().cuda()
    img_p = glob(join(img_path, '*'))
    imgs = [Image.open(p) for p in img_p]
    tensors = torch.stack([tfms(img) for img in imgs])
    return model, imgs, tensors

def setup_model(evaluator_path, img_path=None):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        ])
    model = load_model(evaluator_path)
    model.eval()
    if img_path is not None:
        img_p = glob(join(img_path, '*'))
        imgs = [Image.open(p) for p in img_p]
    else:
        img_p = join(evaluator_path, 'images')
        imgs = [Image.open(join(img_p, f'c{c}_{i}.png')) for c in [1, 2] for i in range(20)]
    tensors = torch.stack([tfms(img) for img in imgs])
    return model, imgs, tensors


def explain_all(model, img, cls=0):
    methods = [
            (
                attr.IntegratedGradients,
                "Integrated Gradients",
                {'n_steps': 200, 'internal_batch_size': 2},
                ),
            # (
            #     lambda model: attr.NoiseTunnel(attr.IntegratedGradients(model)),
            #     "Noise IG",
            #     {'nt_samples': 200, 'nt_samples_batch_size': 1},
            #     ),
            # (attr.Saliency, "Saliency", {}),
            (
                lambda model: attr.NoiseTunnel(attr.Saliency(model)),
                "Saliency",
                {'nt_samples': 200, 'nt_samples_batch_size': 2, 'nt_type': 'smoothgrad_sq'},
                ),
            # (attr.DeepLift, "DeepLift", {}),
            # (attr.InputXGradient, "Input x Gradient", {}),
            (attr.GuidedBackprop, 'Guided Backprop', {}),
            (partial(attr.GuidedGradCam, layer=model.layer4[-1]), 'Guided GradCAM', {}),
            # (partial(attr.LayerGradCam, layer=model.layer4[-1]), 'GradCam', {}),
            # (partial(attr.LayerGradCam, layer=model.layer3[-1]), 'GradCam, l3', {}),
            # (partial(attr.LayerGradCam, layer=model.layer2[-1]), 'GradCam, l2', {}),
            # attr.KernelShap,
            # (attr.LRP, 'LRP', {}),
            # attr.DeepLiftShap,
            # attr.GradientShap,
            # attr.ShapleyValueSampling,
            # attr.Lime,
            ]
    attributes = []
    names = []
    for method, name, params in methods:
        names.append(name)
        torch.cuda.empty_cache()
        print(name)
        a = []
        for I in img:
            torch.cuda.empty_cache()
            at = explain_images(
                    model,
                    I[None],
                    attr_meth=method,
                    cls=cls,
                    params=params,
                    ).detach().cpu().permute(0, 2, 3, 1).numpy()
            a.append(at)
        a = np.concatenate(a)
        attributes.append(a)
    return attributes, names

def plot_all(
    attributes,
    imgs,
    method_names=None,
    sign='all',
    cmap='seismic',
    outlier_perc=2,
    figsize=(4, 12),
    interp='bilinear',
    d='figures',
    dpi=300,
    ):
    if sign == 'all':
        vmin, vmax = -1, 1
    else:
        vmin, vmax = 0, 1
    for i in range(len(imgs)):
        I = imgs[i]
        normed = [
                viz._normalize_image_attr(a[i], sign, outlier_perc)
                    if not np.allclose(a[i], 0) else np.ones_like(a[i])
                for a in attributes]

        fig, axes = plt.subplots(1, 1 + len(normed), figsize=figsize)
        for j, p in enumerate([np.array(I)/255.] + normed):
            if j > 0:
                name = method_names[j-1]
            else:
                name = 'original'
            axes[j].imshow(p, cmap=cmap, vmin=vmin, vmax=vmax, interpolation=interp)
            axes[j].set_title(name)
            axes[j].set_xticks([])
            axes[j].set_yticks([])
        plt.tight_layout()
        plt.show()
        fig.savefig(join(d, f'fig-{i}.jpg'), dpi=dpi)


def explain_images(model, img, attr_meth=attr.IntegratedGradients, dev='cuda:0', cls=0, params={}):
    model.to(dev)
    img = img.to(dev)
    explainer = attr_meth(model)
    img.requires_grad = True
    attribution = explainer.attribute(img, target=cls, **params)
    return attribution


def to_pil(tensors):
    return [
            Image.fromarray((
                    255*((x.permute(1, 2, 0) - x.min()) / (x.max() - x.min()))
                    ).detach().cpu().numpy().astype(np.uint8)
                )
            for x in tensors
            ]

if __name__ == '__main__':
    main()
