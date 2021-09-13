from scipy import io
from os.path import join
import os
import click

from PIL import Image
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torch import nn
import torchvision.models as models
from torchvision import transforms

import pytorch_lightning as pl

torch.multiprocessing.set_sharing_strategy('file_system')

@click.command()
@click.option('--dset', help='which dataset to use', required=True)
@click.option('--dset_path', help='path to dataset; check README for details', required=True)
@click.option('--classes_or_attr', help='class names or celebA attribute. If class names, separate via "-", as in "49-65"', required=True)
@click.option('--target_path', default='.', help='where to output trained model', show_default=True)
@click.option('--freeze_epochs', default=5, help='number of training epochs with frozen convolutional body', show_default=True, type=int)
@click.option('--unfreeze_epochs', default=5, help='number of training epochs with unfrozen convolutional body', show_default=True, type=int)
def main(
        dset: str,
        dset_path: str,
        target_path: str,
        classes_or_attr,
        freeze_epochs: int,
        unfreeze_epochs: int,
        ):
    assert dset in ['celeb', 'flowers102'], 'Only valid dset options are "celeb" and "flowers102"; implement new Dataset class for custom datasets'
    assert os.path.isdir(dset_path), f'{dset_path} does not exist'
    if not os.path.isdir(target_path):
        os.mkdir(target_path)

    if dset == 'celeb':
        train_and_export_celeb_attr(
                d=dset_path,
                freeze_epochs=freeze_epochs,
                unfreeze_epochs=unfreeze_epochs,
                n_imgs=20,
                base=target_path,
                attr=classes_or_attr,
                )
    else:
        cls1, cls2 = [int(c) for c in classes_or_attr.split('-')]
        train_and_export_flowers(
                d=dset_path,
                cls1=cls1,
                cls2=cls2,
                freeze_epochs=freeze_epochs,
                unfreeze_epochs=unfreeze_epochs,
                n_imgs=20,
                base=target_path,
                )


def train(
        d,
        freeze_epochs=10,
        unfreeze_epochs=10,
        bs=10,
        cls1=73,
        cls2=74,
        nw=4,
        dset='flowers102',
        params=dict(),
        lr=3e-4,
        ):
    M = Model(n_targets=1, lr=lr)
    tl, vl = get_data(d=d, cls1=cls1, cls2=cls2, bs=bs, num_workers=nw, dset=dset, **params)

    if freeze_epochs > 0:
        M.freeze()
        trainer = pl.Trainer(
                gpus=1,
                max_epochs=freeze_epochs,
                )
        trainer.fit(M, tl, vl)
    if unfreeze_epochs > 0:
        M.unfreeze()
        trainer = pl.Trainer(
                gpus=1,
                max_epochs=unfreeze_epochs,
                )
        trainer.fit(M, tl, vl)
    return M, tl, vl

def export_model(M, path):
    M = M.model.cpu().eval()
    torch.save(M.state_dict(), path)


def get_data(
        d,
        cls1=73,
        cls2=74,
        size=256,
        bs=10,
        ratio=0.75,
        num_workers=4,
        dset='flowers102',
        attr='Smiling',
        ):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if dset == 'flowers102':
        pad = 200
        mode = 'edge'
        train_tfms = transforms.Compose([
            transforms.Pad(pad, padding_mode=mode),
            transforms.RandomRotation(180, expand=True),
            transforms.CenterCrop(1.8*size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            ])

        val_tfms = transforms.Compose([
            transforms.Pad(pad, padding_mode=mode),
            transforms.CenterCrop(1.8*size),
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            ])
        train_ds = Flowers102(cls1, cls2, train_tfms, d=d)
        val_ds = Flowers102(cls1, cls2, val_tfms, d=d)
        ind = torch.randperm(len(train_ds))
        m = int(len(ind)*ratio)
        train_ind = ind[:m]
        val_ind = ind[m:]
        train_sampler = SubsetRandomSampler(train_ind)
        val_sampler = SubsetRandomSampler(val_ind)

        train_loader = DataLoader(
                train_ds,
                batch_size=bs,
                sampler=train_sampler,
                num_workers=num_workers)
        val_loader = DataLoader(
                val_ds,
                batch_size=bs,
                sampler=val_sampler,
                num_workers=num_workers)
    elif dset == 'celeb':
        train_tfms = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        val_tfms = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_ds = CelebAttr(d=d, attr=attr, split='train', tfms=train_tfms)
        val_ds = CelebAttr(d=d, attr=attr, split='val', tfms=val_tfms)

        train_loader = DataLoader(
                train_ds,
                batch_size=bs,
                shuffle=True,
                num_workers=num_workers)
        val_loader = DataLoader(
                val_ds,
                batch_size=bs,
                shuffle=True,
                num_workers=num_workers)

    return train_loader, val_loader
    

class ImageDataset(Dataset):
    def __len__(self):
        return len(self.pths)

    def __getitem__(self, idx):
        img = Image.open(self.pths[idx])
        y = self.y[idx]
        if self.tfms is not None:
            img = self.tfms(img)
        return img, y


class CelebAttr(ImageDataset):
    def __init__(self, d, attr='Smiling', split='val', tfms=None):
        self.tfms = tfms
        df = (pd.read_csv(join(d, 'CelebAMask-HQ-attribute-anno.txt'), sep='\s+', skiprows=1)[attr] + 1) / 2
        print(f'baseline for attr {attr}: {df.mean():.4f}')
        df = df.sample(frac=1, random_state=42)
        nsplit = int(0.75 * len(df))
        if split == 'train':
            df = df.head(nsplit)
        else:
            df = df.tail(len(df) - nsplit)

        self.pths = [join(d, 'CelebA-HQ-img', p) for p in df.index]
        self.y = torch.from_numpy(df.values).float()

class CelebSmile(CelebAttr):
    def __init__(self, d, split='val', tfms=None):
        super().__init__(d=d, attr="Smiling", split=split, tfms=tfms)

class Flowers102(ImageDataset):
    def __init__(self, cls1, cls2, tfms=None, d='../data/flowers102/'):
        y = io.loadmat(join(d, 'imagelabels.mat'))['labels'].flatten()
        if cls2 is None:
            pth_cls1 = [
                    join(d, 'images', 'image_'+str(i+1).zfill(5) + '.jpg')
                    for i, yy in enumerate(y) if yy == cls1
                    ]
            self.pths = pth_cls1
            self.y = torch.cat([torch.zeros(len(pth_cls1))])
        else:
            pth_cls1 = [
                    join(d, 'images', 'image_'+str(i+1).zfill(5) + '.jpg')
                    for i, yy in enumerate(y) if yy == cls1
                    ]
            pth_cls2 = [
                    join(d, 'images', 'image_'+str(i+1).zfill(5) + '.jpg')
                    for i, yy in enumerate(y) if yy == cls2
                    ]
            self.pths = pth_cls1 + pth_cls2
            self.y = torch.cat([
                torch.zeros(len(pth_cls1)), torch.ones(len(pth_cls2))])
            print(f'baseline: {self.y.mean().item():.3f}')
        self.tfms = tfms

def export_images(loader, dir_path, n_imgs=5):
    dset = loader.dataset
    dset.tfms.transforms = dset.tfms.transforms[:-2]
    c1 = np.where(dset.y.numpy() == 0)[0]
    c2 = np.where(dset.y.numpy() == 1)[0]

    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    for i in range(n_imgs):
        img = dset[c1[i]][0]
        img.save(join(dir_path, f'c1_{i}.png'))

        img = dset[c2[i]][0]
        img.save(join(dir_path, f'c2_{i}.png'))


def train_and_export_flowers(
        d,
        cls1,
        cls2,
        freeze_epochs=30,
        unfreeze_epochs=5,
        n_imgs=20,
        base='.',
        ):
    M, tl, vl = train(
            d=d,
            freeze_epochs=freeze_epochs,
            unfreeze_epochs=unfreeze_epochs,
            cls1=cls1,
            cls2=cls2,
            dset='flowers102',
            )
    d = join(base, f'flowers102_c{cls1}-c{cls2}')
    if not os.path.isdir(d):
        os.mkdir(d)
    export_model(M, join(d, 'discriminator.pt'))
    export_images(vl, join(d, 'images'), n_imgs=n_imgs)

def train_and_export_celeb_attr(
        d,
        freeze_epochs=5,
        unfreeze_epochs=5,
        n_imgs=20,
        base='.',
        attr='Heavy_Makeup',
        ):
    M, tl, vl = train(
            d=d,
            freeze_epochs=freeze_epochs,
            unfreeze_epochs=unfreeze_epochs,
            dset='celeb',
            params={'attr': attr},
            )
    d = join(base, f'celeb-{attr}')
    if not os.path.isdir(d):
        os.mkdir(d)
    export_model(M, join(d, 'discriminator.pt'))
    export_images(vl, join(d, 'images'), n_imgs=n_imgs)

class Model(pl.LightningModule):
    def __init__(self, n_targets=10, lr=3e-4):
        super().__init__()
        self.lr = lr
        
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 1)
        self.freeze()

    def freeze(self):
        for mod in self.model.modules():
            mod.requires_grad_(False)
        self.model.fc.requires_grad_(True)

    def unfreeze(self):
        for mod in self.model.modules():
            mod.requires_grad_(True)

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
        acc = (p.flatten() == y.flatten()).float().mean()
        self.log('valid_loss', loss, on_epoch=True, prog_bar=True)
        self.log('valid_acc', acc, on_epoch=True, prog_bar=True)
        return loss

if __name__ == '__main__':
    main()
