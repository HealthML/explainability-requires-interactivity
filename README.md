# Explainability Requires Interactivity

This repository contains the code to train all custom models used in the paper "Explainability Requires Interactivity," as well as to create all static explanations (heat maps and generative). For our interactive framework, see the [sister repositor](https://github.com/HealthML/StyleGAN2-Hypotheses-Explorer/).

Precomputed generative explanations are located at `static_generative_explanations`.

### Requirements

Install the conda environment via `conda env create -f env.yml` (depending on your system you might need to change some versions, e.g. for `pytorch`, `cudatoolkit` and `pytorch-lightning`).

For some parts you will need the FairFace model, which can be downloaded from the [authors' repo](https://github.com/dchen236/FairFace). You will only need the `res34_fair_align_multi_7_20190809.pt` file.

## Training classification networks

### CelebA dataset
You first need to download and decompress the [CelebAMask-HQ dataset](https://drive.google.com/open?id=1badu11NqxGf6qM3PTTooQDJvQbejgbTv) (or [here](https://github.com/switchablenorms/CelebAMask-HQ)). Then run the training with
```bash
python train.py --dset celeb --dset_path /PATH/TO/CelebAMask-HQ/ --classes_or_attr Smiling --target_path /PATH/TO/OUTPUT
```
`/PATH/TO/FLOWERS102/` should contain a `CelebAMask-HQ-attribute-anno.txt` file and an `CelebA-HQ-img` directory.
Any of the columns in `CelebAMask-HQ-attribute-anno.txt` can be used; in the paper we used `Heavy_Makeup`, `Male`, `Smiling`, and `Young`.


### Flowers102 dataset

You first need to download and decompress the [Flowers102 data](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz). Then run the training with
```bash
python train.py --dset flowers102 --dset_path /PATH/TO/FLOWERS102/ --classes_or_attr 49-65 --target_path /PATH/TO/OUTPUT/
```
`/PATH/TO/FLOWERS102/` should contain an `imagelabels.mat` file and an `images` directory.
Classes 49 and 65 correspond to the "Oxeye daisy" and "California poppy", while 63 and 54 correspond to "Black-eyed Susan" and "Sunflower" as in the paper.


## Generating heatmap explanations

Heatmap explanations are generated using the Captum library. After training, run explanations via
```bash
python static_exp.py --model_path /PATH/TO/MODEL.pt --img_path /PATH/TO/IMGS/ --model_name celeb --fig_dir /PATH/TO/OUTPUT/
```
`/PATH/TO/IMGS/` contains (only) image files and can be omitted in order to run the default images exported by `train.py`. To run on FairFace, choose `--model_name fairface` and add `--attr age` or `--attr gender`. Other explanation methods can be easily added by modifying the `explain_all` function in `static_exp.py`. Explanations are saved to `fig_dir`.
Only tested for the networks trained on the facial images data in the previous step, but any resnet18 with scalar output layer should work just as well.


## Generating generative explanations

First, clone the original [NVIDIA StyleGAN2-ada-pytorch repo](https://github.com/NVlabs/stylegan2-ada-pytorch/). Make sure everything works as expected (e.g. run the getting started code). If the code is stuck at loading TODO, usually `ctrl-C` will let the model fall back to a smaller reference implementation which is good enough for our use case.
Next, export the repo into your `PYTHONPATH` (e.g. via `export PYTHONPATH=$PYTHONPATH:/PATH/TO/stylegan2-ada-pytorch/`).
To generate explanations, you will need to 0) train an image model (see above, or use the FairFace model); 1) create a dataset of latent codes + labels; 2) train a latent space logistic regression models; and 3) create the explanations.
As each of the steps can be very slow, we split them up

### Create labeled latent dataset
First, make sure to either train at least one image model as in the first step and/or download the FairFace model.

```bash
python generative_exp.py --phase 1 --attrs Smiling,ff-skin-color --base_dir /PATH/TO/BASE/ --generator_path /PATH/TO/STYLEGAN2.pkl --n_train 20000 --n_valid 5000
```

The `base_dir` is the directory where all files/sub-directories are stored and should be the same as the `target_path` from `train.py` (e.g., just `.`). It should contain e.g. the `celeb-Smiling` directory and the `res34_fair_align_multi_7_20190809.pt` file if using `--attrs Smiling,ff-skin-color`.


### Train latent space model
After the first step, run
```bash
python generative_exp.py --phase 2 --attrs Smiling,ff-skin-color --base_dir /PATH/TO/BASE/ --epochs 50
```
with same `base_dir` and `attrs`.

### Create generative explanations
Finally, you can generate generative explanations via
```bash
python generative_exp.py --phase 3 --base_dir /PATH/TO/BASE/ --eval_attr Smiling --generator_path /PATH/TO/STYLEGAN2.pkl --attrs Smiling,ff-skin-color --reconstruction_steps 1000 --ampl 0.09 --input_img_dir /PATH/TO/IMAGES/ --output_dir /PATH/TO/OUTPUT/
```
Here, `eval_attr` is the final evaluation model's class that you want to explain; `attrs` are the same as before, the directions in latent space; `input_img_dir` is a directory with (only) image files that are to be explained. Explanations are saved to `output_dir`.

