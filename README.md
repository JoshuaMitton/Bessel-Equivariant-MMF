# Bessel-Equivariant-MMF

This is a PyTorch implementation of a Bessel equivariant model as described in our paper:

https://arxiv.org/abs/2207.12849

## Installation

```
conda create --name roteqseg python=3.8
conda activate mmf
python setup.py install
````

## Data

The dataset used in this project can be downloaded from <ToDo Add Link>.

Download the dataset into the besseleqmmf folder and unzip:

```
unzip BesselEqMMFData.zip
```

## Run the examples from our paper

```
cd besseleqmmf
mkdir outputs
```

For the experiments in the section using a real fibre the below commands train each of the models for fmnist, change the dataset to mnist to run these experiments.

```
python trainer_multimodel.py --savedir 'realfmnist_mlp' --dataset_name 'Real_fmnist' --model 'mlp' --img_size 224 --image_size_out 28 --epochs 200 --lr1 2.0 --lr2 0.01
python trainer_multimodel.py --savedir 'realfmnist_cmlp' --dataset_name 'Real_fmnist' --model 'cmlp' --img_size 224 --image_size_out 28 --epochs 200 --lr1 2.0 --lr2 0.01
python trainer_multimodel.py --savedir 'realfmnist_tmbases' --dataset_name 'Real_fmnist' --model 'tmbases' --img_size 224 --image_size_out 28 --epochs 200 --lr1 100.0 --lr2 0.01
python trainer_multimodel.py --savedir 'realfmnist_tmbases_SR' --loadTM 1 --loaddir 'realfmnist_tmbases' --dataset_name 'Real_fmnist' --model 'tmbases' --img_size 224 --image_size_out 28 --epochs 400 --lr1 100.0 --lr2 0.01
python trainer_multimodel.py --savedir 'realfmnist_tmbases_5offdiag' --blockdiag_mat 5 --dataset_name 'Real_fmnist' --model 'tmbases' --img_size 224 --image_size_out 28 --epochs 200 --lr1 200.0 --lr2 0.01
python trainer_multimodel.py --savedir 'realfmnist_tmbases_5offdiag_SR' --loadTM 1 --loaddir 'realfmnist_tmbases_5offdiag' --blockdiag_mat 5 --dataset_name 'Real_fmnist' --model 'tmbases' --img_size 224 --image_size_out 28 --epochs 400 --lr1 200.0 --lr2 0.01
python trainer_multimodel.py --savedir 'realfmnist_tmbases_10offdiag' --blockdiag_mat 10 --dataset_name 'Real_fmnist' --model 'tmbases' --img_size 224 --image_size_out 28 --epochs 200 --lr1 200.0 --lr2 0.01
python trainer_multimodel.py --savedir 'realfmnist_tmbases_10offdiag_SR' --loadTM 1 --loaddir 'realfmnist_tmbases_10offdiag' --blockdiag_mat 10 --dataset_name 'Real_fmnist' --model 'tmbases' --img_size 224 --image_size_out 28 --epochs 400 --lr1 200.0 --lr2 0.01
python trainer_multimodel.py --savedir 'realfmnist_tmbases_fullfibremat' --tm_full_mat 1 --dataset_name 'Real_fmnist' --model 'tmbases' --img_size 224 --image_size_out 28 --epochs 200 --lr1 25.0 --lr2 0.01
python trainer_multimodel.py --savedir 'realfmnist_tmbases_fullfibremat_SR' --loadTM 1 --loaddir 'realfmnist_tmbases_fullfibremat' --tm_full_mat 1 --dataset_name 'Real_fmnist' --model 'tmbases' --img_size 224 --image_size_out 28 --epochs 400 --lr1 200.0 --lr2 0.01
```

For the experiments in the section using a theoretical fibre the below commands train each of the models for fmnist, change the dataset to mnist to run these experiments.

```
python trainer_multimodel.py --savedir 'theoryfmnist_mlp' --dataset 'orig' --dataset_name 'TheoryTM_fmnist_V3' --model 'mlp' --img_size 180 --image_size_out 28 --epochs 200 --lr1 0.01 --lr2 0.01
python trainer_multimodel.py --savedir 'theoryfmnist_cmlp' --dataset 'orig' --dataset_name 'TheoryTM_fmnist_V3' --model 'cmlp' --img_size 180 --image_size_out 28 --epochs 200 --lr1 0.01 --lr2 0.01
python trainer_multimodel.py --savedir 'theoryfmnist_tmbases' --dataset 'orig' --dataset_name 'TheoryTM_fmnist_V3' --model 'tmbases' --img_size 180 --image_size_out 28 --epochs 200 --lr1 0.01 --lr2 0.01
python trainer_multimodel.py --savedir 'theoryfmnist_tmbases_SR' --loadTM 1 --loaddir 'theoryfmnist_tmbases' --dataset 'orig' --dataset_name 'TheoryTM_fmnist_V3' --model 'tmbases' --img_size 180 --image_size_out 28 --epochs 400 --lr1 0.01 --lr2 0.01
```

We also experiment by adding noise and reducing the number of bases.

```
--data_noise 0.01
--num_bases 931
```

For the experiments in the section using a theoretical fibre on imagenet data.

```
python trainer_multimodel.py --savedir 'theoryimagenet_tmbases' --dataset 'orig' --dataset_name 'TheoryTM_imagenette_grey' --model 'tmbases' --img_size 256 --image_size_out 256 --epochs 200 --lr1 0.5 --lr2 0.01
python trainer_multimodel.py --savedir 'theoryimagenet_tmbases_SR' --loadTM 1 --loaddir 'theoryimagenet_tmbases' --dataset 'orig' --dataset_name 'TheoryTM_imagenette_grey' --model 'tmbases' --img_size 256 --image_size_out 256 --epochs 400 --lr1 0.5 --lr2 0.01
```

We also provide three notebooks files runs_real.ipynb, runs_fmnist.ipynb, and runs_imagenet.ipynb, which perform the post processing we used to generate figures and numeric results used in the paper.

## Cite

Please cite our paper if you make use of our work:

```
@article{mitton2022bessel,
  title={Bessel Equivariant Networks for Inversion of Transmission Effects in Multi-Mode Optical Fibres},
  author={Mitton, Joshua and Mekhail, Simon Peter and Padgett, Miles and Faccio, Daniele and Aversa, Marco and Murray-Smith, Roderick},
  journal={arXiv preprint arXiv:2207.12849},
  year={2022}
}
```



