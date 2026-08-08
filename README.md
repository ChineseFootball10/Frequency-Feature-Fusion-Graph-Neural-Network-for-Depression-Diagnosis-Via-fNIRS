# Frequency-Feature-Fusion-Graph-Neural-Network-for-Depression-Diagnosis-Via-fNIRS
Official implementation of paper 'Frequency Feature Fusion Graph Neural Network for Depression Diagnosis Via fNIRS'.

Chengkai Yang, Xingping Dong, Xiaofen Zong, Mang Ye

News
====
[2026-08-08] Code for paper 'Frequency Feature Fusion Graph Neural Network for Depression Diagnosis Via fNIRS' is released.

⚙ Installation
============
## 1) Clone
    git clone https://github.com/ChineseFootball10/Frequency-Feature-Fusion-Graph-Neural-Network-for-Depression-Diagnosis-Via-fNIRS.git
    cd Frequency-Feature-Fusion-Graph-Neural-Network-for-Depression-Diagnosis-Via-fNIRS

## 2) Create Environment
    conda create -n fnirs python=3.8 -y
    conda activate fnirs
    pip install -r requirements.txt

We strongly recommend installing torch_geometric, torch cluster, torch_scatter, torch_sparse, torch_spline_conv by wheels in https://data.pyg.org/whl/ corresponding to your cuda version and pytorch version.

🔢 Dataset Preparing
=================
## 1) Put fNIRS .mat files under /data folders.
    data/
        |--VFT/
        |   |--N_group/
        |   |--|--N1.mat
        |   |--|--N2.mat
        |   |--|--...
        |   |--P_group/
        |   |--|--P1.mat
        |   |--|--P2.mat
        |   |--|--...
        |   |--nFIRS.xlsx
## 2) Preprocess raw fNIRS .mat data
    python preprocess/preprocess_vft.py -c configs/preprocess_dataset.yml -r xxx -d xxx
This operation preprocesses frequency feature of TGCN brain channel nodes and edges.
Replace "-r xxx" with your dataset dir and "-d xxx" with the folder you would like to store processed fnirs data. Then the preprocessed data will be stored as ".pkl" files under root -d.

🔥Training (Quick Start)
======================
## 1) To train our freqency-based TGCN model, please run script
    python train_eval.py 
If use gpu for training, you should assign "devices" in pl.Trainer() function.

## 2) To train baseline GNN model, please run script
    python train_eval_gnn_baseline.py

## 3) To train baseline machine learning models, please run script
    python train_eval_ml_baselines.py


