import os

from pytorch_lightning.callbacks import ModelCheckpoint

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import torch
import argparse
import random
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger
from models.GCNs import GCN_Based
from datasets.fnirs import Depression
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedShuffleSplit

resting = False
if resting:
    file_name = 'resting'
else:
    file_name = 'VFT'
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_dir", default=f"data/{file_name}/preprocessed",help="Directory to extract data")
args = parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        pl.seed_everything(seed, workers=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    # Training
    dataset = Depression(data_dir=args.data_dir, selected=False, psm=False)
    MAX_EVALS = 1
    random_seed = 2024
    set_seed(random_seed)  # Fixed random seed for reproducing result in our paper.

    y = []
    for data in dataset:
        y.append(data['y'])
    y = np.array(y)

    split_seg_counts = 4

    param_grid = {
        'learning_rate': [7.5e-4],
        'batch_size': [4],
        'weight_decay': [1e-3],
        'step_size': [35],
        'epochs' : [100],
        'hidden_size' :[32],
        'loss_weight': [0.5],   # To promote recall of depression, reduce loss_weight!
        'dropout':[0.2],
    }

    # 记录用
    best_score = 0
    best_hyperparams = {}
    for i in range(MAX_EVALS):
        random.seed(i)  # 设置随机种子，每次搜索设置不同的种子，若种子固定，那每次选取的超参都是一样的
        hyperparameters = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
        print(hyperparameters)

        # hyperparams
        batch_size = hyperparameters['batch_size']
        lr = hyperparameters['learning_rate']
        wd = hyperparameters['weight_decay']
        epochs = hyperparameters['epochs']
        step_size = hyperparameters['step_size']
        hidden_size = hyperparameters['hidden_size']
        loss_weight = hyperparameters['loss_weight']
        dropout = hyperparameters['dropout']

        ss = StratifiedShuffleSplit(n_splits=split_seg_counts, test_size=0.25, train_size=0.75, random_state=random_seed)
        fold_id = 0
        for train_index, test_index in ss.split(dataset, y):
            call_backs = []

            checkpoint_callback = ModelCheckpoint(
                monitor='val_f1_score',  # Replace with your validation metric
                filename='{epoch}-{val_accuracy:.3f}-{val_precision:.3f}-{val_recall:.3f}-{val_f1_score:.3f}',
                save_top_k=150,
                mode='max',  # 'min' for loss/error, 'max' for accuracy
            )

            call_backs.append(checkpoint_callback)

            test_dataset, train_dataset = [dataset[id] for id in test_index], [dataset[id] for id in train_index]

            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
            test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

            mean_sfs = []
            feature_list = ['dxy', 'oxy', 'total']
            sf_f_list = ['cohe', 'corr']
            for data in train_loader:
                tmp = []
                for feature in feature_list:
                    for sf in sf_f_list:
                        tmp.append(data[f'{feature}_whole_{sf}_sf'])
                tmp = torch.stack(tmp, dim=1)
                mean_sfs.append(tmp)
            mean_sfs = torch.cat(mean_sfs, dim=0).mean(dim=0)

            model = GCN_Based(param_grid, mean_sfs)


            # train and val
            # Training
            tensorboard_logger = TensorBoardLogger(save_dir='./', name='baseline_gcn', version=fold_id)
            trainer = pl.Trainer(max_epochs=epochs, devices=[4,5,6,7], accelerator='gpu', logger=tensorboard_logger, callbacks=call_backs)
            trainer.fit(model, train_loader, test_loader)

            fold_id += 1



