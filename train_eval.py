import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import torch
import argparse
import random
import numpy as np
from scipy.stats import pointbiserialr
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint  # Import ModelCheckpoint
from imblearn.over_sampling import SMOTE
from models.DySAT import my_model
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd

from datasets.fnirs import Depression

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
        torch.backends.cudnn.enabled = False

if __name__ == '__main__':
    dataset = Depression(data_dir=args.data_dir, selected=False, psm=False)

    MAX_EVALS = 1
    random_seed = 2024
    set_seed(random_seed)  # Fixed random seed for reproduce results reported in our paper.

    y = []
    for data in dataset:
        y.append(data['y'])
    y = np.array(y)

    # Training
    split_seg_counts = 4

    param_grid = {
        'learning_rate': [2.5e-4],   # 2.5e-4
        'batch_size': [16],
        'weight_decay': [1e-3],
        'epochs' : [60],
        'factor' : [0.5],
        'patience': [3],
        'loss_weight': [0.5],
        'p':[[0.2,0.2]],
        'selected_fre_num':[8],    # < 10  otherwise will raise error
        'hidden_dim':[4],
        'gcn_hidden_dim':[32],
    }
    # 记录用

    for i in range(MAX_EVALS):
        random.seed(i)  # 设置随机种子，每次搜索设置不同的种子，若种子固定，那每次选取的超参都是一样的
        hyperparameters = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
        print(hyperparameters)

        # hyperparams
        batch_size = hyperparameters['batch_size']
        lr = hyperparameters['learning_rate']
        wd = hyperparameters['weight_decay']
        epochs = hyperparameters['epochs']
        factor = hyperparameters['factor']
        patience = hyperparameters['patience']
        loss_weight = hyperparameters['loss_weight']
        p = hyperparameters['p']
        selected_fre_num = hyperparameters['selected_fre_num']
        hidden_dim = hyperparameters['hidden_dim']
        gcn_hidden_dim = hyperparameters['gcn_hidden_dim']

        ss = StratifiedShuffleSplit(n_splits=split_seg_counts, test_size=0.25, train_size=0.75, random_state=random_seed)
        fold_id = 0
        for train_index, test_index in ss.split(dataset, y):
            call_backs = []

            checkpoint_callback = ModelCheckpoint(
                monitor='val_accuracy',  # Replace with your validation metric
                filename='{epoch}-{val_accuracy:.3f}-{val_precision:.3f}-{val_recall:.3f}-{val_f1_score:.3f}',
                save_top_k=150,
                mode='max',  # 'min' for loss/error, 'max' for accuracy
            )

            call_backs.append(checkpoint_callback)


            test_dataset, train_dataset = [dataset[id] for id in test_index], [dataset[id] for id in train_index]
            test_id = [i['id'] for i in test_dataset]
            test_id = np.unique(np.array([int(i[1:]) for i in test_id]))
            hama = np.array(pd.read_excel('data/VFT/P001_P1527_HAMA.xlsx'))
            test_hama = hama[test_id - 1]

            smote = SMOTE(random_state=42)
            smote_x = np.arange(stop=y[train_index].shape[0])
            train_resample_id_list = list(np.squeeze(smote.fit_resample(smote_x[:, None], y[train_index])[0], axis=1))
            train_dataset = [train_dataset[id] for id in train_resample_id_list]
            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,drop_last=False)
            test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,drop_last=False)

            silent_list, task_list, other_list = [], [], []
            time_list = [silent_list, task_list, other_list]
            feature_list = ['dxy', 'oxy', 'total']
            time_period_list = ['s', 't', 'o']
            sf_f_list = ['cohe', 'corr']
            sf_hidden = 'cohe'
            for data in train_loader:
                for (time_idx, time) in enumerate(time_period_list):
                    tmp = []
                    for feature in feature_list:
                        for sf in sf_f_list:
                            tmp.append(data[f'{feature}_{time}_{sf}_sf'])
                    tmp = torch.stack(tmp,dim=1)
                    if sf_hidden == 'corr':
                        tmp[:, [0, 2, 4]] = 0
                    time_list[time_idx].append(tmp)
            silent_list = torch.cat(silent_list,dim=0)
            task_list = torch.cat(task_list,dim=0)
            other_list = torch.cat(other_list,dim=0)

            mean_list = []
            mean_list.append(silent_list.mean(0))
            mean_list.append(task_list.mean(0))
            mean_list.append(other_list.mean(0))
            # biseral weight prior to initialize trainable parameters
            biser_prior_dict = {}
            for f in feature_list:
                for t in time_period_list:
                    label, t_fft, t_fft_loc = [], [], []
                    for data in train_dataset:
                        label.append(data['y'])
                        t_fft.append(data[f'{f}_{t}_fft'])
                        t_fft_loc.append(data[f'{f}_{t}_fre_idx'])
                    t_fft, t_fft_loc = np.stack(t_fft, axis=0), np.stack(t_fft_loc, axis=0)
                    sample_num, select_amplitude, brain_channel = t_fft_loc.shape
                    max_amplitude = t_fft_loc.max() + 1
                    t_fft_total = np.zeros(shape=(sample_num, brain_channel, max_amplitude, 3))
                    t_fft_total[np.arange(sample_num)[:, None, None], np.arange(brain_channel)[None, :, None], np.transpose(t_fft_loc, (0, 2, 1))] = t_fft
                    label = np.array(label)
                    label = (label > 0).astype(int)

                    amplitude = t_fft_total[:, :, :, 0]
                    corr_matrix = np.zeros(shape=(brain_channel, max_amplitude))

                    for channel_idx in range(amplitude.shape[1]):
                        channel_amplitude = amplitude[:, channel_idx, :]
                        for feature_idx in range(channel_amplitude.shape[1]):
                            feature = channel_amplitude[:, feature_idx]
                            corr, _ = pointbiserialr(label, feature)
                            if not np.isnan(corr):
                                corr_matrix[channel_idx, feature_idx] = corr
                    corr_matrix = torch.tensor(corr_matrix)
                    biser_prior_dict[f'{f}_{t}_biserial_prior'] = corr_matrix


            model = my_model(hyper_params=param_grid,sfs_mean_list=mean_list,biser_prior=biser_prior_dict)

            # train and val
            # Training
            tensorboard_logger = TensorBoardLogger(save_dir='./',name='result', version=fold_id)
            trainer = pl.Trainer(max_epochs=epochs,devices=[2], strategy='ddp_find_unused_parameters_true', accelerator='gpu', logger=tensorboard_logger, callbacks=call_backs)
            trainer = pl.Trainer(max_epochs=epochs, accelerator='cpu',logger=tensorboard_logger, callbacks=call_backs)
            trainer.fit(model, train_loader, test_loader)

            # If you would like to load checkpoint to validate, please use trainer.validate() function.
            # trainer.validate(model, test_loader, ckpt_path='checkpoints/best.ckpt')
            fold_id += 1
            break
