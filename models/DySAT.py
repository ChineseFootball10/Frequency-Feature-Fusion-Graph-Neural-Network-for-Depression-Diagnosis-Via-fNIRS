import pickle
from collections import OrderedDict
import numpy as np
import chardet
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch_geometric.nn import GCNConv
import torchmetrics
import matplotlib.pyplot as plt
import os

def weight_init(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        fan_in = m.in_channels / m.groups
        fan_out = m.out_channels / m.groups
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        nn.init.uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight is not None:
            fan_in = m.embed_dim
            fan_out = m.embed_dim
            bound = (6.0 / (fan_in + fan_out)) ** 0.5
            nn.init.uniform_(m.in_proj_weight, -bound, bound)
        else:
            nn.init.xavier_uniform_(m.q_proj_weight)
            nn.init.xavier_uniform_(m.k_proj_weight)
            nn.init.xavier_uniform_(m.v_proj_weight)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        nn.init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)
        if m.bias_k is not None:
            nn.init.normal_(m.bias_k, mean=0.0, std=0.02)
        if m.bias_v is not None:
            nn.init.normal_(m.bias_v, mean=0.0, std=0.02)
    elif isinstance(m, (nn.LSTM, nn.LSTMCell)):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(4, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(4, 0):
                    nn.init.orthogonal_(hh)
            elif 'weight_hr' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)
                nn.init.ones_(param.chunk(4, 0)[1])
    elif isinstance(m, (nn.GRU, nn.GRUCell)):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(3, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(3, 0):
                    nn.init.orthogonal_(hh)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)



class my_model(pl.LightningModule):
    # This is the class of our frequency based TGCN model.
    def __init__(self, hyper_params, sfs_mean_list, biser_prior):
        super(my_model, self).__init__()
        # hyper_params
        self.p = hyper_params['p'][0]
        self.selected_fre_num = hyper_params['selected_fre_num'][0]
        self.hidden_dim = hyper_params['hidden_dim'][0]
        self.gcn_hidden_dim = hyper_params['gcn_hidden_dim'][0]
        self.loss_weight = hyper_params['loss_weight'][0]

        self.sfs_mean_list = sfs_mean_list

        fft_emb_model_s = Position_Encodings(self.p[0], self.selected_fre_num, self.hidden_dim)
        fft_emb_model_t = Position_Encodings(self.p[0], self.selected_fre_num, self.hidden_dim)
        fft_emb_model_o = Position_Encodings(self.p[0], self.selected_fre_num, self.hidden_dim)
        self.fft_emb_model_list = nn.Sequential(fft_emb_model_s, fft_emb_model_t, fft_emb_model_o)

        gcn_model_s = GCN_block(self.p[0], self.selected_fre_num * self.hidden_dim + 20, self.gcn_hidden_dim)
        gcn_model_t = GCN_block(self.p[0], self.selected_fre_num * self.hidden_dim + 20, self.gcn_hidden_dim)
        gcn_model_o = GCN_block(self.p[0], self.selected_fre_num * self.hidden_dim + 20, self.gcn_hidden_dim)
        self.gcn_model_list = nn.Sequential(gcn_model_s, gcn_model_t, gcn_model_o)

        self.rnn1 = nn.GRU(input_size=53 * 2 * 3,hidden_size=64)   # Or Transformer
        self.norm1 = nn.LayerNorm(64)
        self.dropout1 = nn.Dropout(p=self.p[1])
        self.fc1 = nn.Linear(in_features=64,out_features=2)

        self.prior_weights = nn.ParameterDict()
        time_period = ['s', 't', 'o']
        feature_list = ['dxy', 'oxy', 'total']
        for f in feature_list:
            for t in time_period:
                initial_weight_fill = biser_prior[f'{f}_{t}_biserial_prior'].float()
                initial_weight = torch.zeros(size=(initial_weight_fill.shape[0], 30))
                initial_weight[:, :initial_weight_fill.shape[1]] = initial_weight_fill
                self.prior_weights[f'{f}_{t}'] = nn.Parameter(data=initial_weight, requires_grad=True)

        self.apply(weight_init)

        self.train_accuracy = torchmetrics.classification.Accuracy(num_classes=2)
        self.train_conf_mat = torchmetrics.classification.ConfusionMatrix(num_classes=2)
        self.train_f1_score = torchmetrics.classification.F1Score(num_classes=2, average='none')
        self.train_precision = torchmetrics.classification.Precision(num_classes=2, average='none')
        self.train_recall = torchmetrics.classification.Recall(num_classes=2, average='none')

        self.test_accuracy = torchmetrics.classification.Accuracy(num_classes=2)
        self.test_conf_mat = torchmetrics.classification.ConfusionMatrix(num_classes=2)
        self.test_f1_score = torchmetrics.classification.F1Score(num_classes=2, average='none')
        self.test_precision = torchmetrics.classification.Precision(num_classes=2, average='none')
        self.test_recall = torchmetrics.classification.Recall(num_classes=2, average='none')

        # Record results
        self.validation_result = []
    def forward(self, *data):
        if len(data) == 1 and isinstance(data[0], dict):
            data = data[0]
            time_period = ['s', 't', 'o']
            feature_list = ['dxy', 'oxy', 'total']
            dysat_dict = {}
            for time_id, time in enumerate(time_period):
                for feature in feature_list:
                    tfs_scale = data[f'{feature}_tf'].chunk(chunks=len(time_period), dim=-1)[time_id]
                    ffts = data[f'{feature}_{time}_fft'][:,:,:self.selected_fre_num,:]
                    fre_id = data[f'{feature}_{time}_fre_idx'][:,:self.selected_fre_num,:]
                    dysat_dict[f'{feature}_{time}'] = self.fft_emb_model_list[time_id](tfs_scale, ffts, fre_id, self.prior_weights[f'{feature}_{time}'])
                    # dysat_dict[f'{feature}_{time}'] = self.fft_emb_model_list[time_id](tfs_scale, ffts, fre_id, None)
            vft_time_seq = []
            for time_id, time in enumerate(time_period):
                tfs, sfs = [], []
                for feature in feature_list:
                    # if feature == 'dxy':
                    #     tfs.append(torch.zeros_like(dysat_dict[f'{feature}_{time}']))
                    # else:
                    tfs.append(dysat_dict[f'{feature}_{time}'])
                    for statistic in ['cohe', 'corr']:
                        sfs.append(data[f'{feature}_{time}_{statistic}_sf'])
                tfs = torch.stack(tfs, axis=1)
                sfs = torch.stack(sfs, axis=1)
                time_period_out = self.gcn_model_list[time_id](tfs, sfs, self.sfs_mean_list[time_id].to(tfs.device))
                vft_time_seq.append(time_period_out)
            vft_time_seq = torch.stack(vft_time_seq, dim=0)
            _, rnn_out = self.rnn1(vft_time_seq)
            rnn_out = self.norm1(rnn_out.squeeze(0))
            rnn_out = self.dropout1(rnn_out)
            rnn_out = self.fc1(rnn_out)   # [B, 2]
            return rnn_out
        else:  # for SHAP
            time_period = ['s', 't', 'o']
            feature_list = ['dxy', 'oxy', 'total']
            dysat_dict = {}
            data_ids_count = 0
            for time_id, time in enumerate(time_period):
                for feature in feature_list:
                    tfs_scale = data[data_ids_count].chunk(chunks=len(time_period), dim=-1)[time_id]
                    data_ids_count += 1
                    ffts = data[data_ids_count][:, :, :self.selected_fre_num, :]
                    data_ids_count += 1
                    fre_id = data[data_ids_count][:, :self.selected_fre_num, :]
                    data_ids_count += 1
                    dysat_dict[f'{feature}_{time}'] = self.fft_emb_model_list[time_id](tfs_scale, ffts, fre_id,self.prior_weights[f'{feature}_{time}'])

            vft_time_seq = []
            for time_id, time in enumerate(time_period):
                tfs, sfs = [], []
                for feature in feature_list:
                    tfs.append(dysat_dict[f'{feature}_{time}'])
                    for statistic in ['cohe', 'corr']:
                        sfs.append(data[data_ids_count])
                        data_ids_count += 1
                tfs = torch.stack(tfs, axis=1)
                sfs = torch.stack(sfs, axis=1)
                time_period_out = self.gcn_model_list[time_id](tfs, sfs, self.sfs_mean_list[time_id].to(tfs.device))
                vft_time_seq.append(time_period_out)
            vft_time_seq = torch.stack(vft_time_seq, dim=0)
            _, rnn_out = self.rnn1(vft_time_seq)
            rnn_out = self.norm1(rnn_out.squeeze(0))
            rnn_out = self.dropout1(rnn_out)
            rnn_out = F.softmax(self.fc1(rnn_out), dim=-1)  # [B, 2]
            return rnn_out


    def training_step(self, batch, batch_idx):
        y = batch['y'].long()
        y = y.masked_fill(y > 0, 1)
        y_hat = self(batch)

        # focal loss coeficcient
        loss = F.cross_entropy(y_hat, y, reduction='none')
        c = y_hat[:, 0]
        c = torch.masked_scatter(input=c, mask=(y == 1), source=y_hat[y==1, 1])
        c = 1.0 - c
        alpha = torch.full_like(loss, fill_value=self.loss_weight)
        alpha = torch.masked_fill(alpha, mask=(y == 1), value=1.0-self.loss_weight)
        loss = loss * (c ** 1.5) * alpha
        loss = loss.mean()

        self.train_conf_mat.update(y_hat, y)
        self.train_accuracy.update(y_hat, y)
        self.train_precision.update(y_hat, y)
        self.train_recall.update(y_hat, y)
        self.train_f1_score.update(y_hat, y)

        self.log('train_loss', loss, on_step=True)
        self.log('train_accuracy', self.train_accuracy, on_step=False, on_epoch=True)
        self.log('train_precision', self.train_precision.compute()[1], on_step=False, on_epoch=True)
        self.log('train_recall', self.train_recall.compute()[1], on_step=False, on_epoch=True)
        self.log('train_f1_score', self.train_f1_score.compute()[1], on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        y = batch['y'].long()
        y = y.masked_fill(y > 0, 1)
        y_hat = self.forward(batch)
        test_id = np.array([int(i[1:]) for i in batch['id']]) - 1
        classify_probaility = y_hat.cpu().detach().numpy()
        test_result = np.concatenate([test_id[:, None], classify_probaility], axis=-1)
        self.validation_result.append(test_result)
        # focal loss coeficcient
        loss = F.cross_entropy(y_hat, y, reduction='none')
        c = y_hat[:, 0]
        c = c.masked_scatter((y == 1), source=y_hat[y==1, 1])
        c = 1.0 - c
        alpha = torch.full_like(loss, fill_value=self.loss_weight)
        alpha = torch.masked_fill(alpha, mask=(y == 1), value=1.0-self.loss_weight)
        loss = loss * (c ** 1.5) * alpha
        loss = loss.mean()

        if not self.trainer.sanity_checking:
            self.test_conf_mat.update(y_hat, y)
            self.test_accuracy.update(y_hat, y)
            self.test_precision.update(y_hat, y)
            self.test_recall.update(y_hat, y)
            self.test_f1_score.update(y_hat, y)

            self.log('val_loss', loss, on_step=True, batch_size=4)
            self.log('val_accuracy', self.test_accuracy, on_step=False, on_epoch=True, batch_size=4)
            self.log('val_precision', self.test_precision.compute()[1], on_step=False, on_epoch=True, batch_size=4)
            self.log('val_recall', self.test_recall.compute()[1], on_step=False, on_epoch=True, batch_size=4)
            self.log('val_f1_score', self.test_f1_score.compute()[1], on_step=False, on_epoch=True, batch_size=4)


    def on_train_epoch_end(self):
        conf_matrix = self.train_conf_mat.compute()
        self.train_conf_mat.reset()

        accuracy_computed = self.train_accuracy.compute()
        self.train_accuracy.reset()

        precision_computed = self.train_precision.compute()
        self.train_precision.reset()

        recall_computed = self.train_recall.compute()
        self.train_recall.reset()

        f1_computed = self.train_f1_score.compute()
        self.train_f1_score.reset()

        # if self.trainer.is_global_zero:
        #     print(f'\nConf_Matrix={conf_matrix}')
        #     print(f'\nTraining Accuracy={accuracy_computed}')

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            conf_matrix = self.test_conf_mat.compute()
            self.test_conf_mat.reset()


            accuracy_computed = self.test_accuracy.compute()
            self.test_accuracy.reset()


            precision_computed = self.test_precision.compute()
            self.test_precision.reset()


            recall_computed = self.test_recall.compute()
            self.test_recall.reset()


            f1_computed = self.test_f1_score.compute()
            self.test_f1_score.reset()

            self.validation_result = np.concatenate(self.validation_result, axis=0)
            index = np.unique(self.validation_result[:, 0], return_index=True)[1]
            self.validation_result = self.validation_result[index]
        # if self.trainer.is_global_zero:
        #     print(f'\nConf_Matrix={conf_matrix}')
        #     print(f'\nValidation Accuracy={accuracy_computed}')


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3, eps=1e-4)
        scheduler = MultiStepLR(optimizer, milestones=[30,60,90], gamma=0.5, verbose=True)
        return [optimizer], [scheduler]



class Position_Encodings(nn.Module):
    def __init__(self,p,selected_fre_num,hidden_dim):
        super(Position_Encodings, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Sequential(OrderedDict([
            ('linear',nn.Linear(in_features=3,out_features=self.hidden_dim)),
            # ('activate',nn.ELU())
        ]))
        self.dropout1 = nn.Dropout(p=p)
        self.fc2 = nn.Sequential(OrderedDict([
            ('linear',nn.Linear(in_features=6,out_features=20)),
            # ('activate',nn.ELU())
        ]))
        self.dropout2 = nn.Dropout(p=p)
        self.norm = nn.LayerNorm(selected_fre_num * self.hidden_dim + 20)
    def forward(self, tfs, ffts, fft_id, prior_weight, shap=False):
        fft_id = fft_id.long()
        tfs,ffts = tfs.float(),ffts.float()
        # tfs shape:(batch_size,channel_nums,feature_dim(6))
        # ffts shape:(batch_size,channel_nums, selected_fre_num, 3)
        # fft_id shape:(selected_fre_num ,channel_nums)
        out = []
        if shap:
            fft_list = []
        position_emb = self.position_emb(fft_id, d_model=self.hidden_dim)
        batch_ids = 0
        for (tf,fft) in zip(tfs,ffts):
            try:
                fft_emb = F.elu(self.fc1(fft))    # (channel_nums,selected_fre_num,18)
            except:
                print(f'tf.shape {fft.shape}')
                print(f'self.fc1.weight.shape {self.fc1[0].weight.shape}')
                print(f'self.fc1.bias.shape {self.fc1[0].bias.shape}')
            fft_emb = fft_emb + position_emb[batch_ids]
            if prior_weight is not None:
                batch_fft_id = fft_id[batch_ids]
                batch_prior_weight = prior_weight[torch.arange(batch_fft_id.shape[1])[:, None], batch_fft_id.transpose(0, 1)]
                fft_emb = fft_emb * batch_prior_weight.unsqueeze(-1)
            fft_emb = self.dropout1(fft_emb)
            batch_ids += 1
            fft_emb = torch.flatten(fft_emb,start_dim=1,end_dim=-1)   # (channel_nums,selected_fre_num * hidden_dim)
            if shap:
                fft_list.append(fft_emb)
            try:
                tf_emb = F.elu(self.fc2(tf))     # (channel_nums,20)
            except:
                print(f'tf.shape {tf.shape}')
                print(f'self.fc2.weight.shape {self.fc2[0].weight.shape}')
                print(f'self.fc2.bias.shape {self.fc2[0].bias.shape}')
            tf_emb = self.dropout2(tf_emb)
            tfs = torch.cat([fft_emb,tf_emb],dim=-1)   # (channel_nums,selected_fre_num * hidden_dim + 20)
            out.append(tfs)
        out = torch.stack(out,dim=0)
        out = self.norm(out)
        if not shap:
            return out
        else:
            fft_list = torch.stack(fft_list, dim=0)
            return out, fft_list

    def position_emb(self,fft_id,d_model):
        # fft_id shape (6,channel_nums)
        pos = 1e4 ** (torch.repeat_interleave(torch.arange(d_model // 2),repeats=2,dim=0) * 2 / d_model).to(fft_id.device)
        pos = torch.div(fft_id.permute(0,2,1).unsqueeze(-1),pos.reshape(1,1,1,pos.shape[0]))
        pos = pos.reshape(pos.shape[0],pos.shape[1],pos.shape[2],-1,2)
        pos[:,:,:,:,0] = torch.sin(pos[:,:,:,:,0])
        pos[:,:,:,:,1] = torch.cos(pos[:,:,:,:,1])
        pos = pos.flatten(start_dim=-2,end_dim=-1)
        return pos

class GCN_block(nn.Module):
    def __init__(self, p, input_dim, hidden_dim):
        super(GCN_block, self).__init__()
        self.conv_list_1 = nn.Sequential(OrderedDict([
            ('GCNConv1_0', GCNConv(in_channels=input_dim, out_channels=hidden_dim)),
            ('GCNConv1_1', GCNConv(in_channels=input_dim, out_channels=hidden_dim)),
            ('GCNConv1_2', GCNConv(in_channels=input_dim, out_channels=hidden_dim)),
            ('GCNConv1_3', GCNConv(in_channels=input_dim, out_channels=hidden_dim)),
            ('GCNConv1_4', GCNConv(in_channels=input_dim, out_channels=hidden_dim)),
            ('GCNConv1_5', GCNConv(in_channels=input_dim, out_channels=hidden_dim)),
        ]))
        self.norm_1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(p=p)

        self.conv_list_2 = nn.Sequential(OrderedDict([
            ('GCNConv2_0', GCNConv(in_channels=hidden_dim, out_channels=1)),
            ('GCNConv2_1', GCNConv(in_channels=hidden_dim, out_channels=1)),
            ('GCNConv2_2', GCNConv(in_channels=hidden_dim, out_channels=1)),
            ('GCNConv2_3', GCNConv(in_channels=hidden_dim, out_channels=1)),
            ('GCNConv2_4', GCNConv(in_channels=hidden_dim, out_channels=1)),
            ('GCNConv2_5', GCNConv(in_channels=hidden_dim, out_channels=1)),
        ]))
        self.norm_2 = nn.LayerNorm(53)
        self.dropout2 = nn.Dropout(p=p)

        self.register_buffer(name='edge_index', tensor=torch.cat(
            [torch.repeat_interleave(torch.arange(53), repeats=53).unsqueeze(0), \
             torch.arange(53).repeat(53).unsqueeze(0)], dim=0), persistent=False)

    def forward(self,tfs,sfs,sfs_mean):
        # tfs shape (batch_size, 3, num_channels, 128)
        # sfs shape (batch_size, 6, num_channels, 128)
        tfs, sfs = tfs.float(), sfs.float()
        fused_1 = []
        for (tf, sf) in zip(tfs, sfs):
            tmp = []
            for count in range(sf.shape[0]):
                tmp.append(self.conv_list_1[count](tf[count//2],self.edge_index,sfs_mean[count].reshape(-1).float()))
            tmp = torch.stack(tmp,dim=0)
            fused_1.append(tmp)
        tfs = torch.stack(fused_1,dim=0)    # shape (batch_size,6,53,32)
        tfs = F.elu(self.norm_1(tfs))
        tfs = self.dropout1(tfs)

        fused_2 = []
        for (tf,sf) in zip(tfs,sfs):
            tmp = []
            for count in range(sf.shape[0]):
                tmp.append(self.conv_list_2[count](tf[count], self.edge_index,sfs_mean[count].reshape(-1).float()))
            tmp = torch.stack(tmp,dim=0)
            fused_2.append(tmp)
        out = torch.stack(fused_2,dim=0).squeeze(-1)    # (batch_size, 6, channel_nums)
        out = self.norm_2(out)
        out = torch.flatten(out,start_dim=-2,end_dim=-1)
        out = F.elu(out)
        out = self.dropout2(out)
        return out


