from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch_geometric.nn import GCNConv
import pytorch_lightning as pl


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


class GCN_Based(pl.LightningModule):
    def __init__(self, param_grid, mean_sfs):
        super(GCN_Based, self).__init__()

        in_channels = 18
        out_channels = param_grid['hidden_size'][0]
        self.dropout = param_grid['dropout'][0]
        self.loss_weight = param_grid['loss_weight'][0]

        self.register_buffer(name='edge_index', tensor=torch.cat(
            [torch.repeat_interleave(torch.arange(53), repeats=53).unsqueeze(0), \
             torch.arange(53).repeat(53).unsqueeze(0)], dim=0), persistent=False)

        self.register_buffer(name='mean_sfs', tensor=mean_sfs, persistent=False)

        self.conv_list_1 = nn.Sequential(OrderedDict([
            ('GCNConv1_0', GCNConv(in_channels=in_channels,out_channels=out_channels)),
            ('GCNConv1_1', GCNConv(in_channels=in_channels, out_channels=out_channels)),
            ('GCNConv1_2', GCNConv(in_channels=in_channels, out_channels=out_channels)),
            ('GCNConv1_3', GCNConv(in_channels=in_channels,out_channels=out_channels)),
            ('GCNConv1_4', GCNConv(in_channels=in_channels, out_channels=out_channels)),
            ('GCNConv1_5', GCNConv(in_channels=in_channels, out_channels=out_channels)),
        ]))
        self.norm_1 = nn.LayerNorm(out_channels,eps=1e-5)
        self.dropout_1 = nn.Dropout(p=self.dropout)

        self.conv_list_2 = nn.Sequential(OrderedDict([
            ('GCNConv2_0', GCNConv(in_channels=out_channels, out_channels=1)),
            ('GCNConv2_1', GCNConv(in_channels=out_channels, out_channels=1)),
            ('GCNConv2_2', GCNConv(in_channels=out_channels, out_channels=1)),
            ('GCNConv2_3', GCNConv(in_channels=out_channels, out_channels=1)),
            ('GCNConv2_4', GCNConv(in_channels=out_channels, out_channels=1)),
            ('GCNConv2_5', GCNConv(in_channels=out_channels, out_channels=1)),
        ]))
        self.channel_fc = nn.Linear(in_features=6,out_features=1)
        self.norm_2 = nn.LayerNorm(6)
        self.norm_3 = nn.LayerNorm(53)

        self.final_classifier = nn.Sequential(nn.LayerNorm(318),\
                                              nn.Dropout(p=self.dropout),\
                                              nn.Linear(318,2))

        self.apply(weight_init)

        self.train_accuracy = torchmetrics.classification.Accuracy(num_classes=2)
        self.train_conf_mat = torchmetrics.classification.ConfusionMatrix(num_classes=2)
        self.train_f1_score = torchmetrics.classification.F1Score(num_classes=2,average='none')
        self.train_precision = torchmetrics.classification.Precision(num_classes=2,average='none')
        self.train_recall = torchmetrics.classification.Recall(num_classes=2,average='none')

        self.test_accuracy = torchmetrics.classification.Accuracy(num_classes=2)
        self.test_conf_mat = torchmetrics.classification.ConfusionMatrix(num_classes=2)
        self.test_f1_score = torchmetrics.classification.F1Score(num_classes=2,average='none')
        self.test_precision = torchmetrics.classification.Precision(num_classes=2,average='none')
        self.test_recall = torchmetrics.classification.Recall(num_classes=2,average='none')

    def forward(self,data):
        dxy, oxy, total = data['dxy_tf'].float(), data['oxy_tf'].float(), data['total_tf'].float()

        dxy_coh_sf, oxy_coh_sf, total_coh_sf = data['dxy_whole_cohe_sf'].float(), data['oxy_whole_cohe_sf'].float(), \
                                               data['total_whole_cohe_sf'].float()
        dxy_cor_sf, oxy_cor_sf, total_cor_sf = data['dxy_whole_corr_sf'].float(), data['oxy_whole_corr_sf'].float(), \
                                               data['total_whole_corr_sf'].float()
        tfs = torch.cat([dxy, oxy, total], dim=-1)    # [B, c, 54]
        sfs = torch.cat([dxy_coh_sf.unsqueeze(1), dxy_cor_sf.unsqueeze(1), \
                         oxy_coh_sf.unsqueeze(1), oxy_cor_sf.unsqueeze(1), \
                         total_coh_sf.unsqueeze(1), total_cor_sf.unsqueeze(1)], dim=1)   # [B, 6, c, c]

        tfs_new,out = [],[]
        for (tf,sf) in zip(tfs,sfs):
            dxy,oxy,total = torch.chunk(tf,chunks=3,dim=-1)
            ele_1 = []
            for (conv_id,conv) in enumerate(self.conv_list_1):
                if ((conv_id == 0) or (conv_id == 1)):
                    if self.mean_sfs is None:
                        ele_1.append(conv(dxy,self.edge_index,sf[conv_id].reshape(-1)))
                    else:
                        ele_1.append(conv(dxy, self.edge_index, self.mean_sfs[conv_id].reshape(-1)))
                elif ((conv_id == 2) or (conv_id == 3)):
                    if self.mean_sfs is None:
                        ele_1.append(conv(oxy,self.edge_index,sf[conv_id].reshape(-1)))
                    else:
                        ele_1.append(conv(oxy, self.edge_index, self.mean_sfs[conv_id].reshape(-1)))
                else:
                    if self.mean_sfs is None:
                        ele_1.append(conv(total, self.edge_index, sf[conv_id].reshape(-1)))
                    else:
                        ele_1.append(conv(total, self.edge_index, self.mean_sfs[conv_id].reshape(-1)))
            ele_1 = torch.stack(ele_1,dim=0).float()
            ele_1 = F.elu(self.norm_1(ele_1))
            ele_1 = self.dropout_1(ele_1)
            tfs_new.append(ele_1)
        tfs_new = torch.stack(tfs_new,dim=0)   # [batch_size, obstance_nums, channel_nums, feature_dims(6)]
        for (e,sf) in zip(tfs_new,sfs):
            if self.mean_sfs is None:
                ele_2 = [conv(e[conv_id],self.edge_index,sf[conv_id].reshape(-1)) for (conv_id,conv) in enumerate(self.conv_list_2)]
            else:
                ele_2 = [conv(e[conv_id], self.edge_index, self.mean_sfs[conv_id].reshape(-1)) for (conv_id, conv) in enumerate(self.conv_list_2)]

            ele_2 = torch.stack(ele_2,dim=0)
            out.append(ele_2)
        out = torch.stack(out,dim=0)     # (batch_size, num_obs, num_channels, 1)
        out = out.permute(0,1,3,2).flatten(start_dim=1,end_dim=2).float()   # batch_size, obstance_nums (6) * feature_dims (1), channel_nums

        classifier_out = self.final_classifier(out.flatten(1,2))

        return classifier_out


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

        # focal loss coeficcient
        loss = F.cross_entropy(y_hat, y, reduction='none')
        c = y_hat[:, 0]
        c = c.masked_scatter((y == 1), source=y_hat[y==1, 1])
        c = 1.0 - c
        alpha = torch.full_like(loss, fill_value=self.loss_weight)
        alpha = torch.masked_fill(alpha, mask=(y == 1), value=1.0-self.loss_weight)
        loss = loss * (c ** 1.5) * alpha
        loss = loss.mean()

        self.test_conf_mat.update(y_hat, y)
        self.test_accuracy.update(y_hat, y)
        self.test_precision.update(y_hat, y)
        self.test_recall.update(y_hat, y)
        self.test_f1_score.update(y_hat, y)

        self.log('val_loss', loss, on_step=True)
        self.log('val_accuracy', self.test_accuracy, on_step=False, on_epoch=True)
        self.log('val_precision', self.test_precision.compute()[1], on_step=False, on_epoch=True)
        self.log('val_recall', self.test_recall.compute()[1], on_step=False, on_epoch=True)
        self.log('val_f1_score', self.test_f1_score.compute()[1], on_step=False, on_epoch=True)

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

        # if self.trainer.is_global_zero:
        #     print(f'\nConf_Matrix={conf_matrix}')
        #     print(f'\nValidation Accuracy={accuracy_computed}')



    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3, eps=0.0001)
        scheduler = MultiStepLR(optimizer, milestones=[30,60,90], gamma=0.5, verbose=True)
        return [optimizer], [scheduler]

