import os
import pickle
import torch

import pandas as pd
import numpy as np
import scipy
import torch.nn.functional as F
from scipy.signal import welch, csd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns


def batch_corr(bmm_data):
    bmm_data = torch.tensor(bmm_data).permute(0,2,1)
    bmm_data_norm = torch.norm(bmm_data,dim=-1)
    bmm_data_norm = torch.tensor(bmm_data_norm)
    return torch.bmm(bmm_data, bmm_data.permute(0,2,1)) / (bmm_data_norm[:,:,None] * bmm_data_norm[:,None,:])

def preprocess_vft_data(data_root: str,data_dir: str):
    print('Starting preprocessing data...')
    N_group_dir = os.path.join(data_root, 'N_group')
    P_group_dir = os.path.join(data_root, 'P_group')
    nfirs = os.path.join(data_root, 'nFIRS.xlsx')
    feature_list = ['dxy', 'oxy', 'total']
    p_df = pd.read_excel(nfirs, sheet_name='患者组351例信息')
    n_df = pd.read_excel(nfirs, sheet_name='对照组735例信息')
    sample_file_name = '编号'
    label_column_name = '诊断标签1=分裂，2=双相I型，3=双相II型，4=抑郁'
    sex_column = '性别（1男2女）'
    age_column = '年龄'

    total_items_num = p_df.shape[0] + n_df.shape[0]

    dict = {}
    count = 0
    for (p, l, s, a) in zip(p_df[sample_file_name], p_df[label_column_name], p_df[sex_column], p_df[age_column]):
        print(f'Preprocessing {count}/{total_items_num}')
        for feature in feature_list:
            dict.update(load_features_from_mat_file(feature,P_group_dir,p,dict,vft=True))
        label = l
        dict['y'] = label
        dict['age'] = a
        dict['sex'] = s
        dict['id'] = p
        with open(os.path.join(data_dir, p + '.pkl'), 'ab') as w:
            pickle.dump(dict, w)
            w.close()
        count += 1

    # Normal Group
    dict = {}
    for (n,s,a) in zip(n_df[sample_file_name],n_df[sex_column],n_df[age_column]):
        print(f'Preprocessing {count}/{total_items_num}')
        for feature in feature_list:
            dict.update(load_features_from_mat_file(feature,N_group_dir,n,dict,vft=True))
        label = 0
        dict['y'] = label
        dict['age'] = a
        dict['sex'] = s
        dict['id'] = p
        with open(os.path.join(data_dir, n + '.pkl'), 'ab') as w:
            pickle.dump(dict, w)
            w.close()
        count += 1

def preprocess_resting_data(data_root: str,data_dir: str):
    print('Starting preprocessing data...')
    N_group_dir = os.path.join(data_root,'N_group')
    P_group_dir = os.path.join(data_root,'P_group')
    nfirs = os.path.join(data_root,'nFIRS.xlsx')
    feature_list = ['dxy','oxy','total']
    p_df = pd.read_excel(nfirs,sheet_name='青少年病人组名单')
    n_df = pd.read_excel(nfirs, sheet_name='对照组名单')
    sample_file_name = '编号'
    label_column_name = '诊断分类标签1=分裂，2=双相I型，3=双相II型，4=抑郁，5=其它'
    sex_column = '性别1男2女'
    age_column = '年龄'
    total_items_num = p_df.shape[0] + n_df.shape[0]
    print('Calculate mean and std......')
    # Average and std
    norm_dict, norm_res = {}, {}
    for (p, l, s, a) in zip(p_df[sample_file_name], p_df[label_column_name], p_df[sex_column], p_df[age_column]):
        for feature in feature_list:
            norm_dict.update(load_features_from_mat_file(feature, P_group_dir, p, norm_dict, vft=False, avg=True))
    for (n, s, a) in zip(n_df[sample_file_name], n_df[sex_column], n_df[age_column]):

        for feature in feature_list:
            norm_dict.update(load_features_from_mat_file(feature, N_group_dir, n, norm_dict, vft=False, avg=True))
    for (key, value) in norm_dict.items():
        norm_res[key + '_mean'], norm_res[key + '_std'] = value.mean(axis=0, keepdims=True), value.std(axis=0,keepdims=True)
    count = 0
    # Patient Group
    dict = {}
    for (p,l,s,a) in zip(p_df[sample_file_name], p_df[label_column_name],p_df[sex_column], p_df[age_column]):
        print(f'Preprocessing {count}/{total_items_num}')
        for feature in feature_list:
            dict.update(load_features_from_mat_file(feature,P_group_dir,p,dict,norm_res,vft=False,avg=False))

        label = l
        dict['y'] = label
        with open(os.path.join(data_dir, p + '.pkl'),'ab') as w:
            pickle.dump(dict, w)
            w.close()
        count += 1

    # Normal Group
    dict = {}
    for (n,s,a) in zip(n_df[sample_file_name],n_df[sex_column],n_df[age_column]):
        print(f'Preprocessing {count+1}/{total_items_num}')
        for feature in feature_list:
            dict.update(load_features_from_mat_file(feature,N_group_dir,n,dict,norm_res,vft=False,avg=False))
        label = 0
        dict['y'] = label
        with open(os.path.join(data_dir, n + '.pkl'),'ab') as w:
            pickle.dump(dict, w)
            w.close()
        count += 1

def coherence(X, Y,nperseg):
    # 计算归一化的互相关
    # 使用 welch 函数计算功率谱密度
    f_x, Pxx = welch(X, fs=20, nperseg=nperseg)
    f_y, Pyy = welch(Y, fs=20, nperseg=nperseg)

    # 使用 csd 函数计算互功率谱密度
    f, Pxy = csd(X, Y, fs=20, nperseg=nperseg)

    # 计算相干性 coherence
    mask = ((Pxx * Pyy) == 0)
    x = np.ma.array(Pxx * Pyy,mask=mask)
    x = x.filled(1e-8)
    coherence = np.abs(Pxy) ** 2 / (x)
    coherence = coherence.mean()
    return coherence

def get_tf_features(data,mean,std):
    # TFs
    max_element = np.max(data, axis=0, keepdims=True)
    min_element = np.min(data, axis=0, keepdims=True)
    skew = ((data - mean) ** 3).mean(axis=0, keepdims=True) / (np.std(data, axis=0, keepdims=True) ** 3)  # 偏度
    kurtosis = ((data - mean) ** 4).mean(axis=0, keepdims=True) / (((data - mean) ** 2).mean(axis=0, keepdims=True)) ** 2
    design_matrix = np.concatenate([mean, std, max_element, min_element, skew, kurtosis], axis=0).transpose()
    return design_matrix

def get_sf_features(sf_data,nperseg=256):
    f_x, Pxx = welch(sf_data, fs=20, nperseg=nperseg,axis=-1)
    auto_spectrum_power = np.expand_dims(Pxx,axis=0) * np.expand_dims(Pxx,axis=1)
    cross_spectrum_power = csd(np.expand_dims(sf_data,axis=0),np.expand_dims(sf_data,axis=1),fs=20,nperseg=nperseg,axis=-1)
    mask = (auto_spectrum_power == 0)
    x = np.ma.array(auto_spectrum_power, mask=mask)
    x = x.filled(1e-8)
    coherence_matrix = np.abs(cross_spectrum_power[1]) ** 2 / x
    coherence_matrix = np.mean(coherence_matrix,axis=-1)
    coherence_matrix = np.clip(coherence_matrix,a_min=0, a_max=1)

    cor_x, cor_y = np.expand_dims(sf_data,axis=0), np.expand_dims(sf_data,axis=1)
    auto_temporal_desity = ((cor_x ** 2).sum(axis=-1) ** 0.5 * (cor_y ** 2).sum(axis=-1) ** 0.5)
    mask = (auto_temporal_desity == 0)
    x = np.ma.array(auto_temporal_desity, mask=mask)
    x = x.filled(1e-8)
    correlation_matrix = (cor_x * cor_y).sum(axis=-1) / x
    correlation_matrix = np.clip(correlation_matrix,a_min=-1, a_max=1)
    # Normalize coherence and correlation
    correlation_matrix = (correlation_matrix + 1) / 2
    return coherence_matrix, correlation_matrix

def check_sfs_valid(coherence_matrix,correlation_matrix):
    if not ((coherence_matrix >= 0) & (coherence_matrix <= 1)).all():
        print('Coherence value is invalid!')
    if not ((correlation_matrix >= -1) & (correlation_matrix <= 1)).all():
        print('Correlation value is invalid!')

def get_sfs(data,res_dict,feature,name,nperseg):
    data = (data - np.mean(data,axis=0,keepdims=True)) / np.std(data,axis=0,keepdims=True)
    coherence_matrix, correlation_matrix = get_sf_features(data.transpose(),nperseg)
    check_sfs_valid(coherence_matrix,correlation_matrix)
    res_dict[feature + f'_{name}_cohe_sf'] = coherence_matrix
    res_dict[feature + f'_{name}_corr_sf'] = correlation_matrix
    return

def FFT (Fs,data):
    L = len(data)                        # 信号长度
    N = int(np.power(2,np.ceil(np.log2(L))))    # 下一个最近二次幂
    fft_complex = np.fft.fft(data,n=N,axis=0)
    FFT_y1 = np.abs(fft_complex)/L*2      # N点FFT 变化,但处于信号长度
    FFT_angle = np.angle(fft_complex)
    Fre = np.arange(int(N/2))*Fs/N        # 频率坐标
    FFT_y1 = FFT_y1[range(int(N/2))]      # 取一半
    return Fre, FFT_y1, FFT_angle

def load_features_from_mat_file(feature,group_dir,sample_file_name,res_dict,vft=True):
    mat_name = os.path.join(os.path.join(group_dir, feature), sample_file_name + '.mat')
    if not vft:
        data = scipy.io.loadmat(mat_name)[feature + 'data']
    else:
        data = scipy.io.loadmat(mat_name)[feature + 'data'][500:2900]

    if not vft:
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)

        # TFs
        design_matrix = get_tf_features(data, mean, std)
        res_dict[feature + '_tf'] = design_matrix
    else:
        # TFs
        s_mean, s_std, t_mean, t_std, o_mean, o_std = np.mean(data[:100], axis=0, keepdims=True), np.std(data[:100],
                                                                                                         axis=0,
                                                                                                         keepdims=True), \
                                                      np.mean(data[100:1300], axis=0, keepdims=True), np.std(data[100:1300],
                                                                                                             axis=0,
                                                                                                             keepdims=True), \
                                                      np.mean(data[1300:], axis=0, keepdims=True), np.std(data[1300:],
                                                                                                          axis=0,
                                                                                                          keepdims=True)

        silent_period = get_tf_features(data[:100], s_mean, s_std)     # [c, 6]
        task_period = get_tf_features(data[100:1300], t_mean, t_std)   # [c, 6]
        other_silent_period = get_tf_features(data[1300:], o_mean, o_std)   # [c, 6]
        tfs = np.concatenate([silent_period, task_period, other_silent_period], axis=1)    # [c, 18]
        tfs = (tfs - np.mean(tfs, axis=0, keepdims=True)) / np.std(tfs, axis=0, keepdims=True)
        res_dict[feature + '_tf'] = tfs


        # FFT
        s_fre, s_amplitude, s_angle = FFT(Fs=20, data=(data[:100] - s_mean) / s_std)
        t_fre, t_amplitude, t_angle = FFT(Fs=20, data=(data[100:1300] - t_mean) / t_std)
        o_fre, o_amplitude, o_angle = FFT(Fs=20, data=(data[1300:] - o_mean) / o_std)


        s_reserved_num, t_reserved_num, o_reserved_num = 10, 10, 10
        s_selected_amplitude, s_selected_idx = np.flip(np.sort(s_amplitude, axis=0), axis=0)[:s_reserved_num], np.flip(np.argsort(s_amplitude, axis=0), axis=0)[:s_reserved_num]
        s_selected_angle = s_angle[s_selected_idx, np.expand_dims(np.arange(s_amplitude.shape[1]), axis=0)]
        s_feature = np.stack([s_selected_amplitude, np.sin(s_selected_angle), np.cos(s_selected_angle)],axis=-1).transpose(1,0,2)
        s_fre_idx = s_selected_idx     # For Positional Encoding

        t_selected_amplitude, t_selected_idx = np.flip(np.sort(t_amplitude, axis=0), axis=0)[:t_reserved_num], np.flip(np.argsort(t_amplitude, axis=0), axis=0)[:t_reserved_num]
        t_selected_angle = t_angle[t_selected_idx, np.expand_dims(np.arange(t_amplitude.shape[1]), axis=0)]
        t_feature = np.stack([t_selected_amplitude, np.sin(t_selected_angle), np.cos(t_selected_angle)],axis=-1).transpose(1,0,2)
        t_fre_idx = t_selected_idx

        o_selected_amplitude, o_selected_idx = np.flip(np.sort(o_amplitude, axis=0), axis=0)[:o_reserved_num], np.flip(np.argsort(o_amplitude, axis=0), axis=0)[:o_reserved_num]
        o_selected_angle = o_angle[o_selected_idx, np.expand_dims(np.arange(o_amplitude.shape[1]), axis=0)]
        o_feature = np.stack([o_selected_amplitude, np.sin(o_selected_angle), np.cos(o_selected_angle)],axis=-1).transpose(1,0,2)
        o_fre_idx = o_selected_idx

        res_dict[feature+'_s_fft'], res_dict[feature+'_s_fre_idx'] = s_feature, s_fre_idx
        res_dict[feature+'_t_fft'], res_dict[feature+'_t_fre_idx'] = t_feature, t_fre_idx
        res_dict[feature+'_o_fft'], res_dict[feature+'_o_fre_idx'] = o_feature, o_fre_idx

        # SFs
        get_sfs(data,res_dict,feature,name='whole',nperseg=256)
        get_sfs(data[:100],res_dict,feature,name='s',nperseg=32)
        get_sfs(data[100:1300],res_dict,feature,name='t',nperseg=256)
        get_sfs(data[1300:],res_dict,feature,name='o',nperseg=256)
    return res_dict

def load_dfc_features(feature,group_dir,sample_file_name,vft=True):
    mat_name = os.path.join(os.path.join(group_dir, feature), sample_file_name + '.mat')
    if not vft:
        data = scipy.io.loadmat(mat_name)[feature + 'data']
    else:
        data = scipy.io.loadmat(mat_name)[feature + 'data'][500:2900]

    if vft:
        s_mean, s_std, t_mean, t_std, o_mean, o_std = np.mean(data[:100], axis=0, keepdims=True), np.std(data[:100],
                                                                                                         axis=0,
                                                                                                         keepdims=True), \
                                                      np.mean(data[100:1300], axis=0, keepdims=True), np.std(data[100:1300],
                                                                                                        axis=0,
                                                                                                        keepdims=True), \
                                                      np.mean(data[1300:], axis=0, keepdims=True), np.std(data[1300:],
                                                                                                          axis=0,
                                                                                                          keepdims=True)
        silent_statistics = (data[:100] - s_mean) / s_std
        task_statistics = (data[100:1300] - t_mean) / t_std
        other_silent_statistics = (data[1300:] - o_mean) / o_std

    return silent_statistics, task_statistics, other_silent_statistics
