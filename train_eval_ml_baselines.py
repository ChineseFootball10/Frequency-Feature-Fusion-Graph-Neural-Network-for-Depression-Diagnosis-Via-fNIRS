import os

from sklearn.linear_model import LogisticRegression

from k_split_my import set_seed
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import torch
import argparse
import random
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint  # Import ModelCheckpoint

from models.DySAT import my_model
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedShuffleSplit

from datasets.fnirs import Depression

resting = False
if resting:
    file_name = 'resting'
else:
    file_name = 'VFT'
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_dir", default=f"data/{file_name}/preprocessed",help="Directory to extract data")
args = parser.parse_args()
split_seg_counts = 4

if __name__ == '__main__':
    dataset = Depression(data_dir=args.data_dir, selected=False, psm=False)
    MAX_EVALS = 1
    random_seed = 2024
    set_seed(random_seed)  # Fix Random seed for reproducing results in our paper.
    y = []
    for data in dataset:
        y.append(data['y'])
    y = np.array(y)

    temporal = True
    for i in range(MAX_EVALS):
        random.seed(i)
        ss = StratifiedShuffleSplit(n_splits=split_seg_counts, test_size=0.25, train_size=0.75, random_state=random_seed)
        accuracy_list, precision_list, recall_list, f1_score_list, specificity_list = [], [], [], [], []

        for train_index, test_index in ss.split(dataset, y):
            test_dataset, train_dataset = [dataset[id] for id in test_index], [dataset[id] for id in train_index]
            train_feature, test_feature = [], []
            train_label, test_label = [], []
            for data in train_dataset:
                label = data['y']
                if label > 0:
                    train_label.append(1)
                else:
                    train_label.append(0)
                if temporal:
                    dxy_f, oxy_f, total_f = data['dxy_tf'], data['oxy_tf'], data['total_tf']
                    tf_feature = np.concatenate([dxy_f, oxy_f, total_f], axis=-1).flatten()
                    train_feature.append(tf_feature)
                else:
                    dxy_corr, dxy_cohe = data['dxy_whole_corr_sf'], data['dxy_whole_cohe_sf']
                    oxy_corr, oxy_cohe = data['oxy_whole_corr_sf'], data['oxy_whole_cohe_sf']
                    total_corr, total_cohe = data['total_whole_corr_sf'], data['total_whole_cohe_sf']
                    sf_feature = np.stack([dxy_corr, dxy_cohe, oxy_corr, oxy_cohe, total_corr, total_cohe], axis=-1)
                    sf_feature = sf_feature.flatten()
                    train_feature.append(sf_feature)
            for data in test_dataset:
                label = data['y']
                if label > 0:
                    test_label.append(1)
                else:
                    test_label.append(0)
                if temporal:
                    dxy_f, oxy_f, total_f = data['dxy_tf'], data['oxy_tf'], data['total_tf']
                    tf_feature = np.concatenate([dxy_f, oxy_f, total_f], axis=-1).flatten()
                    test_feature.append(tf_feature)
                else:
                    dxy_corr, dxy_cohe = data['dxy_whole_corr_sf'], data['dxy_whole_cohe_sf']
                    oxy_corr, oxy_cohe = data['oxy_whole_corr_sf'], data['oxy_whole_cohe_sf']
                    total_corr, total_cohe = data['total_whole_corr_sf'], data['total_whole_cohe_sf']
                    sf_feature = np.stack([dxy_corr, dxy_cohe, oxy_corr, oxy_cohe, total_corr, total_cohe], axis=-1)
                    sf_feature = sf_feature.flatten()
                    test_feature.append(sf_feature)

            train_feature = np.stack(train_feature, axis=0)
            test_feature = np.stack(test_feature, axis=0)

            train_label = np.array(train_label)
            test_label = np.array(test_label)
            # RF,KNN,LR,SVM
            # classifier = svm.SVC(kernel='rbf', C=4.0, class_weight={0:1, 1:2})
            # classifier = RandomForestClassifier(n_estimators=50, random_state=42, class_weight={0:1, 1:1})   # PSM: class_weight 1:1 not PSM class_weight 1:2 for temporal and not temporal
            # classifier = KNeighborsClassifier(n_neighbors=10,weights='uniform')   # sf->15 tf->10   PSM: tf->20 sf->10
            # classifier = LogisticRegression(max_iter=1000,C=4.0)
            # classifier.fit(train_feature, train_label)
            # y_pred = classifier.predict(test_feature)
            #xgb
            #设置模型参数
            params = {
                'objective': 'binary:logistic',  # Binary Logistic Regression
                'eval_metric': 'logloss',  # The evaluation metric is log-likelihood loss
                'eta': 0.1,  # Learning Rate
                'max_depth': 3,  # Max Depth
                'seed': 42  # Random seed
            }
            dtrain = xgb.DMatrix(train_feature, label=train_label)
            dtest = xgb.DMatrix(test_feature, label=test_label)
            # Train machine learning model
            num_round = 100
            bst = xgb.train(params, dtrain, num_round)
            # Make predictions on the test set
            import shap
            explainer = shap.TreeExplainer(bst)
            shap_values = explainer.shap_values(test_feature)

            name_list = []
            statistics = ['Mean', 'Std', 'Min', 'Max', 'Kurt', 'Skew']
            for i in range(53):
                for j in statistics:
                    name_list.append(str(i) + '_' + j)
            shap_values = np.reshape(shap_values, (272, 53, -1))
            dxy_s, dxy_t, dxy_os, oxy_s, oxy_t, oxy_os, total_s, total_t, total_os = np.array_split(shap_values, 9,
                                                                                                    axis=-1)
            dxy_s, dxy_t, dxy_os, oxy_s, oxy_t, oxy_os, total_s, total_t, total_os = dxy_s.reshape(272,
                                                                                                   -1), dxy_t.reshape(
                272, -1), \
                dxy_os.reshape(272, -1), oxy_s.reshape(272, -1), oxy_t.reshape(272, -1), oxy_os.reshape(272,
                                                                                                        -1), total_s.reshape(
                272, -1), total_t.reshape(272, -1), \
                total_os.reshape(272, -1)

            test_feature = np.reshape(test_feature, (272, 53, -1))
            dxy_s_, dxy_t_, dxy_os_, oxy_s_, oxy_t_, oxy_os_, total_s_, total_t_, total_os_ = np.array_split(
                test_feature, 9, axis=-1)

            dxy_s_, dxy_t_, dxy_os_, oxy_s_, oxy_t_, oxy_os_, total_s_, total_t_, total_os_ = dxy_s_.reshape(272,
                                                                                                             -1), dxy_t_.reshape(
                272, -1), \
                dxy_os_.reshape(272, -1), oxy_s_.reshape(272, -1), oxy_t_.reshape(272, -1), oxy_os_.reshape(272,
                                                                                                            -1), total_s_.reshape(
                272, -1), total_t_.reshape(272, -1), \
                total_os_.reshape(272, -1)


            def plot_shap_beeswarm_top30(shap_values, features, feature_names=None, save_path='shap_beeswarm_top9.png'):
                """
                Using the SHAP library to plot a Beeswarm chart for the top 30 biomarkers

                Parameters:
                -----------
                shap_values : numpy array
                    SHAP matrix，shape (n_samples, n_features)
                features : numpy array
                    shape (n_samples, n_features)
                feature_names : list
                save_path : str
                """
                import matplotlib.pyplot as plt
                # Calculate global SHAP value.
                mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

                # Extract top-30 SHAP indices
                top30_idx = np.argsort(mean_abs_shap)[::-1][:9]

                # Extract top-30 SHAP values
                shap_values_top30 = shap_values[:, top30_idx]
                features_top30 = features[:, top30_idx]

                # Extract top-30 featrues name
                if feature_names is not None:
                    feature_names_top30 = [feature_names[i] for i in top30_idx]
                else:
                    feature_names_top30 = [f'Feature_{i}' for i in range(30)]

                plt.figure(figsize=(12, 10))

                shap.summary_plot(
                    shap_values_top30,
                    features_top30,
                    feature_names=feature_names_top30,
                    show=False,
                    max_display=30,
                    plot_size=(12, 10),
                    color_bar=True,
                    alpha=0.6
                )

                plt.title('Top 30 Biomarkers - Global SHAP Values', fontsize=16, fontweight='bold', pad=20)
                plt.xlabel('SHAP Value (impact on model output)', fontsize=12, fontweight='bold')

                plt.tight_layout()
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.show()

                # 打印前30个特征的信息
                print("\n=== Top 30 Biomarkers (Global Importance) ===")
                print(f"{'Rank':<6}{'Feature Name':<30}{'Mean |SHAP|':<15}{'Contribution Direction'}")
                print("-" * 70)

                for i, idx in enumerate(top30_idx):
                    mean_abs = mean_abs_shap[idx]
                    mean_shap = np.mean(shap_values[:, idx])

                    pos_ratio = np.mean(shap_values[:, idx] > 0)
                    if pos_ratio > 0.6:
                        direction = "Mostly Positive"
                    elif pos_ratio < 0.4:
                        direction = "Mostly Negative"
                    else:
                        direction = "Mixed"

                    print(f"{i + 1:<6}{feature_names_top30[i]:<30}{mean_abs:<15.6f}{direction}")

                return top30_idx, feature_names_top30


            y_pred_prob = bst.predict(dtest)  # Prediction
            y_pred = np.where(y_pred_prob > 0.5, 1, 0)  # Change class distribution to labels

            accuracy, precision, recall, f1 = accuracy_score(test_label, y_pred),\
                                            precision_score(test_label, y_pred),\
                                            recall_score(test_label, y_pred),\
                                            f1_score(test_label, y_pred)
            specificity = 1/ (1 + (1/precision -1) * (1-accuracy) / (accuracy*(1/recall+1/precision-1)-1))
            accuracy_list.append(accuracy)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_score_list.append(f1)
            specificity_list.append(specificity)

        accuracy_list = np.array(accuracy_list)
        precision_list = np.array(precision_list)
        recall_list = np.array(recall_list)
        f1_score_list = np.array(f1_score_list)
        specificity_list = np.array(specificity_list)
        print(f'Accuracy : {accuracy_list.mean()},\n'
              f'Precision : {precision_list.mean()},\n'
              f'Recall : {recall_list.mean()},\n'
              f'F1-score : {f1_score_list.mean()},\n'
              f'Specificity : {specificity_list.mean()},\n')
