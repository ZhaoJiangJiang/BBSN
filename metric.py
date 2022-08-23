import math
import collections
import numpy as np
import pandas as pd


def metric_micro_precision(true_mat, score_mat, k):
    size = len(true_mat)
    corrects = 0
    count = 0
    for i in range(size):
        if not true_mat[i].__contains__(1.):
            continue
        if not type(score_mat) == list:
            cur_score_array = score_mat[i].cpu().detach().numpy()
        else:
            cur_score_array = score_mat[i]
            cur_score_array = np.array(cur_score_array)
        rank_index = np.argsort(-cur_score_array)   # 降序
        for v in range(k):
            if true_mat[i][rank_index[v]] == 1 and score_mat[i][rank_index[v]] >= 0.5:
                corrects += 1
            count += 1
    precision = 100.0 * corrects / count
    return precision

def metric_macro_precision(true_mat, score_mat, k, cate_num):
    size = len(true_mat)
    up = [0] * cate_num
    down = [0] * cate_num
    for i in range(size):
        if not true_mat[i].__contains__(1.):
            continue
        if not type(score_mat) == list:
            cur_score_array = score_mat[i].cpu().detach().numpy()
        else:
            cur_score_array = score_mat[i]
            cur_score_array = np.array(cur_score_array)
        rank_index = np.argsort(-cur_score_array)  # 降序
        for v in range(k):
            cate = rank_index[v]
            if true_mat[i][cate] == 1 and score_mat[i][cate] >= 0.5:
                up[cate] += 1
            down[cate] += 1
    up = np.array(up)
    down = np.array(down)
    precision = up / down
    precision = precision[~np.isnan(precision)]
    precision = np.mean(precision) * 100.0
    return precision


def metric_F1(TP, FP, FN):
    # Micro F1
    if sum(TP) == 0 and sum(FP) == 0:
        precision = 0
    else:
        precision = sum(TP) / (sum(TP) + sum(FP))

    if sum(TP) == 0 and sum(FN) == 0:
        recall = 0
    else:
        recall = sum(TP) / (sum(TP) + sum(FN))

    if precision == 0 and recall == 0:
        entire_micro = 0
    else:
        entire_micro = 2 * precision * recall / (precision + recall)

    # Macro F1
    tmp = []
    for i in range(len(TP)):
        if TP[i] == 0 and FP[i] == 0:
            precision_i = 0
        else:
            precision_i = TP[i] / (TP[i] + FP[i])

        if TP[i] == 0 and FN[i] == 0:
            recall_i = 0
        else:
            recall_i = TP[i] / (TP[i] + FN[i])

        if precision_i == 0 and recall_i == 0:
            macro_i = 0
        else:
            macro_i = 2 * precision_i * recall_i / (precision_i + recall_i)

        tmp.append(macro_i)

    entire_macro = 0
    for item in tmp:
        entire_macro += item
    entire_macro = entire_macro / len(TP)

    return entire_micro, entire_macro

def metric_micro_DCG(y_true, y_pred, k):
    dcg_vector = []
    for ii in range(len(y_true)):
        df = pd.DataFrame({"y_pred": y_pred[ii], "y_true": y_true[ii]})
        df = df.sort_values(by="y_pred", ascending=False)
        df['y_pred'][df['y_pred'] < 0.5] = 0.0       
        df['y_pred'][df['y_pred'] >= 0.5] = 1.0
        df = df.iloc[0:k, :]
        dcg = (2**df["y_true"]-1) / np.log2(np.arange(1, df["y_true"].count()+1)+1)
        dcg_vector.append(np.sum(dcg))
    return dcg_vector

def metric_micro_NDCG(y_true, y_pred, k):
    dcg = metric_micro_DCG(y_true, y_pred, k)
    idcg = metric_micro_DCG(y_true, y_true, k)
    dcg = np.array(dcg)
    idcg = np.array(idcg)
    ndcg = dcg / idcg
    ndcg = ndcg[~np.isnan(ndcg)]
    return np.mean(ndcg) * 100.0
