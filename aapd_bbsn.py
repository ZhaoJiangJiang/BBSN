import os
import math
import torch
import datetime
import argparse
import torchtext
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import torchtext.legacy.data as data
import matplotlib.pyplot as plt

from copy import deepcopy
from sklearn.metrics import f1_score
from sklearn.metrics import ndcg_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

from transformers import BertModel
from pytorch_pretrained_bert import BertAdam

from metric import metric_micro_precision
from metric import metric_macro_precision
from aapd_data_util import util_get_tail_label


class BERT_Encoder(nn.Module):
    def __init__(self):
        super(BERT_Encoder, self).__init__()
        self.encoder = BertModel.from_pretrained("bert-base-cased")

    def encode(self, text, attn):
        out = self.encoder(input_ids=text, attention_mask=attn)
        last_hidden_state = out.last_hidden_state
        text_cls = last_hidden_state[:, 0, :]
        return text_cls


class CLS_BBSN4BRANCH(nn.Module):
    def __init__(self, args):
        super(CLS_BBSN4BRANCH, self).__init__()
        self.d = args.class_size
        self.encoder = BERT_Encoder()
        self.ReLU = nn.ReLU()

        self.siamese1_linear1 = nn.Linear(768, 1024)
        self.siamese1_classifier = nn.Linear(1024, self.d)
        self.siamese1_linear_onehot = nn.Linear(self.d, 1024)
        self.siamese1_contrast = nn.Linear(1024, 1)

        self.siamese2_linear1 = nn.Linear(768, 1024)
        self.siamese2_classifier = nn.Linear(1024, self.d)
        self.siamese2_linear_onehot = nn.Linear(self.d, 1024)
        self.siamese2_contrast = nn.Linear(1024, 1)

        self.bbn_linear_branch2 = nn.Linear(1024, self.d)
        self.bbn_linear_branch3 = nn.Linear(1024, self.d)

    def siamese1_forward_one(self, text, attn_mask):
        text_cls = self.encoder.encode(text, attn_mask)
        f_1024 = self.siamese1_linear1(text_cls)
        pred_out = self.siamese1_classifier(f_1024)
        return f_1024, pred_out

    def siamese1_forward_two(self, text1, attn_mask1, text2, attn_mask2, mask, weight):
        batch_size = mask.shape[0]
        f1, pred_out1 = self.siamese1_forward_one(text1, attn_mask1)
        f2, pred_out2 = self.siamese1_forward_one(text2, attn_mask2)

        f_weight2 = weight * f2

        x1 = torch.sigmoid(f1)
        x2 = torch.sigmoid(f2)
        dis = torch.abs(x1 - x2)    
        dis = dis.unsqueeze(1)      
        dis = dis.repeat(1, self.d, 1)  

        onehot_matrix = torch.eye(self.d).to('cuda')
        onehot_matrix = onehot_matrix.repeat(batch_size, 1, 1)
        q_w = self.siamese1_linear_onehot(onehot_matrix)  
        q_w = self.ReLU(q_w / math.sqrt(self.d))

        tmp = torch.mul(dis, q_w)
        pred_contrast = self.siamese1_contrast(tmp)
        pred_contrast = pred_contrast.squeeze() * mask

        return pred_out1, pred_contrast, f_weight2

    def siamese2_forward_one(self, text, attn_mask):
        text_cls = self.encoder.encode(text, attn_mask)
        f_1024 = self.siamese2_linear1(text_cls)
        pred_out = self.siamese2_classifier(f_1024)
        return f_1024, pred_out

    def siamese2_forward_two(self, text3, attn_mask3, text4, attn_mask4, mask, weight):
        batch_size = mask.shape[0]
        f3, pred_out3 = self.siamese2_forward_one(text3, attn_mask3)
        f4, pred_out4 = self.siamese2_forward_one(text4, attn_mask4)

        f_weight3 = weight * f3

        x3 = torch.sigmoid(f3)
        x4 = torch.sigmoid(f4)
        dis = torch.abs(x3 - x4)
        dis = dis.unsqueeze(1)
        dis = dis.repeat(1, self.d, 1)

        onehot_matrix = torch.eye(self.d).to('cuda')
        onehot_matrix = onehot_matrix.repeat(batch_size, 1, 1)
        q_w = self.siamese2_linear_onehot(onehot_matrix)
        q_w = self.ReLU(q_w / math.sqrt(self.d))

        tmp = torch.mul(dis, q_w)
        pred_contrast = self.siamese2_contrast(tmp)
        pred_contrast = pred_contrast.squeeze() * mask

        return pred_out4, pred_contrast, f_weight3

    def forward_train(self, text1, attn_mask1, text2, attn_mask2, mask1, text3, attn_mask3, text4, attn_mask4, mask2, alpha):
        pred_y1, pred_c1, f_weight2 = self.siamese1_forward_two(text1, attn_mask1, text2, attn_mask2, mask1, weight=alpha)
        pred_y4, pred_c2, f_weight3 = self.siamese2_forward_two(text3, attn_mask3, text4, attn_mask4, mask2, weight=1-alpha)
        bbn_out2 = self.bbn_linear_branch2(f_weight2)
        bbn_out3 = self.bbn_linear_branch3(f_weight3)
        pred_bbn = bbn_out2 + bbn_out3
        return pred_y1, pred_c1, pred_bbn, pred_c2, pred_y4

    def forward_eval(self, text, attn_mask, alpha):
        text_cls = self.encoder.encode(text, attn_mask)
        f_weight2 = alpha * self.siamese1_linear1(text_cls)
        f_weight3 = (1-alpha) * self.siamese2_linear1(text_cls)
        bbn_out2 = self.bbn_linear_branch2(f_weight2)
        bbn_out3 = self.bbn_linear_branch3(f_weight3)
        pred_bbn = bbn_out2 + bbn_out3
        return pred_bbn


def time_print(s):
    print('{}: {}'.format(datetime.datetime.now().strftime('%02y-%02m-%02d %H:%M:%S'), s), flush=True)

def parse_args():
    parser = argparse.ArgumentParser(description="AAPD BBSN")
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--total_epoch', type=int, default=50)
    parser.add_argument('--class_size', type=int, default=54)
    parser.add_argument('--pair_path1', type=str, default='data/aapd/aapd_bbsn1.json')
    parser.add_argument('--pair_path2', type=str, default='data/aapd/aapd_bbsn2.json')
    parser.add_argument('--valid_path', type=str, default='data/aapd/aapd_valid.json')
    parser.add_argument('--test_path', type=str, default='data/aapd/aapd_test.json')
    return parser.parse_args()

def print_args(args):
    print("Parameters:")
    for attr, value in args.__dict__.items():
        print('\t{}={}'.format(attr, value))

def set_seed(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

def read_data(args):
    TOKEN = data.Field(sequential=False, is_target=True, use_vocab=False, dtype=torch.int32)
    ATTN_MASK = data.Field(sequential=False, is_target=True, use_vocab=False, dtype=torch.int32)
    MUTILABEL = data.Field(sequential=False, is_target=True, use_vocab=False, dtype=torch.float32)
    GROUND_TRUTH = data.Field(sequential=False, is_target=True, use_vocab=False, dtype=torch.float32)
    MASK = data.Field(sequential=False, is_target=True, use_vocab=False, dtype=torch.float32)
    common_field = {
        'tokens': ('tokens', TOKEN),
        'attn_mask': ('attn_mask', ATTN_MASK),
        'mutilabel': ('mutilabel', MUTILABEL)
    }
    time_print("read valid data")
    valid_data = data.TabularDataset(path=args.valid_path, format='json', fields=common_field)
    time_print("build valid data iterator")
    valid_data_iter = data.BucketIterator(valid_data, batch_size=2, repeat=False, shuffle=False, device=torch.device('cuda:0'))

    time_print("read test data")
    test_data = data.TabularDataset(path=args.test_path, format='json', fields=common_field)
    time_print("build test data iterator")
    test_data_iter = data.BucketIterator(test_data, batch_size=1, repeat=False, shuffle=False, device=torch.device('cuda:0'))

    pair_field = {
        'tokens1': ('tokens1', TOKEN),
        'attn_mask1': ('attn_mask1', ATTN_MASK),
        'mutilabel1': ('mutilabel1', MUTILABEL),
        'tokens2': ('tokens2', TOKEN),
        'attn_mask2': ('attn_mask2', ATTN_MASK),
        'mutilabel2': ('mutilabel2', MUTILABEL),
        'ground_truth': ('ground_truth', GROUND_TRUTH),
        'mask': ('mask', MASK)
    }
    time_print("read pair1 data")
    pair_train1 = data.TabularDataset(path=args.pair_path1, format='json', fields=pair_field)
    time_print("build pair1 iterator")
    pair_iter1 = data.BucketIterator(pair_train1, batch_size=args.batch_size, repeat=False, shuffle=True, device=torch.device('cuda:0'))

    time_print("read pair2 data")
    pair_train2 = data.TabularDataset(path=args.pair_path2, format='json', fields=pair_field)
    time_print("build pair2 iterator")
    pair_iter2 = data.BucketIterator(pair_train2, batch_size=args.batch_size, repeat=False, shuffle=True, device=torch.device('cuda:0'))

    return pair_iter1, pair_iter2, valid_data_iter, test_data_iter

def loss_fn_mask(logit, truth, mask):
    loss_fn = torch.nn.BCELoss()
    logit = torch.sigmoid(logit)
    logit = torch.mul(logit, mask)
    truth = torch.mul(truth, mask)
    loss = loss_fn(logit, truth)
    return loss

def do_train(model, pair_iter1, pair_iter2, valid_iter, prefix, args):
    model.train()
    lr = args.lr
    total_epoch = args.total_epoch
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.00}
    ]

    accumulate_step = 32 // args.batch_size
    num_total_steps = len(pair_iter1) * (total_epoch+1)
    optimizer = BertAdam(optimizer_grouped_parameters, lr=lr, schedule='warmup_linear', warmup=0.1, t_total=num_total_steps)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    epoch_losses = []
    epoch_micro_f1 = []
    epoch_macro_f1 = []
    best_score = -1
    best_loss = 999
    for epoch in range(total_epoch+1):
        model.train()
        train_losses = []
        optimizer.zero_grad()
        alpha = 1 - (epoch / total_epoch) ** 2
        # beta = 1 - (epoch / total_epoch) ** 2
        beta = math.cos((epoch / total_epoch) * (math.pi / 2))
        for pair_info1, pair_info2 in tqdm(zip(enumerate(pair_iter1), enumerate(pair_iter2))):
            curr_step = pair_info1[0] + 1
            # 第一个孪生网络数据
            x1 = pair_info1[1].tokens1.to('cuda')
            attn_mask1 = pair_info1[1].attn_mask1.to('cuda')
            y1 = pair_info1[1].mutilabel1.to('cuda')
            x2 = pair_info1[1].tokens2.to('cuda')
            attn_mask2 = pair_info1[1].attn_mask2.to('cuda')
            y2 = pair_info1[1].mutilabel2.to('cuda')
            mask1 = pair_info1[1].mask.to('cuda')
            contrast1 = pair_info1[1].ground_truth.to('cuda')
            # 第二个孪生网络数据
            x3 = pair_info2[1].tokens1.to('cuda')
            attn_mask3 = pair_info2[1].attn_mask1.to('cuda')
            y3 = pair_info2[1].mutilabel1.to('cuda')
            x4 = pair_info2[1].tokens2.to('cuda')
            attn_mask4 = pair_info2[1].attn_mask2.to('cuda')
            y4 = pair_info2[1].mutilabel2.to('cuda')
            mask2 = pair_info2[1].mask.to('cuda')
            contrast2 = pair_info2[1].ground_truth.to('cuda')

            pred_y1, pred_c1, pred_bbn, pred_c2, pred_y4 = model.forward_train(x1, attn_mask1, x2, attn_mask2, mask1, x3, attn_mask3, x4, attn_mask4, mask2, alpha)

            loss1 = loss_fn(pred_y1, y1)
            loss2 = loss_fn_mask(pred_c1, contrast1, mask1)
            loss3 = alpha * loss_fn(pred_bbn, y2) + (1-alpha) * loss_fn(pred_bbn, y3)
            loss4 = loss_fn_mask(pred_c2, contrast2, mask2)
            loss5 = loss_fn(pred_y4, y4)
            loss = beta*loss1 + (1-beta)*loss2 + loss3 + (1-beta)*loss4 + beta*loss5

            train_losses.append(loss.item())

            loss = loss / accumulate_step
            loss.backward()
            if curr_step % accumulate_step == 0:
                optimizer.step()
                optimizer.zero_grad()

        train_loss = np.average(train_losses)
        valid_micro_f1, valid_macro_f1, valid_loss = do_eval(model, valid_iter)
        epoch_losses.append(valid_loss)
        epoch_micro_f1.append(valid_micro_f1)
        epoch_macro_f1.append(valid_macro_f1)
        result = "AAPD BBSN [{}/{}] train_loss:{} val_loss:{} val_mi_F1:{} val_ma_F1:{} alpha={:.2f} beta={:.2f}".format(epoch, total_epoch, train_loss, valid_loss, valid_micro_f1, valid_macro_f1, alpha, beta)
        time_print(result)
        if valid_micro_f1 + valid_macro_f1 > best_score:
            time_print("valid macro F1 score increased ({:.2f} --> {:.2f}).  Saving model ...".format(best_score, valid_micro_f1 + valid_macro_f1))
            torch.save(model.state_dict(), prefix)
            best_score = valid_micro_f1 + valid_macro_f1
        model.load_state_dict(torch.load(prefix))

def do_eval(model, valid_iter):
    model.eval()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    valid_losses = []
    entire_true = []
    entire_pred = []
    for valid_info in tqdm(enumerate(valid_iter)):
        valid_tokens = valid_info[1].tokens.to('cuda')
        valid_attn_mask = valid_info[1].attn_mask.to('cuda')
        valid_mutilabel = valid_info[1].mutilabel.to('cuda')
        pred = model.forward_eval(valid_tokens, valid_attn_mask, alpha=0.5)
        loss = loss_fn(pred, valid_mutilabel)
        valid_losses.append(loss.item())
        pred = torch.sigmoid(pred)
        entire_true.extend(valid_mutilabel.cpu().detach().numpy().tolist())
        entire_pred.extend(pred.cpu().detach().numpy().tolist())

    entire_true = torch.tensor(entire_true)
    entire_pred = torch.tensor(entire_pred)
    entire_pred[entire_pred >= 0.5] = 1.0
    entire_pred[entire_pred < 0.5] = 0.0

    entire_true = entire_true.tolist()
    entire_pred = entire_pred.tolist()

    entire_micro_F1 = f1_score(entire_true, entire_pred, average='micro')
    entire_macro_F1 = f1_score(entire_true, entire_pred, average='macro')

    return entire_micro_F1 * 100, entire_macro_F1 * 100, np.average(valid_losses)

def do_test(model, test_iter, result_path):
    model.eval()
    tail_1000 = [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]
    entire_true = []
    entire_pred = []

    for test_info in tqdm(enumerate(test_iter)):
        test_tokens = test_info[1].tokens.to('cuda')
        test_attn_mask = test_info[1].attn_mask.to('cuda')
        test_mutilabel = test_info[1].mutilabel.to('cuda')
        pred = model.forward_eval(test_tokens, test_attn_mask, alpha=0.5)
        pred = torch.sigmoid(pred)
        entire_true.extend(test_mutilabel.cpu().detach().numpy().tolist())
        entire_pred.extend(pred.cpu().detach().numpy().tolist())


    entire_true = torch.tensor(entire_true)
    entire_pred = torch.tensor(entire_pred)
    tail_true = entire_true[:, tail_1000]
    tail_pred = entire_pred[:, tail_1000]

    effect_tail_index = [i for i in range(len(tail_true)) if 1.0 in tail_true[i]]
    tail_true = tail_true[effect_tail_index, :]
    tail_pred = tail_pred[effect_tail_index, :]

    e_true = deepcopy(entire_true)
    e_pred = deepcopy(entire_pred)
    threshold = 0.5
    e_pred[e_pred >= threshold] = 1.0
    e_pred[e_pred < threshold] = 0.0
    t_true = e_true[:, tail_1000]
    t_pred = e_pred[:, tail_1000]
    t_pred[t_pred >= threshold] = 1.0
    t_pred[t_pred < threshold] = 0.0
    e_true = e_true.tolist()
    e_pred = e_pred.tolist()
    t_true = t_true[effect_tail_index, :]
    t_pred = t_pred[effect_tail_index, :]
    t_true = t_true.tolist()
    t_pred = t_pred.tolist()

    entire_micro_F1 = f1_score(e_true, e_pred, average='micro')
    entire_macro_F1 = f1_score(e_true, e_pred, average='macro')

    tail_micro_F1 = f1_score(t_true, t_pred, average='micro')
    tail_macro_F1 = f1_score(t_true, t_pred, average='macro')

    entire_micro_ndcg3 = ndcg_score(entire_true, entire_pred, k=3) * 100
    entire_micro_ndcg5 = ndcg_score(entire_true, entire_pred, k=5) * 100
    entire_micro_precision1 = metric_micro_precision(entire_true, entire_pred, k=1)
    entire_micro_precision3 = metric_micro_precision(entire_true, entire_pred, k=3)
    entire_micro_precision5 = metric_micro_precision(entire_true, entire_pred, k=5)
    entire_macro_precision1 = metric_macro_precision(entire_true, entire_pred, k=1, cate_num=54)
    entire_macro_precision3 = metric_macro_precision(entire_true, entire_pred, k=3, cate_num=54)
    entire_macro_precision5 = metric_macro_precision(entire_true, entire_pred, k=5, cate_num=54)

    tail_micro_ndcg3 = ndcg_score(tail_true, tail_pred, k=3) * 100
    tail_micro_ndcg5 = ndcg_score(tail_true, tail_pred, k=5) * 100
    tail_micro_precision1 = metric_micro_precision(tail_true, tail_pred, k=1)
    tail_micro_precision3 = metric_micro_precision(tail_true, tail_pred, k=3)
    tail_micro_precision5 = metric_micro_precision(tail_true, tail_pred, k=5)
    tail_macro_precision1 = metric_macro_precision(tail_true, tail_pred, k=1, cate_num=23)
    tail_macro_precision3 = metric_macro_precision(tail_true, tail_pred, k=3, cate_num=23)
    tail_macro_precision5 = metric_macro_precision(tail_true, tail_pred, k=5, cate_num=23)

    test_result = "AAPD Single result: \n"
    # Entire - Micro
    print("AAPD BBSN Test - entire_micro_F1 = {:.2f}".format(entire_micro_F1 * 100))
    print("AAPD BBSN Test - entire_micro_precision@1 = {:.2f}".format(entire_micro_precision1))
    print("AAPD BBSN Test - entire_micro_precision@3 = {:.2f}".format(entire_micro_precision3))
    print("AAPD BBSN Test - entire_micro_precision@5 = {:.2f}".format(entire_micro_precision5))
    print("AAPD BBSN Test - entire_micro_NDCG@3 = {:.2f}".format(entire_micro_ndcg3))
    print("AAPD BBSN Test - entire_micro_NDCG@5 = {:.2f}\n".format(entire_micro_ndcg5))
    test_result += "AAPD BBSN Test - entire_micro_F1 = {:.2f}".format(entire_micro_F1 * 100) + "\n"
    test_result += "AAPD BBSN Test - entire_micro_precision@1 = {:.2f}".format(entire_micro_precision1) + "\n"
    test_result += "AAPD BBSN Test - entire_micro_precision@3 = {:.2f}".format(entire_micro_precision3) + "\n"
    test_result += "AAPD BBSN Test - entire_micro_precision@5 = {:.2f}".format(entire_micro_precision5) + "\n"
    test_result += "AAPD BBSN Test - entire_micro_NDCG@3 = {:.2f}".format(entire_micro_ndcg3) + "\n"
    test_result += "AAPD BBSN Test - entire_micro_NDCG@5 = {:.2f}".format(entire_micro_ndcg5) + "\n\n"

    # Entire - Macro
    print("AAPD BBSN Test - entire_macro_F1 = {:.2f}".format(entire_macro_F1 * 100))
    print("AAPD BBSN Test - entire_macro_precision@1 = {:.2f}".format(entire_macro_precision1))
    print("AAPD BBSN Test - entire_macro_precision@3 = {:.2f}".format(entire_macro_precision3))
    print("AAPD BBSN Test - entire_macro_precision@5 = {:.2f}\n".format(entire_macro_precision5))
    test_result += "AAPD BBSN Test - entire_macro_F1 = {:.2f}".format(entire_macro_F1 * 100) + "\n"
    test_result += "AAPD BBSN Test - entire_macro_precision@1 = {:.2f}".format(entire_macro_precision1) + "\n"
    test_result += "AAPD BBSN Test - entire_macro_precision@3 = {:.2f}".format(entire_macro_precision3) + "\n"
    test_result += "AAPD BBSN Test - entire_macro_precision@5 = {:.2f}".format(entire_macro_precision5) + "\n\n"

    # Tail - Micro
    print("AAPD BBSN Test - tail_micro_F1 = {:.2f}".format(tail_micro_F1 * 100))
    print("AAPD BBSN Test - tail_micro_precision@1 = {:.2f}".format(tail_micro_precision1))
    print("AAPD BBSN Test - tail_micro_precision@3 = {:.2f}".format(tail_micro_precision3))
    print("AAPD BBSN Test - tail_micro_precision@5 = {:.2f}".format(tail_micro_precision5))
    print("AAPD BBSN Test - tail_micro_NDCG@3 = {:.2f}".format(tail_micro_ndcg3))
    print("AAPD BBSN Test - tail_micro_NDCG@5 = {:.2f}\n".format(tail_micro_ndcg5))
    test_result += "AAPD BBSN Test - tail_micro_F1 = {:.2f}".format(tail_micro_F1 * 100) + "\n"
    test_result += "AAPD BBSN Test - tail_micro_precision@1 = {:.2f}".format(tail_micro_precision1) + "\n"
    test_result += "AAPD BBSN Test - tail_micro_precision@3 = {:.2f}".format(tail_micro_precision3) + "\n"
    test_result += "AAPD BBSN Test - tail_micro_precision@5 = {:.2f}".format(tail_micro_precision5) + "\n"
    test_result += "AAPD BBSN Test - tail_micro_NDCG@3 = {:.2f}".format(tail_micro_ndcg3) + "\n"
    test_result += "AAPD BBSN Test - tail_micro_NDCG@5 = {:.2f}".format(tail_micro_ndcg5) + "\n\n"

    # Tail - Macro
    print("AAPD BBSN Test - tail_macro_F1 = {:.2f}".format(tail_macro_F1 * 100))
    print("AAPD BBSN Test - tail_macro_precision@1 = {:.2f}".format(tail_macro_precision1))
    print("AAPD BBSN Test - tail_macro_precision@3 = {:.2f}".format(tail_macro_precision3))
    print("AAPD BBSN Test - tail_macro_precision@5 = {:.2f}".format(tail_macro_precision5))
    test_result += "AAPD BBSN Test - tail_macro_F1 = {:.2f}".format(tail_macro_F1 * 100) + "\n"
    test_result += "AAPD BBSN Test - tail_macro_precision@1 = {:.2f}".format(tail_macro_precision1) + "\n"
    test_result += "AAPD BBSN Test - tail_macro_precision@3 = {:.2f}".format(tail_macro_precision3) + "\n"
    test_result += "AAPD BBSN Test - tail_macro_precision@5 = {:.2f}".format(tail_macro_precision5) + "\n\n"

    with open(result_path, "w") as fw:
        fw.write(test_result)

def main():
    args = parse_args()
    print_args(args)
    set_seed(args)

    pair_it1, pair_it2, valid_data_it, test_data_it = read_data(args)

    model = CLS_BBSN4BRANCH(args)
    model.cuda()
    ckpt_prefix = "aapd-checkpoint.pt"

    do_train(model, pair_it1, pair_it2, valid_data_it, ckpt_prefix, args)

    model.load_state_dict(torch.load(ckpt_prefix))
    result_path = "aapd-bbsn.txt"
    do_test(model, test_data_it, result_path=result_path)


if __name__ == '__main__':
    main()