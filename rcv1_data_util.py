import json
import torch
import random
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchtext.legacy import data
from collections import defaultdict

from transformers import BertTokenizer

def get_tail():
    rcv1_train = pd.read_csv('data/rcv1/rcv1-train.csv', sep=',')

    topics_103 = []
    with open('data/rcv1/topics_103.txt', 'r') as f:
        for l in f.readlines():
            topics_103.append(l.strip())

    cate_count = defaultdict(int)
    for i in range(len(rcv1_train)):
        category = eval(rcv1_train['tags'][i])
        for cate in category:
            cate_count[cate] += 1

    tail_100 = []
    for i in range(102):
        key = topics_103[i]
        if cate_count[key] < 100:
            tail_100.append(i)

    return tail_100

def util_mutilabel(topics_103, categories):
    mutilabel = list(np.zeros(103))
    for i in range(len(topics_103)):
        if categories.__contains__(topics_103[i]):
            mutilabel[i] = 1.0
    return mutilabel

def build_train_and_valid_data():
    print("build_train_and_valid_data")
    rcv1_train = pd.read_csv("data/rcv1/rcv1-train.csv", sep=',')
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    total_num = rcv1_train.shape[0]
    train_num = int(total_num * 0.8)
    train_index = random.sample(range(0, total_num), train_num)
    train_index = sorted(train_index)
    valid_index = [e for e in range(total_num) if e not in train_index]

    topics_103 = []
    with open('data/rcv1/topics_103.txt', 'r') as f:
        for l in f.readlines():
            topics_103.append(l.strip())

    with open("data/rcv1/rcv1_train.json", "w") as fw1:
        for i in tqdm(train_index):
            curr_id = rcv1_train['id'][i]
            curr_text = rcv1_train['text'][i]
            curr_label = eval(rcv1_train['tags'][i])
            curr_tokens = tokenizer(curr_text, add_special_tokens=True, max_length=512, truncation=True, padding='max_length', return_tensors='pt')
            curr_mutilabel = util_mutilabel(topics_103, curr_label)
    
            input_dict = dict()
            input_dict['id'] = str(curr_id)
            input_dict['tokens'] = curr_tokens['input_ids'][0].detach().numpy().tolist()
            input_dict['attn_mask'] = curr_tokens['attention_mask'][0].detach().numpy().tolist()
            input_dict['mutilabel'] = curr_mutilabel
            input_obj = json.dumps(input_dict)
            fw1.write(input_obj + "\n")

    with open("data/rcv1/rcv1_valid.json", "w") as fw2:
        for i in tqdm(valid_index):
            curr_id = rcv1_train['id'][i]
            curr_text = rcv1_train['text'][i]
            curr_label = eval(rcv1_train['tags'][i])
            curr_tokens = tokenizer(curr_text, add_special_tokens=True, max_length=512, truncation=True, padding='max_length', return_tensors='pt')
            curr_mutilabel = util_mutilabel(topics_103, curr_label)

            input_dict = dict()
            input_dict['id'] = str(curr_id)
            input_dict['tokens'] = curr_tokens['input_ids'][0].detach().numpy().tolist()
            input_dict['attn_mask'] = curr_tokens['attention_mask'][0].detach().numpy().tolist()
            input_dict['mutilabel'] = curr_mutilabel
            input_obj = json.dumps(input_dict)
            fw2.write(input_obj + "\n")

def build_test_data():
    print("build_test_data")
    rcv1_test = pd.read_csv("data/rcv1/rcv1-test.csv", sep=",")
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    topics_103 = []
    with open('data/rcv1/topics_103.txt', 'r') as f:
        for l in f.readlines():
            topics_103.append(l.strip())

    with open("data/rcv1/rcv1_test.json", "w") as fw:
        for i in tqdm(range(len(rcv1_test))):
            curr_id = rcv1_test['id'][i]
            curr_text = rcv1_test['text'][i]
            if type(curr_text) == float:
                curr_text = ""
            curr_label = eval(rcv1_test['tags'][i])
            curr_tokens = tokenizer(curr_text, add_special_tokens=True, max_length=512, truncation=True, padding='max_length', return_tensors='pt')
            curr_mutilabel = util_mutilabel(topics_103, curr_label)

            input_dict = dict()
            input_dict['id'] = str(curr_id)
            input_dict['tokens'] = curr_tokens['input_ids'][0].detach().numpy().tolist()
            input_dict['attn_mask'] = curr_tokens['attention_mask'][0].detach().numpy().tolist()
            input_dict['mutilabel'] = curr_mutilabel
            input_obj = json.dumps(input_dict)
            fw.write(input_obj + "\n")

def build_bbsn_data(path):
    print("build_bbsn_data")
    rcv1_train = pd.read_csv('data/rcv1/rcv1-train.csv', sep=',')

    topics_103 = []
    with open('data/rcv1/topics_103.txt', 'r') as f:
        for l in f.readlines():
            topics_103.append(l.strip())

    cate_have = defaultdict(list)
    cate_count = defaultdict(int)
    for i in range(len(rcv1_train)):
        for label in eval(rcv1_train['tags'][i]):
            cate_have[label].append(i)
            cate_count[label] += 1

    cate_count = dict(sorted(cate_count.items(), key=lambda x: x[1], reverse=True))
    total_count = sum(cate_count.values())

    p_uniform = []
    cate_keys = list(cate_count.keys())
    for i in range(len(cate_keys)):
        p_uniform.append(cate_count[cate_keys[i]] / total_count)

    def uniform_choice():
        sums = 0
        rand = random.random()
        for cc, pp in zip(cate_keys, p_uniform):
            sums += pp
            if rand < sums:
                return cc

    n_max = list(cate_count.values())[0]
    w = []
    for c in cate_keys:
        w.append(n_max / cate_count[c])
    p_reversed = []
    for i in range(len(w)):
        p_reversed.append(w[i] / sum(w))

    def reversed_choice():
        sums = 0
        rand = random.random()
        for cc, pp in zip(cate_keys, p_reversed):
            sums += pp
            if rand < sums:
                return cc

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    with open(path, "w") as fw:
        for i in tqdm(range(len(rcv1_train))):
            c1 = reversed_choice()
            c2 = uniform_choice()

            index1 = random.choice(cate_have[c1])
            index2 = random.choice(cate_have[c2])

            curr_id1 = rcv1_train['id'][index1]
            curr_text1 = rcv1_train['text'][index1]
            curr_label1 = eval(rcv1_train['tags'][index1])
            curr_tokens1 = tokenizer(curr_text1, add_special_tokens=True, max_length=512, truncation=True, padding='max_length', return_tensors='pt')
            curr_mutilabel1 = util_mutilabel(topics_103, curr_label1)

            curr_id2 = rcv1_train['id'][index2]
            curr_text2 = rcv1_train['text'][index2]
            curr_label2 = eval(rcv1_train['tags'][index2])
            curr_tokens2 = tokenizer(curr_text2, add_special_tokens=True, max_length=512, truncation=True, padding='max_length', return_tensors='pt')
            curr_mutilabel2 = util_mutilabel(topics_103, curr_label2)

            input_dict = dict()
            input_dict['id1'] = str(curr_id1)
            input_dict['tokens1'] = curr_tokens1['input_ids'][0].detach().numpy().tolist()
            input_dict['attn_mask1'] = curr_tokens1['attention_mask'][0].detach().numpy().tolist()
            input_dict['mutilabel1'] = curr_mutilabel1
            input_dict['id2'] = str(curr_id2)
            input_dict['tokens2'] = curr_tokens2['input_ids'][0].detach().numpy().tolist()
            input_dict['attn_mask2'] = curr_tokens2['attention_mask'][0].detach().numpy().tolist()
            input_dict['mutilabel2'] = curr_mutilabel2
            ground_truth = list(set(curr_label1).intersection(set(curr_label2)))
            input_dict['ground_truth'] = util_mutilabel(topics_103, ground_truth)
            curr_label1.extend(curr_label2)
            mask = util_mutilabel(topics_103, curr_label1)
            input_dict['mask'] = mask
            input_obj = json.dumps(input_dict)
            fw.write(input_obj + "\n")



if __name__ == "__main__":
    build_train_and_valid_data()
    build_test_data()
    build_bbsn_data("data/rcv1/rcv1_bbsn1.json")
    build_bbsn_data("data/rcv1/rcv1_bbsn2.json")




































