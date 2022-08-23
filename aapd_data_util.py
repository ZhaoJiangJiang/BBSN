import json
import torch
import random
import collections
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchtext.legacy import data
from collections import defaultdict

from transformers import BertTokenizer

def util_get_tail_label(threshold=1000):
    aapd_train = pd.read_csv("data/aapd/aapd_train.csv", sep="\t", header=None, names=['label', 'text'])
    cate_count = defaultdict(int)
    for i in range(len(aapd_train)):
        curr_mutilabel = eval(aapd_train['label'][i])
        for j in range(len(curr_mutilabel)):
            if curr_mutilabel[j] == 1:
                cate_count[j] += 1

    tail_label = [key for key in cate_count.keys() if cate_count[key] < threshold]
    return tail_label

def build_train_and_valid_data():
    print("build_train_and_valid_data")
    aapd_train = pd.read_csv("data/aapd/aapd_train.csv", sep="\t", header=None, names=['label', 'text'])

    total_num = aapd_train.shape[0]
    train_num = int(total_num * 0.8)
    train_index = random.sample(range(0, total_num), train_num)
    train_index = sorted(train_index)
    valid_index = [e for e in range(total_num) if e not in train_index]

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    with open("data/aapd/aapd_train.json", "w") as fw1:
        for i in tqdm(train_index):
            curr_text = aapd_train['text'][i]
            curr_mutilabel = eval(aapd_train['label'][i])
            curr_tokens = tokenizer(curr_text, add_special_tokens=True, max_length=256, truncation=True, padding='max_length', return_tensors='pt')
            input_dict = dict()
            input_dict['tokens'] = curr_tokens['input_ids'][0].detach().numpy().tolist()
            input_dict['attn_mask'] = curr_tokens['attention_mask'][0].detach().numpy().tolist()
            input_dict['mutilabel'] = curr_mutilabel
            input_obj = json.dumps(input_dict)
            fw1.write(input_obj + "\n")

    with open("data/aapd/aapd_valid.json", "w") as fw2:
        for i in tqdm(valid_index):
            curr_text = aapd_train['text'][i]
            curr_mutilabel = eval(aapd_train['label'][i])
            curr_tokens = tokenizer(curr_text, add_special_tokens=True, max_length=256, truncation=True, padding='max_length', return_tensors='pt')
            input_dict = dict()
            input_dict['tokens'] = curr_tokens['input_ids'][0].detach().numpy().tolist()
            input_dict['attn_mask'] = curr_tokens['attention_mask'][0].detach().numpy().tolist()
            input_dict['mutilabel'] = curr_mutilabel
            input_obj = json.dumps(input_dict)
            fw2.write(input_obj + "\n")

def build_test_data():
    print("build_test_data")
    aapd_test = pd.read_csv("data/aapd/aapd_test.csv", sep="\t", header=None, names=['label', 'text'])
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    with open("data/aapd/aapd_test.json", "w") as fw:
        for i in tqdm(range(len(aapd_test))):
            curr_text = aapd_test['text'][i]
            curr_mutilabel = eval(aapd_test['label'][i])
            curr_tokens = tokenizer(curr_text, add_special_tokens=True, max_length=256, truncation=True, padding='max_length', return_tensors='pt')
            input_dict = dict()
            input_dict['tokens'] = curr_tokens['input_ids'][0].detach().numpy().tolist()
            input_dict['attn_mask'] = curr_tokens['attention_mask'][0].detach().numpy().tolist()
            input_dict['mutilabel'] = curr_mutilabel
            input_obj = json.dumps(input_dict)
            fw.write(input_obj + "\n")


def build_bbsn_data(path):
    print("build_bbsn_data")
    aapd_uniform = []
    with open("data/aapd/aapd_train.json", "r") as fr:
        for line in fr.readlines():
            tmp = json.loads(line)
            aapd_uniform.append(tmp)

    cate_have = defaultdict(list)
    cate_count = defaultdict(int)
    for i in range(len(aapd_uniform)):
        curr_mutilabel = aapd_uniform[i]['mutilabel']
        for j in range(len(curr_mutilabel)):
            if curr_mutilabel[j] == 1:
                cate_have[j].append(i)
                cate_count[j] += 1

    cate_count = dict(sorted(cate_count.items(), key=lambda x: x[1], reverse=True))
    total_count = sum(cate_count.values())

    cate_keys = list(cate_count.keys())

    uniform_p = []
    for i in range(len(cate_keys)):
        uniform_p.append(cate_count[cate_keys[i]] / total_count)

    def uniform_choice():
        sums = 0
        rand = random.random()
        for cc, pp in zip(cate_keys, uniform_p):
            sums += pp
            if rand < sums:
                return cc

    n_max = list(cate_count.values())[0]
    w = []
    for c in cate_keys:
        w.append(n_max / cate_count[c])
    reverse_p = []
    for i in range(len(w)):
        reverse_p.append(w[i] / sum(w))

    def reversed_choice():
        sums = 0
        rand = random.random()
        for cc, pp in zip(cate_keys, reverse_p):
            sums += pp
            if rand < sums:
                return cc

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    with open(path, "w") as fw:
        for i in tqdm(range(len(aapd_uniform))):
            c1 = reversed_choice()

            index1 = random.choice(cate_have[c1])
            index2 = i

            curr_tokens1 = aapd_uniform[index1]['tokens']
            curr_attn_mask1 = aapd_uniform[index1]['attn_mask']
            curr_mutilabel1 = aapd_uniform[index1]['mutilabel']

            curr_tokens2 = aapd_uniform[index2]['tokens']
            curr_attn_mask2 = aapd_uniform[index2]['attn_mask']
            curr_mutilabel2 = aapd_uniform[index2]['mutilabel']

            curr_ground_truth = [0] * 54
            for j in range(54):
                if curr_mutilabel1[j] == 1 and curr_mutilabel2[j] == 1:
                    curr_ground_truth[j] = 1.0

            curr_mask = [0] * 54
            for j in range(54):
                if curr_mutilabel1[j] == 1 or curr_mutilabel2[j] == 1:
                    curr_mask[j] = 1.0

            input_dict = dict()
            input_dict['tokens1'] = curr_tokens1
            input_dict['attn_mask1'] = curr_attn_mask1
            input_dict['mutilabel1'] = curr_mutilabel1
            input_dict['tokens2'] = curr_tokens2
            input_dict['attn_mask2'] = curr_attn_mask2
            input_dict['mutilabel2'] = curr_mutilabel2
            input_dict['ground_truth'] = curr_ground_truth
            input_dict['mask'] = curr_mask
            input_obj = json.dumps(input_dict)
            fw.write(input_obj + "\n")


if __name__ == "__main__":
    build_train_and_valid_data()
    build_test_data()
    build_bbsn_data("data/aapd/aapd_bbsn1.json")
    build_bbsn_data("data/aapd/aapd_bbsn2.json")