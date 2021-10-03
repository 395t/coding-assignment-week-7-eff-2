import csv
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import torch

LABEL_MAPPING = {'positive': 1, 'negative': 0}

def encode_data(data, tokenizer, max_seq_len):
    encoded_data = [tokenizer(i[0], truncation = True) for i in data]
    for i in range(len(encoded_data)):
        encoded_data[i]['label'] = torch.tensor(data[i][1])
        encoded_data[i]['input_ids'] = torch.cat((torch.tensor(encoded_data[i]['input_ids']), torch.zeros(max_seq_len - len(encoded_data[i]['input_ids'])))).int()
        encoded_data[i]['attention_mask'] = torch.cat((torch.tensor(encoded_data[i]['attention_mask']), torch.zeros(max_seq_len - len(encoded_data[i]['attention_mask'])))).int()
    return encoded_data

def create_and_split_data(dataset_path, tokenizer, max_seq_len, train_num=10000, val_num = 5000):
    data = list()
    with open(dataset_path, 'r', encoding='utf-8') as csvfile:
        header = True
        f = csv.reader(csvfile)
        for row in f:
            if header:
                header = False
                continue
            data.append((row[0], LABEL_MAPPING[row[1]]))
    indices = np.arange(0, train_num + val_num)
    data = encode_data(data, tokenizer, max_seq_len)
    random.shuffle(indices)
    return [data[i] for i in indices[:train_num]], [data[i] for i in indices[train_num:train_num + train_num]]


class IMDBDataset(Dataset):
    def __init__(self, dataset):
        self.labels = dataset

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.labels[idx]['input_ids'], self.labels[idx]['attention_mask'], self.labels[idx]['label']


def load_data(dataset_path, tokenizer, max_seq_len, num_workers=8, batch_size=128):
    random.seed(0)
    train, validation = create_and_split_data(dataset_path, tokenizer, max_seq_len)
    train_dataset = IMDBDataset(train)
    val_dataset = IMDBDataset(validation)
    return DataLoader(train_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True), \
           DataLoader(val_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()

from datasets import load_dataset, load_metric

def load_ds():
    return load_dataset("imdb", ignore_verifications=True)

def preprocess_function(examples, tokenizer):
    return tokenizer(examples['text'], truncation=True)

def prepare_data_and_metric(tokenizer):
    from utils import load_data, preprocess_function
    dataset = load_ds()
    return dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True), load_metric("accuracy")