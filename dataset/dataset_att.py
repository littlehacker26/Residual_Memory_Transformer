from torch.utils.data import Dataset, DataLoader
import json
import random
from tqdm import tqdm
import numpy as np
import torch
import copy
import re
import string
import sys
import random
import jsonlines


class Sentiment_Dataset(Dataset):

    def __init__(self, json_path, tokenizer, is_training=True, args=None):
        super(Sentiment_Dataset, self).__init__()

        self.tokenizer = tokenizer
        self.is_training = is_training
        np.set_printoptions(threshold=sys.maxsize)
        self.args = args
        
        self.tokenizer.padding_side = "right"
        self.read_content(json_path)
        
        
    def read_content(self, json_path):
        print("reading data from %s ..." % json_path)
        self.record = []
        with open(json_path) as out:

            lines = out.readlines()
            
            for item in tqdm(lines):
                
                c = item
                c = c.strip()
                c_output_ids = self.tokenizer(c, return_tensors="np")['input_ids'][0].tolist()
                    
                concept_set_input_ids = self.tokenizer(f"Sentiment: Positive", return_tensors="np")['input_ids'][0].tolist()
                concept_set_input_ids_ = self.tokenizer(f"Sentiment: Negative", return_tensors="np")['input_ids'][0].tolist()
                
                if len(c_output_ids)>150:
                    continue

                self.record.append({
                        "encode_input":concept_set_input_ids,
                        "encode_input_":concept_set_input_ids_,
                        "context":c_output_ids,
                        "input_ids":c_output_ids})
                    
        if self.is_training: random.shuffle(self.record)

    def __len__(self):
        return len(self.record)

    def __getitem__(self, index):
        item = self.record[index]

        return item


# class Sentiment_Dataset(Dataset):

#     def __init__(self, json_path, tokenizer, is_training=True, args=None):
#         super(Sentiment_Dataset, self).__init__()

#         self.tokenizer = tokenizer
#         self.is_training = is_training
#         np.set_printoptions(threshold=sys.maxsize)
#         self.args = args
        
#         self.tokenizer.padding_side = "right"
#         self.read_content(json_path)
        
        
#     def read_content(self, json_path):
#         print("reading data from %s ..." % json_path)
#         self.record = []
#         with open(json_path) as out:

#             # lines = json.load(out)
#             lines = out.readlines()
            
#             for item in tqdm(lines):
                
#                 c = item["content"]
#                 c = c.strip()
#                 c_output_ids = self.tokenizer(c, return_tensors="np")['input_ids'][0].tolist()
                
#                 t = item["target"]
#                 t = t.strip()                
                
#                 if len(c_output_ids)==0:
#                     c_output_ids_ = self.tokenizer(t, return_tensors="np")['input_ids'][0].tolist()                    
#                 else:
#                     c_output_ids_ = self.tokenizer(" "+t, return_tensors="np")['input_ids'][0].tolist()
                
#                 if (len(c_output_ids_)+len(c_output_ids))>200:
#                     continue
                    
#                 concept_set_input_ids = self.tokenizer(f"Sentiment: Positive", return_tensors="np")['input_ids'][0].tolist()
#                 concept_set_input_ids_ = self.tokenizer(f"Sentiment: Negative", return_tensors="np")['input_ids'][0].tolist()

#                 self.record.append({
#                         "encode_input":concept_set_input_ids,
#                         "encode_input_":concept_set_input_ids_,
#                         "context":c_output_ids,
#                         "input_ids":c_output_ids+c_output_ids_})
                    
              
#         if self.is_training: random.shuffle(self.record)

#     def __len__(self):
#         return len(self.record)

#     def __getitem__(self, index):
#         item = self.record[index]

#         return item

    

    
class Senti_Prompt_Data(Dataset):
    def __init__(self, json_path, tokenizer, is_training=False, args=None):
        super(Senti_Prompt_Data, self).__init__()

        self.tokenizer = tokenizer
        np.set_printoptions(threshold=sys.maxsize)
        self.args = args
        
        self.is_training = False
        
        self.record = []
        self.read_content(json_path)
        
    def read_content(self, json_path):
        print("reading data from %s ..." % json_path)
        
        
        with open(str(json_path), "r+", encoding="utf8") as f:
            for item in jsonlines.Reader(f):
                prompt = item["prompt"]["text"]
                
                context = self.tokenizer(prompt.strip(), return_tensors="np")['input_ids'][0].tolist()
                
                if len(context)<1:
                    continue
                
                concept_set_input_ids = self.tokenizer(f"Sentiment: Positive", return_tensors="np")['input_ids'][0].tolist()
                concept_set_input_ids_ = self.tokenizer(f"Sentiment: Negative", return_tensors="np")['input_ids'][0].tolist()
            
                self.record.append({
                        "encode_input":concept_set_input_ids,
                        "encode_input_":concept_set_input_ids_,
                        "context":context,
                        "input_ids":context})

    def __len__(self):
        return len(self.record)

    def __getitem__(self, index):
        item = self.record[index]
        return item    



def data_wrapper(dataset, tokenizer, is_training):
    batch_size = len(dataset)
    new_dataset = {'encode_input_': [d['encode_input_'] for d in dataset],
                   'encode_input': [d['encode_input'] for d in dataset], 
                   'input_ids': [d['input_ids'] for d in dataset],
                   'context': [d['context'] for d in dataset]}

    _PAD = tokenizer.eos_token_id
    
    if is_training:
        max_concept_len = max([len(d['input_ids']) for d in dataset])
        concept_set_input = np.full((batch_size, max_concept_len), _PAD, dtype=np.int64)
        for i, d in enumerate(dataset):
            data = d['input_ids'][:max_concept_len]
            concept_set_input[i, :len(data)] = data
        new_dataset['input_ids'] = torch.from_numpy(concept_set_input)
        
        
        max_output_len = max([len(d['input_ids']) for d in dataset])
        mask_ids = np.full((batch_size, max_output_len), 0, dtype=np.int64)
        for i, d in enumerate(dataset):
            mask_ids[i, :len(d['input_ids'])] = 1
        new_dataset['attention_mask'] = torch.from_numpy(mask_ids)
        
    else:
        max_concept_len = max([len(d['input_ids']) for d in dataset])
        concept_set_input = np.full((batch_size, max_concept_len), _PAD, dtype=np.int64)
        for i, d in enumerate(dataset):
            data = d['input_ids'][:max_concept_len]
            concept_set_input[i, -len(data):] = data
        new_dataset['input_ids'] = torch.from_numpy(concept_set_input)
        
        
        max_output_len = max([len(d['input_ids']) for d in dataset])
        mask_ids = np.full((batch_size, max_output_len), 0, dtype=np.int64)
        for i, d in enumerate(dataset):
            mask_ids[i, -len(d['input_ids']):] = 1
        new_dataset['attention_mask'] = torch.from_numpy(mask_ids)        
        

    
       
    max_concept_len = max([len(d['encode_input']) for d in dataset])
    encode_input = np.full((batch_size, max_concept_len), _PAD, dtype=np.int64)
    for i, d in enumerate(dataset):
        data = d['encode_input'][:max_concept_len]
        encode_input[i, :len(data)] = data
    new_dataset['encode_input'] = torch.from_numpy(encode_input)
    
    
    max_concept_len = max([len(d['encode_input_']) for d in dataset])
    encode_input = np.full((batch_size, max_concept_len), _PAD, dtype=np.int64)
    for i, d in enumerate(dataset):
        data = d['encode_input_'][:max_concept_len]
        encode_input[i, :len(data)] = data
    new_dataset['encode_input_'] = torch.from_numpy(encode_input)
    
    return new_dataset





def get_data_loader(dataset, batch_size):
    collate_fn = lambda d: data_wrapper(d, dataset.tokenizer, dataset.is_training)
    return DataLoader(dataset, 
        batch_size=batch_size,
        num_workers=4,
        collate_fn=collate_fn
    )