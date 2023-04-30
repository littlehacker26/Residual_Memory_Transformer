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

    
# class C2Gen(Dataset):
#     def __init__(self, json_path, tokenizer, args=None):
#         super(C2Gen, self).__init__()

#         self.tokenizer = tokenizer
#         np.set_printoptions(threshold=sys.maxsize)
#         self.args = args
        
#         self.record = []
#         self.read_content(json_path)
        
#     def read_content(self, json_path):
#         print("reading data from %s ..." % json_path)
        
#         data = json.load(open(json_path,"r"))
#         for r in data["rows"]:            
#             keyword = ', '.join(r["row"]["keywords"])
#             context =  r["row"]["context"]
#             concept = '#'.join(r["row"]["keywords"])

            
#             context = self.tokenizer(context, return_tensors="np")['input_ids'][0].tolist()
#             concept_set_input_ids = self.tokenizer(f"Include:{keyword}; Context:{len(context)}; Target:20", return_tensors="np")['input_ids'][0].tolist()
            
#             self.record.append({
#                         "item": [0],
#                         "concept_set":concept,
#                         "encode_input":concept_set_input_ids,
#                         "input_ids":context,
#                         "context":context})
#         random.shuffle(self.record)
        

#     def __len__(self):
#         return len(self.record)

#     def __getitem__(self, index):
#         item = self.record[index]
#         return item
    

class Length_CommonGenDataset(Dataset):

    def __init__(self, json_path, tokenizer, is_training=False, args=None):
        super(Lencontrol_CommonGenDataset, self).__init__()

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
            # if self.is_training:
                # lines = out.readlines()[:100]
            #     random.seed(42)
            #     lines = random.sample(out.readlines(), self.args.training_sample_num)
            # else:
            lines = out.readlines()
                
            for l in tqdm(lines):
                item = json.loads(l.strip())
                concept_set = ', '.join(item['concept_set'].split('#'))
                s

                gt = copy.deepcopy(item['scene'])
                if self.is_training:
                    for c in item['scene']:
                        c = c.strip()
                        # if c.endswith('.'):
                        #     c= c[:-1].strip()                   
                        c_output_ids = self.tokenizer(c, return_tensors="np")['input_ids'][0].tolist()
    
                        _concept_set_input_ids = self.tokenizer(f"Include:{concept_set}; Context:0; Target:{len(c_output_ids)}", return_tensors="np")['input_ids'][0].tolist()               

                        self.record.append({
                            "concept_set":item['concept_set'],
                            "encode_input":_concept_set_input_ids,
                            "input_ids":c_output_ids,
                            "context":None,
                            "item":0})
                else:
                    self.record.append({
                            "concept_set":item['concept_set'],
                            "encode_input":self.tokenizer(f"Include:{concept_set}; Context:1; Target:{self.args.generated_len}", return_tensors="np")['input_ids'][0].tolist(),
                            "input_ids":[464],
                            "context":[464],
                            "item": 0})
                        
        if self.is_training: random.shuffle(self.record)

    def __len__(self):
        return len(self.record)

    def __getitem__(self, index):
        item = self.record[index]

        return item
    
    
    
class Length_kw_Dataset(Dataset):

    def __init__(self, json_path, tokenizer, is_training=False, args=None):
        super(Length_kw_Dataset, self).__init__()

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

            lines = json.load(out)
            
            for item in tqdm(lines):
                
                c = item["content"]
                c = c.strip()
                c_output_ids = self.tokenizer(c, return_tensors="np")['input_ids'][0].tolist()
                
                                
                concept_set = ', '.join(item['keywords'].split('#'))
                t = item["target"]
                t = t.strip()
                
                if len(c_output_ids)==0:
                    c_output_ids_ = self.tokenizer(t, return_tensors="np")['input_ids'][0].tolist()                    
                else:
                    c_output_ids_ = self.tokenizer(" "+t, return_tensors="np")['input_ids'][0].tolist()
                
                if (len(c_output_ids_)+len(c_output_ids))>200:
                    continue
                
                if self.is_training:
                    concept_set_input_ids = self.tokenizer(f"Include:{concept_set}; Context:{len(c_output_ids)}; Target:{len(c_output_ids_)}", return_tensors="np")['input_ids'][0].tolist()               
                    self.record.append({
                        "item": [0],
                        "concept_set":item['keywords'],
                        "encode_input":concept_set_input_ids,
                        "context":c_output_ids,
                        "input_ids":c_output_ids+c_output_ids_})
                    
                else:
                    concept_set_input_ids = self.tokenizer(f"Include:{concept_set}; Context:{len(c_output_ids)}; Target:16", return_tensors="np")['input_ids'][0].tolist()
                    self.record.append({
                            "item": [0],
                            "concept_set":item['keywords'],
                            "encode_input":concept_set_input_ids,
                            "context":c_output_ids,
                            "input_ids":c_output_ids})
                        
        if self.is_training: random.shuffle(self.record)

    def __len__(self):
        return len(self.record)

    def __getitem__(self, index):
        item = self.record[index]

        return item
    
    
    
    
class keyword_CommonGenDataset(Dataset):

    def __init__(self, json_path, tokenizer, is_training=False, args=None):
        super(keyword_CommonGenDataset, self).__init__()

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

            lines = json.load(out)
            
            for item in tqdm(lines):
                
                c = item["content"]
                c = c.strip()
                c_output_ids = self.tokenizer(c, return_tensors="np")['input_ids'][0].tolist()
                
                                
                concept_set = ', '.join(item['keywords'].split('#'))
                
                t = item["target"]
                t = t.strip()                
                
                if len(c_output_ids)==0:
                    c_output_ids_ = self.tokenizer(t, return_tensors="np")['input_ids'][0].tolist()                    
                else:
                    c_output_ids_ = self.tokenizer(" "+t, return_tensors="np")['input_ids'][0].tolist()
                
                if (len(c_output_ids_)+len(c_output_ids))>200:
                    continue
                
                if self.is_training:
                    # concept_set_input_ids = self.tokenizer(f"Include:{concept_set}; Context:{len(c_output_ids)}", return_tensors="np")['input_ids'][0].tolist()
                    
                    concept_set_input_ids = self.tokenizer(f"Include:{concept_set}; Context:{len(c_output_ids)}; Target:{len(c_output_ids_)}", return_tensors="np")['input_ids'][0].tolist()
                    # concept_set_input_ids = self.tokenizer(f"Include:{concept_set}", return_tensors="np")['input_ids'][0].tolist()

                    self.record.append({
                        "item": [0],
                        "concept_set":item['keywords'],
                        "encode_input":concept_set_input_ids,
                        "context":c_output_ids,
                        "input_ids":c_output_ids+c_output_ids_})
                    
                else:
                    concept_set_input_ids = self.tokenizer(f"Include:{concept_set}; Context:{len(c_output_ids)}", return_tensors="np")['input_ids'][0].tolist()
                    
                    # concept_set_input_ids = self.tokenizer(f"Include:{concept_set}; Context:{len(c_output_ids)}; Target:{len(c_output_ids_)}", return_tensors="np")['input_ids'][0].tolist()

                    self.record.append({
                            "item": [0],
                            "concept_set":item['keywords'],
                            "encode_input":concept_set_input_ids,
                            "context":c_output_ids,
                            "input_ids":c_output_ids+c_output_ids_})
                        
        if self.is_training: random.shuffle(self.record)

    def __len__(self):
        return len(self.record)

    def __getitem__(self, index):
        item = self.record[index]

        return item
    
    
    
class CommonGenDataset(Dataset):

    def __init__(self, json_path, tokenizer, is_training=False, args=None):
        super(CommonGenDataset, self).__init__()

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
                for l in tqdm(lines):
                    item = json.loads(l.strip())
                    concept_set = ', '.join(item['concept_set'].split('#'))

                    gt = copy.deepcopy(item['scene'])
                    
                    if self.is_training:
                        for c in item['scene']:
                            c = c.strip()                 
                            c_output_ids = self.tokenizer(c, return_tensors="np")['input_ids'][0].tolist()

                            _concept_set_input_ids = self.tokenizer(f"Include:{concept_set}; Context:0; Target:{len(c_output_ids)}", return_tensors="np")['input_ids'][0].tolist()

                            self.record.append({
                                "concept_set":item['concept_set'],
                                "encode_input":_concept_set_input_ids,
                                "input_ids":c_output_ids,
                                "context":None,
                                "item":0})
                    else:
                        self.record.append({
                                "concept_set":item['concept_set'],
                                "encode_input":self.tokenizer(f"Include:{concept_set}; Context:0; Target:{self.args.generated_len}", return_tensors="np")['input_ids'][0].tolist(),
                                # "encode_input":self.tokenizer(f"Include:{concept_set}", return_tensors="np")['input_ids'][0].tolist(),
                                "input_ids":[464],
                                "context":[464],
                                "item": 0})
                        
        if self.is_training: random.shuffle(self.record)

    def __len__(self):
        return len(self.record)

    def __getitem__(self, index):
        item = self.record[index]

        return item
    
    
    
class C2Gen(Dataset):
    def __init__(self, json_path, tokenizer, args=None):
        super(C2Gen, self).__init__()

        self.tokenizer = tokenizer
        np.set_printoptions(threshold=sys.maxsize)
        self.args = args
        
        self.record = []
        self.read_content(json_path)
        
    def read_content(self, json_path):
        print("reading data from %s ..." % json_path)
        
        data = json.load(open(json_path,"r"))
        for r in data["rows"]:            
            keyword = ', '.join(r["row"]["keywords"])
            context =  r["row"]["context"]
            concept = '#'.join(r["row"]["keywords"])

            
            context = self.tokenizer(context, return_tensors="np")['input_ids'][0].tolist()
            concept_set_input_ids = self.tokenizer(f"Include:{keyword}; Context:{len(context)}; Target:{self.args.generated_len}", return_tensors="np")['input_ids'][0].tolist()
            
            self.record.append({
                        "item": [0],
                        "concept_set":concept,
                        "encode_input":concept_set_input_ids,
                        "input_ids":context,
                        "context":context})
        random.shuffle(self.record)
        

    def __len__(self):
        return len(self.record)

    def __getitem__(self, index):
        item = self.record[index]
        return item
    

def data_wrapper(dataset, tokenizer):
    batch_size = len(dataset)
    new_dataset = {'concept_set': [d['concept_set'] for d in dataset], 
                   'encode_input': [d['encode_input'] for d in dataset], 
                   'input_ids': [d['input_ids'] for d in dataset],
                   'context': [d['context'] for d in dataset],
                   'item': [d['item'] for d in dataset]}

    _PAD = tokenizer.eos_token_id
    
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
    
    
    max_concept_len = max([len(d['encode_input']) for d in dataset])
    encode_input = np.full((batch_size, max_concept_len), _PAD, dtype=np.int64)
    for i, d in enumerate(dataset):
        data = d['encode_input'][:max_concept_len]
        encode_input[i, :len(data)] = data
    new_dataset['encode_input'] = torch.from_numpy(encode_input)
    
    
    max_output_len = max([len(d['input_ids']) for d in dataset])
    mask_ids = np.full((batch_size, max_output_len), 1, dtype=np.int64)
    for i, d in enumerate(dataset):
        if d['context']!=None:
            mask_ids[i, :len(d['context'])] = 0
    new_dataset['mask_ids'] = torch.from_numpy(mask_ids)
                                  
    return new_dataset



def get_data_loader(dataset, batch_size):
    collate_fn = lambda d: data_wrapper(d, dataset.tokenizer)
    return DataLoader(dataset, 
        batch_size=batch_size,
        num_workers=4,
        collate_fn=collate_fn
    )
