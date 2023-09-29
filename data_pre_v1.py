# %%
import os
import json
import spacy
import nltk
from nltk.tokenize import sent_tokenize
import random
from glob import glob
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd


def encoder_data_target(data):
    
    nn = ['VERB', 'NOUN'] 
    nlp = spacy.load('en_core_web_sm')# 加载预训练模型
    res = []
    
    for line in data:
        line_content  = ' '.join(line[:-1])
        target = line[-1]
        
        if len(target.split())<6:
            continue
                
        doc = nlp(target.lower())
        tokens = [token for token in doc]           # 将句子切分成单词
        pos = [token.pos_ for token in doc]         # 词性标注
        lem = [token.lemma_ for token in doc]       # 词性还原
        
        sen = []
        for t,p,l in zip(tokens, pos, lem):
            if (p in nn) and (len(str(l))>=3) and ('-'  not in str(l)) and ('.' not in str(l)):
                sen.append(str(l))
                
        sen = list(set(sen))
        
        if len(sen)<3:
            continue
            
        if len(sen)>5:
            
            sen =  random.sample(sen, 5)
            
        random.shuffle(sen)
 
        res.append({
                    "content":line_content,
                    "target":target,
                    "keywords": '#'.join(sen)})
        
    return res



def encoder_data(data):
    
    nn = ['VERB', 'NOUN'] 
    nlp = spacy.load('en_core_web_sm')# 加载预训练模型

    res = []
    
    for line in data:
        
        cut_pos = random.randint(8,16)
        line = line.strip(' \n\t\r),\\')
        
        if "amp" in line:
            continue
        
        split_line= line.split()
        
        if cut_pos>len(split_line) or len(split_line)>150 or len(split_line)<3:
            continue
        
        
        line_content  = ' '.join(split_line[:-cut_pos])
        target = ' '.join(split_line[-cut_pos:])
                
        doc = nlp(target.lower())
        tokens = [token for token in doc]           # 将句子切分成单词
        pos = [token.pos_ for token in doc]         # 词性标注
        lem = [token.lemma_ for token in doc]       # 词性还原
        
        sen = []
        for t,p,l in zip(tokens, pos, lem):
            if (p in nn) and (len(str(l))>=3) and ('-'  not in str(l)) and ('.' not in str(l)):
                sen.append(str(l))
                
        sen = list(set(sen))
        
        if len(sen)<3:
            continue
            
        if len(sen)>5:
            
            sen =  random.sample(sen, random.randint(3,5))
            
        random.shuffle(sen)
 
        res.append({
                    "content":line_content,
                    "target":target,
                    "keywords": '#'.join(sen)})
        
    return res 

json_path = '/home2/zhanghanqing/formatted_wikipedia'

out_file_train = os.path.join('./data/', 'wiki_train.json')
out_file_train = open(out_file_train, 'w', encoding='utf8')


out_file_val = os.path.join('./data/', 'wiki_val.json')
out_file_val = open(out_file_val, 'w', encoding='utf8')


#######################################wiki dataset########################################

print("reading data from %s ..." % json_path)

filenames = []                
filenames += glob(os.path.join(json_path,'wiki**.format'))

data =  []

for data_file in filenames:
    data +=[x for x in Path(data_file).open().readlines()[-3000000:]]
    
    
d_0_50 = 0  #6w
d_50_90 = 0 #3w
d_90=0  #1w

    
filter_data = []  

for d in data:
    l_d = len(d.split())
    
    if d_0_50>80000 and d_50_90>30000 and d_90>10000:
        break
    
    if l_d<50 and d_0_50<=80000:
        d_0_50 +=1
        filter_data.append(d)
            
        
    elif l_d>=50 and l_d<90 and d_50_90<=30000:
        d_50_90 +=1
        filter_data.append(d)
        
    elif l_d>=90 and d_90<=10000:
        d_90+=1
        filter_data.append(d)
             
print(d_0_50,d_50_90, d_90)
        
data = filter_data
print("data num is:", len(data))

n = int(len(data)/48)
data_list =[data[i:i + n] for i in range(0, len(data), n)]

with Pool(48) as p:

    data =list(tqdm(p.imap(encoder_data, data_list), total=len(data_list)))

record = [item for subl in data for item in subl]

print("the record is:",len(record))

#######################################roc dataset########################################
# roc = pd.read_csv("../data/roc/roc.csv")
# data = []
# for row in roc.itertuples():
#     # if random.randint(1,5)%5==1:
#     cut_pos = random.randint(3,5)
#     # else:
#     #     cut_pos = 5
#     context = [row[i+3] for i in range(cut_pos)]
#     data.append(context)
    
# random.shuffle(data)


# n = int(len(data)/48)
# data_list =[data[i:i + n] for i in range(0, len(data), n)]

# with Pool(48) as p:

#     data =list(tqdm(p.imap(encoder_data_target, data_list), total=len(data_list)))

# record += [item for subl in data for item in subl]

#######################################hc dataset########################################
# data_file = "../data/hc/train.src"
# data =[eval(x.strip(' \n\t\r),\\')) for x in Path(data_file).open().readlines()]

# data_file = "../data/hc/valid.src"
# data +=[eval(x.strip(' \n\t\r),\\')) for x in Path(data_file).open().readlines()]

# print("data num is:", len(data))

# n = int(len(data)/48)
# data_list =[data[i:i + n] for i in range(0, len(data), n)]

# with Pool(48) as p:

#     data =list(tqdm(p.imap(encoder_data, data_list), total=len(data_list)))

# record += [item for subl in data for item in subl]

#######################################commonsence dataset########################################


json_path = "../data/commongen.train.jsonl"
with open(json_path) as out:
    lines = out.readlines()

    for l in tqdm(lines):
        item = json.loads(l.strip())
        concept_set = item['concept_set']
        for c in item['scene']:
            c = c.strip()
        record.append({
                        "content":"",
                        "target":c,
                        "keywords":concept_set})

#######################################end########################################


random.shuffle(record)

print("total data number is:", len(record))
out_file_train.write(json.dumps(record[:int(len(record)*0.95)])+'\n')
out_file_train.flush()

out_file_val.write(json.dumps(record[int(len(record)*0.95):])+'\n')
out_file_val.flush()