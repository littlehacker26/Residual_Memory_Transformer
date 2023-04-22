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


def encoder_data(data):
    
    nn = ['VBG', 'NOUN']
    nlp = spacy.load('en_core_web_sm')# 加载预训练模型

    res = []
    
    for line in data:
        
        cut_pos = random.randint(9,15)
        line = line.strip(' \n\t\r),\\')
        
        split_line= line.split()
        
        if cut_pos>len(split_line) or len(split_line)>150 or len(split_line)<3:
            continue
        
        line_content  = ' '.join(split_line[:-cut_pos])
        target = ' '.join(split_line[-cut_pos:])
                
        doc = nlp(target)
        tokens = [token for token in doc]           # 将句子切分成单词
        pos = [token.pos_ for token in doc]         # 词性标注
        lem = [token.lemma_ for token in doc]       # 词性还原
        
        sen = []
        for t,p,l in zip(tokens, pos, lem):
            if p in nn:
                sen.append(str(l))
                
        sen = list(set(sen))
        
        if len(sen)<3:
            continue
            
        if len(sen)>5:
            
            sen =  random.sample(sen, 5)
            
        random.shuffle(sen)
 
        res.append({
                    "content":line_content,
                    "target":target ,
                    "keywords": '#'.join(sen)})
        
    return res 


json_path = '/home2/zhanghanqing/formatted_wikipedia'

out_file_train = os.path.join('./data/', 'wiki_train.json')
out_file_train = open(out_file_train, 'w', encoding='utf8')


out_file_val = os.path.join('./data/', 'wiki_val.json')
out_file_val = open(out_file_val, 'w', encoding='utf8')



print("reading data from %s ..." % json_path)

filenames = []                
filenames += glob(os.path.join(json_path,'wiki**.format'))

data =  []

for data_file in filenames:
    data +=[x for x in Path(data_file).open().readlines()[-40000:]]

print("data num is:", len(data))

n = int(len(data)/48)
data_list =[data[i:i + n] for i in range(0, len(data), n)]

with Pool(48) as p:

    data =list(tqdm(p.imap(encoder_data, data_list), total=len(data_list)))

record = [item for subl in data for item in subl]

random.shuffle(record)

out_file_train.write(json.dumps(record[:int(len(record)*0.95)])+'\n')
out_file_train.flush()

out_file_val.write(json.dumps(record[int(len(record)*0.95):])+'\n')
out_file_val.flush()