import os
import json
import spacy
import nltk
from nltk.tokenize import sent_tokenize
import random
from glob import glob
from pathlib import Path
from multiprocessing import Pool
import pandas as pd
from utils import *
from eval_metric import *




import transformers
from transformers import pipeline, set_seed
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    BertTokenizer,
    GPT2Tokenizer
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers import GPT2LMHeadModel, AutoTokenizer, AutoModelForMaskedLM

seed = 54
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)#as reproducibility docs
torch.manual_seed(seed)# as reproducibility docs
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False# as reproducibility docs
torch.backends.cudnn.deterministic = True# as reproducibility docs
set_seed(seed)


result = pd.read_csv("./eval/common_gen/generated_result_15_seed_40.csv")
gts = {}
for index, row in enumerate(result.itertuples()):
    text = row[-1].strip()
    text = text.split(".")[0]
    gts[str(row[1])] =  [text]
coverage = evaluator_coverage(gts)
self_bleu = evaluator_selfbleu(gts)
ppls = evaluator_ppl_all(gts, "/home2/zhanghanqing/pretrained_model/gpt2/large")
print(coverage, self_bleu, ppls)


result = pd.read_csv("./eval/common_gen/generated_result_18_seed_40.csv")
gts = {}
for index, row in enumerate(result.itertuples()):
    text = row[-1].strip()
    text = text.split(".")[0]
    gts[str(row[1])] =  [text]
coverage = evaluator_coverage(gts)
self_bleu = evaluator_selfbleu(gts)
ppls = evaluator_ppl_all(gts, "/home2/zhanghanqing/pretrained_model/gpt2/large")
print(coverage, self_bleu, ppls)



result = pd.read_csv("./eval/common_gen/generated_result_20_seed_40.csv")
gts = {}
for index, row in enumerate(result.itertuples()):
    text = row[-1].strip()
    text = text.split(".")[0]
    gts[str(row[1])] =  [text]
coverage = evaluator_coverage(gts)
self_bleu = evaluator_selfbleu(gts)
ppls = evaluator_ppl_all(gts, "/home2/zhanghanqing/pretrained_model/gpt2/large")
print(coverage, self_bleu, ppls)