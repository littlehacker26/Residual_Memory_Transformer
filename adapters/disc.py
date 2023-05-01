import torch
from torch.nn.utils.rnn import pad_sequence
from os.path import join
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import AutoModelWithLMHead, AutoTokenizer, top_k_top_p_filtering

import re
import datetime

from transformers import AutoTokenizer

from .models import get_embedding_layer, create_model, _create_model
from .prompt_encoder import PromptEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F

SMALL_CONST = 1e-10
BIG_CONST = -1e15

class DISC(torch.nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        
        self.args.template_disc = eval(args.template_disc)
        

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        
        self.label_token ={"positive":'good',"negative":'bad'}
        self.label_token_ids ={}
        
        for k, v in self.label_token.items():
            print(k,v,self.tokenizer.encode(v))
            self.label_token_ids[k] = self.tokenizer.encode(v)

        ### load discriminator
        
        if self.args.disc_embedding_checkpoint!=None:
            self.disc_model = _create_model(self.args.disc_embedding_checkpoint[:-5]).to(self.args.device)
            self.spell_length_disc = sum(self.args.template_disc)
            self.disc_embedding = self.disc_model.get_input_embeddings()
            self.prompt_encoder_disc = PromptEncoder(self.args.template_disc, self.disc_embedding.embedding_dim, self.tokenizer, args)
            self.prompt_encoder_disc = self.prompt_encoder_disc.to(self.args.device)
            self.prompt_encoder_disc.load_state_dict(self.load_prompt(self.args.disc_embedding_checkpoint))
            self.pseudo_token_id = self.tokenizer.convert_tokens_to_ids(self.args.pseudo_token)

            
    def load_prompt(self, embedding_checkpoint):
        checkpoint = torch.load(embedding_checkpoint)
        prompt_embedding = checkpoint['embedding']
        return prompt_embedding        

    
    def _predict_scores(self, x_hs, att_mask):
        bz = len(x_hs)
        # construct query ids
        prompt_tokens = [self.pseudo_token_id]
        
        queries = self.get_query(x_hs, prompt_tokens)
        # construct label ids
        attention_mask = torch.cat([att_mask, torch.ones([att_mask.shape[0], self.spell_length_disc]).long().to(self.args.device)], dim=1)
        # get embedded input
        inputs_embeds = self.embed_input(queries)
        
        position_ids = attention_mask.long().cumsum(-1)- 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        
        with torch.no_grad():
            output = self.disc_model(inputs_embeds = inputs_embeds,
                            attention_mask = attention_mask,
                            position_ids = position_ids,
                            labels=None)

        logits = output.logits[:,-1,:].squeeze(1)
        
        binary_prob = torch.softmax(logits[:,[11274,14774]], dim=-1)
        
        
        
        return binary_prob[:,0] # return positive scores
        
        # if self.args.corpus_type == "negative":
        #     return binary_prob[:,1]
        # else:
        #     return binary_prob[:,0]
    
    
    def get_query(self, x_h, prompt_tokens, x_t = None):
        
        prompt_tensor = torch.tensor(prompt_tokens* (self.spell_length_disc)).to(self.args.device)
        prompt_tensor = prompt_tensor.expand(x_h.shape[0],-1)
        if x_t != None:
            x_t = x_t.unsqueeze(1)
            return  torch.cat([x_h, prompt_tensor, x_t], dim =1)
        else:
            return  torch.cat([x_h, prompt_tensor], dim =1)
        

    def embed_input(self, queries):
        bz = queries.shape[0]
        queries_for_embedding = queries.clone()
        raw_embeds = self.disc_embedding(queries_for_embedding)
        
        replace_embeds = self.prompt_encoder_disc()
        
        replace_embeds = replace_embeds.unsqueeze(0).expand(bz,-1, -1)
        
        raw_embeds[:,-self.prompt_encoder_disc.spell_length:,: ] = replace_embeds
                
        return raw_embeds
    

    def  feedback_from_discriminator(self, input_ids, logits_seq, desired_att):
        
        top_logits, top_indices = torch.topk(logits_seq, self.args.ranking_scope) # batch x topk
        
        candidates = []
        for logit_id,  ids  in  zip(top_indices, input_ids):
            data = ids.expand(self.args.ranking_scope, -1)
            new_input_candidates = torch.cat([data, logit_id.unsqueeze(1)], dim=1) # batch x topk x seq+1
            candidates.append(new_input_candidates)
            
        
        candidates = torch.cat(candidates, dim=0)
        
        if candidates.shape[1]<30:
            pad_tensor =torch.empty(candidates.shape[0],30 - candidates.shape[1]).long().fill_(self.tokenizer.eos_token_id).to(self.args.device)
            candidates = torch.cat([pad_tensor,candidates], dim=1)
                    
        pred_scores = []
        for new_input_candidates in torch.split(candidates, 150, dim=0):
            musk =  (new_input_candidates != self.tokenizer.eos_token_id).type(torch.uint8)
            pred_score  = self._predict_scores(new_input_candidates, musk)
            pred_scores.append(pred_score)

        pred_scores = torch.cat(pred_scores, dim=0)
        pred_scores = pred_scores.reshape(input_ids.shape[0], -1)

        res_logits = logits_seq.clone().detach()
        res_logits.scatter_(-1, top_indices, pred_scores)
        return res_logits
    
    
        
    def get_ranked_logtis(self, inputs, logits, desired_att):
        
        return_logits = []
        for i  in range(inputs.shape[1]):
            tmp = self.feedback_from_discriminator(inputs[:, :i+1], logits[:,i,:],  desired_att)
            return_logits.append(tmp)
                        
        return torch.stack(return_logits, dim=1).detach().clone()
    
    
       


        

        
        

 