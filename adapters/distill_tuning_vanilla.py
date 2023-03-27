import torch
from torch.nn.utils.rnn import pad_sequence
from os.path import join
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import AutoModelWithLMHead, AutoTokenizer, top_k_top_p_filtering

import torch
from torch.nn import GRU


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
    

class Distill_Tuning(torch.nn.Module):

    def __init__(self, args, template):
        super().__init__()
        self.args = args

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # model setting
        self.model = create_model(self.args)
        self.embeddings = self.model.get_input_embeddings()

        if self.args.tuning_mode == "pt":
            for param in self.model.parameters():
                param.requires_grad = False
            
        self.template = template
        # load prompt encoder
        self.hidden_size = self.embeddings.embedding_dim
        
        self.pseudo_token_id = self.tokenizer.convert_tokens_to_ids(self.args.pseudo_token)

        self.spell_length = sum(self.template)
        self.prompt_encoder = PromptEncoder(self.template, self.hidden_size, self.tokenizer, args)
 
        
        self.cross_entry_loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        
    
            
    def load_prompt(self, embedding_checkpoint):
        checkpoint = torch.load(embedding_checkpoint)
        prompt_embedding = checkpoint['embedding']
        return prompt_embedding
    
    
        
    
    def top_k_top_p_filtering(self,
        logits,
        top_k = 0,
        top_p = 1.0,
        filter_value = -1e15 ,
        min_tokens_to_keep = 1,
    ):
        """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        if top_k > 0:
            top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value
            
        return logits
    

    def get_query_head(self, x_h, prompt_tokens, x_t = None):
        
        mix_tensor_head =  torch.zeros(x_h.shape[0], self.template[0]).to(x_h.device).fill_(prompt_tokens[0]).long()
        mix_tensor_tail =  torch.zeros(x_h.shape[0], self.template[1]).to(x_h.device).fill_(prompt_tokens[0]).long()

        mix_tensor = torch.cat([mix_tensor_head,x_h,mix_tensor_tail], dim=1)
        
        if x_t != None:
            return  torch.cat([mix_tensor, x_t], dim =1)
        else:
            return mix_tensor
        
        
        
    def embed_hybird_inputs(self, queries, x_h):
      
        bz = queries.shape[0]
        
        queries_for_embedding = queries.clone()
        
        queries_for_embedding[(queries == self.pseudo_token_id)] = self.tokenizer.unk_token_id
        
        raw_embeds = self.embeddings(queries_for_embedding)
        
        replace_embeds = self.prompt_encoder().expand(x_h.shape[0],-1,-1).contiguous()
        
        raw_embeds[:,:self.template[0],:] = replace_embeds[:,:self.template[0],:]
        raw_embeds[:,self.template[0]+x_h.shape[1]:self.spell_length+x_h.shape[1],:] = replace_embeds[:,self.template[0]:,:]
        
        return raw_embeds
    
    
    
    
                
    def generate(self, prompts_ids, max_length, context=None, desired_att = None,  beta = 0.5):
        """
        generation forward based on given prompt tokens, 
        Args:
            prompt_ids: the  prompt tokens
            max_length: the max len of the generation
        Returns:
            generated_texts:[generated tokens]
        """
        cur_len = 0
        logits = []
        output_ids = prompts_ids
        
        if context != None:
            output_ids = torch.cat([output_ids, context], dim=1)
            
        return_dict = {}
        eos_flag = torch.ones([prompts_ids.shape[0]]).type(torch.uint8).to(self.args.device)
        
        prompt_tokens = [self.pseudo_token_id]
        
        queries = self.get_query_head(prompts_ids, prompt_tokens,context)
        inputs_embeds = self.embed_hybird_inputs(queries, prompts_ids)
        
        attention_mask = torch.cat([queries!= self.tokenizer.pad_token_id, torch.ones([prompts_ids.shape[0], prompts_ids.shape[1]+self.prompt_encoder.spell_length + max_length]).long().to(self.args.device)], dim=1)
        
        
        while cur_len <= max_length:
            outputs = self.model(inputs_embeds=inputs_embeds,
                                 attention_mask = attention_mask[:,:inputs_embeds.shape[1]],
                                 return_dict=True)
                
            next_token_logits = outputs.logits[:, -1, :]

            next_token_logits_ = self.top_k_top_p_filtering(next_token_logits,  top_k=0, top_p= self.args.top_p, filter_value=BIG_CONST)

            next_token_logits_prob = torch.softmax(next_token_logits_, dim=1)
            next_tokens = torch.multinomial(next_token_logits_prob, num_samples=1).squeeze(1)
            
            eos_flag = eos_flag.mul((next_tokens != self.tokenizer.eos_token_id).type(torch.uint8))# if flag = 0, it means the generation is over 
            next_tokens = next_tokens.mul(eos_flag)
            next_tokens[next_tokens == 0] = self.tokenizer.eos_token_id            
            output_ids = torch.cat([output_ids, next_tokens.unsqueeze(1)], dim=1)

            
            queries = self.get_query_head(prompts_ids, prompt_tokens, output_ids[:,prompts_ids.shape[1]:])
            
            inputs_embeds = self.embed_hybird_inputs(queries, prompts_ids)

            cur_len = cur_len + 1
        
        return_dict = {"generated_tokens":output_ids[:,prompts_ids.shape[1]:]}
        return return_dict
    
    
    def forward(self, x_hs, x_ts):
        # construct query ids
        prompt_tokens = [self.pseudo_token_id]
        queries = self.get_query_head(x_hs, prompt_tokens, x_ts)
                
        logits = []
        labels = []
      
        inputs_embeds = self.embed_hybird_inputs(queries, x_hs)
        attention_mask = (queries!= self.tokenizer.pad_token_id).long().to(x_hs.device)
        label_mask = torch.zeros([queries.shape[0],queries.shape[1]]).to(queries.device)

        label_mask[:,-x_ts.shape[1]:] = 1
        labels = torch.clone(queries)

        labels.masked_fill_(label_mask==0, -100)
        labels.masked_fill_(attention_mask==0, -100)

        output = self.model(inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask,
                            labels = labels)
        return None, output.loss
 