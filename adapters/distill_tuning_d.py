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


class Residual_Model(nn.Module):
    def __init__(self, input_size, n_head, n_layer):
        super(Residual_Model, self).__init__()
        
        self.encoder_block = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=input_size, nhead=n_head, batch_first=True) for i in range(n_layer)]
        )
            
        self.decoder_block = nn.ModuleList(
            [nn.TransformerDecoderLayer(d_model=input_size, nhead=n_head, batch_first=True) for i in range(n_layer)]
        )
        
        self.lm_head = nn.Linear(input_size, 50257,bias=False)
        
        
    def forward(self, tgt, memory, tgt_mask, memory_mask, att_mask):
        
        for layer_module in  self.encoder_block:
            memory = layer_module(src = memory, src_key_padding_mask = memory_mask)
            
            
        for layer_module in  self.decoder_block:
            tgt = layer_module(tgt =tgt , memory=memory, tgt_key_padding_mask = tgt_mask, memory_key_padding_mask=memory_mask, tgt_mask=att_mask)
            
        return self.lm_head(tgt)
    

class Distill_Tuning(torch.nn.Module):

    def __init__(self, args, template):
        super().__init__()
        self.args = args

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # model setting
        self.model = create_model(self.args)
        # self.embeddings = self.model.get_input_embeddings()
        if self.args.tuning_mode == "pt":
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.embeddings = self.model.get_input_embeddings()
        
        
        self.position_embedings = self.model.transformer.wpe
        self.position_embedings.requires_grad = False
        
        self.template = template
        # load prompt encoder
        self.hidden_size = self.embeddings.embedding_dim
        self.pseudo_token_id = self.tokenizer.convert_tokens_to_ids(self.args.pseudo_token)
        

        self.spell_length = sum(self.template)
        self.prompt_encoder_ = PromptEncoder(self.template, self.hidden_size, self.tokenizer, args)
        
            
        self.hidden_size = self.model.get_input_embeddings().embedding_dim
        self.pseudo_token_id = self.tokenizer.convert_tokens_to_ids(self.args.pseudo_token)
 
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        
        self.prompt_encoder = Residual_Model(input_size = self.hidden_size, n_head = 8, n_layer=self.args.num_layer)        

    
    def load_prompt(self, embedding_checkpoint):
        checkpoint = torch.load(embedding_checkpoint)
        prompt_embedding = checkpoint['embedding']
        return prompt_embedding
    
    
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
        control_len = x_h.shape[1] + self.spell_length #- self.template[1]
       
        queries_for_embedding = queries.clone()
        
        queries_for_embedding[(queries == self.pseudo_token_id)] = self.tokenizer.unk_token_id
        
        raw_embeds = self.embeddings(queries_for_embedding)
        
        replace_embeds = self.prompt_encoder_().expand(x_h.shape[0],-1,-1).contiguous()
        
        raw_embeds[:,:self.template[0],:] = replace_embeds[:,:self.template[0],:]
        raw_embeds[:,self.template[0]+x_h.shape[1]:self.spell_length+x_h.shape[1],:] = replace_embeds[:,self.template[0]:,:]
        
        return raw_embeds
    
    
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
        
    
    
    
                
    def generate(self, prompts_ids ,max_length, context= None, desired_att = None,  beta = 0.5):
        """
        generation forward based on given prompt tokens, 
        Args:
            prompt_ids: the  prompt tokens
            max_length: the max len of the generation
        Returns:
            generated_texts:[generated tokens]
        """
        cur_len = 0
        
        if context== None:
            # context_len = prompts_ids.shape[1]
            decoder_input_ids = None
        else:
            # context_len = prompts_ids.shape[1] + context.shape[1]
            decoder_input_ids = context

        
        return_dict = {}
        eos_flag = torch.ones([prompts_ids.shape[0]]).type(torch.uint8).to(self.args.device)
        
        control_input_ids = prompts_ids
        attention_mask_control = (control_input_ids!= self.tokenizer.pad_token_id).to(prompts_ids.device).bool()
        position_ids_control = attention_mask_control.long().cumsum(-1)- 1
        control_hidden = self.get_gpt_embeddings(control_input_ids, position_ids_control)
       
        
        prompt_tokens = [self.pseudo_token_id]
        queries = self.get_query_head(prompts_ids, prompt_tokens, decoder_input_ids)        
        inputs_embeds = self.embed_hybird_inputs(queries, prompts_ids)
        
        while cur_len <= max_length:
            attention_mask = (queries != self.tokenizer.pad_token_id).to(queries.device).bool()
                
            output_decoder = self.model(inputs_embeds=inputs_embeds,
                                    attention_mask=attention_mask,
                                    output_hidden_states = True,
                                    return_dict= True)
            decoder_hidden = output_decoder.hidden_states[-5]   # batch*seq*hidden
            
            att_mask = self._generate_square_subsequent_mask(decoder_hidden.shape[1], self.args.device)
        
            logits = 0.9*self.prompt_encoder(tgt=decoder_hidden, memory = control_hidden, tgt_mask = ~attention_mask, memory_mask=~attention_mask_control, att_mask=att_mask) + 0.1*output_decoder.logits
            
            next_token_logits = logits[:, -1, :]
            
            next_token_logits_ = self.top_k_top_p_filtering(next_token_logits,  top_k=0, top_p= self.args.top_p, filter_value=BIG_CONST)

            next_token_logits_prob = torch.softmax(next_token_logits_, dim=1)
            next_tokens = torch.multinomial(next_token_logits_prob, num_samples=1).squeeze(1)
            
            eos_flag = eos_flag.mul((next_tokens != self.tokenizer.eos_token_id).type(torch.uint8))# if flag = 0, it means the generation is over 
            next_tokens = next_tokens.mul(eos_flag)
            next_tokens[next_tokens == 0] = self.tokenizer.eos_token_id
            
            if decoder_input_ids==None:
                decoder_input_ids = next_tokens.unsqueeze(1)
                
            else:
                decoder_input_ids = torch.cat([decoder_input_ids, next_tokens.unsqueeze(1)], dim=1)
            
            queries = self.get_query_head(prompts_ids, prompt_tokens, decoder_input_ids)
            
            inputs_embeds = self.embed_hybird_inputs(queries, prompts_ids)

            cur_len = cur_len + 1
        
        return_dict = {"generated_tokens":decoder_input_ids}
        return return_dict
    
    
    
    def prepare_decoder_input_ids(self, control_ids= None, target_ids = None):
        
        start_of_token =  torch.zeros(control_ids.shape[0], 1).to(self.args.device).fill_(self.pseudo_token_id).long()
        
        if target_ids == None:
            return torch.cat([control_ids, start_of_token], dim=1)
        else:
            return torch.cat([control_ids,start_of_token,target_ids], dim=1)
        
                    
    def _generate_square_subsequent_mask(self, sz: int, device='cpu'):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        # return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)
        return torch.triu(torch.full((sz, sz), True, device=device), diagonal=1)
    
    
    def prepare_fixed_GPT2_information(self, inputs_ids):
        
        
        attention_mask = (inputs_ids!= self.tokenizer.pad_token_id).to(inputs_ids.device).bool()

        position_ids = attention_mask.long().cumsum(-1)- 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        
        return self.model(input_ids= inputs_ids,
                            attention_mask= attention_mask,
                            position_ids=position_ids,
                            output_hidden_states = True,
                            return_dict= True).logits
    
    
    @torch.no_grad()
    def get_gpt_embeddings(self, input_ids, position_ids):
                        
        position_embeds = self.position_embedings(position_ids)
        inputs_embeds = self.embeddings(input_ids)
        hidden_states = inputs_embeds + position_embeds 
        
        return hidden_states.detach()
    
    
    def forward(self, x_hs, x_ts):
        # construct query ids
        # prompt_tokens = [self.pseudo_token_id]
        prompt_tokens = [self.pseudo_token_id]

                
        control_input_ids = x_hs
        attention_mask_control = (control_input_ids!= self.tokenizer.pad_token_id).to(x_hs.device).bool()
        position_ids_control = attention_mask_control.long().cumsum(-1)-1
        control_hidden = self.get_gpt_embeddings(control_input_ids, position_ids_control)
        
        
        decoder_input_ids = self.get_query_head(x_hs, prompt_tokens, x_ts)
        inputs_embeds = self.embed_hybird_inputs(decoder_input_ids, x_hs)
        attention_mask = (decoder_input_ids!= self.tokenizer.pad_token_id).bool().to(x_hs.device)
        
        label_mask = torch.zeros([decoder_input_ids.shape[0],decoder_input_ids.shape[1]]).to(decoder_input_ids.device)
        label_mask[:,-x_ts.shape[1]:] = 1
        
        labels = torch.clone(decoder_input_ids)
        labels.masked_fill_(label_mask==0, -100)
        labels.masked_fill_(attention_mask==0, -100)

        output_decoder = self.model(inputs_embeds=inputs_embeds,
                                    attention_mask=attention_mask,
                                    output_hidden_states = True,
                                    return_dict= True)
        decoder_hidden = output_decoder.hidden_states[-5]   #batch*seq*hidden
        
        att_mask = self._generate_square_subsequent_mask(decoder_hidden.shape[1],self.args.device).bool()
        
        logits = 0.9*self.prompt_encoder(tgt=decoder_hidden, memory=control_hidden, tgt_mask=~attention_mask, memory_mask=~attention_mask_control, att_mask=att_mask) + 0.1*output_decoder.logits

        shift_loigt = logits[:,:-1,:].reshape(-1, logits.shape[-1])
        shift_label = labels[:,1:].reshape(-1)
        
        loss = self.loss_fct(shift_loigt, shift_label)
                    
        return shift_loigt, loss 