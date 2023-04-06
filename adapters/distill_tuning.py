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
            [nn.TransformerDecoderLayer(d_model=input_size, nhead=n_head, batch_first=True) for i in range(1)]
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
        
        
        if self.args.tuning_mode == "pt":
            for param in self.model.parameters():
                param.requires_grad = False
            
            
        self.position_embedings = self.model.transformer.wpe
        self.position_embedings.requires_grad = False
        
        self.embeddings = self.model.get_input_embeddings()
        self.embeddings.requires_grad = False
        
        # load prompt encoder
        self.hidden_size = self.embeddings.embedding_dim
            
        
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        
        self.prompt_encoder = Residual_Model(input_size = self.hidden_size, n_head = 8, n_layer=self.args.residual_layer)
        

    @torch.no_grad()
    def get_gpt_embeddings(self, input_ids, position_ids):
        
                        
        position_embeds = self.position_embedings(position_ids)
        inputs_embeds = self.embeddings(input_ids)
        hidden_states = inputs_embeds + position_embeds 
        
        return hidden_states.detach()
                
        
    
    def load_prompt(self, embedding_checkpoint):
        checkpoint = torch.load(embedding_checkpoint)
        prompt_embedding = checkpoint['embedding']
        return prompt_embedding    
    
    
    def prepare_decoder_input_ids(self, control_ids= None, target_ids = None):
        
        start_of_token =  torch.zeros(control_ids.shape[0], 1).to(self.args.device).fill_(self.tokenizer.pad_token).long()
        
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
    
    
    def forward(self, x_hs, x_ts):

        control_input_ids = x_hs
        attention_mask_control = (control_input_ids!= self.tokenizer.pad_token_id).to(x_hs.device).bool()
        position_ids_control = attention_mask_control.long().cumsum(-1)-1
        # position_ids_control.masked_fill_(attention_mask_control == 0, 0)

        control_hidden = self.get_gpt_embeddings(control_input_ids, position_ids_control)
        
        context = x_ts
        attention_mask = (x_ts!= self.tokenizer.pad_token_id).bool()
        
        labels = torch.clone(context)
        labels.masked_fill_(attention_mask==0, -100)
        
        with torch.no_grad():
        
            output_decoder = self.model(input_ids=context,
                                        attention_mask=attention_mask,
                                        output_hidden_states = True,
                                        return_dict= True)
            
            decoder_hidden = output_decoder.hidden_states[-self.args.num_layer-1]  #batch*seq*hidden
            
        
        att_mask = self._generate_square_subsequent_mask(decoder_hidden.shape[1],self.args.device).bool()
        
      
        logits = self.prompt_encoder(tgt=decoder_hidden, memory=control_hidden, tgt_mask=~attention_mask, memory_mask=~attention_mask_control, att_mask=att_mask)

        shift_loigt = logits[:,:-1,:].reshape(-1, logits.shape[-1])
        shift_label = labels[:,1:].reshape(-1)
        
        loss = self.loss_fct(shift_loigt, shift_label)
                    
        return logits, loss #+kl_loss