import os
from pathlib import Path

from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch
import torch.nn as nn



from .models import get_embedding_layer, create_model, create_model_t5
from .prompt_encoder import PromptEncoder


class T5PromptTuning(torch.nn.Module):
    
    def __init__(self, args, template):
        super().__init__()
        self.args = args

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # model setting
        self.model = create_model_t5(self.args)
        # self.model = self.model.to(self.args.device)
        
#         for param in self.model.parameters():
#             param.requires_grad = False

        self.embeddings = self.model.get_input_embeddings()
        # label information
        
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
    

    def non_params_simility(self, x, key_value, mask_value):

        key_value = torch.transpose(key_value,-1,-2)

        simility_matrix = x.bmm(key_value) # b*seq_x*seq_k_v

        simility_matrix_mean = - torch.mean(simility_matrix, dim=-1, keepdim =False)
        simility_matrix_max = - torch.max(simility_matrix, dim=-1, keepdim= False).values

        mask = torch.zeros(x.shape[0],x.shape[1]).to(x.device)
        mask.masked_fill_(mask_value == 1, -1000) # mask the pading token, simility to 1 direction

        # out = -torch.cosine_similarity(x, key_value, dim=-1) + mask # batch*seq
        softmax_max =  F.softmax((simility_matrix_max+mask)/0.1, dim=1).unsqueeze(1) # batch*seq
        
        softmax_mean =  F.softmax((simility_matrix_mean+mask)/0.1, dim=1).unsqueeze(1) # batch*seq

        return softmax_max, softmax_mean
    
    
    def get_feedback_prompt(self, query, generated_text):
        ""
        ""
        k_v = self.embeddings(generated_text)
        q = self.embeddings(query) # batch*seq*dim
        
        mask_value = (query == self.tokenizer.pad_token_id).long().to(query.device)
        
        softmax_max, softmax_mean = self.non_params_simility(q, k_v, mask_value)# batch*1*seq; the bigger the value is, the more attention will be allocated
        
        # print(attn_output.shape, attn_output)
        
        out_max = softmax_max.bmm(q)#batch*dim
        
        out_meam = softmax_mean.bmm(q) #batch*dim


        return out_max, out_meam
    
    
    def get_query_head(self, x_h, prompt_tokens, x_t = None):
        
        mix_tensor_head =  torch.zeros(x_h.shape[0], self.template[0]).to(x_h.device).fill_(prompt_tokens[0]).long()
        mix_tensor_tail =  torch.zeros(x_h.shape[0], self.template[1]).to(x_h.device).fill_(prompt_tokens[0]).long()

        mix_tensor = torch.cat([mix_tensor_head,x_h,mix_tensor_tail], dim=1)
        
        if x_t != None:
            return  torch.cat([mix_tensor, x_t], dim =1)
        else:
            return mix_tensor
        
        
        
    def embed_hybird_inputs(self, queries, x_h, generated_text = None):
      
        bz = queries.shape[0]
        control_len = x_h.shape[1] + self.spell_length #- self.template[1]
        
        if generated_text ==  None:
            generated_text = torch.tensor([[self.tokenizer.unk_token_id]]*bz).to(queries.device)
        
        queries_for_embedding = queries.clone()
        
        queries_for_embedding[(queries == self.pseudo_token_id)] = self.tokenizer.unk_token_id
        
        # raw_embeds = self.embeddings(queries_for_embedding)
        raw_embeds = self.model.encoder.embed_tokens(queries_for_embedding)
        
        
        replace_embeds = self.prompt_encoder().expand(x_h.shape[0],-1,-1).contiguous()
        
        raw_embeds[:,:self.template[0],:] = replace_embeds[:,:self.template[0],:]
        raw_embeds[:,self.template[0]+x_h.shape[1]:self.spell_length+x_h.shape[1],:] = replace_embeds[:,self.template[0]:,:]
        
        if self.args.pattern != "vanilla":

            feed_data_max, feed_data_mean = self.get_feedback_prompt(x_h, generated_text)
            
            if self.args.pattern == "dynamic_prompt_max":
                raw_embeds = torch.cat([feed_data_max ,raw_embeds], dim=1)
                
            elif self.args.pattern == "dynamic_prompt_mean":
                raw_embeds = torch.cat([feed_data_mean ,raw_embeds], dim=1)
                
            elif self.args.pattern == "dynamic_prompt_hybird":
                raw_embeds = torch.cat([feed_data_max ,feed_data_mean, raw_embeds], dim=1)
            
            else:
                raise Exception('This is the error pattern.')
        
        return raw_embeds    
    
    

    
    
 

        
    
    def generate(self, *args, **kwargs):
        # This fixes CUDA for some reason
        #print("device = ", self.device)
        kwargs['input_ids'] = kwargs['input_ids'].to(self.device)
        kwargs['inputs_embeds'] = self._cat_learned_embedding_to_input(kwargs['input_ids']).to(self.device)
        del kwargs['input_ids']
            
        #print("the shape input embeds = ", kwargs['inputs_embeds'].shape)

        return super().generate( *args, **kwargs)
    

        
    def forward(
        self,
        x_hs, 
        x_ts):
                
        prompt_tokens = [self.pseudo_token_id]
        queries = self.get_query_head(x_hs, prompt_tokens)
        attention_mask = (queries!= self.tokenizer.pad_token_id).long().to(x_hs.device)
        
        logits  = []
        labels = []
        for i in range(x_ts.shape[1]):

            if i ==0:
                _x_ts = torch.tensor([[self.tokenizer.pad_token_id]]*x_hs.shape[0]).to(queries.device)
            else:
                _x_ts = x_ts[:,:i]
                
            inputs_embeds = self.embed_hybird_inputs(queries, x_hs, _x_ts)

            
            output = self.model(attention_mask=attention_mask,
                                inputs_embeds=inputs_embeds,
                                decoder_input_ids=_x_ts,
                                return_dict=True)
            
            
            logits.append(output.logits[:,-1,:])
            labels.append(x_ts[:,i])
            
        logits = torch.cat(logits, dim=0)
        labels = torch.cat(labels, dim=0)
            
        loss = self.cross_entry_loss(logits, labels)
        
        return loss
        