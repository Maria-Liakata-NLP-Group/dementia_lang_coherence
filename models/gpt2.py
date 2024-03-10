import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2Model

class Gpt2(torch.nn.Module):
    def __init__(self):
        super(Gpt2, self).__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2", bos_token='<startoftext>', eos_token='<endoftext>', pad_token='<pad>')
        
        self.Gpt2 = GPT2Model.from_pretrained('gpt2')
        
        self.Gpt2.resize_token_embeddings(len(self.tokenizer))
        
        self.classifier = nn.Linear(768,len(self.tokenizer))
        
        
       

    def forward(self, input_ids, attention_masks):
        gpt = self.Gpt2(input_ids = input_ids, attention_mask=attention_masks)
        pooler = gpt.last_hidden_state
    
        
        pooler = F.dropout(pooler, 0.3)
        output = self.classifier(pooler)
        
        return output
        
