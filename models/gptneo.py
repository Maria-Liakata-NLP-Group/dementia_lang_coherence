import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPTNeoModel,GPT2Tokenizer

class GptNeo(torch.nn.Module):
    def __init__(self):
        super(GptNeo, self).__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B", bos_token='<startoftext>', eos_token='<endoftext>', pad_token='<pad>')
       
        
        self.GptNEO = GPTNeoModel.from_pretrained("EleutherAI/gpt-neo-1.3B")
        
        self.GptNEO.resize_token_embeddings(len(self.tokenizer))
        
        self.classifier = nn.Linear(2048,len(self.tokenizer))
        
        
       

    def forward(self, input_ids, attention_masks):
        gpt = self.GptNEO(input_ids = input_ids, attention_mask=attention_masks)
        pooler = gpt.last_hidden_state
    
        
        pooler = F.dropout(pooler, 0.3)
        output = self.classifier(pooler)
        
        return output
        
