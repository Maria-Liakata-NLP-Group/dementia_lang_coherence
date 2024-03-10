from transformers import BertModel,BertConfig
import torch
import torch.nn as nn

class SBert(torch.nn.Module):
    def __init__(self):
        super(SBert, self).__init__()
        self.config = BertConfig.from_pretrained("bert-base-uncased")
        self.Bert = BertModel.from_pretrained('bert-base-uncased')
        self.num_classes = 2
    
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.config.hidden_size*3, self.num_classes)
        
        
    def mean_pool(self,token_embeds, attention_mask):
        # reshape attention_mask to cover 768-dimension embeddings
        in_mask = attention_mask.unsqueeze(-1).expand(
            token_embeds.size()
        ).float()
        # perform mean-pooling but exclude padding tokens (specified by in_mask)
        pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(
            in_mask.sum(1), min=1e-9
        )
        return pool 
    
    
    def forward(self, input_ids_1,attention_masks_1,input_ids_2,attention_masks_2):
        
        u_bert = self.Bert(input_ids = input_ids_1, attention_mask=attention_masks_1)
        u_hs = u_bert[0]         
        u = self.mean_pool(u_hs,attention_masks_1)
        
        v_bert = self.Bert(input_ids = input_ids_2, attention_mask=attention_masks_2)
        v_hs = v_bert[0]         
        v = self.mean_pool(v_hs,attention_masks_2)
        
        uv_abs = torch.abs(torch.sub(u, v))  # produces |u-v| tensor
        
        # then we concatenate
        x = torch.cat([u, v, uv_abs], dim=-1)
        
        # Compute logits. Output shape: (b, n_classes)
        logits = self.fc(self.dropout (x))
        
        return logits
        
        
 
