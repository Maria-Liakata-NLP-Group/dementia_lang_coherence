from transformers import BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import numpy as np

class BertCNN(torch.nn.Module):
    def __init__(self):
        super(BertCNN, self).__init__()
        self.Bert = BertModel.from_pretrained('bert-large-uncased')
        
        self.num_filters = num_filters=[100, 100, 100,100]
        #Filter size acts as sliding window of tokens, e.g., bi-grams, tri-grams etc
        self.filter_sizes=[2, 3, 4, 5]
        self.embed_dim = 1024
        self.num_classes = 2
        
        self.conv1d_list = nn.ModuleList([
           nn.Conv1d(in_channels=self.embed_dim,
                     out_channels=self.num_filters[i],
                     kernel_size=self.filter_sizes[i])
           for i in range(len(self.filter_sizes))
       ])
        
    
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(np.sum(self.num_filters), self.num_classes)
        
       
        
        
    def forward(self, input_ids,attention_masks):
        _, _, all_layers = self.Bert(input_ids, attention_mask=attention_masks, output_hidden_states=True,return_dict=False)
        # all_layers  = [heads, bacth, sequence, output]
        x = torch.transpose(torch.cat(tuple([t.unsqueeze(0) for t in all_layers]), 0), 0, 1)
        # x  = [bacth, heads, sequence, output]
        #Get CLS tokens from each layer
        x = x[:,:,:1,:].squeeze(2)
        
        # Output shape: (b, embed_dim, max_len)
        x = x.permute(0, 2, 1)
        
        del all_layers
        torch.cuda.empty_cache()
        
        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list = [F.relu(conv1d(x)) for conv1d in self.conv1d_list]
        
        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
          for x_conv in x_conv_list]
        
        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
                         dim=1)
        
   
        
        # Compute logits. Output shape: (b, n_classes)
        logits = self.fc(self.dropout (x_fc))
        
        return logits