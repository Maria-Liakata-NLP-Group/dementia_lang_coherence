from tqdm import *
from transformers import GPT2Tokenizer
from transformers import T5Tokenizer
import torch
import pandas as pd
import pickle
import os
from torch.utils.data import TensorDataset
from random import shuffle, random, randrange


class Extract_features():
     def __init__(self, data_path: str, model_name = str, task = str, negative_samples = str,  max_seq_length=512):
         self.data_path = data_path
         self.max_seq_length = max_seq_length
         self.model_name  = model_name
         self.task = task
         self.negative_samples = negative_samples
         
         if 'gpt' in self.model_name:
             self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2", bos_token='<startoftext>', eos_token='<endoftext>', pad_token='<pad>')
             print('All special tokens:', self.tokenizer.all_special_tokens)
             print('All special ids:', self.tokenizer.all_special_ids)
         elif 't5' in self.model_name: 
             self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
             
             
     def tokenize(self):
         df = pd.read_csv(self.data_path, sep=',')
         
         source_ids = []
         source_masks = []
         target_ids = []
         target_masks = []
         
         
         for index, row in tqdm(df.iterrows(), total=df.shape[0]):
             source = df.iloc[index]['Source'].lower()
             target = df.iloc[index]['Target'].lower()
             
             
             if 'gpt' in self.model_name:
                 prep_source = f'<startoftext> {source} <endoftext>'
                 prep_target = f'<startoftext> {target} <endoftext>'
             elif 't5' in self.model_name: 
                 prep_source = f'{source}'
                 prep_target = f'{target}'
                 
             encoded_source = self.tokenizer(prep_source, truncation=True, max_length=self.max_seq_length, padding='max_length', return_tensors="pt")
             encoded_target = self.tokenizer(prep_target, truncation=True, max_length=self.max_seq_length, padding='max_length',return_tensors="pt")
             
             
             
             
             source_ids.append(encoded_source['input_ids'])
             source_masks.append(encoded_source['attention_mask'])
             #add labels
             target_ids.append(encoded_target['input_ids'])
             target_masks.append(encoded_target['attention_mask'])
             
         # Convert the lists into tensors.
         source_ids = torch.stack(source_ids, dim=0)
         source_masks = torch.stack(source_masks, dim=0)
         target_ids = torch.stack(target_ids, dim=0)
         target_masks = torch.stack(target_masks, dim=0)
         
         source_ids = source_ids.view(-1,source_ids.size(-1))
         source_masks = source_masks.view(-1,source_masks.size(-1))
         target_ids = target_ids.view(-1,target_ids.size(-1))
         target_masks = target_masks.view(-1,target_masks.size(-1))
           
         
         dataset = TensorDataset(source_ids, source_masks, target_ids,target_masks)
             
         if 'train' in self.data_path:
             subset = 'train'
         elif 'test'in self.data_path:
             subset = 'test'
         elif 'val' in self.data_path:
             subset = 'val'
            
         with open(os.path.join('../data', self.task + '_' + subset +  '_' + negative_samples + '.pkl'), 'wb') as f:
            pickle.dump(dataset, f)
            
     def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
           """Truncates a sequence pair in place to the maximum length."""
           # This is a simple heuristic which will always truncate the longer sequence
           # one token at a time. This makes more sense than truncating an equal percent
           # of tokens from each, since if one sequence is very short then each token
           # that's truncated likely contains more information than a longer sequence.
           while True:
               total_length = len(tokens_a) + len(tokens_b)
               if total_length <= max_length:
                   break
               if len(tokens_a) > len(tokens_b):
                   tokens_a.pop()
               else:
                  tokens_b.pop()       
         
         
         
if __name__ == "__main__":   
    data_path = '../data/gen_train_same_.csv'
    negative_samples = 'same' #Could be same 
    model_name = 't5-base' #can be either gpt or t5-base or t5-large
    task = 'gen'    #can be either gen or 
    max_seq_length = 512
   
    
    self = Extract_features(data_path,model_name, task, negative_samples, max_seq_length)
    self.tokenize()                 
                         
    


