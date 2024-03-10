import os
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import pickle
import torch

class Load_Data():
     def __init__(self, task: str, subdataset:str, negative_samplels:str, batch_size: int):
         self.task = task
         self.subdataset = subdataset
         self.batch_size = batch_size
         self.negative_samplels = negative_samplels
         self.data_path = os.path.join('data', self.task+ '_'+ self.subdataset  + '_' + self.negative_samplels  +'.pkl')
        

     def loader(self):
        #Load data
        dataset_file = open(self.data_path,'rb')
        dataset = pickle.load(dataset_file)
        dataset_file.close()
        
       
         #Load data on DataLoader
        if 'train' in self.subdataset:
             # Create the DataLoaders for our training and validation sets.
             # We'll take training samples in random order. 
             dataloader = DataLoader(
                 dataset,  # The training samples.
                 sampler = RandomSampler(dataset), # Select batches randomly
                 batch_size = self.batch_size, # Trains with this batch size.
                 drop_last=True
             )
        else:
             # For validation and test the order doesn't matter, so we'll just read them sequentially.
             dataloader = DataLoader(
                dataset, # The validation samples.
                #sampler = RandomSampler(dataset),
                sampler = SequentialSampler(dataset), # Pull out batches sequentially.
                batch_size = self.batch_size, # Evaluate with this batch size.
                drop_last=True
            )
            
        #Returns input_ids, masks, token_type_ids, and labels    
        return dataloader      