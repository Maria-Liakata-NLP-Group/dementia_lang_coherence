from torch.utils.data import Dataset
import random

class Dataload(Dataset):
    
    def __init__(self, df):
        self.df = df
        self.anchors = df[df['Label']==1]['Anchor'].values
        self.positives = df[df['Label']==1]['Sentence'].values
        self.clients = df[df['Label']==1]['Client'].values
        self.sessions = df[df['Label']==1]['Session'].values
        self.index = df.index.values
        
    def __getitem__(self, item):
        anchor = self.anchors[item]
        positive = self.positives[item]
        
        negative_list = self.df[(self.df['Client']==self.clients[item]) & (self.df['Session']==self.sessions[item]) & (self.df['Label']==0) & (self.df['Anchor']==anchor)]
        
        negative = negative_list.sample(1)['Sentence'].to_list()[0]
        return (anchor, positive, negative)
        
    def __len__ (self):
        return len(self.anchors)# -*- coding: utf-8 -*-