import os
import random
import numpy as np
import torch
import torch.nn as nn
from argument_parser import parse_arguments
from load_data import Load_Data
from models.coh_model import Coh_base
from load_data import Load_Data
from torch.optim import AdamW,Adam
import torch.multiprocessing as mp
import torch.distributed as dist
from transformers import get_linear_schedule_with_warmup
from tqdm import tnrange, tqdm
from sklearn.metrics import classification_report,accuracy_score
import pandas as pd
from data_loader import Dataload
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from scipy.special import expit
from transformers import BertModel,BertTokenizer
# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

class MarginRankingLoss(nn.Module):
    def __init__(self, margin):
        super(MarginRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, p_scores, n_scores, weights=None):
        scores = self.margin - p_scores + n_scores
        scores = scores.clamp(min=0)
        if weights is not None:
            scores = weights * scores
        return scores.mean()
    
    
class Sentence_features(nn.Module):
    def __init__(self):
        super(Sentence_features, self).__init__()
        self.model = SentenceTransformer('sentence-transformers/distilroberta-base-msmarco-v1')     
    
        
    def forward(self, sentences):
        
        ids = []
        for sentence in sentences:
            ids.append(torch.tensor(self.model.encode(sentence)))
        ids = torch.stack(ids, dim=0)
        return ids
    
class Word_features(nn.Module):
    def __init__(self):
        super(Word_features, self).__init__()
        self.tokenizer  = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
    def forward(self, sentences):
        
        ids = []
        for sentence in sentences:
            tokenized_text = self.tokenizer.tokenize(sentence)
            sent_len = len(tokenized_text)
            tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            tokens = tokens + [0]*(512-len(tokens))
            tokens_tensor = torch.tensor(tokens)
            attention_mask = torch.tensor(len(tokenized_text)*[1] + (512-len(tokenized_text))*[0])
            
            with torch.no_grad():
                last_hidden = self.model(tokens_tensor.unsqueeze(0),attention_mask=attention_mask.unsqueeze(0))[0]
                sent_rep = torch.mean(last_hidden.squeeze(0)[:sent_len],0, False)
                ids.append(sent_rep)
            
        ids = torch.stack(ids, dim=0)
        return ids    
    
        
def train(args,features):
    
    
    
    model_bin = [args.task, args.model_name, args.train_task, args.negative_samples, args.batch_size, args.gradient_accumulation_steps, args.lr, 'train' ]
    model_bin.append(features)
    df = pd.DataFrame(columns = ['Epoch','Tr Loss', 'Val Loss', 'Val Acc'])
    
    if args.model_name == 'coh_model':
        model = Coh_base()
    else:
        print('Undefined model')
    print(model)    
       
        
    if torch.cuda.device_count() > 2:
        print('You use %d GPUs'%torch.cuda.device_count())
        model = nn.DataParallel(model, device_ids=[0,1,2,3])
        #Run the model on a specified GPU and run the operations to multiple GPUs
        torch.cuda.set_device(int(args.cuda))
        model.cuda(int(args.cuda))
        loss_fn= MarginRankingLoss(5.0).cuda(int(args.cuda))

    else:
        print('You use only one device')  
        device = torch.device("cuda" if USE_CUDA else "cpu")
        model.to(device)
        loss_fn= MarginRankingLoss(5.0)
       
     
        
    print('The model has {:} different named parameters.\n'.format(len(list(model.named_parameters()))))
    
    
    if features == 'SBert':
        features = Sentence_features()
    elif features == 'Bert':
        features = Word_features()
    
    
    
    #AdamW gets better results than Adam
    optimizer = AdamW([{'params': model.parameters(), 'lr': args.lr}], weight_decay=1.0)
    
    # Total number of training steps is [number of batches] x [number of epochs]. 
    # (Note that this is not the same as the number of training samples).
    #total_steps = len(train_data) * args.epochs
    # Create the learning rate scheduler.
    #However scheduler is not needed for fine-tuning
    #scheduler = get_linear_schedule_with_warmup(optimizer, 
    #            num_warmup_steps = 0, # Default value in run_glue.py
    #            num_training_steps = total_steps)
    
    
    model.zero_grad()
    best_loss = 10000
    trigger_times = 0
    #train_iterator = tqdm.notebook.tnrange(int(args.epochs), desc="Epoch")
    for epoch in range(0, args.epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, args.epochs))
        
        #Load data
        df_train = pd.read_csv('data/disc_train_same_.csv')
        train_data = Dataload(df_train)
        train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size,shuffle=True,drop_last=True)
        
        df_val = pd.read_csv('data/disc_val_same_.csv')
        val_data = Dataload(df_val)
        val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size,shuffle=False,drop_last=True)
        
        
        
        print('Training...')
        model.train()
        
        # Reset the total loss for this epoch.
        tr_loss = 0.0
        for step, (anchor, pos, neg) in tqdm(enumerate(train_loader)):
            
            anchor_ids = features(anchor)
            pos_ids = features(pos)
            neg_ids = features(neg)
            
            
            #Load data
            if torch.cuda.device_count() > 1:
                anchor_ids = anchor_ids.cuda(non_blocking=True)
                pos_ids  = pos_ids.cuda(non_blocking=True)
                neg_ids = neg_ids.cuda(non_blocking=True)
            else:    
                anchor_ids = anchor_ids.to(device)
                pos_ids = pos_ids.to(device)
                neg_ids = neg_ids.to(device)
                
            pos_scores = model(anchor_ids, pos_ids)    
            neg_scores = model(anchor_ids, neg_ids)    
            
            
            
            loss = loss_fn(pos_scores, neg_scores)
            # accumulate train loss
            tr_loss += loss.item()
            
            
            loss = loss/args.gradient_accumulation_steps
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                #scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                
            if (step+1)%args.statistic_step==0: 
                print("Average train loss per %d steps: %0.3f"%((step+1),tr_loss/(step+1)))
                
                
        print("")
        print('Training resuts')   
        print("Average train loss: %0.3f"%(tr_loss/len(train_loader)))
        

        model.eval()
        val_loss = 0.0
        
        positive = []
        negative = []
        
        for step, (anchor, pos, neg) in tqdm(enumerate(val_loader)):
            #Load data
            anchor_ids = features(anchor)
            pos_ids = features(pos)
            neg_ids = features(neg)
            
            if torch.cuda.device_count() > 1:
                anchor_ids = anchor_ids.cuda(non_blocking=True)
                pos_ids  = pos_ids.cuda(non_blocking=True)
                neg_ids = neg_ids.cuda(non_blocking=True)
            else:    
                anchor_ids = anchor_ids.to(device)
                pos_ids = pos_ids.to(device)
                neg_ids = neg_ids.to(device)
        
        
            with torch.no_grad():  
                pos_scores = model(anchor_ids, pos_ids)    
                neg_scores = model(anchor_ids, neg_ids)    
                    
                loss = loss_fn(pos_scores, neg_scores)
                val_loss += loss.item()
                
                positive.extend(expit(pos_scores.cpu().detach().numpy().flatten().tolist())) #sigmoid function
                negative.extend(expit(neg_scores.cpu().detach().numpy().flatten().tolist()))
                
                
            
                if (step+1)%args.statistic_step==0: 
                    print("Average val loss per %d steps: %0.3f"%((step+1),val_loss/(step+1)))
                    correct = 0
                    for i in range(len(positive)):
                        if positive[i]>negative[i]:
                            correct+=1
                    print("Accuracy per %d steps: %0.3f"%((step+1),correct/len(positive)))         
        
        
        
        print("")
        print('Val resuts')
        print("Average val loss: %0.3f"%(val_loss/len(val_data)))
        correct = 0
        for i in range(len(positive)):
            if positive[i]>negative[i]:
                correct+=1
        val_accuracy =  correct/len(positive)       
        print("Val Accuracy: %0.3f"%(correct/len(positive))) 
            
        df.loc[epoch] = pd.Series({'Epoch':int(epoch), 'Tr Loss':round(tr_loss/len(train_data),4), 'Val Loss':round(val_loss/len(val_data),4), 'Val Acc':round(val_accuracy,4)})

        if val_loss/len(val_data)<best_loss:
            trigger_times = 0
            best_loss = val_loss/len(val_data)
            print('Found better model')
            print("Saving model to %s"%os.path.join(args.model_path,'_'.join([str(i) for i in model_bin])))
            model_to_save = model
            torch.save(model_to_save, os.path.join(args.model_path,'_'.join([str(i) for i in model_bin])))
        else:
            trigger_times += 1
            if trigger_times>= args.patience:
                df.to_csv(os.path.join(args.model_path,'_'.join([str(i) for i in model_bin]))+'.csv')
                print('Early Stopping!')
                break 
    
    print("")
    df.to_csv(os.path.join(args.model_path,'_'.join([str(i) for i in model_bin]))+'.csv')
    print("Training complete!")  


    
if __name__ == "__main__":
    args = parse_arguments()
    features = 'Bert' #Can be either SBert or Bert
    print(args)
    
  
    USE_CUDA = torch.cuda.is_available()
    train(args,features)
    
# =============================================================================
# from transformers import AutoTokenizer, AutoModel    
# from sentence_transformers import SentenceTransformer, models
# from torch import nn
# model = AutoModel.from_pretrained('sentence-transformers/distilroberta-base-msmarco-v1')
# 
#     
# 
# =============================================================================
