import os
import random
import numpy as np
import torch
import torch.nn as nn
from argument_parser import parse_arguments
from load_data import Load_Data
from models.bert import Bert
from models.roberta import Roberta
from models.bert_cnn import BertCNN
from models.bert_small import Bert_base
from models.cnn import CNN
from data_loader import Dataload
from torch.utils.data import DataLoader
from torch.optim import AdamW,Adam
import torch.multiprocessing as mp
import torch.distributed as dist
from transformers import get_linear_schedule_with_warmup
from tqdm import tnrange, tqdm
from sklearn.metrics import classification_report,accuracy_score
import pandas as pd
from transformers import BertModel,BertTokenizer,RobertaTokenizer

# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)



class Extract_feautres(nn.Module):
    def __init__(self,  model):
        super(Extract_feautres, self).__init__()
        self.model = model
        
        if self.model =='BERT_base':
            self.tokenizer  = BertTokenizer.from_pretrained('bert-base-uncased')
        elif self.model =='RoBERTa':
            self.tokenizer  = RobertaTokenizer.from_pretrained('roberta-base')
        else:
            print('Undefined model')
    def forward(self, anchors, sentences):
        
        ids = []
        masks = []
        for i,anchor in enumerate(anchors):
            
            sen1 = anchor
            sen2 = sentences[i]
            
            token1 = self.tokenizer.tokenize(sen1)
            token2 = self.tokenizer.tokenize(sen2)
            
            tokens = [self.tokenizer.cls_token] + token1 + [self.tokenizer.sep_token] + token2
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            
            input_ids = input_ids + [0]*(512-len(input_ids))
            input_ids = torch.tensor(input_ids)
            attention_mask = torch.tensor(len(tokens)*[1] + (512-len(tokens))*[0])
            
            ids.append(input_ids)
            masks.append(attention_mask)
        
            
        ids = torch.stack(ids, dim=0)
        masks = torch.stack(masks, dim=0)
        return ids,masks    
    
    
class Extract_CNN_feautres(nn.Module):  
    def __init__(self,  model):
        super(Extract_CNN_feautres, self).__init__()
        
        self.tokenizer  = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        
    def forward(self, anchors, sentences): 
        
         batch = []
         for i,anchor in enumerate(anchors):
             
             sen1 = anchor
             sen2 = sentences[i]
             
             token1 = self.tokenizer.tokenize(sen1)
             token2 = self.tokenizer.tokenize(sen2)
             
             tokens =  token1 + [self.tokenizer.sep_token] + token2
             input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
             
             segments_ids = [1] * len(tokens)
             
             input_ids = input_ids + [self.tokenizer.pad_token_id]*(512-len(input_ids))
             segments_ids = segments_ids + [self.tokenizer.pad_token_id]*(512-len(segments_ids))
             
             input_ids = torch.tensor(input_ids).unsqueeze(0)
             segments_ids = torch.tensor(segments_ids).unsqueeze(0)
             
             self.model.eval()
             
             with torch.no_grad():
                 encoded_layers = self.model(input_ids,segments_ids)[0]
                 batch.append(encoded_layers)
         batch = torch.stack(batch, dim=0).squeeze(1)    
         
         return batch

def train(args):
    
    
    
    model_bin = [args.task, args.model_name, args.train_task, args.negative_samples, args.batch_size, args.gradient_accumulation_steps, args.lr, 'train' ]
    df = pd.DataFrame(columns = ['Epoch','Tr Loss', 'Val Loss', 'Tr Acc', 'Val Acc', 'lr'])
    
    if args.model_name == 'BERT_large':
        model = Bert()
    elif args.model_name == 'RoBERTa':
        model = Roberta()
    elif args.model_name == 'BertCNN':
        model = BertCNN()    
    elif args.model_name == 'BERT_base':
        model = Bert_base()  
    elif args.model_name == 'CNN':
        model = CNN()       
    else:
        print('Undefined model')
    print(model)    
       
        
    if torch.cuda.device_count() > 1:
        print('You use %d GPUs'%torch.cuda.device_count())
        model = nn.DataParallel(model, device_ids=[0,1,2,3])
        #Run the model on a specified GPU and run the operations to multiple GPUs
        torch.cuda.set_device(int(args.cuda))
        model.cuda(int(args.cuda))
        loss_fct = torch.nn.BCELoss().cuda(int(args.cuda))
    else:
        print('You use only one device')  
        device = torch.device("cuda" if USE_CUDA else "cpu")
        model.to(device)
        loss_fct = torch.nn.BCELoss()
     
        
    print('The model has {:} different named parameters.\n'.format(len(list(model.named_parameters()))))
    
    #Load data
    df_train = pd.read_csv('data/nsp_train_same_.csv')
    train_data = Dataload(df_train)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size,shuffle=True,drop_last=True)
    
    
    df_val = pd.read_csv('data/nsp_val_same_.csv')
    val_data = Dataload(df_val)
    val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size,shuffle=False,drop_last=True)
    
   
    #AdamW gets better results than Adam
    optimizer = AdamW([{'params': model.parameters(), 'lr': args.lr}], weight_decay=1.0)
    
    # Total number of training steps is [number of batches] x [number of epochs]. 
    # (Note that this is not the same as the number of training samples).
    #total_steps = len(train_data) * args.epochs
    # Create the learning rate scheduler.
    #However scheduler is not needed for fine-tuning
    #scheduler = get_linear_schedule_with_warmup(optimizer, 
     #           num_warmup_steps = 0, # Default value in run_glue.py
      #          num_training_steps = total_steps)
    if args.model_name == 'RoBERTa' or args.model_name == 'BERT_base':
        features = Extract_feautres(args.model_name)
    elif args.model_name == 'CNN':
        features = Extract_CNN_feautres(args.model_name)
        
    model.zero_grad()
    best_loss = 1000
    trigger_times = 0
    
   
    #train_iterator = tqdm.notebook.tnrange(int(args.epochs), desc="Epoch")
    for epoch in range(0, args.epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, args.epochs))
        
        
        
        
        print('Training...')
        model.train()
        
        # Reset the total loss for this epoch.
        tr_loss = 0.0
        n_labels = []
        n_predict = []
        for step, (anchor, pos, neg) in tqdm(enumerate(train_loader)):
            if args.model_name == 'RoBERTa' or args.model_name == 'BERT_base':
                pos_ids, pos_masks = features(anchor,pos)
                neg_ids, neg_masks = features(anchor,neg)
                input_ids = torch.cat((pos_ids,neg_ids), dim=0)
                attention_masks = torch.cat((pos_masks,neg_masks), dim=0)
                labels = torch.tensor(len(pos)*[1] + len(neg)*[0]).to(torch.float32)
                #Load data
                if torch.cuda.device_count() > 1:
                    input_ids = input_ids.cuda(non_blocking=True)
                    attention_masks  = attention_masks.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)
                else:    
                    input_ids = input_ids.to(device)
                    attention_masks  = attention_masks.to(device)
                    labels = labels.to(device)
                
                logits = torch.squeeze(model(input_ids,attention_masks))  
                
            elif args.model_name == 'CNN':  
                pos_ids = features(anchor,pos)
                neg_ids = features(anchor,neg)
                input_ids = torch.cat((pos_ids,neg_ids), dim=0)
                labels = torch.tensor(len(pos)*[1] + len(neg)*[0]).to(torch.float32)
                #Load data
                if torch.cuda.device_count() > 1:
                    input_ids = input_ids.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)
                else:    
                    input_ids = input_ids.to(device)
                    labels = labels.to(device)
                
                logits  = torch.squeeze(model(input_ids))
            
            
            
            loss = loss_fct(logits, labels)
            # accumulate train loss
            tr_loss += loss.item()
            
            
            loss = loss/args.gradient_accumulation_steps
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            
            
            # calculate preds
            predicted = logits.round().detach().cpu().numpy()
            # move logits and labels to CPU
            y_true = labels.detach().cpu().numpy()
            n_predict.extend(predicted)
            n_labels.extend(y_true)
            tr_accuracy = accuracy_score(n_labels, n_predict)
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                #scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                
                
            if (step+1)%args.statistic_step==0: 
                print("Average train loss per %d steps: %0.3f"%((step+1),tr_loss/(step+1)))
                print('Train accuracy per %d steps:%0.3f'%((step+1),tr_accuracy))
                print(classification_report(n_labels, n_predict, zero_division=0))
                
                
        print("")
        print('Training resuts')   
        print("Average train loss: %0.3f"%(tr_loss/len(train_data)))
        print('Train accuracy:%0.3f'%(tr_accuracy))
        print(classification_report(n_labels, n_predict, zero_division=0))     


        model.eval()
        n_predict = []
        n_labels = []
        val_loss = 0.0
        
        
        for step, (anchor, pos, neg) in tqdm(enumerate(val_loader)):
            
            if args.model_name == 'RoBERTa' or args.model_name == 'BERT_base':
                pos_ids, pos_masks = features(anchor,pos)
                neg_ids, neg_masks = features(anchor,neg)
                input_ids = torch.cat((pos_ids,neg_ids), dim=0)
                attention_masks = torch.cat((pos_masks,neg_masks), dim=0)
                labels = torch.tensor(len(pos)*[1] + len(neg)*[0]).to(torch.float32)
                #Load data
                if torch.cuda.device_count() > 1:
                    input_ids = input_ids.cuda(non_blocking=True)
                    attention_masks  = attention_masks.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)
                else:    
                    input_ids = input_ids.to(device)
                    attention_masks  = attention_masks.to(device)
                    labels = labels.to(device)
                
                with torch.no_grad():  
                    logits = torch.squeeze(model(input_ids,attention_masks))    
                
            elif args.model_name == 'CNN':  
                pos_ids = features(anchor,pos)
                neg_ids = features(anchor,neg)
                input_ids = torch.cat((pos_ids,neg_ids), dim=0)
                labels = torch.tensor(len(pos)*[1] + len(neg)*[0]).to(torch.float32)
                #Load data
                if torch.cuda.device_count() > 1:
                    input_ids = input_ids.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)
                else:    
                    input_ids = input_ids.to(device)
                    labels = labels.to(device)
                
                with torch.no_grad():  
                    logits = torch.squeeze(model(input_ids))    
        
        
            
                    
            loss = loss_fct(logits, labels)
            val_loss += loss.item()
                
                
            # calculate preds
            predicted = logits.round().detach().cpu().numpy()
            y_true = labels.detach().cpu().numpy()
            n_predict.extend(predicted)
            n_labels.extend(y_true)
            val_accuracy = accuracy_score(n_labels, n_predict)
                
            
            if (step+1)%args.statistic_step==0: 
                print("Average val loss per %d steps: %0.3f"%((step+1),val_loss/(step+1)))
                print('Val accuracy per %d steps:%0.3f'%((step+1),val_accuracy))
                print(classification_report(n_labels, n_predict, zero_division=0))
            
        print("")
        print('Val resuts')
        print("Average val loss: %0.3f"%(val_loss/len(val_data)))
        print('Val accuracy:%0.3f'%(val_accuracy)) 
        print(classification_report(n_labels, n_predict, zero_division=0))
            
        df.loc[epoch] = pd.Series({'Epoch':int(epoch), 'Tr Loss':round(tr_loss/len(train_data),4), 'Val Loss':round(val_loss/len(val_data),4), 'Tr Acc':tr_accuracy, 'Val Acc':val_accuracy})

        if (val_loss/len(val_data))<best_loss:
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
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    USE_CUDA = torch.cuda.is_available()
    train(args)
