import os
import random
import numpy as np
import torch
import torch.nn as nn
from argument_parser import parse_arguments
from load_data import Load_Data
from models.sbert import SBert
from load_data import Load_Data
from torch.optim import AdamW,Adam
import torch.multiprocessing as mp
import torch.distributed as dist
from transformers import get_linear_schedule_with_warmup
from tqdm import tnrange, tqdm
from sklearn.metrics import classification_report,accuracy_score
import pandas as pd

# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

def train(args):
    
    
    
    model_bin = [args.task, args.model_name, args.train_task, args.negative_samples, args.batch_size, args.gradient_accumulation_steps, args.lr, 'train' ]
    df = pd.DataFrame(columns = ['Epoch','Tr Loss', 'Val Loss', 'Tr Acc', 'Val Acc', 'lr'])
    
    model = SBert()    
    print(model)    
       
        
    if torch.cuda.device_count() > 1:
        print('You use %d GPUs'%torch.cuda.device_count())
        model = nn.DataParallel(model, device_ids=[0,1])
        #Run the model on a specified GPU and run the operations to multiple GPUs
        torch.cuda.set_device(int(args.cuda))
        model.cuda(int(args.cuda))
        loss_fct = nn.CrossEntropyLoss().cuda(int(args.cuda))
    else:
        print('You use only one device')  
        device = torch.device("cuda" if USE_CUDA else "cpu")
        model.to(device)
        loss_fct = nn.CrossEntropyLoss()
     
        
    print('The model has {:} different named parameters.\n'.format(len(list(model.named_parameters()))))
    
    
    #Load Data
    train_dataloader = Load_Data(args.task, 'train' , args.negative_samples, args.batch_size)
    train_data = train_dataloader.loader()
    
    val_dataloader = Load_Data(args.task, 'val' ,  args.negative_samples, args.batch_size)
    val_data = val_dataloader.loader()
    
    #AdamW gets better results than Adam
    optimizer = AdamW([{'params': model.parameters(), 'lr': args.lr}], weight_decay=1.0)
    
    # Total number of training steps is [number of batches] x [number of epochs]. 
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_data) * args.epochs
    # Create the learning rate scheduler.
    #However scheduler is not needed for fine-tuning
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                num_warmup_steps = 0, # Default value in run_glue.py
                num_training_steps = total_steps)
    
    
    model.zero_grad()
    best_loss = 1000
    
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
        for step, batch in tqdm(enumerate(train_data)):
            #Load data
            if torch.cuda.device_count() > 1:
                input_ids_1 = batch[0].cuda(non_blocking=True)
                attention_masks_1  = batch[1].cuda(non_blocking=True)
                input_ids_2 = batch[2].cuda(non_blocking=True)
                attention_masks_2 = batch[3].cuda(non_blocking=True)
                labels = batch[4].cuda(non_blocking=True)
            else:    
                input_ids_1 = batch[0].to(device)
                attention_masks_1  = batch[1].to(device)
                input_ids_2 = batch[2].to(device)
                attention_masks_2  = batch[3].to(device)
                labels = batch[4].to(device)
                
            logits = model(input_ids_1,attention_masks_1,input_ids_2,attention_masks_2)  
              
            
            loss = loss_fct(logits, labels)
            # accumulate train loss
            tr_loss += loss.item()
            
            
            loss = loss/args.gradient_accumulation_steps
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            # calculate preds
            _, predicted = torch.max(logits, 1)
            # move logits and labels to CPU
            predicted = predicted.detach().cpu().numpy()
            y_true = labels.detach().cpu().numpy()
            n_predict.extend(predicted)
            n_labels.extend(y_true)
            tr_accuracy = accuracy_score(n_labels, n_predict)
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                
                
            if (step+1)%args.statistic_step==0: 
                print('-lr: %.5f after %d steps'%(scheduler.get_last_lr()[0],(step+1)))
                print("Average train loss per %d steps: %0.3f"%((step+1),tr_loss/(step+1)))
                print('Train accuracy per %d steps:%0.3f'%((step+1),tr_accuracy))
                print(classification_report(n_labels, n_predict, target_names=['Non-sequential','Sequential'], zero_division=0))
                
                
        print("")
        print('Training resuts')   
        print("Average train loss: %0.3f"%(tr_loss/len(train_data)))
        print('Train accuracy:%0.3f'%(tr_accuracy))
        print(classification_report(n_labels, n_predict, target_names=['Non-sequential','Sequential'], zero_division=0))     


        model.eval()
        n_predict = []
        n_labels = []
        val_loss = 0.0
        
        for step, batch in tqdm(enumerate(val_data)):
            #Load data
            if torch.cuda.device_count() > 1:
                input_ids_1 = batch[0].cuda(non_blocking=True)
                attention_masks_1  = batch[1].cuda(non_blocking=True)
                input_ids_2 = batch[2].cuda(non_blocking=True)
                attention_masks_2 = batch[3].cuda(non_blocking=True)
                labels = batch[4].cuda(non_blocking=True)
            else:    
                input_ids_1 = batch[0].to(device)
                attention_masks_1  = batch[1].to(device)
                input_ids_2 = batch[2].to(device)
                attention_masks_2  = batch[3].to(device)
                labels = batch[4].to(device)
                
        
        
            with torch.no_grad():  
            
                logits = model(input_ids_1,attention_masks_1,input_ids_2,attention_masks_2)  
                  
                    
                loss = loss_fct(logits, labels)
                
                val_loss += loss.item()
                
                # calculate preds
                _, predicted = torch.max(logits, 1)
                # move logits and labels to CPU
                predicted = predicted.detach().cpu().numpy()
                y_true = labels.detach().cpu().numpy()
                n_predict.extend(predicted)
                n_labels.extend(y_true)
                val_accuracy = accuracy_score(n_labels, n_predict)
                
            
                if (step+1)%args.statistic_step==0: 
                    print("Average val loss per %d steps: %0.3f"%((step+1),val_loss/(step+1)))
                    print('Val accuracy per %d steps:%0.3f'%((step+1),val_accuracy))
                    print(classification_report(n_labels, n_predict, target_names=['Non-sequential','Sequential'], zero_division=0))
            
        print("")
        print('Val resuts')
        print("Average val loss: %0.3f"%(val_loss/len(val_data)))
        print('Val accuracy:%0.3f'%(val_accuracy)) 
        print(classification_report(n_labels, n_predict, target_names=['Non-sequential','Sequential'], zero_division=0))
            
        df.loc[epoch] = pd.Series({'Epoch':int(epoch), 'Tr Loss':round(tr_loss/len(train_data),4), 'Val Loss':round(val_loss/len(val_data),4), 'Tr Acc':tr_accuracy, 'Val Acc':val_accuracy, 'lr':scheduler.get_last_lr()[0]})

        if (val_loss/len(val_data))<best_loss:
            best_loss = val_loss/len(val_data)
            print('Found better model')
            print("Saving model to %s"%os.path.join(args.model_path,'_'.join([str(i) for i in model_bin])))
            model_to_save = model
            torch.save(model_to_save, os.path.join(args.model_path,'_'.join([str(i) for i in model_bin])))
    
    
    print("")
    df.to_csv(os.path.join(args.model_path,'_'.join([str(i) for i in model_bin]))+'.csv')
    print("Training complete!")  


    
if __name__ == "__main__":
    args = parse_arguments()
    print(args)
    
  
    USE_CUDA = torch.cuda.is_available()
    train(args)
