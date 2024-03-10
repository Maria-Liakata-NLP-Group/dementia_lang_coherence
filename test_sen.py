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

def test(args):
    
    model_bin = [args.task, args.model_name, args.train_task, args.negative_samples, args.batch_size, args.gradient_accumulation_steps, args.lr, 'train' ]
    
    
    model = SBert()    
    print(model)    
       
       
        
    if torch.cuda.device_count() > 1:
        print('You use %d GPUs'%torch.cuda.device_count())
        model = nn.DataParallel(model, device_ids=[0,1])
        if os.path.exists(os.path.join(args.model_path,   '_'.join([str(i) for i in model_bin]))):
            model = torch.load(os.path.join(args.model_path, '_'.join([str(i) for i in model_bin])))
            print('Pretrained model has been loaded')
        else:
            print('Pretrained model does not exist!!!')
            
        #Run the model on a specified GPU and run the operations to multiple GPUs
        torch.cuda.set_device(int(args.cuda))
        model.cuda(int(args.cuda))
        loss_fct = nn.CrossEntropyLoss().cuda(int(args.cuda))
    else:
        print('You use only one device')  
        device = torch.device("cuda" if USE_CUDA else "cpu")
        if os.path.exists(os.path.join(args.model_path,  '_'.join([str(i) for i in model_bin]))):
            model = torch.load(os.path.join(args.model_path, '_'.join([str(i) for i in model_bin])))
            print('Pretrained model %s has been loaded'%(os.path.join(args.model_path,  '_'.join([str(i) for i in model_bin]))))
        else:
            print('Pretrained model does not exist!!!')
        model.to(device)
        loss_fct = nn.CrossEntropyLoss()
     
        
    print('The model has {:} different named parameters.\n'.format(len(list(model.named_parameters()))))
    print('Pretrained model %s has been loaded'%(os.path.join(args.model_path,  '_'.join([str(i) for i in model_bin]))))

    
    #Load Data
    test_dataloader = Load_Data(args.task, 'test' , args.negative_samples, args.batch_size)
    test_data = test_dataloader.loader()
    
    n_labels = []
    n_predict = []
    test_loss = 0.0
    for step, batch in tqdm(enumerate(test_data)):
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
            test_loss += loss.item()
            
            # calculate preds
            _, predicted = torch.max(logits, 1)
            # move logits and labels to CPU
            predicted = predicted.detach().cpu().numpy()
            y_true = labels.detach().cpu().numpy()
            n_predict.extend(predicted)
            n_labels.extend(y_true)
            test_accuracy = accuracy_score(n_labels, n_predict)
            
                
            if (step+1)%args.statistic_step==0: 
                print("Average test loss per %d steps: %0.3f"%((step+1),test_accuracy/(step+1)))
                print('Test accuracy per %d steps:%0.3f'%((step+1),test_accuracy))
                print(classification_report(n_labels, n_predict, target_names=['Non-sequential','Sequential'], zero_division=0))
        
    print("")
    print('Test resuts')
    print("Average test loss: %0.3f"%(test_loss/len(test_data)))
    print('Val accuracy:%0.3f'%(test_accuracy)) 
    print(classification_report(n_labels, n_predict, target_names=['Non-sequential','Sequential'], zero_division=0))
     

    
if __name__ == "__main__":
    args = parse_arguments()
    print(args)
    
  
    USE_CUDA = torch.cuda.is_available()
    test(args)
