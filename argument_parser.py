import argparse


def parse_arguments(): 
     parser = argparse.ArgumentParser()
     
    
     
     parser.add_argument('--cuda', default="0", type=str, help='enables cuda')
     parser.add_argument('--epochs', default=50, type=int, help='number of epochs to train for')
     parser.add_argument('--model_path', default="best", help='path to save the model')
     parser.add_argument('--checkpoint_path', default="tmp", help='path to save the checkpoint model')
     parser.add_argument('--model_name', default="RoBERTa", choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'BERT_large', 'BERT_base', 'BertCNN', 'RoBERTa', 'SBert', 'CNN', 'gptneo', 't5-base', 't5-large', 'coh_model'], help='model name')
     parser.add_argument('--negative_samples', default="same", choices=['across', 'same'], help='type of negative samples')
     parser.add_argument('--train_task', default="fine-tune", choices=['fine-tune', 'train', 'zero_shot'], help='type of negative samples')
     
     
     parser.add_argument('--batch_size', type=int, default=16, help='batch size')
     parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
     parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help= 'Gradient accumulation loss')
     parser.add_argument('--max_grad_norm', type=float, default=1.0, help= 'Gradient clip')
     parser.add_argument('--patience', type=int, default=4, help= 'Early stop ')
     
     parser.add_argument('--statistic_step', type=int, default=20, help='show statistics per a number of step')
     parser.add_argument('--task', type=str, default='nsp', choices=['context', 'nsp', 'gen', 'disc'], help='Name of task')
     
     parser.add_argument('-n', '--nodes', default=1, type=int, help="number of machines")
     parser.add_argument('-g', '--gpus', default=4, type=int,help='number of gpus per machine')
     parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')
     
     opt = parser.parse_args()
     return opt
