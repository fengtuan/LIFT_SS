#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 20:45:29 2022

@author: weiyang
"""

import torch
from networks import LIFT_SS_DML,T5LayerNorm_
from  datasets import *
import os
import torch.nn as nn
import argparse
import torch.backends.cudnn as cudnn
import utils
import time
import numpy as np
from loss import SoftmaxLoss
from transformers.adapters import ConfigUnion, ParallelConfig, PrefixTuningConfig
from transformers.adapters import CompacterConfig,CompacterPlusPlusConfig
from transformers.adapters import AdapterConfig,LoRAConfig,IA3Config
from torch.cuda.amp import autocast,GradScaler
from transformers.optimization import Adafactor,AdafactorSchedule
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

parser = argparse.ArgumentParser()
LookupChoices = type('', (argparse.Action, ), dict(__call__ = lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))
parser.add_argument('--SequenceIdentity',type=int,default=60)
parser.add_argument('--seed', default = 0, type = int)
parser.add_argument('--maxEpochs', default =100, type = int)
parser.add_argument('--batch_size', default = 4, type = int)
parser.add_argument('--num_class',type=int,default=9)
parser.add_argument('--proj_dim', default = 32, type = int)
parser.add_argument('--num_channels',type=int,default=1024)
parser.add_argument('--depth',type=int,default=2) 
parser.add_argument('--gradient_accumulation_steps',type=int,default=8)
parser.add_argument('--dropout', default = 0.2, type = float)
parser.add_argument('--tau', default = 18.0, type = float)

# parser.add_argument('--adapter',type=str,default="prefix_tuning")
# parser.add_argument('--adapter',type=str,default="lora")
#parser.add_argument('--adapter',type=str,default="ia3")
parser.add_argument('--adapter',type=str,default="mam_adapter")
# parser.add_argument('--adapter',type=str,default="bottleneck")
#parser.add_argument('--adapter',type=str,default="parallel_bottleneck")
#parser.add_argument('--adapter',type=str,default="compacter")
# parser.add_argument('--adapter',type=str,default="parallel_compacter")
# parser.add_argument('--adapter',type=str,default="unipelt")

parser.add_argument('--reduction_factor',type=int,default=64)
parser.add_argument('--non_linearity',type=str,default="relu")
parser.add_argument('--r',type=int,default=1)
parser.add_argument('--prefix_len',type=int,default=48)
parser.add_argument('--lora_type',type=str,default="selfattn")
parser.add_argument('--adapter_type',type=int,default=0)
parser.add_argument('--leave_out_num',type=int,default=0)
parser.add_argument('--lr', default = 1e-3, type = float, help = 'Learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-3)
parser.add_argument('--res_dir',type=str,default="dml_adapter_model")

args = parser.parse_args()
args.use_amp=True
torch.manual_seed(args.seed)
np.random.seed(args.seed) 
use_cuda =torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
if use_cuda:     
    torch.cuda.manual_seed_all(args.seed)
if(os.path.isdir(args.res_dir)==False ):
  os.mkdir(args.res_dir) 
if args.num_class==3:
    isThreeClass=True
else:
    isThreeClass=False
if args.adapter=="lora":
    if args.lora_type=='selfattn':
        config=LoRAConfig(r=args.r,alpha=args.r,leave_out=list(range(args.leave_out_num)))
        res_path= args.res_dir+'/'+str(args.SequenceIdentity)+'_'+str(args.proj_dim)+'_'+str(args.depth)+'_'+args.adapter+'_'+str(args.r)+'_'+args.lora_type+'_'+str(args.leave_out_num) 
    else:
        config=LoRAConfig(r=args.r,alpha=args.r, selfattn_lora=False,intermediate_lora=True,output_lora=True,leave_out=list(range(args.leave_out_num)))
        res_path= args.res_dir+'/'+str(args.SequenceIdentity)+'_'+str(args.proj_dim)+'_'+str(args.depth)+'_'+args.adapter+'_'+str(args.r)+'_'+"FF"+'_'+str(args.leave_out_num) 
    adapter_name="lora"
elif args.adapter=="ia3":
    config=IA3Config(leave_out=list(range(args.leave_out_num)))
    res_path= args.res_dir+'/'+str(args.SequenceIdentity)+'_'+str(args.proj_dim)+'_'+str(args.depth)+'_'+args.adapter+'_'+str(args.leave_out_num) 
    adapter_name="ia3"    
elif args.adapter=="bottleneck": 
    if args.adapter_type==0: 
        config = AdapterConfig(ln_before=True,mh_adapter=False, output_adapter=True, reduction_factor=args.reduction_factor, non_linearity=args.non_linearity,leave_out=list(range(args.leave_out_num)))   
    elif  args.adapter_type==1: 
        config = AdapterConfig(ln_before=True,mh_adapter=True, output_adapter=False, reduction_factor=args.reduction_factor, non_linearity=args.non_linearity,leave_out=list(range(args.leave_out_num)))   
    else:
        config = AdapterConfig(ln_before=True,mh_adapter=True, output_adapter=True, reduction_factor=args.reduction_factor, non_linearity=args.non_linearity,leave_out=list(range(args.leave_out_num)))   
    adapter_name="bottleneck_adapter"
    res_path= args.res_dir+'/'+str(args.SequenceIdentity)+'_'+str(args.proj_dim)+'_'+str(args.depth)+'_'+args.adapter+'_'+str(args.reduction_factor)+'_'+args.non_linearity+'_'+str(args.adapter_type)+'_'+str(args.leave_out_num) 
elif args.adapter=="parallel_bottleneck":
    if args.adapter_type==0:
        config = AdapterConfig(ln_before=True,mh_adapter=False, output_adapter=True,reduction_factor=args.reduction_factor,non_linearity=args.non_linearity,
                           init_weights="bert", is_parallel=True, scaling=1.0,leave_out=list(range(args.leave_out_num))) 
    elif  args.adapter_type==1: 
        config = AdapterConfig(ln_before=True,mh_adapter=True, output_adapter=False, reduction_factor=args.reduction_factor,non_linearity=args.non_linearity,
                           init_weights="bert", is_parallel=True, scaling=1.0,leave_out=list(range(args.leave_out_num))) 
    else:
        config = AdapterConfig(ln_before=True,mh_adapter=True, output_adapter=True, reduction_factor=args.reduction_factor, non_linearity=args.non_linearity,
                           init_weights="bert", is_parallel=True, scaling=1.0,leave_out=list(range(args.leave_out_num))) 
    adapter_name="parallel_bottleneck"
    res_path= args.res_dir+'/'+str(args.SequenceIdentity)+'_'+str(args.proj_dim)+'_'+str(args.depth)+'_'+args.adapter+'_'+str(args.reduction_factor)+'_'+args.non_linearity+'_'+str(args.adapter_type)+'_'+str(args.leave_out_num) 
elif args.adapter=="compacter":
    if args.adapter_type==0:
       config = AdapterConfig(ln_before=True,mh_adapter=False, output_adapter=True, reduction_factor=args.reduction_factor, non_linearity=args.non_linearity,
                           init_weights="bert", is_parallel=False, scaling=1.0,
                           phm_layer=True,leave_out=list(range(args.leave_out_num)))   
    elif  args.adapter_type==1: 
       config = AdapterConfig(ln_before=True,mh_adapter=True, output_adapter=False, reduction_factor=args.reduction_factor, non_linearity=args.non_linearity,
                           init_weights="bert", is_parallel=False, scaling=1.0,
                           phm_layer=True,leave_out=list(range(args.leave_out_num)))   
    else:
       config = AdapterConfig(ln_before=True,mh_adapter=True, output_adapter=True, reduction_factor=args.reduction_factor, non_linearity=args.non_linearity,
                           init_weights="bert", is_parallel=False, scaling=1.0,
                           phm_layer=True,leave_out=list(range(args.leave_out_num)))
    adapter_name="compacter"
    res_path= args.res_dir+'/'+str(args.SequenceIdentity)+'_'+str(args.proj_dim)+'_'+str(args.depth)+'_'+args.adapter+'_'+str(args.reduction_factor)+'_'+args.non_linearity+'_'+str(args.adapter_type)+'_'+str(args.leave_out_num) 
elif args.adapter=="parallel_compacter":
    if args.adapter_type==0:
       config = AdapterConfig(ln_before=True,mh_adapter=False, output_adapter=True, reduction_factor=args.reduction_factor, non_linearity=args.non_linearity,
                           init_weights="bert", is_parallel=True, scaling=1.0,
                           phm_layer=True,leave_out=list(range(args.leave_out_num))) 
    elif  args.adapter_type==1: 
       config = AdapterConfig(ln_before=True,mh_adapter=True, output_adapter=False, reduction_factor=args.reduction_factor, non_linearity=args.non_linearity,
                           init_weights="bert", is_parallel=True, scaling=1.0,
                           phm_layer=True,leave_out=list(range(args.leave_out_num))) 
    else:
       config = AdapterConfig(ln_before=True,mh_adapter=True, output_adapter=True, reduction_factor=args.reduction_factor, non_linearity=args.non_linearity,
                           init_weights="bert", is_parallel=True, scaling=1.0,
                           phm_layer=True,leave_out=list(range(args.leave_out_num))) 
    adapter_name="parallel_compacter" 
    res_path= args.res_dir+'/'+str(args.SequenceIdentity)+'_'+str(args.proj_dim)+'_'+str(args.depth)+'_'+args.adapter+'_'+str(args.reduction_factor)+'_'+args.non_linearity+'_'+str(args.adapter_type)+'_'+str(args.leave_out_num) 
elif args.adapter=="mam_adapter":    
    config = ConfigUnion(
    PrefixTuningConfig(cross_prefix=False,flat=True, prefix_length=args.prefix_len,leave_out=list(range(args.leave_out_num))),
    AdapterConfig(ln_before=True,mh_adapter=False, output_adapter=True, reduction_factor=args.reduction_factor, non_linearity=args.non_linearity,
                        init_weights="bert", is_parallel=True, scaling=1.0,
                        phm_layer=True,leave_out=list(range(args.leave_out_num))), 
    )
    adapter_name="mam_adapter"
    res_path= args.res_dir+'/'+str(args.SequenceIdentity)+'_'+str(args.proj_dim)+'_'+str(args.depth)+'_'+args.adapter+'_'+args.non_linearity+'_'+str(args.reduction_factor)+'_'+str(args.prefix_len)+'_'+str(args.leave_out_num)    
elif args.adapter=="unipelt":    
    config = ConfigUnion(
    LoRAConfig(r=args.r,alpha=args.r,leave_out=list(range(args.leave_out_num)),use_gating=True),
    PrefixTuningConfig(cross_prefix=False,flat=True, prefix_length=args.prefix_len,leave_out=list(range(args.leave_out_num)),use_gating=True),
    AdapterConfig(ln_before=True,mh_adapter=False, output_adapter=True, reduction_factor=args.reduction_factor, non_linearity=args.non_linearity,
                        init_weights="bert", is_parallel=True, scaling=1.0,
                        phm_layer=True,leave_out=list(range(args.leave_out_num)),use_gating=True),
    )
    adapter_name="unipelt"
    res_path= args.res_dir+'/'+str(args.SequenceIdentity)+'_'+str(args.proj_dim)+'_'+str(args.depth)+'_'+args.adapter+'_'+args.non_linearity+'_'+str(args.reduction_factor)+'_'+str(args.prefix_len)+'_'+str(args.r)+'_'+str(args.leave_out_num)    
else:
    config = PrefixTuningConfig(cross_prefix=False,flat=True, prefix_length=args.prefix_len,leave_out=list(range(args.leave_out_num)))
    adapter_name="prefix_tuning"
    res_path= args.res_dir+'/'+str(args.SequenceIdentity)+'_'+str(args.proj_dim)+'_'+str(args.depth)+'_'+args.adapter+'_'+str(args.prefix_len)+'_'+str(args.leave_out_num) 

if(os.path.isdir(res_path)==False ):
  os.mkdir(res_path) 

train_list,valid_list=load_train_valid(args.SequenceIdentity,isThreeClass)  
split_sequence(train_list)
model=LIFT_SS_DML(config,adapter_name,args.num_channels,args.depth,args.proj_dim,args.dropout).to(device)
criterion = SoftmaxLoss(args.num_class,args.proj_dim,args.tau).to(device)
def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


opt_model=model
ALL_LAYERNORM_LAYERS.append(T5LayerNorm_)
decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
decay_parameters = [name for name in decay_parameters if "bias" not in name]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in opt_model.named_parameters() if n in decay_parameters and p.requires_grad==True],
        "weight_decay": args.weight_decay,
    },
    {
        "params": [p for n, p in opt_model.named_parameters() if n not in decay_parameters and p.requires_grad==True],
        "weight_decay": 0.0,
    },
    {"params":criterion.parameters(),
     "weight_decay":args.weight_decay,
    }
]   

FileName=res_path+'/log_'+str(args.num_class)+time.strftime('_%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
ModelName='%s/model_%d'%(res_path,args.num_class)
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr,amsgrad=False)


scaler=GradScaler(enabled=args.use_amp)
#############################################################
f = open(FileName+'.txt', 'w')
# early-stopping parameters
decrease_patience=5
best_accuracy = 0
best_epoch = 0
epoch = 0
Num_Of_decrease=0
done_looping = False
while (epoch < args.maxEpochs) and (not done_looping):
    epoch = epoch + 1
    start_time = time.time()  
    average_loss=utils.half_train(args,model,scaler,device,train_list,optimizer,criterion)
    # scheduler.step()
    print("{}th Epoch took {:.3f}s".format(epoch, time.time() - start_time))
    f.write("{}th Epoch took {:.3f}s\n".format(epoch, time.time() - start_time))
    print("  training loss:\t\t{:.3f}".format(average_loss))
    f.write("  training loss:\t\t{:.3f}\n".format(average_loss))  
    accuracy=utils.eval(args,model,device,valid_list,args.batch_size,criterion) 
    print("  validation accuracy:\t\t{:.2f}".format(accuracy[-1]))
    f.write("  validation accuracy:\t\t{:.2f}\n".format(accuracy[-1]))        
    f.flush()
    # if we got the best validation accuracy until now
    if accuracy[-1] > best_accuracy:
      #improve patience if loss improvement is good enough
        best_accuracy = accuracy[-1]
        best_epoch = epoch
        Num_Of_decrease=0
        model.save(ModelName,criterion)      
    else:
        Num_Of_decrease=Num_Of_decrease+1
    if (Num_Of_decrease>=decrease_patience):
        done_looping = True    
print('The validation accuracy %.2f %% of the best model in the %i th epoch' 
            %(best_accuracy,best_epoch))
f.write('The validation accuracy %.2f %% of the best model in the %i th epoch\n' 
            %(best_accuracy,best_epoch))

model.load(ModelName,criterion)
test_datasets=['CB433','CASP12','CASP13','CASP14']
for l in range(len(test_datasets)):
    dataset=test_datasets[l]
    FileName=res_path+'/'+dataset+'_'+ str(args.num_class) 
    test_list=load_test_data(dataset,isThreeClass) 
    args.dataset=dataset
    args.res_dir=res_path
    F1,accuracy,sov_results=utils.eval_sov(args,model,device,test_list,1,criterion)
    if not isThreeClass:
        print('Q9: %.2f %%, F1: %.2f %%  on the dataset  %s'%(accuracy[-1],100*F1,dataset))
        print('L: %.2f,B: %.2f,E: %.2f,G: %.2f,I: %.2f,H: %.2f,S: %.2f,T: %.2f,P: %.2f '%(accuracy[0],accuracy[1],accuracy[2],accuracy[3],accuracy[4],accuracy[5],accuracy[6],accuracy[7],accuracy[8]))
        print('9-state SOV results:')
        print(sov_results)
        f.write('Q9:%.2f %%, F1:%.2f %% on the dataset  %s:\n'%(accuracy[-1],100*F1,dataset))
        f.write('  L: %.2f,B: %.2f,E: %.2f,G: %.2f,I: %.2f,H: %.2f,S: %.2f,T: %.2f,P: %.2f \n'%(accuracy[0],accuracy[1],accuracy[2],accuracy[3],accuracy[4],accuracy[5],accuracy[6],accuracy[7],accuracy[8]))
        f.write('  [%.2f,%.2f,%.2f, %.2f,%.2f,%.2f, %.2f,%.2f,%.2f] \n'%(accuracy[0],accuracy[1],accuracy[2],accuracy[3],accuracy[4],accuracy[5],accuracy[6],accuracy[7],accuracy[8]))
        f.write('9-state SOV results:\n')
        f.write(' %s\n'%(sov_results))            
    else:
        print('Q3: %.2f %%, F1: %.2f %%  on the dataset  %s'%(accuracy[-1],100*F1,dataset))
        print('C: %.2f,E: %.2f,H: %.2f'%(accuracy[0],accuracy[1],accuracy[2]))
        print('3-state SOV results:')
        print(sov_results)
        f.write('Q3:%.2f %%, F1:%.2f %% on the dataset  %s:\n'%(accuracy[-1],100*F1,dataset))
        f.write('  C: %.2f,E: %.2f,H: %.2f\n'%(accuracy[0],accuracy[1],accuracy[2]))
        f.write('  [%.2f,%.2f,%.2f] \n'%(accuracy[0],accuracy[1],accuracy[2]))
        f.write('3-state SOV results:\n')
        f.write(' %s\n'%(sov_results))     
    accuracy.append(100*F1)
    utils.save_excel(accuracy,FileName,sov_results)
f.close()
