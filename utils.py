#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 16:43:28 2020

@author: weiyang
"""
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import itertools
import subprocess
import os
from torch.cuda.amp import autocast
import torch.nn as nn
import torch.nn.functional as F

        
def iterate_minibatches(ProteinLists, batchsize,shuffle=True):
    N= len(ProteinLists)
    last_size=N % batchsize
    indices = np.arange(N)
    if shuffle:        
        np.random.shuffle(indices)   
    maxLength=0
    masks=torch.zeros(size=(batchsize,4096),dtype=torch.long)
    targets=torch.zeros(size=(batchsize,4096),dtype=torch.long)
    PrimarySeqs=torch.zeros(size=(batchsize,4096),dtype=torch.long)
    for idx in range(N):
        if idx % batchsize==0:           
            masks.fill_(0)
            PrimarySeqs.fill_(0)
            targets.fill_(-100)
            batch_idx=0          
            maxLength=0
        length=ProteinLists[indices[idx]].ProteinLen+1        
        masks[batch_idx,:length]=1      
        targets[batch_idx,:length]=ProteinLists[indices[idx]].SecondarySeq[:]
        PrimarySeqs[batch_idx,:length]=ProteinLists[indices[idx]].PrimarySeq[:]
        batch_idx+=1
        if length>maxLength:
                maxLength=length
        if (idx+1) % batchsize==0:
            yield PrimarySeqs[:,:maxLength],targets[:,:maxLength],masks[:,:maxLength],
    if last_size !=0:        
        yield PrimarySeqs[:last_size,:maxLength],targets[:last_size,:maxLength],masks[:last_size,:maxLength]


def embedding_feature_iterate_minibatches(ProteinLists, batchsize,shuffle=True):
    N= len(ProteinLists)
    last_size=N % batchsize    
    num_features=ProteinLists[0].Profile.shape[1]
    indices = np.arange(N)
    if shuffle:        
        np.random.shuffle(indices)   
    maxLength=0
    inputs=torch.zeros(size=(batchsize,4096,num_features),dtype=torch.float32)
    masks=torch.zeros(size=(batchsize,4096),dtype=torch.long)
    targets=torch.zeros(size=(batchsize,4096),dtype=torch.long)
    for idx in range(N):
        if idx % batchsize==0:
            inputs.fill_(0)
            masks.fill_(0)
            targets.fill_(-100)
            batch_idx=0          
            maxLength=0
        length=ProteinLists[indices[idx]].ProteinLen+1        
        masks[batch_idx,:length]=1
        inputs[batch_idx,:length,:]=ProteinLists[indices[idx]].Profile[:,:]
        targets[batch_idx,:length]=ProteinLists[indices[idx]].SecondarySeq[:length]
        batch_idx+=1
        if length>maxLength:
                maxLength=length
        if (idx+1) % batchsize==0:
            yield inputs[:,:maxLength,:].transpose(1,2),targets[:,:maxLength],masks[:,:maxLength]
    if last_size !=0:        
        yield inputs[:last_size,:maxLength,:].transpose(1,2),targets[:last_size,:maxLength],masks[:last_size,:maxLength]


def half_train(args,model,scaler,device,train_list,optimizer,criterion):
    model.train()
    total_loss=0.0
    count=0
    num_batches=(len(train_list)+args.batch_size-1)//args.batch_size
    gradient_accumulation_steps=args.gradient_accumulation_steps
    optimizer.zero_grad() 
    batch_loss=0
    for batch in iterate_minibatches(train_list,args.batch_size, shuffle=True):
        inputs, targets,masks = batch
        inputs=inputs.to(device)
        targets=targets.to(device)
        masks=masks.to(device)       
        with autocast(enabled=args.use_amp):           
            outputs=model(inputs,masks)
            outputs=outputs.view(-1,outputs.size(2))
            loss=criterion(outputs,targets.flatten())/gradient_accumulation_steps
        batch_loss+=loss.item()
        scaler.scale(loss).backward()
        if ((count+ 1)%gradient_accumulation_steps== 0) or (count == num_batches-1):
            total_loss += batch_loss                
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            batch_loss =0           
        count+=1
        if count % 1000==0:
            print(count)
            
    return total_loss/num_batches*gradient_accumulation_steps

def generate_embedding_feature(model,device,data_list,batch_size):
    model.eval()
    idx=0
    with torch.no_grad():
        for batch in iterate_minibatches(data_list,batch_size, shuffle=False):
            inputs, targets,masks = batch
            n=inputs.shape[0]
            inputs=inputs.to(device)
            targets=targets.to(device)
            masks=masks.to(device)
            outputs=model(inputs,masks)[0]
            outputs=outputs.cpu()
            print(idx)
            for i in range(n):
                data_list[idx+i].Profile=outputs[i,:data_list[idx+i].ProteinLen+1,:]        
            idx+=n

def train_embedding_feature(args,model,device,train_list,optimizer,criterion):
    model.train()
    total_loss=0.0
    count=0
    for batch in embedding_feature_iterate_minibatches(train_list,args.batch_size, shuffle=True):
        inputs, targets,masks = batch
        inputs=inputs.to(device)
        targets=targets.to(device)
        masks=masks.to(device)
        optimizer.zero_grad()
        outputs=model(inputs,masks)
        outputs=outputs.view(-1,outputs.size(2))
        loss=criterion(outputs,targets)       
        total_loss += loss.item()
        count+=1
        loss.backward()
        optimizer.step()
    return total_loss/count 

# Each class is represnted by its centroid, with test samples classified to the 
#     class with the nearest centroid.    
def NearestCentroidClassifier(x,centroids):        
    assert x.size()[1]==centroids.size()[1],"The dimension must match."
    with torch.no_grad():
        dist=torch.matmul(x,centroids.t())
        idx=torch.topk(dist,1,dim=1,largest=True)[1]
        pred_labels=idx.squeeze()
    return pred_labels 


def eval(args,model,device,eval_list,batch_size,criterion=None):    
    model.eval()
    labels_true=np.array([])
    labels_pred=np.array([])
    with torch.no_grad():
        if criterion is not None:
            centroids=criterion.getNormalizedCentrioids()         
        for batch in iterate_minibatches(eval_list,batch_size, shuffle=False):
            inputs, targets,masks = batch
            inputs=inputs.to(device)
            targets=targets.to(device)
            masks=masks.to(device)
            with autocast(enabled=args.use_amp):
                outputs=model(inputs,masks)            
            if criterion is None:
                pred_labels=torch.argmax(outputs, dim=2)
            else:
                outputs=outputs.view(-1,outputs.size(2))
                pred_labels=NearestCentroidClassifier(outputs,centroids).view(inputs.shape[0],-1)
            Lengths=masks.sum(dim=-1).cpu().numpy()
            for i in range(len(Lengths)):
                L=Lengths[i]-1
                if L>0: 
                    labels_true=np.hstack((labels_true,targets[i,:L].cpu().numpy()))
                    labels_pred=np.hstack((labels_pred,pred_labels[i,:L].cpu().numpy()))     
    class_correct=list(0. for i in range(args.num_class))
    class_total=list(0. for i in range(args.num_class))
    idxs=np.where(labels_true!=-100)# def NearestCentroidClassifier(x,centroids):        
#     assert x.size()[1]==centroids.size()[1],"The dimension must match."
#     with torch.no_grad():
#         dist=1.0-torch.matmul(x,centroids.t())
#         idx=torch.topk(dist,1,dim=1,largest=False)[1]
#         pred_labels=idx.squeeze()
#     return pred_labels
    labels_true=labels_true[idxs]
    labels_pred=labels_pred[idxs]
    for i in range(len(labels_true)):
        label=int(labels_true[i])
        class_total[label]+=1
        if label==labels_pred[i]:
            class_correct[label]+=1
#    classes=['L','B','E','G','I','H','S','T','P'] or ['C','E','H']
    accuracy=[]
    for i in range(args.num_class):
        accuracy.append(100.0*class_correct[i]/(class_total[i]+1e-12))
    accuracy.append(100.0*sum(class_correct)/sum(class_total))
    return accuracy

def eval_embedding_feature(args,model,device,eval_list,batch_size,criterion=None):    
    model.eval()
    labels_true=np.array([])
    labels_pred=np.array([])
    with torch.no_grad():
        if criterion is not None:
            centroids=criterion.getNormalizedCentrioids()         
        for batch in embedding_feature_iterate_minibatches(eval_list,batch_size, shuffle=False):
            inputs, targets,masks = batch
            inputs=inputs.to(device)
            targets=targets.to(device)
            masks=masks.to(device)            
            outputs=model(inputs,masks)            
            if criterion is None:
                pred_labels=torch.argmax(outputs, dim=2)
            else:
                outputs=outputs.view(-1,outputs.size(2))
                pred_labels=NearestCentroidClassifier(outputs,centroids).view(inputs.shape[0],-1)
            Lengths=masks.sum(dim=-1).cpu().numpy()
            for i in range(len(Lengths)):
                L=Lengths[i]-1
                if L>0: 
                    labels_true=np.hstack((labels_true,targets[i,:L].cpu().numpy()))
                    labels_pred=np.hstack((labels_pred,pred_labels[i,:L].cpu().numpy()))     
    class_correct=list(0. for i in range(args.num_class))
    class_total=list(0. for i in range(args.num_class))
    idxs=np.where(labels_true!=-100)
    labels_true=labels_true[idxs]
    labels_pred=labels_pred[idxs]
    for i in range(len(labels_true)):
        label=int(labels_true[i])
        class_total[label]+=1
        if label==labels_pred[i]:
            class_correct[label]+=1
#    classes=['L','B','E','G','I','H','S','T','P'] or ['C','E','H']
    accuracy=[]
    for i in range(args.num_class):
        accuracy.append(100.0*class_correct[i]/(class_total[i]+1e-12))
    accuracy.append(100.0*sum(class_correct)/sum(class_total))
    return accuracy

def eval_sov_embedding_feature(args,model,device,eval_list,batch_size,criterion=None):
    model.eval()
    labels_true=np.array([])
    labels_pred=np.array([])
    Eval_FileName=args.res_dir+'/'+args.dataset
    f = open(Eval_FileName+'_Q%d.txt'%(args.num_class), 'w')
    count=0
    if args.num_class!=3:
       SS_dict={0:'L', 1:'B', 2:'E', 3:'G', 4:'I', 5:'H', 6:'S', 7:'T', 8:'P',-100:'X'}
    else:
       SS_dict={0:'C', 1:'E', 2:'H',-100:'X'} 
    with torch.no_grad():
        if criterion is not None:
            centroids=criterion.getNormalizedCentrioids()        
        for batch in embedding_feature_iterate_minibatches(eval_list,batch_size, shuffle=False):
            inputs, targets,masks = batch
            inputs=inputs.to(device)
            targets=targets.to(device)
            masks=masks.to(device)                    
            outputs=model(inputs,masks)           
            if criterion is None:
                pred_labels=torch.argmax(outputs, dim=2)
            else:
                outputs=outputs.view(-1,outputs.size(2))
                pred_labels=NearestCentroidClassifier(outputs,centroids).view(inputs.shape[0],-1)
            Lengths=masks.sum(dim=-1).cpu().numpy()
            for i in range(len(Lengths)):
                L=Lengths[i]-1
                if L>0:
                    label_t=targets[i,:L].cpu().numpy()
                    label_p=pred_labels[i,:L].cpu().numpy()
                    labels_true=np.hstack((labels_true,targets[i,:L].cpu().numpy()))
                    labels_pred=np.hstack((labels_pred,pred_labels[i,:L].cpu().numpy()))
                    label_t_=''                    
                    for i in label_t:                        
                        label_t_+=SS_dict[i] 
                    label_p_=''
                    for i in label_p:
                        label_p_+=SS_dict[i]                   
                    f.write('>%s %d\n'%(eval_list[count].ProteinID,eval_list[count].ProteinLen))
                    f.write('%s\n'%(label_t_))
                    f.write('%s\n'%(label_p_))
                    count+=1
    
    f.close()
    commands="perl SOV.pl "+Eval_FileName+'_Q%d.txt'%(args.num_class)    
    subprocess.call(commands, stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)   
    f=open(Eval_FileName+'_Q%d_Eval.txt'%(args.num_class),'rt')
    line=f.readline()
    sov_results=line.strip() 
    f.close()          
    if os.path.exists(Eval_FileName+'_Q%d_Eval.txt'%(args.num_class)):
      os.remove(Eval_FileName+'_Q%d_Eval.txt'%(args.num_class)) 
    idxs=np.where(labels_true!=-100)
    labels_true=labels_true[idxs]
    labels_pred=labels_pred[idxs]
    F1=f1_score(labels_true,labels_pred,average='macro',labels=np.unique(labels_pred))
    class_correct=list(0. for i in range(args.num_class))
    class_total=list(0. for i in range(args.num_class))
    for i in range(len(labels_true)):
        label=int(labels_true[i])
        class_total[label]+=1
        if label==labels_pred[i]:
            class_correct[label]+=1
#    classes=['L','B','E','G','I','H','S','T'] or ['C','E','H']
    accuracy=[]
    for i in range(args.num_class):
        accuracy.append(100.0*class_correct[i]/(class_total[i]+1e-12))       
    accuracy.append(100.0*sum(class_correct)/sum(class_total))
    return F1,accuracy,sov_results

def eval_sov(args,model,device,eval_list,batch_size,criterion=None):
    model.eval()
    labels_true=np.array([])
    labels_pred=np.array([])
    Eval_FileName=args.res_dir+'/'+args.dataset
    f = open(Eval_FileName+'_Q%d.txt'%(args.num_class), 'w')
    count=0
    if args.num_class!=3:
       SS_dict={0:'L', 1:'B', 2:'E', 3:'G', 4:'I', 5:'H', 6:'S', 7:'T', 8:'P',-100:'X'}
    else:
       SS_dict={0:'C', 1:'E', 2:'H',-100:'X'} 
    with torch.no_grad():
        if criterion is not None:
            centroids=criterion.getNormalizedCentrioids()        
        for batch in iterate_minibatches(eval_list,batch_size, shuffle=False):
            inputs, targets,masks = batch
            inputs=inputs.to(device)
            targets=targets.to(device)
            masks=masks.to(device)
            with autocast(enabled=args.use_amp):            
                outputs=model(inputs,masks)           
            if criterion is None:
                pred_labels=torch.argmax(outputs, dim=2)
            else:
                outputs=outputs.view(-1,outputs.size(2))
                pred_labels=NearestCentroidClassifier(outputs,centroids).view(inputs.shape[0],-1)
            Lengths=masks.sum(dim=-1).cpu().numpy()
            for i in range(len(Lengths)):
                L=Lengths[i]-1
                if L>0:
                    label_t=targets[i,:L].cpu().numpy()
                    label_p=pred_labels[i,:L].cpu().numpy()
                    labels_true=np.hstack((labels_true,targets[i,:L].cpu().numpy()))
                    labels_pred=np.hstack((labels_pred,pred_labels[i,:L].cpu().numpy()))
                    label_t_=''                    
                    for i in label_t:                        
                        label_t_+=SS_dict[i] 
                    label_p_=''
                    for i in label_p:
                        label_p_+=SS_dict[i]                   
                    f.write('>%s %d\n'%(eval_list[count].ProteinID,eval_list[count].ProteinLen))
                    f.write('%s\n'%(label_t_))
                    f.write('%s\n'%(label_p_))
                    count+=1
    
    f.close()
    commands="perl SOV.pl "+Eval_FileName+'_Q%d.txt'%(args.num_class)    
    subprocess.call(commands, stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)   
    f=open(Eval_FileName+'_Q%d_Eval.txt'%(args.num_class),'rt')
    line=f.readline()
    sov_results=line.strip() 
    f.close()          
    if os.path.exists(Eval_FileName+'_Q%d_Eval.txt'%(args.num_class)):
      os.remove(Eval_FileName+'_Q%d_Eval.txt'%(args.num_class)) 
    idxs=np.where(labels_true!=-100)
    labels_true=labels_true[idxs]
    labels_pred=labels_pred[idxs]
    F1=f1_score(labels_true,labels_pred,average='macro',labels=np.unique(labels_pred))
    class_correct=list(0. for i in range(args.num_class))
    class_total=list(0. for i in range(args.num_class))
    for i in range(len(labels_true)):
        label=int(labels_true[i])
        class_total[label]+=1
        if label==labels_pred[i]:
            class_correct[label]+=1
#    classes=['L','B','E','G','I','H','S','T'] or ['C','E','H']
    accuracy=[]
    for i in range(args.num_class):
        accuracy.append(100.0*class_correct[i]/(class_total[i]+1e-12))       
    accuracy.append(100.0*sum(class_correct)/sum(class_total))
    return F1,accuracy,sov_results

def eval_sov_9_3(args,model,device,eval_list,batch_size,criterion=None):
    model.eval()
    labels_true=np.array([])
    labels_pred=np.array([])
    Eval_FileName=args.res_dir+'/'+args.dataset
    f = open(Eval_FileName+'_Q3.txt', 'w')
    count=0
    SS_dict={0:'C', 1:'E', 2:'H',-100:'X'} 
    with torch.no_grad():
        if criterion is not None:
            centroids=criterion.getNormalizedCentrioids()        
        for batch in iterate_minibatches(eval_list,batch_size, shuffle=False):
            inputs, targets,masks = batch
            inputs=inputs.to(device)
            targets=targets.to(device)
            masks=masks.to(device)
            with autocast(enabled=args.use_amp):            
                outputs=model(inputs,masks)           
            if criterion is None:
                pred_probs=F.softmax(outputs,dim=-1)   
            else:
                outputs=outputs.view(-1,outputs.size(2))                
                dist=args.tau*torch.matmul(outputs,centroids.t()).view(inputs.size(0),inputs.size(1),-1)
                pred_probs=F.softmax(dist,dim=-1)           
            helix=torch.index_select(pred_probs,dim=-1,index=torch.tensor([3,4,5,8]).cuda()).sum(dim=-1,keepdim=True)
            strand=torch.index_select(pred_probs, dim=-1, index=torch.tensor([1,2]).cuda()).sum(dim=-1,keepdim=True)
            coil=torch.index_select(pred_probs, dim=-1, index=torch.tensor([0,6,7]).cuda()).sum(dim=-1,keepdim=True)           
            probs=torch.cat([coil,strand,helix],dim=-1)          
            pred_labels=torch.argmax(probs, dim=-1)              
            Lengths=masks.sum(dim=-1).cpu().numpy()
            for i in range(len(Lengths)):
                L=Lengths[i]-1
                if L>0:
                    label_t=targets[i,:L].cpu().numpy()
                    label_p=pred_labels[i,:L].cpu().numpy()
                    labels_true=np.hstack((labels_true,targets[i,:L].cpu().numpy()))
                    labels_pred=np.hstack((labels_pred,pred_labels[i,:L].cpu().numpy()))
                    label_t_=''                    
                    for i in label_t:                        
                        label_t_+=SS_dict[i] 
                    label_p_=''
                    for i in label_p:
                        label_p_+=SS_dict[i]                   
                    f.write('>%s %d\n'%(eval_list[count].ProteinID,eval_list[count].ProteinLen))
                    f.write('%s\n'%(label_t_))
                    f.write('%s\n'%(label_p_))
                    count+=1
    
    f.close()
    commands="perl SOV.pl "+Eval_FileName+'_Q3.txt'   
    subprocess.call(commands, stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)   
    f=open(Eval_FileName+'_Q3_Eval.txt','rt')
    line=f.readline()
    sov_results=line.strip() 
    f.close()          
    if os.path.exists(Eval_FileName+'_Q3_Eval.txt'):
      os.remove(Eval_FileName+'_Q3_Eval.txt') 
    idxs=np.where(labels_true!=-100)
    labels_true=labels_true[idxs]
    labels_pred=labels_pred[idxs]
    F1=f1_score(labels_true,labels_pred,average='macro',labels=np.unique(labels_pred))
    class_correct=list(0. for i in range(3))
    class_total=list(0. for i in range(3))
    for i in range(len(labels_true)):
        label=int(labels_true[i])
        class_total[label]+=1
        if label==labels_pred[i]:
            class_correct[label]+=1
    # classes= ['C','E','H']
    accuracy=[]
    for i in range(3):
        accuracy.append(100.0*class_correct[i]/(class_total[i]+1e-12))       
    accuracy.append(100.0*sum(class_correct)/sum(class_total))
    return F1,accuracy,sov_results

def Ensemble_eval_sov(args,model,model_names,device,eval_list,batch_size,criterion):
    num_models=len(model_names)
    labels_true=np.array([])
    labels_pred=np.array([])
    Eval_FileName=args.res_dir+'/'+args.dataset
    f = open(Eval_FileName+'_Q%d.txt'%(args.num_class), 'w')
    count=0
    if args.num_class!=3:
       SS_dict={0:'L', 1:'B', 2:'E', 3:'G', 4:'I', 5:'H', 6:'S', 7:'T', 8:'P',-100:'X'}
    else:
       SS_dict={0:'C', 1:'E', 2:'H',-100:'X'} 
      
    for batch in iterate_minibatches(eval_list,batch_size, shuffle=False):
        inputs, targets,masks = batch
        inputs=inputs.to(device)
        targets=targets.to(device)
        masks=masks.to(device)
        dists_=[]
        for l in range(num_models):
            model.load(model_names[l],criterion)
            model.eval()
            with torch.no_grad():
                if criterion is not None:
                    centroids=criterion.getNormalizedCentrioids()
                with autocast(enabled=args.use_amp):
                    outputs=model(inputs,masks)     
                outputs=outputs.view(-1,outputs.size(2))
                S=torch.matmul(outputs,centroids.t())                
            dists_.append(S)                
        dist=dists_[0]
        for j in range(num_models-1):
            dist+=dists_[j+1]
        dist=dist/num_models            
        idx=torch.topk(dist,1,dim=1,largest=True)[1]
        pred_labels=idx.squeeze().view(batch_size,-1)            
        Lengths=masks.sum(dim=-1).cpu().numpy()
        for i in range(len(Lengths)):
            L=Lengths[i]-1
            if L>0:
                label_t=targets[i,:L].cpu().numpy()
                label_p=pred_labels[i,:L].cpu().numpy()
                labels_true=np.hstack((labels_true,targets[i,:L].cpu().numpy()))
                labels_pred=np.hstack((labels_pred,pred_labels[i,:L].cpu().numpy()))
                label_t_=''                    
                for i in label_t:                        
                    label_t_+=SS_dict[i] 
                label_p_=''
                for i in label_p:
                    label_p_+=SS_dict[i]
               
                f.write('>%s %d\n'%(eval_list[count].ProteinID,eval_list[count].ProteinLen))
                f.write('%s\n'%(label_t_))
                f.write('%s\n'%(label_p_))
                count+=1
    
    f.close()
    commands="perl SOV.pl "+Eval_FileName+'_Q%d.txt'%(args.num_class)    
    subprocess.call(commands, stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)   
    f=open(Eval_FileName+'_Q%d_Eval.txt'%(args.num_class),'rt')
    line=f.readline()
    sov_results=line.strip() 
    f.close()          
    if os.path.exists(Eval_FileName+'_Q%d_Eval.txt'%(args.num_class)):
      os.remove(Eval_FileName+'_Q%d_Eval.txt'%(args.num_class)) 
    idxs=np.where(labels_true!=-100)
    labels_true=labels_true[idxs]
    labels_pred=labels_pred[idxs]
    F1=f1_score(labels_true,labels_pred,average='macro',labels=np.unique(labels_pred))
    class_correct=list(0. for i in range(args.num_class))
    class_total=list(0. for i in range(args.num_class))
    for i in range(len(labels_true)):
        label=int(labels_true[i])
        class_total[label]+=1
        if label==labels_pred[i]:
            class_correct[label]+=1
#    classes=['L','B','E','G','I','H','S','T'] or ['C','E','H']
    accuracy=[]
    for i in range(args.num_class):
        accuracy.append(100.0*class_correct[i]/(class_total[i]+1e-12))       
    accuracy.append(100.0*sum(class_correct)/sum(class_total))
    return F1,accuracy,sov_results

def save_excel(eval_accuracy,FileName,SOV_Res):
    import xlwt
    workbook = xlwt.Workbook(encoding='utf-8') 
    sheet1 = workbook.add_sheet("result")    
    Header_8=['Q_L',"Q_B","Q_E","Q_G","Q_I","Q_H","Q_S","Q_T","Q8","F₁-Score"]
    Header_9=['Q_L',"Q_B","Q_E","Q_G","Q_I","Q_H","Q_S","Q_T","Q_P","Q9","F₁-Score"]
    Header_3=['Q_C',"Q_E","Q_H","Q3","F₁-Score"]
    row=0
    if len(eval_accuracy)==len(Header_8):
        for i in range(len(Header_8)):
            sheet1.write(row,i,Header_8[i]) 
    elif  len(eval_accuracy)==len(Header_9):
        for i in range(len(Header_9)):
            sheet1.write(row,i,Header_9[i])         
    else:
        for i in range(len(Header_3)):
            sheet1.write(row,i,Header_3[i])         
    row+=1
    for i in range(len(eval_accuracy)):
        sheet1.write(row,i,round(eval_accuracy[i],2))
    sheet2 = workbook.add_sheet("SOV")
    split_res=SOV_Res.split()
    Header2=[]
    eval_accuracy2=[]
    for i in range(len(split_res)):
        if i %2==0:
            Header2.append(split_res[i][:-1])
        else:
            eval_accuracy2.append(float(split_res[i]))    
    row=0
    for i in range(len(Header2)):
        sheet2.write(row,i,Header2[i])    
    row+=1
    for i in range(len(eval_accuracy2)):
        sheet2.write(row,i,eval_accuracy2[i])     
    workbook.save(r'%s.xls'%(FileName)) 
