#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 21:45:24 2021

@author: weiyang
"""
import copy
import torch
class ProteinNode:
    def __init__(self,ProteinID,ProteinLen,PrimarySeq=None,SecondarySeq=None):
        self.ProteinID=ProteinID
        self.ProteinLen=ProteinLen
        self.PrimarySeq=PrimarySeq        
        self.SecondarySeq=SecondarySeq
        self.Profile=None


def split_sequence(train_list,MaxLen=512):
    N=len(train_list)
    for idx in range(N):
        length=train_list[idx].ProteinLen
        if (length>MaxLen-1):            
            len2=length//2
            len1=length-len2
            PrimarySeq=copy.deepcopy(train_list[idx].PrimarySeq[:MaxLen])
            SecondarySeq=copy.deepcopy(train_list[idx].SecondarySeq[:MaxLen])
            SecondarySeq[len1:]=-100
            Node=ProteinNode(train_list[idx].ProteinID+"_",MaxLen-1,PrimarySeq,SecondarySeq)
            train_list.append(Node)
            train_list[idx].ProteinLen=MaxLen-1
            train_list[idx].PrimarySeq=copy.deepcopy(train_list[idx].PrimarySeq[-MaxLen:])
            train_list[idx].SecondarySeq=copy.deepcopy(train_list[idx].SecondarySeq[-MaxLen:])
            train_list[idx].SecondarySeq[:MaxLen-len2-1]=-100            
    

# {'<pad>': 0,  '</s>': 1,  '<unk>': 2,  '▁A': 3,  '▁L': 4,  '▁G': 5,  '▁V': 6,
#   '▁S': 7,  '▁R': 8,  '▁E': 9,  '▁D': 10,  '▁T': 11,  '▁I': 12,  '▁P': 13,
#   '▁K': 14,  '▁F': 15,  '▁Q': 16,  '▁N': 17,  '▁Y': 18,  '▁M': 19,  '▁H': 20, 
#   '▁W': 21,  '▁C': 22,  '▁X': 23,  '▁B': 24,  '▁O': 25,  '▁U': 26,  '▁Z': 27,
def Load_Data(FastaFile,isThreeClass):
    f=open(FastaFile,'r')
    SS=['L','B','E','G','I','H','S','T','P','X']
    SS9_Dict=dict(zip(SS,range(len(SS))))
    SS9_Dict['X']=-100
    SS3_Dict={'L': 0, 'B': 1, 'E': 1, 'G': 2, 'I': 2, 'H': 2, 'S': 0, 'T': 0, 'P':2,'X':-100}
    Standard_AAS=['A', 'L', 'G', 'V', 'S', 'R', 'E', 'D', 'T', 'I', 'P','K','F','Q', 'N', 'Y', 'M', 'H',  'W', 'C']
    AA_Dict={}
    Non_Standard_AAS=list('BJOUXZ')
    for i,AA in enumerate(Standard_AAS):
        AA_Dict.update({AA:i+3})
    for i,AA in enumerate(Non_Standard_AAS):
        AA_Dict.update({AA:23})    
    Data=[]    
    while True:       
        PDBID=f.readline().strip()
        if len(PDBID)==0:
            break
        PrimarySeq=f.readline().strip()
        ProteinLen=len(PrimarySeq)
        SecondarySeq=f.readline().strip()
        PrimarySeq=[AA_Dict[e] for e in PrimarySeq]
        # add special_token "eos_token":'</s>': 1
        PrimarySeq.append(1)        
        if isThreeClass:
            SecondarySeq=[SS3_Dict[e] for e in SecondarySeq]            
        else:
            SecondarySeq=[SS9_Dict[e] for e in SecondarySeq]
        SecondarySeq.append(-100)
        PrimarySeq=torch.tensor(PrimarySeq,dtype=torch.long)
        SecondarySeq=torch.tensor(SecondarySeq,dtype=torch.long)
        Node=ProteinNode(PDBID[1:],ProteinLen,PrimarySeq,SecondarySeq)
        Data.append(Node)    
    f.close()
    return Data

def load_train_valid(SequenceIdentity=25,isThreeClass=False):  
   train_list=Load_Data('data/PISCES_%d_train.txt'%(SequenceIdentity),isThreeClass)
   valid_list=Load_Data('data/PISCES_25_valid.txt',isThreeClass)
   valid_list.sort(key=lambda Node: Node.ProteinLen,reverse=False)
   return train_list,valid_list

def load_train_valid_for_CB634(isThreeClass=False):  
   train_list=Load_Data('data/CB634_train.txt',isThreeClass)
   valid_list=Load_Data('data/CB634_valid.txt',isThreeClass)
   valid_list.sort(key=lambda Node: Node.ProteinLen,reverse=False)
   return train_list,valid_list

def load_spot_1d_train_valid(isThreeClass=False):  
   train_list=Load_Data('data/SPOT_1D_Train.txt',isThreeClass) 
   valid_list=Load_Data('data/SPOT_1D_Valid.txt',isThreeClass)
   valid_list.sort(key=lambda Node: Node.ProteinLen,reverse=False)
   return train_list,valid_list

def load_test_data(dataset,isThreeClass): 
   if dataset not in ['CB634','CB433','CASP12','CASP13','CASP14','Test2016','Test2018']:
        print('Dataset:'+dataset+' does not exist.')
        return None   
   test_list=Load_Data('data/%s.txt'%(dataset),isThreeClass)
   return test_list

