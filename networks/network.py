#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 16:47:39 2021

@author: weiyang
"""
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5EncoderModel
import copy


class T5LayerNorm_(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1,hidden_size,1))
        self.variance_epsilon = eps

    def forward(self, hidden_states):

        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(torch.float32).pow(2).mean(1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class BasicConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size-1)//2)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x, masks=None):
        x = self.bn(self.conv(x))
        x = F.hardswish(x)
        return x
    
class BasicConv1d_LN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(BasicConv1d_LN, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size-1)//2, bias=True)
        self.ln = T5LayerNorm_(out_channels)
        self.kernel_size = kernel_size

    def forward(self,x,masks):
        out = x
        if self.kernel_size >1:        
           out = torch.where(masks,out,torch.zeros(size=(1,),device=out.device)) 
        out = self.conv(out)
        out = F.hardswish(out)
        if out.shape==x.shape:        
           out = self.ln(out)+x
        else:
           out = self.ln(out)
        return out  


class build_block(nn.Module):
    def __init__(self, BasicConv,in_channels, out_channels,dropout=0.2):
        super(build_block, self).__init__()       
        self.conv =BasicConv(in_channels, out_channels, kernel_size=1)
        self.branch_dropout = nn.Dropout(dropout)
        self.branch_conv =BasicConv(out_channels//2, out_channels//2, kernel_size=3)        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x,masks=None):        
        out = self.conv(x,masks)
        branch1,branch2 = out.chunk(2, dim=1)
        branch2 = self.branch_conv(self.branch_dropout(branch2),masks)        
        out=torch.cat([branch1, branch2], 1)
        out=self.dropout(out)
        return out


class inception_unit(nn.Module):
    def __init__(self, BasicConv, in_channels,out_channels,dropout=0.2):
        super(inception_unit, self).__init__()       
        self.input_layers = nn.ModuleList(
            [build_block(BasicConv, in_channels,out_channels//2,dropout) for i in range(2)])
        self.intermediate_layer =build_block(BasicConv, out_channels//2,out_channels//2,dropout)
        

    def forward(self, x,masks=None):
        branch1 = self.input_layers[0](x,masks)
        branch2 = self.intermediate_layer(self.input_layers[1](x,masks),masks)       
        output = torch.cat([branch1, branch2], 1)
        return output

# class HeadNet(nn.Module):
#     def __init__(self,feature_dim,num_channels,depth=2,num_classes=9,dropout=0.2):
#         super(HeadNet, self).__init__()
#         self.layer1 = inception_unit(BasicConv1d,feature_dim,num_channels,dropout)
#         self.layer2 = nn.ModuleList(
#                 [inception_unit(BasicConv1d,num_channels,num_channels,dropout) for i in range(depth-1)])
#         self.conv = nn.Conv1d(num_channels, 96, 9, padding=4)
#         self.fc = nn.Linear(96, 32, bias=False)          
#         self.classifier=nn.Linear(32,num_classes, bias=False)
#         self.depth=depth
#         self.dropout=nn.Dropout(0.2) 
#     def forward(self, x,masks=None):
#         out = self.dropout(x)           
#         out=self.layer1(out)
#         for i in range(self.depth-1):
#             out = self.layer2[i](out)          
#         out=F.relu(self.conv(out))
#         out=out.transpose(1,2).contiguous()
#         out=F.relu(self.fc(out))
#         out=self.classifier(out)
#         return out

class HeadNet_cls(nn.Module):
    def __init__(self,feature_dim,num_channels,depth=2,num_classes=9,dropout=0.2):
        super(HeadNet_cls, self).__init__()
        self.layer1 = inception_unit(BasicConv1d_LN,feature_dim,num_channels,dropout)
        self.layer2 = nn.ModuleList(
                [inception_unit(BasicConv1d_LN,num_channels,num_channels,dropout) for i in range(depth-1)])
        self.conv = nn.Conv1d(num_channels, 96, 9, padding=4)
        self.fc = nn.Linear(96, 32, bias=False)          
        self.classifier=nn.Linear(32,num_classes, bias=False)
        self.depth=depth
        self.dropout=nn.Dropout(0.2) 
    def forward(self, x,masks=None):
        masks_=copy.deepcopy(masks)
        cols=masks_.sum(dim=-1)-1
        rows=[i for i in range(masks_.shape[0])] 
        masks_[rows,cols]=0
        masks_=masks_.to(torch.bool).unsqueeze(dim=1) 
        out = self.dropout(x)   
        out=self.layer1(out,masks_)
        for i in range(self.depth-1):
            out = self.layer2[i](out,masks_)
        out = torch.where(masks_,out,torch.zeros(size=(1,),device=out.device))
        out=F.relu(self.conv(out))
        out=out.transpose(1,2).contiguous()
        out=F.relu(self.fc(out))
        out=self.classifier(out)
        return out    
    

class LIFT_SS_cls(nn.Module):
    def __init__(self,adapter_config,adapter_name,num_channels,depth=2,num_class=9,dropout=0.2):
        super().__init__()
        model_name = "ProtTrans/Rostlab/prot_t5_xl_uniref50"
        T5Encoder = T5EncoderModel.from_pretrained(model_name)
        T5Encoder.add_adapter(adapter_name, config=adapter_config)
        T5Encoder.train_adapter(adapter_name)
        self.T5Encoder=T5Encoder
        self.adapter_name=adapter_name
        feature_dim=T5Encoder.encoder.block[-1].layer[1].DenseReluDense.wo.out_features
        self.head=HeadNet_cls(feature_dim,num_channels,depth,num_class,dropout)

    def forward(self,inputs=None,masks=None):       
        hidden_out=self.T5Encoder(input_ids=inputs,attention_mask=masks)[0]
        out=hidden_out.transpose(1,2).contiguous()
        out=self.head(out,masks)        
        return out
    def save(self,model_path,criterion=None):
        if criterion is None:
            checkpoint = { 'head': self.head.state_dict()}
            torch.save(checkpoint,'%s_head.pth'%(model_path))
        else:
            checkpoint = { 'head': self.head.state_dict(),'criterion': criterion.state_dict()}
            torch.save(checkpoint,'%s_head.pth'%(model_path))            
        self.T5Encoder.save_adapter('%s_adapter.pt'%(model_path),self.adapter_name) 
    def load(self,model_path,criterion=None):
        if criterion is None:        
            checkpoint=torch.load('%s_head.pth'%(model_path))        
            self.head.load_state_dict(checkpoint['head'])
        else:
            checkpoint=torch.load('%s_head.pth'%(model_path))        
            self.head.load_state_dict(checkpoint['head'])
            criterion.load_state_dict(checkpoint['criterion'])            
        self.T5Encoder.load_adapter('%s_adapter.pt'%(model_path)) 


def Normalize(x):    
    out=F.normalize(x-x.mean(dim=-1,keepdim=True),2,dim=-1)
    return out


class DML_SS(nn.Module):
    def __init__(self,feature_dim,num_channels,depth=2,proj_dim=32,dropout=0.2):
        super(DML_SS, self).__init__()
        self.layer1 = inception_unit(BasicConv1d,feature_dim,num_channels,dropout)
        self.layer2 = nn.ModuleList(
                [inception_unit(BasicConv1d,num_channels,num_channels,dropout) for i in range(depth-1)])        
        self.conv = nn.Conv1d(num_channels, 96, 9, padding=4)       
        self.fc = nn.Linear(96, proj_dim, bias=False)  
        self.depth=depth
        self.dropout=nn.Dropout(0.2)        
    def forward(self, x,masks=None): 
        out = self.dropout(x)        
        out=self.layer1(out)
        for i in range(self.depth-1):
            out = self.layer2[i](out)         
        out=F.relu(self.conv(out))
        out=out.transpose(1,2).contiguous()            
        out = self.fc(out)        
        out=Normalize(out)
        return out
    
    
class HeadNet_DML(nn.Module):
    def __init__(self,feature_dim,num_channels,depth=2,proj_dim=32,dropout=0.2):
        super(HeadNet_DML, self).__init__()
        self.layer1 = inception_unit(BasicConv1d_LN,feature_dim,num_channels,dropout)
        self.layer2 = nn.ModuleList(
                [inception_unit(BasicConv1d_LN,num_channels,num_channels,dropout) for i in range(depth-1)])        
        self.conv = nn.Conv1d(num_channels, 96, 9, padding=4, bias=True)
        self.fc = nn.Linear(96, proj_dim, bias=False)  
        self.depth=depth
        self.dropout=nn.Dropout(0.2)
    def forward(self, x,masks=None):
        masks_=copy.deepcopy(masks)
        cols=masks_.sum(dim=-1)-1
        rows=[i for i in range(masks_.shape[0])] 
        masks_[rows,cols]=0
        masks_=masks_.to(torch.bool).unsqueeze(dim=1) 
        out = self.dropout(x)
        out=self.layer1(out,masks_)
        for i in range(self.depth-1):
            out = self.layer2[i](out,masks_) 
        out = torch.where(masks_,out,torch.zeros(size=(1,),device=out.device))
        out=F.relu(self.conv(out))
        out=out.transpose(1,2).contiguous()            
        out = self.fc(out)        
        out=Normalize(out)
        return out 

    
class LIFT_SS_DML(nn.Module):
    def __init__(self,adapter_config,adapter_name,num_channels,depth=2,proj_dim=32,dropout=0.2):
        super().__init__()
        model_name = "ProtTrans/Rostlab/prot_t5_xl_uniref50"
        T5Encoder = T5EncoderModel.from_pretrained(model_name)       
        T5Encoder.add_adapter(adapter_name, config=adapter_config)
        T5Encoder.train_adapter(adapter_name)
        self.T5Encoder=T5Encoder
        self.adapter_name=adapter_name
        feature_dim=T5Encoder.encoder.block[-1].layer[1].DenseReluDense.wo.out_features
        self.head=HeadNet_DML(feature_dim,num_channels,depth,proj_dim,dropout)

    def forward(self,inputs=None,masks=None):       
        hidden_out=self.T5Encoder(input_ids=inputs,attention_mask=masks)[0]
        out=hidden_out.transpose(1,2).contiguous()
        out=self.head(out,masks)        
        return out
    def save(self,model_path,criterion=None):
        if criterion is None:
            checkpoint = { 'head': self.head.state_dict()}
            torch.save(checkpoint,'%s_head.pth'%(model_path))
        else:
            checkpoint = { 'head': self.head.state_dict(),'criterion': criterion.state_dict()}
            torch.save(checkpoint,'%s_head.pth'%(model_path)) 
        self.T5Encoder.save_adapter('%s_adapter.pt'%(model_path),self.adapter_name) 
    def load(self,model_path,criterion=None):
        if criterion is None:        
            checkpoint=torch.load('%s_head.pth'%(model_path))        
            self.head.load_state_dict(checkpoint['head'])
        else:
            checkpoint=torch.load('%s_head.pth'%(model_path))        
            self.head.load_state_dict(checkpoint['head'])
            criterion.load_state_dict(checkpoint['criterion'])            
        self.T5Encoder.load_adapter('%s_adapter.pt'%(model_path)) 



class DML_full_finetuning(nn.Module):
    def __init__(self,num_channels,depth=2,proj_dim=32,dropout=0.2,skiped_blocks=0):
        super().__init__()
        model_name = "ProtTrans/Rostlab/prot_t5_xl_uniref50"
        T5Encoder = T5EncoderModel.from_pretrained(model_name) 
        # T5Encoder.gradient_checkpointing_enable()
        self.T5Encoder=T5Encoder
        feature_dim=T5Encoder.encoder.block[-1].layer[1].DenseReluDense.wo.out_features
        self.head=HeadNet_DML(feature_dim,num_channels,depth,proj_dim,dropout)
        for i,layer in enumerate(T5Encoder.encoder.block):
            if i < skiped_blocks:
                for param in layer.parameters():
                    param.requires_grad=False
        if skiped_blocks>0:
            self.T5Encoder.shared.weight.requires_grad=False

    def forward(self,inputs=None,masks=None):       
        hidden_out=self.T5Encoder(input_ids=inputs,attention_mask=masks)[0]
        out=hidden_out.transpose(1,2).contiguous()
        out=self.head(out,masks)        
        return out
    
class cls_full_finetuning(nn.Module):
    def __init__(self,num_channels,depth=2,num_class=9,dropout=0.2,skiped_blocks=0):
        super().__init__()
        model_name = "ProtTrans/Rostlab/prot_t5_xl_uniref50"
        T5Encoder = T5EncoderModel.from_pretrained(model_name)
        # T5Encoder.gradient_checkpointing_enable()
        self.T5Encoder=T5Encoder
        feature_dim=T5Encoder.encoder.block[-1].layer[1].DenseReluDense.wo.out_features
        self.head=HeadNet_cls(feature_dim,num_channels,depth,num_class,dropout)
        for i,layer in enumerate(T5Encoder.encoder.block):
            if i < skiped_blocks:
                for param in layer.parameters():
                    param.requires_grad=False
        if skiped_blocks>0:
            self.T5Encoder.shared.weight.requires_grad=False        

    def forward(self,inputs=None,masks=None):       
        hidden_out=self.T5Encoder(input_ids=inputs,attention_mask=masks)[0]
        out=hidden_out.transpose(1,2).contiguous()
        out=self.head(out,masks)        
        return out



    
    
