import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class LDAM_SoftmaxLoss(torch.nn.Module):
    def __init__(self, num_classes, embed_dim, tau,cls_num_list, max_m=1.0):
        super(LDAM_SoftmaxLoss, self).__init__()        
        self.centroids = nn.Parameter(torch.FloatTensor(num_classes, embed_dim))
        self.centroids.data.normal_(0, 1).renorm_(2, 1, 1e-5).mul_(1e5)        
        self.tau = tau
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list)) 
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list       
    def getNormalizedCentrioids(self):
        P=F.normalize(self.centroids-self.centroids.mean(dim=-1,keepdim=True),2,dim=-1)
        return P  
        
    def forward(self,x, targets):
        targets=targets.flatten() 
        masks=(targets!=-100)
        X=torch.masked_select(x,masks.unsqueeze(1)).view(-1,x.shape[1])
        T=torch.masked_select(targets,masks)
        
        P=self.getNormalizedCentrioids()
        S=self.tau*torch.matmul(X,P.t())        

        index = torch.zeros_like(S, dtype=torch.bool)
        index.scatter_(1, T.view(-1, 1), 1)
        batch_m = self.m_list[T].view(-1,1)
        s_m = S- batch_m
        S_ = torch.where(index, s_m, S)   
    
        T=F.one_hot(T,num_classes=self.centroids.shape[0])
        loss = torch.sum(- T * F.log_softmax(S_, -1), -1)
        loss = loss.mean()
        return loss  




class SoftmaxLoss(torch.nn.Module):
    def __init__(self, num_classes, embed_dim, tau):
        super(SoftmaxLoss, self).__init__()        
        self.centroids = nn.Parameter(torch.FloatTensor(num_classes, embed_dim))
        self.centroids.data.normal_(0, 1).renorm_(2, 1, 1e-5).mul_(1e5)        
        self.tau = tau/9.0 
       
    def getNormalizedCentrioids(self):
        P=F.normalize(self.centroids-self.centroids.mean(dim=-1,keepdim=True),2,dim=-1)
        return P  
        
    def forward(self,x, targets):
        targets=targets.flatten() 
        masks=(targets!=-100)
        X=torch.masked_select(x,masks.unsqueeze(1)).view(-1,x.shape[1])
        T=torch.masked_select(targets,masks)
        
        P=self.getNormalizedCentrioids()
        S=torch.matmul(3.0*X,3.0*P.t())
        T=F.one_hot(T,num_classes=self.centroids.shape[0])
        loss = torch.sum(- T * F.log_softmax(self.tau*S, -1), -1)
        loss = loss.mean()
        return loss    


def Normalize(x):    
    out=F.normalize(x-x.mean(dim=-1,keepdim=True),2,dim=-1)
    return out

    
if __name__ == '__main__':  
    data=torch.randn(8,32)
    label=torch.randint(high=4,size=(8,))
    mask=torch.randn(8).ge(0.5)
    label[0]=-100
    label[4]=-100
    data=Normalize(data)
    
    data=data.cuda()
    label=label.cuda()

    
    criterion=SoftmaxLoss(4,32,18).cuda()
    loss=criterion(data,label)
    print(loss)    
    cls_num_list=[1134520,65450,1287266,235506,33687,2023843,497267,632229,103119]
    criterion2=LDAM_SoftmaxLoss(4,32,18,cls_num_list).cuda()
    loss2=criterion2(data,label)

    print(loss2)



