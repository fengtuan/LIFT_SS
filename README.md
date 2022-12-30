
# Lightweight Fine-tuning a Pretrained Protein Language Model for Protein Secondary Structure Prediction
=

**Abstract**
<br>

Pretrained large-scale protein language models, such as ESM-1b and ProtTrans, are becoming the fundamental infrastructure for various protein-related biological modeling tasks. Existing works use mainly pretrained protein language models in feature extraction. However, the knowledge contained in the embedding features directly extracted from a pretrained model is task-agnostic. To obtain task-specific feature representations, a reasonable approach is to fine-tune a pretrained model based on labeled datasets from downstream tasks. To this end, we investigate the fine-tuning of a given pretrained protein language model for protein secondary structure prediction tasks. Specifically, we propose a novel end-to-end protein secondary structure prediction framework involving the lightweight fine-tuning of a pretrained model. The framework first introduces a few new parameters for each transformer block in the pretrained model, then updates only the newly introduced parameters, and then keeps the original pretrained parameters fixed during training. Extensive experiments on seven test sets, namely, CASP12, CASP13, CASP14, CB433, CB634, TEST2016, and TEST2018, show that the proposed framework outperforms existing predictors and achieves new state-of-the-art prediction performance. Furthermore, we also experimentally demonstrate that lightweight fine-tuning significantly outperforms full model fine-tuning and feature extraction in enabling models to predict secondary structures. Further analysis indicates that only a few top transformer blocks need to introduce new parameters, while skipping many lower transformer blocks has little impact on the prediction accuracy of secondary structures. Our code and datasets are publicly available at https://github.com/fengtuan/LIFT_SS.

### 1. Requirement
> * Python >=3.8  
> * Pytorch >=1.10
> * adapter-transformers >=3.1.0
The codes are tested under Ubuntu.

### 2. Citation
If you use our code in your study, please cite as:
[1] Wei Yang,Chun Liu, Zheng Li and Lei Zhang, Lightweight Fine-tuning a Pretrained Protein Language Model for Protein Secondary Structure Prediction

### 3. Contact
For questions and comments, feel free to contact :yang0sun@gmail.com. 


