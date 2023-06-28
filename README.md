# Learning to Understand the Vague Graph for Stock Prediction with Momentum Spillovers
This is the source code of our manuscript, 
named Learning to Understand the Vague Graph forStock Prediction with Momentum Spillovers

## Model architechture
![image](./Pictures/model_architecture.jpeg) 

## Overview
* `layers.py` contains three modules of the VGNN model: Vague Node Representation (`TensorFusionLayer`, `MatrixFusionLayer`),
Vague Node Links (`ImplicitLayer`, `ExplicitLayer`), 
  Attribute-sensitive Message Passing (`AttributeGate`);
  
* `models.py` contains the VGNN model;

* `utils.py` contains data loading (`load_data`) 
  and evaluation metrics (`R2_score_calculate`, `IC_ICIR_score_calculate`);

* `train_evaluation.py` puts all of the above together and is uesd to execute
a full training run on our dataset.

* `BackTesting.ipynb` contains the investment simulation experiments.

* `Simulation.ipynb` contains a simple test based on simulated data to vividly illustrate the effectiveness of the proposed tensor-based
fusion for solving dynamic interactions.
## Environment
* Python==3.8.5 
  
* PyTorch==1.6.0 
  
* Numpy==1.19.2 
  
* Pandas==1.1.3

## Run
```
$ python train_evaluation.py 
```

## Data
### Transcational Data
Due to space limitations, we can only provide a small portion of the data here(`./data`), considering that the raw data size is around 3.8GB. You can access the complete raw data by downloading it from either of the following links: https://dachxiu.chicagobooth.edu<sup><a href="#ref1">[1]</a></sup> or http://quant.zxlearn.cn.

### Company Relations
Firm relations can be found at `./data/relations`, including industry information and headquarter location.

<br/>
<br/>



[1] <span name = "ref1">S. Gu, B. Kelly, and D. Xiu, “Empirical asset pricing via machine learning,” Review of Financial Studies, vol. 33, no. 5, pp. 2223–2273, 2020.</span>
 

## Contact
jiwenhuangfic@gmail.com

