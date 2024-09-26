# Fast-AMPs-Discovery-Projects
![](GraphicalAbstract.png)
### Brief Introduction of Developers

#### Developer Introduction

**Sizhe Chen**, PhD Student (Medical Sciences) at Chinese University of Hong Kong (1155202847@link.cuhk.edu.hk). Supervisor: **Professor CHAN Ka Leung Francis** and **Research Assistant Professor Qi Su**. Welcome to contact **Sizhe Chen** via the aforementioned email if you have any questions or suggestions.

This work is supervised by **Professor Tian Liu**, **Professor Cangzhi Jia**, and **Research Associate Fellow Qiuyu Gong**. The research work is primarily finished by **Sizhe Chen** (PhD student), **Huitang Qi** (PhD student), and **Xingzhuo Zhu** (MPhil student) with equal contributions.

#### Acknowledgement
We highly appreciate the academic suggestions and manuscript proofreading by **Professor CHAN Ka Leung Francis**, and **Research Assistant Professor Qi Su** at the Department of Medicine and Therapeutics, Faculty of Medicine, Chinese University of Hong Kong. And we also appreciate the linguistic revision by native citizen **Mrs. Jean O’Donnell** from New York University, U.S.A.

### AI Tool Introduction and Guideline Review
This is a new deep-learning pipeline for AMP predictions. On the independent test dataset, it showed robust prediction performance and potentially avoided high false-positive conditions. Regarding the overall time costs, it only needs 4.27±0.05 hours for training and predictions on a normal laptop. Applying it to the gut microbiomes of cockroach _Blattella germanica_ successfully identified a series of new AMPs with low cytotoxicity and antimicrobial activities in vitro and in vivo.

The preprint version of our research paper is now available at BioRexiv (The discovery of antimicrobial peptides from the gut microbiome of cockroach Blattella germanica using deep learning pipeline. Sizhe Chen, Huitang Qi, Xingzhuo Zhu, Tianxiang Liu, Yingda Teng, Qiuyu Gong, Cangzhi Jia, Tian Liu. bioRxiv 2024.02.12.580024; doi: https://doi.org/10.1101/2024.02.12.580024). The formal version of this research is currently under review and will be published soon.

By opening and implementing the document Training1.py, you can directly utilize AMPidentifer to train and predict potential AMPs, the details of the created model are written in utils.py.

This repository contains models and data for predicting AMPs described in our paper.

## Requirements
- python 3.9.7 or higher
- keras==2.10.0
- pandas==1.5.2
- matplotlib==3.0.3
- propy3 (tutorial: https://propy3.readthedocs.io/en/latest/UserGuide.html)
- numpy==1.23.5
- sklearn=1.2.0
## Implementation details:

1. The training sequences were deposited in the document TrainingAMP.csv
2. The validation sequences were deposited in the document Validation.csv
3. The test sequences were deposited in the document Non-AMPsfilter.csv

This algorithm demands one-hot code matrix (sequential information，50×20) and physical/chemical descriptors matrix (91×17) as input.
The one-hot code can be calculated by the three .csv documents aforementioned.

For example:\
  ```train_file_name = 'TrainingAMP.csv'  # Training dataset```\
  ```win1 = 50```\
  ```X1, T, rawseq, length = getMatrixLabelh(train_file_name, win1)```

The physical/chemical descriptors matrix can also be calculated by three .csv documents aforementioned,

For example, you can calculate the physical/chemical descriptors matrix by code:\
  ```Matr=getMatrixLabelFingerprint(train_file_name, win1)```
\
And the pre-calculated physical/chemical descriptors matrixes (91×17) have been deposited in 3 .npy documents. You can directly load them by codes:\
\
```X2 = np.load(file="Training_vector.npy")# Descriptor of Training dataset```\
```X2tt = np.load(file="Test_vector.npy")# Descriptor of Test dataset```\
```X2_val = np.load(file="5810_vector.npy")# Descriptor of Validation dataset```\
\
Due to the size limitations of physiochemical descriptors of all sequences, the .npy documents containing these datasets were not submitted to Git Hub. For convenience, you can calculate it by the codes provided in Training1.py. Or you can contact Sizhe Chen for these documents. 

## AMPidentifer pipeline
It is very easy to train this model. You can open Training1.py in Spyder or Pytorch and just run it. More details or explanations can be found in the annotations of the Training1.py document. 

## Model Prediction Results
The predictions will change slightly for your conditions. If you want to maintain the results of our model reported in our paper, you can load the pre-trained model (.h5 document) deposited in AMPfinder.rar. Other .h5 files of each model strategy generated from iterative training and comparisons can be found in the “Model” directory.

## The Core Part of the Model
![](1.png)

Here we designed a new DL pipeline to establish AMP identification tools, including two N-layer computational blocks and one delicately embedded new self-attention module proposed in this work.

## The Mechanisms of the Attention Part
![](FigureS1.png)

The Attention Part designed in this work improved the overall performance of the AMP prediction task. We assumed that the self-attention module may enhance the recognition of critical features and prevent information loss during training.

## Utilizations Procedures
1. You can implement AMPidentifier on your own laptop by opening Training1.py in Spyder or Pytorch.
2. Import all necessary functions and packages. (This part has been included in the Training1.py)
3. Calculate descriptors for each given sequence provided in csv document. (The exact format is the same as the csv document provided in this github link)
4. Model training by using the training data and the validation dataset, and assess performances by using and the test dataset.
5. Make predictions and obtain the probability of being AMP or non-AMP for a given sequence.

Please feel free to contact us if you have any suggestions. Thank you for your attention.
