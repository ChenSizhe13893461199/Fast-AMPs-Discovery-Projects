# Fast-AMPs-Discovery-Projects

This is a lightweight Deep Learning pipeline for AMPs predictions.
By open and implement the document Training1.py, you can directly utilize it to train and predict potential AMPs, the details of the created model are written in utils.py.

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

1. The training sequences were deposited in document TrainingAMP.csv
2. The validation sequences were deposited in document Validation.csv
3. The test sequences were deposited in document Non-AMPsfilter.csv

This algorithm demand one-hot code matrix (sequential information，50×20) and physical/chemical descriptors matrix (91×17) as input.
The one-hot code can be calculated by the three .csv documents aforementioned,\

For example:\
  ```train_file_name = 'TrainingAMP.csv'  # Training dataset```\
  ```win1 = 50```\
  ```X1, T, rawseq, length = getMatrixLabelh(train_file_name, win1)```

The physical/chemical descriptors matrix can also be calculated by three .csv documents aforementioned,\

For example:\
```Matr=getMatrixLabelFingerprint(train_file_name, win1)```/
\
And the pre-calculated physical/chemical descriptors matrix (91×17) have been deposited in 3 .npy documents. You can directly load them by codes:\
\
```X2 = np.load(file="Training_vector.npy")# Descriptor of Training dataset```\
```X2tt = np.load(file="Test_vector.npy")# Descriptor of Test dataset```\
```X2_val = np.load(file="5810_vector.npy")# Descriptor of Validation dataset```\
\
Due to the size limitations of physiochemical descriptors of all sequences, the .npy document containing these dataset were not submitted to github. For convenience, you can calculate it by codes provided in Training1.py. Or you can email chen2422679942@163.com for these documents. \
\
You can calculate the physical/chemical descriptors matrix by code:\
  ```Matr=getMatrixLabelFingerprint(train_file_name, win1)```

## AMPfinder pipeline
It is very easy to train this model. You can open Training1.py in Spyder or Pytorch, and just run it. And more details or explanations can be found in annotations of Training1.py document. 

The predictions will change slightly for your conditions. If you want to maintain the consistence of our current model. You can load the pre-trained model (.h5 document) deposited in AMPfinder.rar.
