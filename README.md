# Fast-AMPs-Discovery-Projects

This is a new lightweight DL pipeline by using the currently largest scale of physicochemical descriptors. Unlike multiple model hybrids, our framework was lightweight and we mathematically designed a new self-attention module for the first time, potentially avoiding risks of overfitting.
By open and implement the document Training1.py, you can directly utilize it to train and predict potential AMPs, the details are written in utils.py.

Due to the size limitations of physiochemical descriptors of all sequences, the .npy document containing these dataset were not submitted to github. For convenience, you can calculate it by codes provided in Training1.py. This process usually needs 10 hours if you implement it on your normal laptop. Or you can email chen2422679942@163.com for these document. 

The pre-trained model can be found in AMPfinder.rar. 

Implementation details:

1. The training sequences were deposited in document TrainingAMP.csv
2. The validation sequences were deposited in document Validation.csv
3. The test sequences were deposited in document Non-AMPsfilter.csv

This algorithm demand one-hot code matrix (sequential information) and physical/chemical descriptors matrix as input.
The one-hot code can be calculated by 3 .csv documents aforementioned by code,

For example:\
  ```train_file_name = 'TrainingAMP.csv'  # Training dataset```\
  ```win1 = 50```\
  ```X1, T, rawseq, length = getMatrixLabelh(train_file_name, win1)```

The physical/chemical descriptors matrix can also be calculated by 3 .csv documents aforementioned by code,
For example:\
  ```Matr=getMatrixLabelFingerprint(train_file_name, win1)```

### AMPfinder codes

