# Fast-AMPs-Discovery-Projects
This is a new lightweight DL pipeline by using the currently largest scale of physicochemical descriptors. Unlike multiple model hybrids, our framework was lightweight and we mathematically designed a new self-attention module for the first time, potentially avoiding risks of overfitting.
By open and implement the document Training1.py, you can directly utilize it to train and predict potential AMPs, the details are written in utils.py.

Due to the size limitations of physiochemical descriptors of all sequences, the .npy document containing these dataset were not submitted to github. For convenience, you can calculate it by codes provided in Training1.py. This process usually needs 10 hours if you implement it on your normal laptop.
This is a pure python algorithm based on python 3.9.
For utilizing this algorithm, you need to download and install all mentioned packages in Training1.py via Anaconda

If you want to implement it on your local equipment, please be noted that our pipeline was implemented by python 3.9.7 with tensorflow packages.

First, create your own conda environment:
conda create -n tf2 python=3.9.7

Activate your environment
conda activate tf2

Install tensorflow
pip install tensorflow==2.0.0







