# Fast-AMPs-Discovery-Projects
This is a new lightweight DL pipeline by using the currently largest scale of physicochemical descriptors. Unlike multiple model hybrids, our framework was lightweight and we mathematically designed a new self-attention module for the first time, potentially avoiding risks of overfitting.
By open and implement the document Training1.py, you can directly utilize it to train and predict potential AMPs, the details are written in utils.py.

Due to the size limitations of physiochemical descriptors of all sequences, the .npy document containing these dataset were not submitted to github. For convenience, you can calculate it by codes provided in Training1.py. This process usually needs 10 hours if you implement it on your normal laptop. The pre-trained model can be found in AMPfinder.rar

This is a pure python algorithm based on python 3.9.
For utilizing this algorithm, you need to download and install all mentioned packages in Training1.py via Anaconda

If you want to implement it on your local equipment, please be noted that our pipeline was implemented by python 3.9.7 with tensorflow packages.

#First, create your own conda environment:

conda create -n tf2 python=3.9.7

#Activate your environment

conda activate tf2

#Install tensorflow

pip install tensorflow==2.0.0

#Activate your environment

conda activate tf2

#Install corresponding keras

conda install mingw libpython
pip install theano
pip install keras==2.3.1

Please be noted that the version of tensorflow and keras need to be matched, and you can find more information about that at 
(https://cn.bing.com/images/search?view=detailV2&ccid=UrgoE5i0&id=0FAB48A265C885F25CD0DB203CE1F60EE1858E5D&thid=OIP.UrgoE5i0OZzU0ZH7sP0ClAHaEW&mediaurl=https%3a%2f%2fimg-blog.csdnimg.cn%2f2021041919594759.png%3fx-oss-process%3dimage%2fwatermark%2ctype_ZmFuZ3poZW5naGVpdGk%2cshadow_10%2ctext_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTMyODkyNTQ%3d%2csize_16%2ccolor_FFFFFF%2ct_70&exph=709&expw=1207&q=keras%e4%b8%8etensorflow%e7%89%88%e6%9c%ac%e5%af%b9%e5%ba%94%e5%ae%98%e7%bd%91&simid=607998796419704018&FORM=IRPRST&ck=F5BA70A6D0DEC0AE9807A91CF2A9FCC9&selectedIndex=1&ajaxhist=0&ajaxserp=0)

#In the next step, you can install the corresponding descriptors packages.

pip install propy3

More information of this descriptor package can be found at https://propy3.readthedocs.io/en/latest/UserGuide.html

There may be some differences between codes provided by descriptor package and the utilization of codes in Training1.py. Please be careful. This may be caused by delaying update on https://propy3.readthedocs.io/en/latest/UserGuide.html. You can adjust these codes when necessary.

For biopython packages analyzing sequences physiochemical features,

#Installation From Source

pip install biopython

However, if you need to compile Biopython yourself, the following are required at compile time:

xcode-select --install

Then either download and decompress our source code, or fetch it using git. Now change directory to the Biopython source code folder and run:

pip install -e .
python setup.py test
sudo python setup.py install

#For Testing, you can find more information about this part at https://github.com/biopython/biopython

All these packages mentioned above besides biopython are necessary during the model training and prediction.
The biopython can be very useful in sequences physiochemical features analysis towards the novel AMPs identified by our pipeline.

