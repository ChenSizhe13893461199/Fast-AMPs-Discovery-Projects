# Fast-AMPs-Discovery-Projects
This is a new lightweight DL pipeline by using the currently largest scale of physicochemical descriptors. Unlike multiple model hybrids, our framework was lightweight and we mathematically designed a new self-attention module for the first time, potentially avoiding risks of overfitting.
By open and implement the document Training1.py, you can directly utilize it to train and predict potential AMPs, the details are written in utils.py.

Due to the size limitations of physiochemical descriptors of all sequences, the .npy document containing these dataset were not submitted to github. For convenience, you can calculate it by codes provided in Training1.py. This process usually needs 10 hours if you implement it on your normal laptop.
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

Please be noted that the version of tensorflow and keras need to be matched, and you can find more information about that at https://cn.bing.com/images/search?view=detailV2&ccid=kiyLnG6o&id=4D6497BAA97A6711FEBBAB468953B69107D91D65&thid=OIP.kiyLnG6ofEmD2PhjdLqroAHaEP&mediaurl=https%3a%2f%2fimg-blog.csdnimg.cn%2fimg_convert%2fd264f64f857f13e5d8c6f2022e668728.png&exph=413&expw=720&q=keras%e4%b8%8etensorflow%e7%89%88%e6%9c%ac%e5%af%b9%e5%ba%94%e5%ae%98%e7%bd%91&simid=608007347696256376&FORM=IRPRST&ck=62370D51BB9A5F00BAB50F8716AA0793&selectedIndex=0&ajaxhist=0&ajaxserp=0

In the next step, you can install the corresponding descriptors packages.










