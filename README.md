# Fast-AMPs-Discovery-Projects
Abstract:

Antimicrobial peptides (AMPs) are candidates for use against antibiotic-resistant bacteria. However, the existing AMPs cannot meet the requirements due to their high mammalian toxicities and low efficiencies in biological environments. The gut microbiome of insects like cockroach Blattella germanica is a valuable resource for AMPs and more efficient tool is desirable to overcome the challenges of giant data scale and high false-positive condition. Here, AMPfinder, a lightweight AI pipeline, was proposed for AMPs identification by using densely connected convolutional neural networks and 3D self-attention module.  It overcame the challenges of high false-positive problems and outperformed all previously published tools. The AMPfinder was applied for excavating AMPs from the gut microbiome of cockroach B. germanica , with 79 sequences identified as possible AMPs. Compared to traditional AMPs, the analysis indicated the potential balances between low toxicities and antimicrobial activities of these 79 sequences. By AI scores and other selection criteria, two sequences were chemically synthesized, and showed promising antimicrobial activities and low mammalian toxicities. This work not only provides a new tool for mining AMPs, but also shows its application prospects by discovering novel AMPs from natural resources.

Relevant academic paper has been submitted to Nucleic Acids Research

This is a new lightweight DL pipeline by using the currently largest scale of physicochemical descriptors. Unlike multiple model hybrids, our framework was lightweight and we mathematically designed a new self-attention module for the first time, potentially avoiding risks of overfitting.
By open and implement the document Training1.py, you can directly utilize it to train and predict potential AMPs, the details are written in utils.py.

Due to the size limitations of physiochemical descriptors of all sequences, the .npy document containing these dataset were not submitted to github. For convenience, you can calculate it by codes provided in Training1.py. This process usually needs 10 hours if you implement it on your normal laptop. Or you can email chen2422679942@163.com for these document. 

The pre-trained model can be found in AMPfinder.rar. 

This is a pure python algorithm based on python 3.9.
For utilizing this algorithm, you need to download and install all mentioned packages in Training1.py via Anaconda

If you want to implement it on your local equipment, please be noted that our pipeline was implemented by python 3.9.7 with tensorflow packages.

Implementation details
