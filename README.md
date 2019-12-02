# Palmprint-Recognition-in-Uncontrolled-and-Uncooperative-Environment
## Paper
Wojciech Michal Matkowski, Tingting Chai and Adams Wai Kin Kong. “Palmprint Recognition in Uncontrolled and Uncooperative Environment.” IEEE Transactions on Information Forensics and Security, October 2019, DOI: 10.1109/TIFS.2019.2945183.

Preprint can be found now on arXiv [here](https://arxiv.org/ftp/arxiv/papers/1911/1911.12514.pdf)
Paper (Early Access) can be found on IEEE Xplore [here](https://ieeexplore.ieee.org/document/8854829)

## Database
### How to acquire the dataset?
To acquire the NTU Palmprints from the Internet (NTU-PI-v1) database (which was used in the paper), download and fill in the "Data Release Agreement.pdf" file. Print the agreement and sign on page 2. Scan the signed copy and send back to matk0001@e.ntu.edu.sg, xpxu@ntu.edu.sg or adamskong@ntu.edu.sg with title "Application for NTU Palmprints from the Internet database (NTU-PI-v1)". A download link to the database will be send to you once after we receive the signed agreement file.


### Image examples
![alt text](https://github.com/matkowski-voy/Palmprint-Recognition-in-the-Wild/blob/master/Fig4a.png)\
Example images from NTU-PI-v1. Images in the same column are from the same palm. Images are resized to the same size. 
![alt text](https://github.com/matkowski-voy/Palmprint-Recognition-in-the-Wild/blob/master/Fig5.png)\
Examples of original images downloaded from the Internet. Hands are higlited in red boxes in the original images.



## Code
- download matconvnet [here](http://www.vlfeat.org/matconvnet/download/matconvnet-1.0-beta25.tar.gz) unpack and put into maltab folder
- download pretrained alignemnt network (ROI-LAnet) [here](https://www.dropbox.com/s/ktoylrk90yyeo0b/AlignNet-epoch-25.mat?dl=0) and put into preTrainedNetworks folder
- download pretrained end to end network (EE-PRnet) [here](https://www.dropbox.com/s/8nt8nlo1mburzn9/EE_PRnet-epoch-60.mat?dl=0) and put into preTrainedNetworks folder
- see and run demo codes in ROI-LAnet and EE-PRnet folders

## Questions
If you have any questions about the paper please email me on wojciech.matkowski@ntu.edu.sg, matk0001@e.ntu.edu.sg or maskotky@gmail.com
