This folder contains the source code and weights for the experiments of our work, `Probability Trust Intervals for Out of Distribution Input Detection`. 

### Directory Structure

1. The files Cn.py (n denotes the architecture number) contains the code for the Perturbed-NNs.
2. The files, Cn_Eval.py (n denotes the architecture number) contains the code for evaluation of all the considered baselines and our proposed approach.

### Experimental Setup

All the experiments are done on [Google Colaboratory](https://colab.research.google.com/). The links to original files have been removed for the purpose of double blind review. If you are interested in executing our code, you may follow the steps given below,

1. Upload the code files to the above website. 
2. Create a `weights` folder in your Google drive. Under this folder, create, `mnist` subfolder for weights of `mnist`, `cifar10` subfolder for weights of `cifar10` and so on for other datasets.
3. Mount Google drive to Google colaboratory. You may use [these instructions](https://colab.research.google.com/notebooks/io.ipynb#scrollTo=u22w3BFiOveA) in order to do the same.
4. Change the `prefix` variable (search for `prefix = ""`) in the uploaded code file to the path of parent folder of `weights` in your Google drive. For example if your `weights` folder is somewhere in the path, `./drive/My Drive/reviewCode/PNN/weights` then value of `prefix` should be `./drive/My Drive/reviewCode/PNN/` (observe that the path is to `PNN`, the parent folder of `weights`). 
5. Now you should be able to run the notebook and re-generate the results.

### Note

We are unable to provide weights of CNNs and PNNs due to the maximum file size limit imposed for the supplementary material.