# Backdrop: stochastic backpropagation [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1252464.svg)](https://doi.org/10.5281/zenodo.1252464) [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/dexgen/backdrop/master)

* Siavash Golkar
* Kyle Cranmer

## Summary
[Backdrop](https://www.dropbox.com/s/qgckopo5zt7be6a/Backdrop.pdf?dl=0) is a flexible and easy to implement tool for introducing stochasticity to the gradient and can be  intuitively  described as dropout along the backpropagation pipeline. Backdrop can lead to a significant improvement in optimization performance in problems where stochastic gradient descent is not suitable or well defined. These generally fall into two main classes: 1) problems with non-decomposable losses and 2) problems where each sample can be intuitively considered as being hierarchically comprised of many smaller subsamples. 

<p align="center">
  <img src="masking-conv-net.jpg" width="750"/>
</p>

Similar to dropout, backdrop is implemented via the insertion of one or more masking layers in the network. Backdrop masking layers are transparent during the forward pass but block random parts of the gradient along the backward pass. Above is an example of two backdrop masking layers in a convolutional network. An example of the masks generated during the backward pass of this network is given below. Note that in this case stochasticity is introduced because the gradient is propagated for only a fraction of the various parts of the input.

<p align="center">
  <img src="masks.jpg" width="500"/>
</p>

For more details and results from experiments that demonstrate performance improvements with backdrop see the paper ([draft](https://www.dropbox.com/s/qgckopo5zt7be6a/Backdrop.pdf?dl=0)).

## Motivation

<img  src="millenium.jpg"  width="320" align="right" />

There are two general classes of problems for which stochastic gradient descent is not suitable.  First, optimization of loss functions that cannot be decomposed into a sum of individual sample losses. For example global performance measures such as $F$ measures and ROC AUC fall into this calss. Similarly, problems where we would like to impose some desired structure on the latent space are often not decomposable. For these class of problems, the loss cannot be evaluated on individual samples so stochastic gradient descent is not applicable. Moreover, often they can not be well approximated with small batch sizes; therefore, minibatch SGD is also not well suited. For details on how backdrop helps in this class of problems see the [paper](https://www.dropbox.com/s/qgckopo5zt7be6a/Backdrop.pdf?dl=0). 

A second class of problems for which SGD is not suitable are problems that have a small number of information-rich samples, each of which is comprised of many smaller subsamples. Examples include sequences, graphs, images, time-series as well as many physics based scenarios. For instance, the Millenium Run (see image), a cosmology simulation that traces the evolution of the matter distribution of the Universe, is a demonstrative example of this class of problems. At the largest scale, the image is a web of filaments. As we zoom in, we start seeing more structure, corresponding to the dark matter halos and eventually we see many galaxies inside each halo.  For this class of problems, one would like to take advantage of the many realizations of the objects at the different scales in the problem. 

## Gaussian process texture dataset

As a benchmark example for this class of problems, we created the two-scale Gaussian process dataset, where each sample is comprised of a number of large and small blobs. Each large or small blob can be considered an independent realization at that scale. We considered a classifiction problem for four classes defined by the length scale associated to the large and small blobs. The 2-scale nature of the data poses a problem for approaches based on cropping at a single scale. For a demonstration of the GP dataset generator see [gp-generate-demo.ipynb](gp-generate-demo.ipynb)

<p align="center">
  <img src="GP-blobs.JPG" width="600"/>
</p>

### Backdrop implementation

With backdrop we can approach this problem using a convolutional network which incorporates two backdrop masking layers inserted at points along the network that correspond to convolutional layers with receptor fields comparable to the size of the blobs. During the forward pass of the network, the objective is evaluated using the entire sample but during the backward pass, the gradients are only propagated for a random patches of the data. Specifically, the weights of the coarser convolutional layers will get updated according to the surviving patches of the large scale mask, and  the weights of the finer convolutional layers will get updated according to the surviving patches of the small scale mask.

<p align="center">
  <img src="masking-conv-net.jpg" width="750"/>
</p>

An example of the masks generated during the backward pass is given in the next image. Note that because of the multiplicative nature of backpropagation the masks generated are multiplicative as well, i.e. the patches that survive in the small-scale mask are a subset of the surviving patches in the large-scale mask. For details of the implementation see [backdrop-demo.ipynb](backdrop-demo.ipynb). 

<p align="center">
  <img src="masks.jpg" width="500"/>
</p>

<hr>

For more details and results regarding performance improvements with backdrop see the paper ([draft](https://www.dropbox.com/s/qgckopo5zt7be6a/Backdrop.pdf?dl=0)). For implementation of backdrop via a masking layer see [backdrop-demo.ipynb](backdrop-demo.ipynb). For a demonstration of the GP dataset generator see [gp-generate-demo.ipynb](gp-generate-demo.ipynb). The dataset used in the paper is [provided on Zenodo](https://zenodo.org/record/1252464#.WwcpbkgvwuU) under dataset.zip. 


For backdrop ([draft](https://www.dropbox.com/s/qgckopo5zt7be6a/Backdrop.pdf?dl=0)) , please cite using the following BibTex entry:

```
@article{golkar2018,
    author = {{Golkar}, s. and {Cranmer}, K.},
    title = "{Backdrop: stochastic backpropagation}",
    journal = {ArXiv e-prints},
    archivePrefix = "arXiv",
    eprint = {1805.XXXX},
    primaryClass = "stat.ML",
    year = 2018,
    month = may,
}

```
For the multi-scale Gaussian process dataset, please cite:
 
```
@misc{golkar_siavash_2018_1252464,
  author       = {Golkar, Siavash and
                  Cranmer, Kyle},
  title        = {Multi-scale Gaussian process dataset},
  month        = may,
  year         = 2018,
  doi          = {10.5281/zenodo.1252464},
  url          = {https://doi.org/10.5281/zenodo.1252464}
}
```