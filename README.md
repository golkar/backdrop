# Backdrop: stochastic backpropagation [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1252464.svg)](https://doi.org/10.5281/zenodo.1252464) [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/dexgen/backdrop/master)

* Siavash Golkar
* Kyle Cranmer 

<img  src="https://wwwmpa.mpa-garching.mpg.de/galform/virgo/millennium/poster_large.jpg"  width="320" align="right" />

[Backdrop](https://www.dropbox.com/s/wuvzafu1pgihd8f/Backdrop.pdf?dl=0) is a tool for introducing stochasticity which intuitively can be thought of as dropout along the backpropagation pipeline. Backdrop can lead to a significant improvement in optimization performance in problems where stochastic gradient descent is not suitable or well defined. Problems of this sort include optimization of loss functions that cannot be decomposed into a sum of individual sample losses (e.g. F measures, area under ROC or any other global performance measure) and problems which have a small number of samples where each sample is information rich and can be intuitively considered as being hierarchically comprised of many smaller subsamples. Examples of the latter include graphs, images, time-series as well as many physics based scenarios. 

The Millenium Run, a cosmology simulation that traces the evolution of the matter distribution of the Universe, is a demonstrative example of this class of problems. At the largest scale, the image is a web of filaments. As we zoom in, we start seeing more structure, corresponding to the dark matter halos and eventually we see many galaxies inside each halo.  For this class of problems, one would like to take advantage of the many realizations of the objects at the different scales in the problem. 

As a stand-in for this class of problems, we created the two-scale Gaussian process dataset, where each sample is comprised of a number of large and small blobs. Each large or small blob can be considered an independent realization of the GP at that scale and if we want to use the size of these blobs for classification or inference, we would be well served to look at each one individually. 

<p align="center">
  <img src="/GP-blobs.JPG" width="600"/>
</p>

Using backdrop we can approach this problem using a convolutional network which incorporates two backdrop masking layers inserted at points along the network that correspond to convolutional layers with receptor fields comparable to the size of the blobs. During the forward pass of the network, the objective is evaluated using the entire sample but during the backward pass, the gradients are only propagated for a random patches of the data. Specifically, the weights of the coarser convolutional layers will get updated according to the surviving patches of the large scale mask, and  the weights of the finer convolutional layers will get updated according to the surviving patches of the small scale mask.

<p align="center">
  <img src="/masking-conv-net.jpg" width="750"/>
</p>

An example of the masks generated during the backward pass is given in the next image. Note that because of the multiplicative nature of backpropagation the masks generated are multiplicative as well, i.e. the patches that survive in the small-scale mask are a subset of the surviving patches in the large-scale mask.

<p align="center">
  <img src="/masks.jpg" width="650"/>
</p>

<hr>

For more details and results regarding performance improvements with backdrop see the paper ([draft](https://www.dropbox.com/s/wuvzafu1pgihd8f/Backdrop.pdf?dl=0)). For implementation of backdrop via a masking layer see [backdrop-demo.ipynb](backdrop-demo.ipynb). For a demonstration of the GP dataset generator see [gp-generate-demo.ipynb](gp-generate-demo.ipynb). The dataset used in the paper is [provided on Zenodo](https://zenodo.org/record/1252464#.WwcpbkgvwuU) under dataset.zip. 


For backdrop ([draft](https://www.dropbox.com/s/wuvzafu1pgihd8f/Backdrop.pdf?dl=0)) , please cite using the following BibTex entry:

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
