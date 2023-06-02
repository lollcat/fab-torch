# Flow Annealed Importance Sampling Bootstrap (FAB)

## Overview

Normalizing flows can approximate complicated Boltzmann distributions of physical systems. However, 
current methods for training flows either suffer from mode-seeking behavior, use samples from the 
target generated beforehand by expensive MCMC simulations, or use stochastic losses that have very 
high variance. We tackle this challenge by augmenting flows with annealed importance sampling (AIS) 
and minimize the mass covering $\alpha$-divergence with $\alpha = 2$, which minimizes importance 
weight variance. Our method, Flow AIS Bootstrap (FAB), uses AIS to generate samples in regions 
where the flow is a poor approximation of the target, facilitating the discovery of new modes. 

In this repository, we implement FAB and provide the code to reproduce our experiments. For more
details about our method and the results of our experiments, please read
[our paper](https://arxiv.org/abs/2208.01893).

**Note**: The most important thing to get right when applying FAB to a given problem is to make sure that AIS is returning reasonable samples,
where by reasonable we mean that the samples from AIS are closer to the target than the flow. 
See [About the code](#about-the-code) for further details on how to use the FAB codebase on new problems.


## Methods of Installation

The  package can be installed via pip by navigating in the repository directory and running

```
pip install --upgrade .
```

To run the alanine dipeptide experiments, you will need to install the [OpenMM Library](http://openmm.org/)
as well as [`openmmtools`](https://openmmtools.readthedocs.io/en/stable/). This can be done via conda.

```
conda install -c conda-forge openmm openmmtools
```

## Experiments

### Gaussian Mixture Model

<a href="https://colab.research.google.com/github/lollcat/fab-torch/blob/master/experiments/gmm/fab_gmm.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

For this problem we use a mixture of 40 two dimensional Gaussian distributions. 
This allows for easy visualisation of the various methods for training the flow.
We provide a [colab notebook](experiments/gmm/fab_gmm.ipynb) with an example of training a flow on the GMM problem, 
comparing FAB to training a flow with KL divergence minimisation.
This can be run in a short period of time (10 min) and provides a clear visualisation of how FAB is 
able to discover new modes and fit them. 

To run the experiment for the FAB with a prioritised replay buffer (for the first seed), use the following command:
```
python experiments/gmm/run.py training.use_buffer=True training.prioritised_buffer=True
```
To run the full set of experiments see the [README](experiments/gmm/README.md) for the GMM experiments. 

The below plot shows samples from various trained models, with the GMM problem target contours in the background.
![Gaussian Mixture Model samples vs contours](experiments/gmm/plots/MoG.png)

### Many Well distribution
<a href="https://colab.research.google.com/github/lollcat/fab-torch/blob/master/experiments/many_well/fab_many_well.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


The Many Well distribution is made up of multiple repeats of the Double Well distribution, 
from the [original Boltzmann generators paper](https://www.science.org/doi/10.1126/science.aaw1147).

We provide a [colab notebook](experiments/many_well/fab_many_well.ipynb) comparing FAB to training a flow via KL divergence minimisation, on the 
6 dimensional Many Well problem, where the difference between the two methods is apparent after a 
short (<5 min) training period. This experiment can be run locally on a laptop using just CPU. 

To run the experiment for the FAB with a prioritised replay buffer (for the first seed) on the 
32 dimensional Many Well problem, use the following command:
```
python experiments/many_well/run.py training.use_buffer=True training.prioritised_buffer=True
```
To run the full set of experiments see the [README](experiments/many_well/README.md) for the Many Well experiments. 

The below plot shows samples for our model (FAB) vs training a flow by reverse KL divergence 
minimisation, with the Many Well problem target contours in the background. 
This visualisation is for the marginal pairs of the distributions for the first four elements of the x.
![Many Well distribution FAB vs training by KL divergence minimisation](experiments/many_well/plots/many_well.png)

### Alanine dipeptide

In our final experiment, we approximate the Boltzmann distribution of alanine dipeptide in an 
implicit solvent, which is a molecule with 22 atoms and a popular model system. The molecule
is visualized in the figure below. The right figure shows the probability density of for the
dihedral angle $\phi$ comparing the ground truth, which was obtrained with a molecular dynamics
(MD) simulation, the models trained with our method as well as maximum likelihood on MD samples.

![Alanine dipeptide and its dihedral angles; Comparison of probability densities](experiments/aldp/plots/aldp_phi.png)

Furthermore, we compared the Ramachandran plots of the different methods in the following figure.

![Ramachandran plot of alanine dipeptide](experiments/aldp/plots/ramachandran.png)

To reproduce our experiment, use the [`experiments/aldp/train.py`](experiments/aldp/train.py) script.
The respective configuration files are located in [`experiments/aldp/config`](experiments/aldp/config).
We used the seeds 0, 1, and 2 in our runs.

The data used to evaluate our models and to train the flow model with maximum likelihood is provided 
on [Zenodo](https://zenodo.org/record/6993124#.YvpugVpBy5M). If you want to use the configuration files
in [`experiments/aldp/config`](experiments/aldp/config) as is, you should put the data in the 
[`experiment/aldp/data`](experiments/aldp/data) folder.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6993124.svg)](https://doi.org/10.5281/zenodo.6993124)


### About the code 
The main FAB loss can be found in [core.py](fab/core.py), and we provide a simple training loop to 
train a flow with this loss (or other flow - loss combinations that meet the spec) in [train.py](fab/train.py) 
The FAB training algorithm **with** the prioritised buffer can be found in [train_with_prioritised_buffer.py](fab/train_with_prioritised_buffer.py). 
Additionally, we provide the code for running the SNR/dimensionality analysis with p and q set to independent Gaussians.
in the [fab-jax](https://github.com/lollcat/fab-jax-old) repository.
For training the CRAFT model on the GMM problem we forked the 
[Annealed Flow Transport repository](https://github.com/deepmind/annealed_flow_transport). 
This fork may be found [here](https://github.com/lollcat/annealed_flow_transport), and may be used for training the CRAFT model.

As we are still adding improvements to the efficiency and stability of the code, make sure you use the latest version.
Additionally, if you spot any areas of the code that could be improved then make an issue and we will be more 
than happy to fix it.
For the version of the code that was used in the paper see our [releases](https://github.com/lollcat/fab-torch/releases).

#### Applying FAB to a new problem:
The most important thing to get right when applying FAB to a given problem is to make sure that AIS is returning reasonable samples,
where by reasonable we just mean that the samples from AIS are closer to the target than the flow. 
Simply visualising the samples from the flow and AIS provides a good check for whether this is the case.
Making sure that the transition kernel (e.g. HMC) is working well (e.g. has well tuned step size) is key for AIS to work well.

An additional source of instability can be if the target energy function gives spurious values to points that have extreme values.
For example, evaluating the density of a zero-mean unit variance Gaussian on a point that has a value of 100 will give a spurious values. 
One can fix this by manually setting the log prob of the target to be -inf for regions that are 
known to be far outside of where samples from the target lie. 

Feel free to contact us if you would like any help getting FAB to work nicely!


### Normalizing Flow Libraries
We offer a simple wrapper that allows for various normalising flow libraries to be plugged into 
this repository. The main library we rely on is 
[normflows](https://github.com/VincentStimper/normalizing-flows). 


## Citation

If you use this code in your research, please cite it as:

> Laurence I. Midgley, Vincent Stimper, Gregor N. C. Simm, Bernhard Schölkopf, José Miguel Hernández-Lobato.
> Flow Annealed Importance Sampling Bootstrap. The Eleventh International Conference on Learning Representations. 2023.

**Bibtex**

```
@inproceedings{
midgley2023flow,
title={Flow Annealed Importance Sampling Bootstrap},
author={Laurence Illing Midgley and Vincent Stimper and Gregor N. C. Simm and Bernhard Sch{\"o}lkopf and Jos{\'e} Miguel Hern{\'a}ndez-Lobato},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=XCTVFJwS9LJ}
}
```

