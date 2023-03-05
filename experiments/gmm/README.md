# GMM Problem
## Experiments
The following commands can each be used to train the methods from the paper:
```
# FAB with prioritised buffer.
python experiments/gmm/run.py -m training.seed=0,1,2 training.use_buffer=True training.prioritised_buffer=True 

# FAB without the prioritised buffer.
python experiments/gmm/run.py -m training.seed=0,1,2 fab.loss_type=fab_alpha_div 

# Flow using ground truth samples, training by maximum likelihood/forward KL divergence minimiation.
python experiments/gmm/run.py -m training.seed=0,1,2 fab.loss_type=target_forward_kl

# Flow using alpha-divergence, with alpha=2
python experiments/gmm/run.py -m training.seed=0,1,2 fab.loss_type=flow_alpha_2_div_nis

# Flow using reverse KL divergence
python experiments/gmm/run.py -m training.seed=0,1,2 fab.loss_type=flow_reverse_kl

# Flow using reverse KL divergence with resampled base distribution
python experiments/gmm/run.py -m training.seed=0,1,2 fab.loss_type=flow_reverse_kl flow.resampled_base=True

# SNF using reverse KLD
python experiments/gmm/run.py -m training.seed=0,1,2 flow.use_snf=True fab.loss_type=flow_reverse_kl
```

The config file for this experiment is [here](../config/gmm.yaml), where you can change the hyper-parameters.
These commands will (1) save plots of the model throughout training, (2) save metrics logged via 
the logger, and (3) save the model parameters, which may be loaded and analysed with the 
further scripts provided.
The location where these are saved may be adjusted by editing the config file.
By default the logger just writes all info to a pandas dataframe, however we 
provide a simple logger definition that allows for other loggers to be plugged in, 
such as a wandb logger.

Additionally, the CRAFT experiments are run using a fork of the original CRAFT code 
[here](https://github.com/lollcat/annealed_flow_transport). 

**Further notes** This will use hydra-multirun to run the random seeds in parallel. 
However, if you just want to run locally and get a general idea of the results, 
you can run a single random seed for a much lower number of iterations. 

## Evaluation
Trained models may be evaluated using the code in
[`evaluation.py`](evaluation.py) and [`evaluation_expectation_quadratic_func.py`](evaluation_expectation_quadratic_func.py).
Furthermore [`results_vis.py`](results_vis.py) may be used to obtain the plot from the paper
visualising each of the modes. 

## Sample notebook

<a href="https://colab.research.google.com/github/lollcat/fab-torch/blob/dev-loll/experiments/gmm/fab_gmm.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

We also provide a [colab notebook](experiments/gmm/fab_gmm.ipynb) with an example of training 
a flow on the GMM problem, comparing FAB to training a flow with KL divergence minimisation.

