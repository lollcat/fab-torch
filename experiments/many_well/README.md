# Many Well Problem
The Many Well problem is created by repeating the double well problem from the 
[original Boltzmann generators paper](https://www.science.org/doi/10.1126/science.aaw1147) multiple times.
We provided a [colab notebook](fab_many_well.ipynb) for training a flow using FAB, as well as training a flow
with KL divergence minimisation on the 6 dimensional Many Well problem. This notebook shows
the strong performance improvement from FAB in a relatively short training time.

## Experiments: Many Well 32
The following commands can each be used to train the methods from the paper:
```
# FAB with prioritised buffer.
python experiments/many_well/run.py -m training.seed=0,1,2 training.use_buffer=True training.prioritised_buffer=True 

# FAB without the prioritised buffer.
python experiments/many_well/run.py -m training.seed=0,1,2 fab.loss_type=fab_alpha_div 

# Flow using ground truth samples, training by maximum likelihood/forward KL divergence minimiation.
python experiments/many_well/run.py -m training.seed=0,1,2 fab.loss_type=target_forward_kl

# Flow using alpha-divergence, with alpha=2
python experiments/many_well/run.py -m training.seed=0,1,2 fab.loss_type=flow_alpha_2_div_nis

# Flow using reverse KL divergence
python experiments/many_well/run.py -m training.seed=0,1,2 fab.loss_type=flow_reverse_kl

# Flow using reverse KL divergence with resampled base distribution
python experiments/many_well/run.py -m training.seed=0,1,2 fab.loss_type=flow_reverse_kld flow.resampled_base=True

# SNF using reverse KLD
python experiments/many_well/run.py -m training.seed=0,1,2 flow.use_snf=True fab.loss_type=flow_reverse_kl
```
The config file for this experiment is [here](../config/many_well.yaml), where you can change the hyper-parameters.
These commands will (1) save plots of the model throughout training, (2) save metrics logged via 
the logger, and (3) save the model parameters, which may be loaded and analysed with the 
further scripts provided.
The location where these are saved may be adjusted by editing the config file.
By default the logger just writes all info to a pandas dataframe, however we 
provide a simple logger definition that allows for other loggers to be plugged in, 
such as a wandb logger.

**Further notes**: This will use hydra-multirun to run the random seeds in parallel. 
However, if you just want to run locally and get a general idea of the results, 
you can run a single random seed for a much lower number of iterations. 


## Evaluation
Trained models may be evaluated using the code in [`evaluation.py`](evaluation.py).
Furthermore [`results_vis.py`](results_vis.py) may be used to obtain the plot from the paper
visualising each of the modes. 