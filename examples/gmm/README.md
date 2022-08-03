# GMM Problem
## Experiments
The following commands can each be used to train the methods from the paper:
```
# FAB with prioritised buffer.
python examples/gmm/run.py -m training.seed=0,1,2 training.use_buffer=True training.prioritised_buffer=True 

# FAB without the prioritised buffer.
python examples/gmm/run.py -m training.seed=0,1,2 fab.loss_type=p2_over_q_alpha_2_div 

# Flow using ground truth samples, training by maximum likelihood/forward KL divergence minimiation.
python examples/gmm/run.py -m training.seed=0,1,2 fab.loss_type=target_forward_kl

# Flow using alpha-divergence, with alpha=2
python examples/gmm/run.py -m training.seed=0,1,2 fab.loss_type=flow_alpha_2_div_nis

# Flow using reverse KL divergence
python examples/gmm/run.py -m training.seed=0,1,2 fab.loss_type=flow_reverse_kld

# SNF using reverse KLD
python examples/gmm/run.py -m training.seed=0,1,2 flow.use_snf=True
```

**Further notes** This will use hydra-multirun to run the random seeds in parallel. 
However, if you just want to run locally and get a general idea of the results, 
you can run a single random seed for a much lower number of iterations. 
The config file for this experiment is [here](../config/gmm.yaml), where you can change the hyper-parameters.

## Evaluation
Trained models may be evaluated using the code in
[`evaluation.py`](evaluation.py) and [`evaluation_expectation_quadratic_func.py`](evaluation_expectation_quadratic_func.py).
Furthermore [`results_vis.py`](results_vis.py) may be used to obtain the plot from the paper
visualising each of the modes. 