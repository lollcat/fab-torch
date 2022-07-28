# GMM Problem
## Training
All training scripts are provided in the Makefile

## Evaluation
We provide the trained model parameters in the models folder. These can then be evaluated using
[`evaluation.py`](evaluation.py) and [`evaluation_expectation_quadratic_func.py`](evaluation_expectation_quadratic_func.py) 
to obtain the results from the paper. Furthermore [`results_vis.py`](results_vis.py) may be used to obtain the plot from the paper
visualising each of the modes. 