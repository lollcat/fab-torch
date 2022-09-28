# Alanine Dipeptide Problem

The config files for this experiment are given in the folder [`config`](config). 
Each experiment can be run by executing the following command:

```
python train.py config/<your-config>.yaml
```

The seed parameter was set to 0, 1, and 2 for the three runs. The data used to evaluate our models and to train the flow model with maximum likelihood is provided 
on [Zenodo](https://zenodo.org/record/6993124#.YvpugVpBy5M).

For evaluation, we use the script [`sample.py`](sample.py) to generate samples from the models and do AIS with them.