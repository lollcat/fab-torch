import os
import pathlib
import wandb
from omegaconf import DictConfig

from datetime import datetime
import time
import matplotlib.pyplot as plt
import torch

from fab import Trainer
from fab.target_distributions.base import TargetDistribution
from fab.utils.plotting import plot_history
from fab.types_ import Model
from fab.utils.numerical import effective_sample_size

from examples.make_flow import make_normflow_snf_model
from examples.setup_run import SetupPlotterFn, setup_logger, get_n_iterations, get_load_checkpoint_dir



class SNFModel(Model):
    def __init__(self, snf, target, dim):
        self.snf = snf
        self.target_distribution = target
        self.loss = self.snf.reverse_kld
        # hack so that plotting with self.flow.sample works
        self.flow = type('', (), {})()
        self.flow.sample = lambda shape: snf.sample(shape[0])[0]
        self.flow.log_prob = lambda x: self.snf.log_prob(x)
        self.annealed_importance_sampler = type('', (), {})()
        self.annealed_importance_sampler.sample_and_log_weights = \
            lambda batch_size, *args, **kwargs: (
            torch.zeros(batch_size, dim), torch.zeros(batch_size))


    def get_iter_info(self):
        return {}

    def get_eval_info(self, outer_batch_size: int, inner_batch_size: int):
        base_samples = []
        base_log_w_s = []
        assert outer_batch_size % inner_batch_size == 0
        n_batches = outer_batch_size // inner_batch_size
        for i in range(n_batches):
            # Initialise AIS with samples from the base distribution.
            x, base_log_q_hat = self.snf.sample(inner_batch_size)
            base_log_w = self.target_distribution.log_prob(x) - base_log_q_hat
            # append base samples and log probs
            base_samples.append(x.detach().cpu())
            base_log_w_s.append(base_log_w.detach().cpu())

        base_samples = torch.cat(base_samples, dim=0)
        base_log_w_s = torch.cat(base_log_w_s, dim=0)
        # Z_estimate = torch.exp(torch.logsumexp(base_log_w_s, axis=0) - np.log(base_log_w_s.shape[0]))
        info = {"eval_ess_flow": effective_sample_size(log_w=base_log_w_s, normalised=False).item()}
        # note here that log prob of test set is actually the pi(x_0) - \sum_i S(x_i).
        # So the value of KL divergence will still be correct.
        target_info = self.target_distribution.performance_metrics(base_samples, base_log_w_s,
                                                                 self.snf.log_prob,
                                                                 batch_size=inner_batch_size)
        info.update(target_info)
        return info


    def parameters(self):
        return self.snf.parameters()

    def save(self,
             path: "str"
             ):
        """Save FAB model to file."""
        torch.save({'flow': self.snf.state_dict()},
                   path)

    def load(self,
             file_path,
             map_location):
        checkpoint = torch.load(file_path, map_location=map_location)
        try:
            self.snf.load_state_dict(checkpoint['flow'])
        except RuntimeError:
            print('Flow could not be loaded. '
                  'Perhaps there is a mismatch in the architectures.')


def setup_trainer_and_run_snf(cfg: DictConfig, setup_plotter: SetupPlotterFn,
                               target: TargetDistribution):
    """Create and trainer and run."""
    if cfg.training.tlimit:
        start_time = time.time()
    else:
        start_time = None
    if cfg.training.checkpoint_load_dir is not None:
        if not os.path.exists(cfg.training.checkpoint_load_dir):
            print("no checkpoint loaded, starting training from scratch")
            chkpt_dir = None
            iter_number = 0
        else:
            chkpt_dir, iter_number = get_load_checkpoint_dir(cfg.training.checkpoint_load_dir)
    else:
        chkpt_dir = None
        iter_number = 0
    dim = cfg.target.dim  # applies to flow and target
    save_path = os.path.join(cfg.evaluation.save_path, str(datetime.now().isoformat()))
    logger = setup_logger(cfg, save_path)
    if hasattr(cfg.logger, "wandb"):
        # if using wandb then save to wandb path
        save_path = os.path.join(wandb.run.dir, save_path)
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(save_path, "config.txt"), "w") as file:
        file.write(str(cfg))

    snf = make_normflow_snf_model(dim, n_flow_layers=cfg.flow.n_layers,
                                  layer_nodes_per_dim=cfg.flow.layer_nodes_per_dim,
                                  act_norm=cfg.flow.act_norm,
                                  target=target)

    # use GPU if available
    if torch.cuda.is_available() and cfg.training.use_gpu:
        snf.cuda()
        print("utilising GPU")

    model = SNFModel(snf, target, dim)

    optimizer = torch.optim.Adam(snf.parameters(), lr=cfg.training.lr)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
    scheduler = None

    if chkpt_dir is not None:
        map_location = "cuda" if torch.cuda.is_available() and cfg.training.use_gpu else "cpu"
        model.load(os.path.join(chkpt_dir, "model.pt"), map_location)
        opt_state = torch.load(os.path.join(chkpt_dir, 'optimizer.pt'), map_location)
        optimizer.load_state_dict(opt_state)
        print(f"\n\n****************loaded checkpoint: {chkpt_dir}*******************\n\n")



    plot = setup_plotter(cfg, target)

    trainer = Trainer(model=model, optimizer=optimizer, logger=logger, plot=plot,
                      optim_schedular=scheduler, save_path=save_path,
                      max_gradient_norm=cfg.training.max_grad_norm
                      )

    n_iterations = get_n_iterations(
        n_training_iter=cfg.training.n_iterations,
        n_flow_forward_pass=cfg.training.n_flow_forward_pass,
        batch_size=cfg.training.batch_size,
        loss_type=cfg.fab.loss_type,
        n_transition_operator_inner_steps=cfg.fab.transition_operator.n_inner_steps,
        n_intermediate_ais_dist=cfg.fab.n_intermediate_distributions,
        transition_operator_type=cfg.fab.transition_operator.type,
        use_buffer=cfg.training.use_buffer,
        min_buffer_length=cfg.training.min_buffer_length,
    )

    trainer.run(n_iterations=n_iterations, batch_size=cfg.training.batch_size,
                n_plot=cfg.evaluation.n_plots,
                n_eval=cfg.evaluation.n_eval, eval_batch_size=cfg.evaluation.eval_batch_size,
                save=True,
                n_checkpoints=cfg.evaluation.n_checkpoints,
                tlimit=cfg.training.tlimit,
                start_time=start_time,
                start_iter=iter_number
                )

    if hasattr(cfg.logger, "list_logger"):
        plot_history(trainer.logger.history)
        plt.show()
        print(trainer.logger.history['eval_ess_flow_p_target'][-10:])
        print(trainer.logger.history['eval_ess_ais_p_target'][-10:])
        print(trainer.logger.history['test_set_mean_log_prob_p_target'][-10:])