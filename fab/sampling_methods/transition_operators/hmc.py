import torch
from typing import Union, Callable

from fab.sampling_methods.transition_operators.base import TransitionOperator, Point
from fab.types_ import LogProbFunc


class HamiltonianMonteCarlo(TransitionOperator):
    def __init__(self,
                 n_ais_intermediate_distributions: int,
                 dim: int,
                 base_log_prob: LogProbFunc,
                 target_log_prob: LogProbFunc,
                 alpha: float = None,
                 p_target: bool = False,
                 epsilon: float = 1.0,
                 n_outer: int = 1,
                 L: int = 5,
                 mass_init: Union[float, torch.Tensor] = 1.0,
                 target_p_accept: float = 0.65,
                 max_grad: float = 1e3,
                 tune_period: bool = False,
                 common_epsilon_init_weight: float = 0.1,
                 eval_mode: bool = False
                 ):
        """
        Step tuning with p_accept if used.
        """
        super(HamiltonianMonteCarlo, self).__init__(
            n_ais_intermediate_distributions, dim, base_log_prob, target_log_prob,
            alpha=alpha, p_target=p_target)
        if isinstance(mass_init, torch.Tensor):
            assert mass_init.shape == (dim, )  # check mass_init dim is correct if a vector
        self.tune_period = tune_period
        # we weakly tie the step size parameters by utilising a shared component.
        self.register_buffer("common_epsilon", torch.tensor([epsilon * common_epsilon_init_weight]))
        self.register_buffer("epsilons", torch.ones([n_ais_intermediate_distributions, n_outer]) * epsilon *
                             (1 - common_epsilon_init_weight))
        self.register_buffer("mass_vector", torch.ones(dim) * mass_init)
        self.n_outer = n_outer
        self.L = L
        self.target_p_accept = target_p_accept
        self.max_grad = max_grad  # max grad used when taking steps
        self.first_dist_p_accepts = [torch.tensor([0.0]) for _ in range(n_outer)]
        self.last_dist_p_accepts = [torch.tensor([0.0]) for _ in range(n_outer)]
        self.average_distance_first_dist: torch.Tensor
        self.average_distance_last_dist: torch.Tensor
        self.eval_mode = eval_mode  # turn off step size tuning


    @property
    def uses_grad_info(self) -> bool:
        return True

    def set_eval_mode(self, eval_setting: bool):
        """When eval_mode is turned on, no tuning of epsilon or the mass matrix occurs."""
        self.eval_mode = eval_setting

    def get_logging_info(self) -> dict:
        """Log the total distance moved during HMC, as well as some of the tuned parameters."""
        interesting_dict = {}
        for i, val in enumerate(self.first_dist_p_accepts):
            interesting_dict[f"dist0_p_accept_{i}"] = val.item()
        if self.n_ais_intermediate_distributions > 1:
            for i, val in enumerate(self.last_dist_p_accepts):
                interesting_dict[f"dist{self.n_ais_intermediate_distributions - 1}_p_accept_{i}"]\
                    = val.item()
        epsilon_first_dist_first_loop = self.get_epsilon(0, 0)
        if epsilon_first_dist_first_loop.numel() == 1:
            interesting_dict[f"epsilons_dist0_loop0"] = epsilon_first_dist_first_loop.cpu().item()
        else:
            interesting_dict[f"epsilons_dist0_0_dim0"] = epsilon_first_dist_first_loop[0].cpu().item()
            interesting_dict[f"epsilons_dist0_0_dim-1"] = epsilon_first_dist_first_loop[-1].cpu().item()

        if self.n_ais_intermediate_distributions > 1:
            last_dist_n = self.n_ais_intermediate_distributions - 1
            epsilon_last_dist_first_loop = self.get_epsilon(last_dist_n, 0)
            if epsilon_last_dist_first_loop.numel() == 1:
                interesting_dict[f"epsilons_dist{last_dist_n}_loop0"] = epsilon_last_dist_first_loop.cpu().item()
            else:
                interesting_dict[f"epsilons_dist{last_dist_n}_0_dim0"] = epsilon_last_dist_first_loop[0].cpu().item()
                interesting_dict[f"epsilons_dist{last_dist_n}_0_dim-1"] = epsilon_last_dist_first_loop[-1].cpu().item()

        interesting_dict["average_distance_dist0"] = self.average_distance_first_dist.cpu().item()
        if hasattr(self, f"average_distance_last_dist"):
            interesting_dict[f"average_distance_dist_{self.n_ais_intermediate_distributions - 1}"] \
                = self.average_distance_last_dist.cpu().item()
        return interesting_dict

    def get_epsilon(self, i: int, n: int) -> torch.Tensor:
        """
        Args:
            i: AIS intermediate distribution number
            n: HMC outer loop step number

        Returns: A scalar for the HMC step size hyper-parameter.

        """
        index = i - 1
        return self.epsilons[index, n] + self.common_epsilon

    def joint_log_prob(self, point: Point, p, mass_matrix, U):
        return - U(point) - self.kinetic_energy(p, mass_matrix)

    def metropolis_accept(self, point_proposed: Point, point_current: Point,
                          p_proposed: torch.Tensor, p_current: torch.Tensor,
                          mass_matrix: torch.Tensor, U: Callable):
        log_prob_current = self.joint_log_prob(point_current, p_current, mass_matrix, U)
        log_prob_proposed = self.joint_log_prob(point_proposed, p_proposed, mass_matrix, U)
        with torch.no_grad():
            log_acceptance_prob = log_prob_proposed - log_prob_current
            # reject samples with nan acceptance probability
            valid_samples = torch.isfinite(log_acceptance_prob)
            log_acceptance_prob = torch.nan_to_num(log_acceptance_prob,
                                                      nan=-float('inf'),
                                                      posinf=-float('inf'),
                                                      neginf=-float('inf'))
            accept = log_acceptance_prob > -torch.distributions.Exponential(1.).sample(
                log_acceptance_prob.shape).to(log_acceptance_prob.device)
            accept = accept & valid_samples
            log_acceptance_prob = torch.clamp(log_acceptance_prob, max=0.0)  # prob can't be higher than 1.
            log_p_accept_mean = torch.logsumexp(log_acceptance_prob, dim=-1) - \
                                torch.log(torch.tensor(log_acceptance_prob.shape[0])).to(log_acceptance_prob.device)
            return accept, log_p_accept_mean

    def kinetic_energy(self, p, mass_matrix):
        return torch.sum(p**2 / mass_matrix, dim=-1) / 2

    def HMC_func(self, U, point: Point, grad_U, i):
        current_point = point
        for n in range(self.n_outer):
            original_point = current_point
            epsilon = self.get_epsilon(i, n)
            p = torch.randn_like(point.x) * self.mass_vector
            current_p = p
            grad_u = grad_U(point)
            # Now loop through position and momentum leapfrogs
            for l in range(self.L):
                # make momentum half step
                p = p - epsilon * grad_u / 2
                # Make full step for position
                x = point.x + epsilon / self.mass_vector * p
                point = self.create_new_point(x)
                # update grad_u
                grad_u = grad_U(point)
                # make momentum half step
                p = p - epsilon * grad_u / 2

            accept, log_p_accept_mean = self.metropolis_accept(
                point_proposed=point, point_current=current_point,
                p_proposed=p, p_current=current_p,
                U=U, mass_matrix=self.mass_vector
            )
            current_point[accept] = point[accept]
            self.store_info(i=i, n=n, p_accept_mean=torch.exp(log_p_accept_mean), current_x=point.x,
                            original_x=original_point.x)
            if not self.eval_mode:
                self.adjust_step_size_p_accept(log_p_accept_mean=log_p_accept_mean,
                                               i=i, n=n)
        return current_point

    def adjust_step_size_p_accept(self, log_p_accept_mean, i, n):
        """Adjust step size to reach the target p-accept."""
        index = i - 1
        if log_p_accept_mean > torch.log(torch.tensor(self.target_p_accept).to(log_p_accept_mean.device)):  # too much accept
            self.epsilons[index, n] = self.epsilons[index, n] * 1.05
            self.common_epsilon = self.common_epsilon * 1.02
        else:
            self.epsilons[index, n] = self.epsilons[index, n] / 1.05
            self.common_epsilon = self.common_epsilon / 1.02


    def store_info(self, i, n, p_accept_mean, current_x, original_x):
        """Store info that will be retrieved for logging."""
        if i == 1:  # save info from the first AIS distribution.
            # save as interesting info for plotting
            self.first_dist_p_accepts[n] = p_accept_mean.cpu().detach()
            distance = torch.linalg.norm((original_x - current_x), ord=2, dim=-1)
            self.average_distance_first_dist = torch.mean(distance).detach().cpu()
        elif i == self.n_ais_intermediate_distributions:
            self.last_dist_p_accepts[n] = p_accept_mean.cpu().detach()
            distance = torch.linalg.norm((original_x - current_x), ord=2, dim=-1)
            self.average_distance_last_dist = torch.mean(distance).detach().cpu()


    def transition(self, point: Point, i: int, beta: float) -> Point:
        """
        Perform HMC transition.
        """

        def U(point: Point):
            return - self.intermediate_target_log_prob(point, beta)

        def grad_U(point: Point):
            grad = - self.grad_intermediate_target_log_prob(point, beta)
            return torch.nan_to_num(
                torch.clamp(grad,
                max=self.max_grad, min=-self.max_grad),
                nan=0.0, posinf=0.0, neginf=0.0)

        point = self.HMC_func(U, point, grad_U, i)
        return point

    def save_model(self, save_path, epoch=None):
        model_description = str(self.class_args)
        if epoch is None:
            summary_results_path = str(save_path / "HMC_model_info.txt")
            model_path = str(save_path / "HMC_model")
        else:
            summary_results_path = str(save_path / f"HMC_model_info_epoch{epoch}.txt")
            model_path = str(save_path / f"HMC_model_epoch{epoch}")
        with open(summary_results_path, "w") as g:
            g.write(model_description)
        torch.save(self.state_dict(), model_path)

    def load_model(self, save_path, epoch=None, device='cpu'):
        if epoch is None:
            model_path = str(save_path / "HMC_model")
        else:
            model_path = str(save_path / f"HMC_model_epoch{epoch}")
        self.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        print("loaded HMC model")