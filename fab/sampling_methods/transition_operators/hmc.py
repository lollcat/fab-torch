import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple

from fab.sampling_methods.transition_operators.base import TransitionOperator
from fab.types_ import LogProbFunc

HMC_STEP_TUNING_METHODS = ["p_accept", "Expected_target_prob", "No-U"]

class HamiltoneanMonteCarlo(TransitionOperator):
    def __init__(self, n_ais_intermediate_distributions: int,
                 dim: int,
                 epsilon: float = 1.0,
                 n_outer: int = 1,
                 L: int = 5,
                 mass_init: Union[float, torch.Tensor] = 1.0,
                 step_tuning_method: str = "p_accept",
                 target_p_accept: float = 0.65,
                 lr: float = 1e-3,
                 tune_period: bool = False,
                 common_epsilon_init_weight: float = 0.1,
                 max_grad: float = 1e3,
                 eval_mode: bool = False):
        """
        The options for the step_uning_method are as follows:
            - p_accept: Tune the step size (epsilon) to obtain an average acceptance probability of
                65%.
            - No-U: Maximise expected distance moved, using gradient descent based on
                the last HMC inner step. Additionally with scaling of the distance moved according
                to the standard deviation of samples w.r.t each dimension.
                Following ideas from https://arxiv.org/pdf/1711.09268.pdf.
            - Expected_target_prob: Maximise the expected target prob by gradient descent, from
                http://proceedings.mlr.press/v139/campbell21a/campbell21a.pdf
        """
        super(HamiltoneanMonteCarlo, self).__init__()
        assert step_tuning_method in HMC_STEP_TUNING_METHODS
        if isinstance(mass_init, torch.Tensor):
            assert mass_init.shape == (dim, )  # check mass_init dim is correct if a vector
        self.dim = dim
        self.tune_period = tune_period
        self.max_grad = max_grad
        self.step_tuning_method = step_tuning_method
        if step_tuning_method in ["Expected_target_prob", "No-U"]:
            self.train_params = True
            self.counter = 0
            self.mass_vector = nn.ParameterDict()
            self.epsilons = nn.ParameterDict()
            self.epsilons["common"] = nn.Parameter(
                torch.log(torch.tensor([epsilon])) * common_epsilon_init_weight)
            # have to store epsilons like this otherwise we get weird erros
            for i in range(n_ais_intermediate_distributions):
                for n in range(n_outer):
                    self.epsilons[f"{i}_{n}"] = nn.Parameter(torch.log(
                        torch.ones(dim)*epsilon*(1 - common_epsilon_init_weight)))
                    mass_after_softplus = torch.ones(dim) * mass_init
                    # now invert the softplus
                    self.mass_vector[f"{i}_{n}"] = nn.Parameter(
                        torch.log(torch.exp(mass_after_softplus) - 1.))
            if self.step_tuning_method == "No-U":
                self.register_buffer("characteristic_length",
                                     torch.ones(n_ais_intermediate_distributions, dim))
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        else:
            self.train_params = False
            # we weakly tie the step size parameters by utilising a shared component.
            self.register_buffer("common_epsilon", torch.tensor([epsilon * common_epsilon_init_weight]))
            self.register_buffer("epsilons", torch.ones([n_ais_intermediate_distributions, n_outer]) * epsilon *
                                 (1 - common_epsilon_init_weight))
            self.register_buffer("mass_vector", torch.ones(dim) * mass_init)
        self.n_outer = n_outer
        self.L = L
        self.n_intermediate_ais_distributions = n_ais_intermediate_distributions
        self.target_p_accept = target_p_accept
        self.first_dist_p_accepts = [torch.tensor([0.0]) for _ in range(n_outer)]
        self.last_dist_p_accepts = [torch.tensor([0.0]) for _ in range(n_outer)]
        self.average_distance_first_dist: torch.Tensor
        self.average_distance_last_dist: torch.Tensor
        self.eval_mode = eval_mode  # turn off step size tuning

    def set_eval_mode(self, eval_setting: bool):
        """When eval_mode is turned on, no tuning of epsilon or the mass matrix occurs."""
        self.eval_mode = eval_setting

    def get_logging_info(self) -> dict:
        """Log the total distance moved during HMC, as well as some of the tuned parameters."""
        interesting_dict = {}
        for i, val in enumerate(self.first_dist_p_accepts):
            interesting_dict[f"dist0_p_accept_{i}"] = val.item()
        if self.n_intermediate_ais_distributions > 1:
            for i, val in enumerate(self.last_dist_p_accepts):
                interesting_dict[f"dist{self.n_intermediate_ais_distributions - 1}_p_accept_{i}"]\
                    = val.item()
        epsilon_first_dist_first_loop = self.get_epsilon(0, 0)
        mass_first_dist_first_loop = self.get_mass(0, 0)
        if epsilon_first_dist_first_loop.numel() == 1:
            interesting_dict[f"epsilons_dist0_loop0"] = epsilon_first_dist_first_loop.cpu().item()
        else:
            interesting_dict[f"epsilons_dist0_0_dim0"] = epsilon_first_dist_first_loop[0].cpu().item()
            interesting_dict[f"epsilons_dist0_0_dim-1"] = epsilon_first_dist_first_loop[-1].cpu().item()
        interesting_dict[f"mass_dist0_0_dim0"] = mass_first_dist_first_loop[0].cpu().item()
        interesting_dict[f"mass_dist0_0_dim-1"] = mass_first_dist_first_loop[-1].cpu().item()

        if self.n_intermediate_ais_distributions > 1:
            last_dist_n = self.n_intermediate_ais_distributions - 1
            epsilon_last_dist_first_loop = self.get_epsilon(last_dist_n, 0)
            mass_last_dist_first_loop = self.get_mass(last_dist_n, 0)
            if epsilon_last_dist_first_loop.numel() == 1:
                interesting_dict[f"epsilons_dist{last_dist_n}_loop0"] = epsilon_last_dist_first_loop.cpu().item()
            else:
                interesting_dict[f"epsilons_dist{last_dist_n}_0_dim0"] = epsilon_last_dist_first_loop[0].cpu().item()
                interesting_dict[f"epsilons_dist{last_dist_n}_0_dim-1"] = epsilon_last_dist_first_loop[-1].cpu().item()

            interesting_dict[f"mass_dist{last_dist_n}_0_dim0"] = mass_last_dist_first_loop[0].cpu().item()
            interesting_dict[f"mass_dist{last_dist_n}_0_dim-1"] = mass_last_dist_first_loop[-1].cpu().item()

        interesting_dict["average_distance_dist0"] = self.average_distance_first_dist.cpu().item()
        if hasattr(self, f"average_distance_last_dist"):
            interesting_dict[f"average_distance_dist_{self.n_intermediate_ais_distributions - 1}"] \
                = self.average_distance_last_dist.cpu().item()
        return interesting_dict

    def get_epsilon(self, i: int, n: int) -> torch.Tensor:
        """
        Args:
            i: AIS intermediate distribution number
            n: HMC outer loop step number

        Returns: The HMC step size hyper-parameter, either a vector or scalar.

        """
        if self.train_params:
            return torch.exp(self.epsilons[f"{i}_{n}"]) + torch.exp(self.epsilons["common"])
        else:
            return self.epsilons[i, n] + self.common_epsilon


    def get_mass(self, i: int, n: int) -> torch.Tensor:
        """
        Args:
            i: AIS intermediate distribution number
            n: HMC outer loop step number

        Returns: The HMC mass (or Metric) hyper-parameter, of the same length as the
        dimension of the position vector.
        """
        if self.train_params:
            return F.softplus(self.mass_vector[f"{i}_{n}"])
        else:
            return self.mass_vector

    def setup_inner_step_mass_and_epsilon(self, l, i, n) -> Tuple[torch.Tensor, torch.Tensor]:
        """For No-U based methods we only backpropogate through the final step. For maximising
        expected target log prob we backpropogate through all the steps"""
        epsilon = self.get_epsilon(i, n)
        mass_matrix = self.get_mass(i, n)
        if (l != self.L - 1 and self.step_tuning_method in ["No-U", "No-U-unscaled"]) or \
                not self.train_params or self.eval_mode:
            epsilon = epsilon.detach()
            mass_matrix = mass_matrix.detach()
        return epsilon, mass_matrix

    def joint_log_prob(self, theta, p, mass_matrix, U):
        return - U(theta) - self.kinetic_energy(p, mass_matrix)

    def metropolis_acceptance_prob(self, theta_proposed, theta_current, p_proposed, p_current,
                                   mass_matrix, U):
        log_prob_current = self.joint_log_prob(theta_current, p_current, mass_matrix, U)
        log_prob_proposed = self.joint_log_prob(theta_proposed, p_proposed, mass_matrix, U)
        acceptance_prob = torch.exp(log_prob_proposed - log_prob_current)
        # reject samples with nan acceptance probability
        acceptance_probability = torch.nan_to_num(acceptance_prob,
                                                  nan=0.0,
                                                  posinf=0.0,
                                                  neginf=0.0)
        acceptance_prob = torch.clamp(acceptance_probability, min=0.0, max=1.0)
        return acceptance_prob.detach()

    def kinetic_energy(self, p, mass_matrix):
        return torch.sum(p**2 / mass_matrix, dim=-1) / 2

    def HMC_func(self, U, current_theta, grad_U, i):
        current_theta = current_theta.detach()  # block grads from previous HMC steps
        loss = 0
        if self.train_params:
            self.optimizer.zero_grad()
        for n in range(self.n_outer):
            original_theta = torch.clone(current_theta).detach()
            theta = current_theta
            mass_matrix = self.get_mass(i, n)
            if self.step_tuning_method not in ["Expected_target_prob"] or self.eval_mode:
                mass_matrix = mass_matrix.detach()
                theta = theta.detach()  # only backprop through final step to theta.
            p = torch.randn_like(theta) * mass_matrix
            current_p = p
            grad_u = grad_U(theta)

            # Now loop through position and momentum leapfrogs
            for l in range(self.L):
                epsilon, mass_matrix = self.setup_inner_step_mass_and_epsilon(l=l, i=i, n=n)
                # make momentum half step
                p = p - epsilon * grad_u / 2
                # Make full step for position
                theta = theta + epsilon / mass_matrix * p
                # update grad_u
                grad_u = grad_U(theta)
                # Make a full step for the momentum if not at end of trajectory
                # make momentum half step
                p = p - epsilon * grad_u / 2

            acceptance_probability = self.metropolis_acceptance_prob(
                theta_proposed=theta, theta_current=current_theta,
                p_proposed=p, p_current=current_p,
                U=U, mass_matrix=mass_matrix
            )
            accept = acceptance_probability > torch.rand(acceptance_probability.shape).to(theta.device)
            current_theta[accept] = theta[accept]
            p_accept_mean = torch.mean(acceptance_probability)
            self.store_info(i=i, n=n, p_accept_mean=p_accept_mean, current_theta=current_theta,
                            original_theta=original_theta)
            if self.eval_mode:
                # in eval mode we don't perform any tuning of the step size.
                return current_theta.detach()  # stop gradient flow
            else:
                if self.step_tuning_method == "p_accept":
                    self.adjust_step_size_p_accept(p_accept_mean=p_accept_mean, i=i, n=n)
                else:
                    self.adjust_step_size_min_p_accept(p_accept_mean=p_accept_mean, i=i, n=n)
                    if self.step_tuning_method == "No-U":
                        loss = loss + self.no_u_turn_loss(
                            i=i, current_theta=current_theta, original_theta=original_theta,
                            acceptance_probability=acceptance_probability)
        if self.train_params:
            if self.tune_period is False or self.counter < self.tune_period:
                if self.step_tuning_method == "Expected_target_prob":
                    loss = torch.mean(U(current_theta))
                if not (torch.isinf(loss) or torch.isnan(loss)):
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
                    assert torch.isfinite(grad_norm)
                    torch.nn.utils.clip_grad_value_(self.parameters(), 1)
                    # torch.autograd.grad(loss, self.epsilons["0_1"], retain_graph=True)
                    self.optimizer.step()

        if self.step_tuning_method == "No-U":
            # set next characteristc lengths
            self.characteristic_length.data[i, :] = torch.std(current_theta.detach(), dim=0)
        return current_theta.detach()  # stop gradient flow

    def adjust_step_size_p_accept(self, p_accept_mean, i, n):
        """Adjust step size to reach the target p-accept."""
        if p_accept_mean > self.target_p_accept:  # too much accept
            self.epsilons[i, n] = self.epsilons[i, n] * 1.05
            self.common_epsilon = self.common_epsilon * 1.02
        else:
            self.epsilons[i, n] = self.epsilons[i, n] / 1.05
            self.common_epsilon = self.common_epsilon / 1.02

    def adjust_step_size_min_p_accept(self, p_accept_mean, i, n):
        """ For gradient based tuning methods, we make sure that p_accept is at least 0.1."""
        if p_accept_mean < 0.2:
            # if p_accept is very low manually decrease step size, as this means that no acceptances so no
            # gradient flow to use
            self.epsilons[f"{i}_{n}"].data = self.epsilons[f"{i}_{n}"].data - 0.05
            self.epsilons["common"].data = self.epsilons["common"].data - 0.05
        if i == 0:
            self.counter += 1

    def no_u_turn_loss(self, i, current_theta, original_theta, acceptance_probability):
        """Estimate loss based on maximising expected distance moved."""
        distance_scaled = torch.linalg.norm(
            (original_theta - current_theta) / self.characteristic_length[i, :], ord=2, dim=-1)
        weighted_scaled_mean_square_distance = acceptance_probability * distance_scaled ** 2
        if (self.tune_period is False or self.counter < self.tune_period) and \
                self.step_tuning_method == "No-U":
            # remove zeros so we don't get infs when we divide
            weighted_scaled_mean_square_distance[weighted_scaled_mean_square_distance == 0.0] = 1.0
            loss = torch.mean(1.0 / weighted_scaled_mean_square_distance -
                                     weighted_scaled_mean_square_distance)
        else:
            loss = 0.0
        return loss

    def store_info(self, i, n, p_accept_mean, current_theta, original_theta):
        """Store info that will be retrieved for logging."""
        if i == 0:  # save info from the first AIS distribution.
            # save as interesting info for plotting
            self.first_dist_p_accepts[n] = p_accept_mean.cpu().detach()
            distance = torch.linalg.norm((original_theta - current_theta), ord=2, dim=-1)
            self.average_distance_first_dist = torch.mean(distance).detach().cpu()
        elif i == self.n_intermediate_ais_distributions - 1:
            self.last_dist_p_accepts[n] = p_accept_mean.cpu().detach()
            distance = torch.linalg.norm((original_theta - current_theta), ord=2, dim=-1)
            self.average_distance_last_dist = torch.mean(distance).detach().cpu()

    def transition(self, x: torch.Tensor, log_p_x: LogProbFunc, i: int) -> torch.Tensor:
        # currently mainly written with grad_log_q_x = None in mind
        # using diagonal, would be better to use vmap (need to wait until this is released doh)
        """
        U is potential energy, theta is position, p is momentum
        """
        current_theta = x
        def U(theta: torch.Tensor):
            return - log_p_x(theta)

        def grad_U(theta: torch.Tensor):
            theta = theta.clone().requires_grad_(True) #  need this to get gradients
            y = U(theta)
            return torch.nan_to_num(
                torch.clamp(
                torch.autograd.grad(y, theta, grad_outputs=torch.ones_like(y))[0],
                max=self.max_grad, min=-self.max_grad),
                nan=0.0, posinf=0.0, neginf=0.0)

        current_theta = self.HMC_func(U, current_theta, grad_U, i)
        return current_theta


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