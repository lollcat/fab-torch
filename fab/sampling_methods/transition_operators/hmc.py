import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple

from fab.sampling_methods.transition_operators.base import TransitionOperator
from fab.types_ import LogProbFunc

HMC_STEP_TUNING_METHODS = ["p_accept", "Expected_target_prob", "No-U", "No-U-unscaled"]

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
                 max_grad: float = 1e3):
        """
        The options for the step_uning_method are as follows:
            - p_accept: Tune the step size (epsilon) to obtain an average acceptance probability of
                65%.
            - No-U-unscaled: Maximise expected distance moved, using gradient descent based on
                the last HMC inner step.
            - No-U: Like No-U-unscaled with scaling of the distance moved according to the standard
                deviation of samples w.r.t each dimension.
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
        if step_tuning_method in ["Expected_target_prob", "No-U", "No-U-unscaled"]:
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
                    self.mass_vector[f"{i}_{n}"] = nn.Parameter(torch.ones(dim) * mass_init)
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
        self.average_distance: torch.Tensor

    def get_logging_info(self) -> dict:
        """Log the total distance moved during HMC, as well as some of the tuned parameters."""
        interesting_dict = {}
        for i, val in enumerate(self.first_dist_p_accepts):
            interesting_dict[f"dist1_p_accept_{i}"] = val.item()
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

        interesting_dict["average_distance"] = self.average_distance.cpu().item()
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
                not self.train_params:
            epsilon = epsilon.detach()
            mass_matrix = mass_matrix.detach()
        return epsilon, mass_matrix


    def HMC_func(self, U, current_theta, grad_U, i):
        if self.step_tuning_method == "Expected_target_prob":
            # need this for grad function
            current_theta = current_theta.clone().detach().requires_grad_(True)
            current_theta = torch.clone(current_theta)  # so we can do in place operations
        else:
            current_theta = current_theta.detach()  # otherwise just need to block grad flow
        loss = 0
        # need this for grad function
        # base function for HMC written in terms of potential energy function U
        for n in range(self.n_outer):
            original_theta = torch.clone(current_theta).detach()
            if self.train_params:
                self.optimizer.zero_grad()
            theta = current_theta
            mass_matrix = self.get_mass(i, n)
            if mass_matrix not in ["Expected_target_prob"]:
                mass_matrix = mass_matrix.detach()
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


            U_current = U(current_theta)
            U_proposed = U(theta)
            current_K = torch.sum(current_p**2, dim=-1) / 2
            proposed_K = torch.sum(p**2, dim=-1) / 2

            # Accept or reject the state at the end of the trajectory, returning either the position at the
            # end of the trajectory or the initial position
            acceptance_probability = torch.exp(U_current - U_proposed + current_K - proposed_K)
            # reject samples with nan acceptance probability
            acceptance_probability = torch.nan_to_num(acceptance_probability,
                                                      nan=0.0,
                                                      posinf=0.0,
                                                      neginf=0.0)
            acceptance_probability = torch.clamp(acceptance_probability, min=0.0, max=1.0)
            accept = acceptance_probability > torch.rand(acceptance_probability.shape).to(theta.device)
            current_theta[accept] = theta[accept]
            p_accept = torch.mean(acceptance_probability)
            if self.step_tuning_method == "p_accept":
                if p_accept > self.target_p_accept: # too much accept
                    self.epsilons[i, n] = self.epsilons[i, n] * 1.05
                    self.common_epsilon = self.common_epsilon * 1.02
                else:
                    self.epsilons[i, n] = self.epsilons[i, n] / 1.05
                    self.common_epsilon = self.common_epsilon / 1.02
            else: # self.step_tuning_method == "No-U":
                if p_accept < 0.01: # or (self.counter < 100 and p_accept < 0.4):
                    # if p_accept is very low manually decrease step size, as this means that no acceptances so no
                    # gradient flow to use
                    self.epsilons[f"{i}_{n}"].data = self.epsilons[f"{i}_{n}"].data - 0.05
                    self.epsilons["common"].data = self.epsilons["common"].data - 0.05
                if i == 0:
                    self.counter += 1
            if i == 0: # save fist and last distribution info
                # save as interesting info for plotting
                self.first_dist_p_accepts[n] = torch.mean(acceptance_probability).cpu().detach()
            elif i == self.n_intermediate_ais_distributions - 1:
                self.last_dist_p_accepts[n] = torch.mean(acceptance_probability).cpu().detach()

            if i == 0 or self.step_tuning_method == "No-U-unscaled":
                distance = torch.linalg.norm((original_theta - current_theta), ord=2, dim=-1)
                if i == 0:
                    self.average_distance = torch.mean(distance).detach().cpu()
                if self.step_tuning_method == "No-U-unscaled":
                    # torch.autograd.grad(torch.mean(weighted_mean_square_distance), self.epsilons[f"{i}_{n}"], retain_graph=True)
                    # torch.autograd.grad(loss, self.epsilons[f"{i}_{n}"], retain_graph=True)
                    weighted_mean_square_distance = acceptance_probability * distance ** 2
                    loss = loss - torch.mean(weighted_mean_square_distance)
            if self.step_tuning_method == "No-U":
                distance_scaled = torch.linalg.norm((original_theta - current_theta) / self.characteristic_length[i, :], ord=2, dim=-1)
                weighted_scaled_mean_square_distance = acceptance_probability * distance_scaled ** 2
                if (self.tune_period is False or self.counter < self.tune_period) and self.step_tuning_method == "No-U":
                    # remove zeros so we don't get infs when we divide
                    weighted_scaled_mean_square_distance[weighted_scaled_mean_square_distance == 0.0] = 1.0
                    loss = loss + torch.mean(1.0/weighted_scaled_mean_square_distance -
                                             weighted_scaled_mean_square_distance)

        if self.train_params:
            if self.tune_period is False or self.counter < self.tune_period:
                if self.step_tuning_method == "Expected_target_prob":
                    loss = torch.mean(U(current_theta))
                if not (torch.isinf(loss) or torch.isnan(loss)):
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
                    torch.nn.utils.clip_grad_value_(self.parameters(), 1)
                    # torch.autograd.grad(loss, self.epsilons["0_1"], retain_graph=True)
                    self.optimizer.step()
        if self.step_tuning_method == "No-U":
            # set next characteristc lengths
            self.characteristic_length.data[i, :] = torch.std(current_theta.detach(), dim=0)
        return current_theta.detach()  # stop gradient flow

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