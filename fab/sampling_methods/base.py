from typing import Mapping, Any, NamedTuple, Optional
import torch
import torch.nn as nn

from fab.types_ import LogProbFunc


class Point(NamedTuple):
    x: torch.Tensor
    log_q: torch.Tensor
    log_p: torch.Tensor
    grad_log_q: Optional[torch.Tensor] = None
    grad_log_p: Optional[torch.Tensor] = None

def grad_and_value(x, forward_fn):
    """Calculate the forward pass of a function y = f(x) as well as its gradient w.r.t x."""
    x = x.detach()
    x.requires_grad = True
    y = forward_fn(x)
    grad = torch.autograd.grad(y, x,  grad_outputs=torch.ones_like(y))[0]
    return grad.detach(), y.detach()

def create_point(x: torch.Tensor, log_q_fn: LogProbFunc, log_p_fn: LogProbFunc,
                 with_grad: bool) -> Point:
    if with_grad:
        grad_log_p, log_p = grad_and_value(x, log_p_fn)
        grad_log_q, log_q = grad_and_value(x, log_q_fn)
        return Point(x=x, log_p=log_p, log_q=log_q, grad_log_p=grad_log_p, grad_log_q=grad_log_q)
    else:
        return Point(x=x, log_q=log_q_fn(x), log_p=log_p_fn(x))


def get_intermediate_log_prob(x: Point,
                              beta: float,
                              p_sq_over_q_target: bool) -> torch.Tensor:
    """Get log prob of point according to intermediate AIS distribution.
    Set AIS final target g=p^2/q if p_sq_over_q_target else g=p.
    log_prob = (1 - beta) log_q + beta log_g
    """
    if p_sq_over_q_target:
        return (1 - 2*beta) * x.log_q + 2*beta*x.log_p
    else:
        return (1-beta) * x.log_q + beta * x.log_p


def get_grad_intermediate_log_prob(
        x: Point,
        beta: float,
        p_sq_over_q_target: bool) -> torch.Tensor:
    """Get gradient of intermediate AIS distribution for a point.
    Set AIS final target g=p^2/q if p_sq_over_q_target else g=p.
    \nabla_x log_prob = (1 - beta) \nabla_x log_q + beta \nabla_x log_g
    """
    if p_sq_over_q_target:
        return (1 - 2*beta) * x.grad_log_q + 2*beta*x.grad_log_p
    else:
        return (1-beta) * x.grad_log_q + beta * x.grad_log_p

