from collections import deque
from math import sqrt

import torch
from torch.optim import Optimizer


class AdaShift(Optimizer):
  def __init__(self, params, lr=1e-3, keep_num=10, betas=(0.9, 0.999),
               eps=1e-10, reduce_func=torch.max):
    exp_avg_weights = [betas[0] ** i for i in range(keep_num - 1, -1, -1)]
    weights_sum = sum(exp_avg_weights)
    exp_avg_weights = [weight / weights_sum for weight in exp_avg_weights]
    defaults = dict(lr=lr, keep_num=keep_num, betas=betas,
                    eps=eps, reduce_func=reduce_func,
                    exp_avg_weights=exp_avg_weights)
    super(AdaShift, self).__init__(params, defaults)

  def step(self, closure=None):
    loss = None
    if closure is not None:
      loss = closure()

    for group in self.param_groups:
      for p in group["params"]:
        if p.grad is None:
          continue
        grad = p.grad.data
        if grad.is_sparse:
          raise RuntimeError("AdaShift does not support sparse gradients.")

        state = self.state[p]
        if len(state) == 0:
          state["step"] = 1
          state["grad_deque"] = deque([grad], maxlen=group["keep_num"])
          state["exp_avg_sq"] = torch.zeros_like(p.data)
          continue

        grad_deque, exp_avg_sq = state["grad_deque"], state["exp_avg_sq"]
        _, beta2 = group["betas"]

        state["step"] += 1
        grad_apply = len(grad_deque) == group["keep_num"]
        offset_grad = grad_deque[0]
        grad_deque.append(grad)
        if not grad_apply:
          continue

        exp_avg = sum(weight * g for weight, g
                      in zip(group["exp_avg_weights"], grad_deque))
        reduce_func = group["reduce_func"]
        reduced_grad_sq = reduce_func(offset_grad.mul(offset_grad))
        exp_avg_sq.mul_(beta2).add_(1 - beta2, reduced_grad_sq)
        denom = exp_avg_sq.sqrt().add_(group["eps"])
        denom_bias_correction = 1 - beta2 ** state["step"]

        step_size = group["lr"] * sqrt(denom_bias_correction)
        p.data.addcdiv_(-step_size, exp_avg, denom)
    return loss
