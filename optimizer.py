from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary.
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary.
                alpha = group["lr"]

                # Complete the implementation of AdamW here, reading and saving
                # your state in the `state` dictionary above.
                # The hyperparameters can be read from the `group` dictionary
                # (they are lr, betas, eps, weight_decay, as saved in the constructor).
                #
                # To complete this implementation:
                # 1. Update the first and second moments of the gradients.
                # 2. Apply bias correction
                #    (using the "efficient version" given in https://arxiv.org/abs/1412.6980;
                #     also given in the pseudo-code in the project description).
                # 3. Update parameters (p.data).
                # 4. Apply weight decay after the main gradient-based updates.
                # Refer to the default project handout for more details.

                beta_1, beta_2 = group["betas"]
                eps = group["eps"]
                w_lambda = group["weight_decay"]
                correct_bias = group["correct_bias"]


                # State initialization (if not already initialized)
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                # 1. Update the first and second moments of the gradients.

                # m is the first moment, v is the second moment
                m, v = state["m"], state["v"]
                t = state["step"] + 1  # Update step

                # m = beta_1 * m + (1 - beta_1) * grad
                m.mul_(beta_1).add_(grad, alpha = 1 - beta_1)

                # v = beta_2 * v + (1 - beta_2) * grad^2
                v.mul_(beta_2).addcmul_(grad, grad, value = 1 - beta_2)

                # 2. Apply bias correction
                if correct_bias:
                    m_hat = m / (1 - beta_1 ** t)
                    v_hat = v / (1 - beta_2 ** t)
                else:
                    m_hat, v_hat = m, v

                # Update parameters
                update = m_hat / (v_hat.sqrt() + eps)
                
                # if weight_decay is enabled, add the weight decay term
                if w_lambda > 0:
                    update.add_(p.data, alpha=w_lambda * alpha)  # Weight decay

                # p.data.add_(-alpha, update)  # Parameter update
                p.data.add_(update, alpha=-alpha)

                # Update state
                state["step"] = t

                # # If correct_bias is True, then the bias correction is applied.
                # # if group["correct_bias"]:
                # #     m /= (1 - beta_1)
                # #     v /= (1 - beta_2)
                # if group["correct_bias"]:
                #     m_hat = m / (1 - beta_1)
                #     v_hat = v / (1 - beta_2)
                # else:
                #     m_hat = m
                #     v_hat = v
                 
                # # 3. Update parameters (p.data).
                # # p = p - alpha * m_hat / (sqrt(v_hat) + eps)
                # p.data.addcdiv_(m_hat, (v_hat.sqrt() + group["eps"]), value=-alpha)

                # # 4. Apply weight decay after the main gradient-based updates.
                # # p = p * (1 - alpha * weight_decay)
                # # p -= alpha * lambda * p
                # w_lambda = group["weight_decay"]
                # if w_lambda > 0:
                #     p.data.add_(p.data, alpha=-alpha * w_lambda)

                # p.data.mul_(1 - group["lr"] * group["weight_decay"])

                # 3. Update parameters (p.data).
                # p.data.addcdiv_(m_hat, (v_hat.sqrt() + group["eps"]), value=-alpha)

                # 4. Apply weight decay after the main gradient-based updates.
                # p.data.mul_(1 - group["lr"] * group["weight_decay"])
                # if group["weight_decay"] > 0:
                #     p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

                # print("loss: ", loss)
                
                # if "m" not in state:
                #     state["m"] = torch.zeros_like(p.data)
                # if "v" not in state:    
                #     state["v"] = torch.zeros_like(p.data)

        return loss
