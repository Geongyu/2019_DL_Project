import torch
import numpy as np 

def optimize_linear(grad, eps, norm=np.inf):

    """
      Solves for the optimal input to a linear function under a norm constraint.
      Optimal_perturbation = argmax_{eta, ||eta||_{norm} < eps} dot(eta, grad)
      :param grad: Tensor, shape (N, d_1, ...). Batch of gradients
      :param eps: float. Scalar specifying size of constraint region
      :param norm: np.inf, 1, or 2. Order of norm constraint.
      :returns: Tensor, shape (N, d_1, ...). Optimal perturbation
      """

    red_ind = list(range(1, len(grad.size())))
    avoid_zero_div = torch.tensor(1e-12, dtype=grad.dtype, device=grad.device)
    if norm == np.inf:
        optimal_perturbation = torch.sign(grad)
    elif norm == 1:
        abs_grad = torch.abs(grad)
        sign = torch.sign(grad)
        red_ind = list(range(1, len(grad.size())))
        abs_grad = torch.abs(grad)
        ori_shape = [1] * len(grad.size())
        ori_shape[0] = grad.size(0)

        max_abs_grad, _ = torch.max(abs_grad.view(grad.size(0), -1), 1)
        max_mask = abs_grad.eq(max_abs_grad.view(ori_shape)).to(torch.float)
        num_ties = max_mask
        for red_scalar in red_ind:
            num_ties = torch.sum(num_ties, red_scalar, keepdim=True)
        optimal_perturbation = sign * max_mask / num_ties
        # TODO integrate below to a test file
        # check that the optimal perturbations have been correctly computed
        opt_pert_norm = optimal_perturbation.abs().sum(dim=red_ind)
        assert torch.all(opt_pert_norm == torch.ones_like(opt_pert_norm))
    elif norm == 2:
        square = torch.max(
            avoid_zero_div,
            torch.sum(grad ** 2, red_ind, keepdim=True)
        )
        optimal_perturbation = grad / torch.sqrt(square)
        # TODO integrate below to a test file
        # check that the optimal perturbations have been correctly computed
        opt_pert_norm = optimal_perturbation.pow(2).sum(dim=red_ind, keepdim=True).sqrt()
        one_mask = (
                (square <= avoid_zero_div).to(torch.float) * opt_pert_norm +
                (square > avoid_zero_div).to(torch.float))
        assert torch.allclose(opt_pert_norm, one_mask, rtol=1e-05, atol=1e-08)
    else:
        raise NotImplementedError("Only L-inf, L1 and L2 norms are currently implemented.")

    # Scale perturbation to be the solution for the norm=eps rather than
    # norm=1 problem
    scaled_perturbation = eps*optimal_perturbation

    return scaled_perturbation
