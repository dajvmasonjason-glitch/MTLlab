import copy
import random
from abc import abstractmethod
from typing import Dict, List, Tuple, Union

import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize, least_squares, linprog
from kmeans_pytorch import kmeans  # 直接导入kmeans函数

from methods.min_norm_solvers import MinNormSolver, gradient_normalizers
from methods.utils import (
    list_of_float,
    get_task_weight,
    get_robust_step_size,
    check_device,
    normalize_weights,
    compute_task_adjusted_weights,
    create_assignment_matrix,
    compute_group_weights
)
from methods.cluster_methods import *
from methods.cons_city import ConsMTLCity, ConsCityFAMO 
from methods.famo_lbfgs import FamoLBFGS
from collections import deque
EPS = 1e-8 # for numerical stability


class PIVRG(WeightMethod):
    """
    PIVRG (Proportional Inverse Variance Regularized Gradient) method for multi-task learning.
    This method adjusts task weights based on task scores and gradient information.
    
    Args:
        n_tasks: Number of tasks
        device: Torch device
        max_norm: Maximum norm for gradient clipping
        bound: Bound parameter for temperature calculation
        mintemp: Minimum temperature for softmax
    """
    def __init__(self, n_tasks, device: torch.device, max_norm=1.0, bound=2.0, mintemp=10):
        super().__init__(n_tasks, device=device)
        self.max_norm = max_norm
        self.bound = bound
        self.mintemp = mintemp

    def get_weighted_loss(
        self,
        losses,
        shared_parameters,
        task_specific_parameters,
        last_shared_parameters,
        representation,
        scores
    ):
        """
        Calculate weighted loss using PIVRG algorithm.
        
        Parameters:
        ----------
        losses : Task losses
        shared_parameters : Shared parameters of the model
        task_specific_parameters : Task-specific parameters
        last_shared_parameters : Last shared layer parameters
        representation : Shared representation
        scores : Task scores used for weight adjustment
        
        Returns:
        -------
        GTG matrix and computed weights
        """
        # Compute gradient dimensions and store gradients for each task
        grad_dims = []
        for param in shared_parameters:
            grad_dims.append(param.data.numel())
        grads = torch.Tensor(sum(grad_dims), self.n_tasks).to(self.device)

        # Compute gradients for each task
        for i in range(self.n_tasks):
            if i < self.n_tasks - 1:
                losses[i].backward(retain_graph=True)
            else:
                losses[i].backward()
            self.grad2vec(shared_parameters, grads, grad_dims, i)
            # Reset gradients
            for p in shared_parameters:
                p.grad = None

        # Compute gradient inner product matrix
        GG = grads.t().mm(grads).cpu()  # [num_tasks, num_tasks]

        # Initialize optimization
        x_start = np.ones(self.n_tasks) / self.n_tasks
        A = GG.data.cpu().numpy()

        # Calculate temperature based on scores
        # 如果scores为None，则使用均匀权重 【FAIRGRAD特有：添加了对scores为None的处理】
        if scores is None:
            w = np.ones(self.n_tasks) / self.n_tasks
        else:
            max_score = np.max(scores)
            min_score = np.min(scores)
            
            def softmax_witht(x, temperature=1.0):
                e_x = np.exp((x - np.max(x)) / temperature)
                return e_x / e_x.sum(axis=0)

            # 处理max_score == min_score的情况 【FAIRGRAD特有：添加了对max_score等于min_score的处理】
            if max_score == min_score:
                temp = self.mintemp
            else:
                temp = np.max([(max_score - min_score) / np.log(self.bound), self.mintemp])
            w = self.n_tasks * softmax_witht(scores, temperature=temp)

        # Define objective function for optimization
        def objfn(x):
            return np.dot(A, x) - np.power(w / x, 0.5)

        # Solve optimization problem
        res = least_squares(objfn, x_start, bounds=(0, np.inf))
        w_cpu = res.x
        ww = torch.Tensor(w_cpu).to(grads.device)
        
        # Compute weighted gradient
        g = (grads * ww.view(1, -1)).sum(1)
        GTG = GG.data.cpu().numpy()
        
        # Overwrite gradients with weighted gradients
        self.overwrite_grad(shared_parameters, g, grad_dims)
        return GTG, w_cpu

    @staticmethod
    def grad2vec(shared_params, grads, grad_dims, task):
        """Convert gradients to vector form"""
        grads[:, task].fill_(0.0)
        cnt = 0
        
        for param in shared_params:
            grad = param.grad
            if grad is not None:
                grad_cur = grad.data.detach().clone()
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[: cnt + 1])
                grads[beg:en, task].copy_(grad_cur.data.view(-1))
            cnt += 1

    def overwrite_grad(self, shared_parameters, newgrad, grad_dims):
        """Overwrite model gradients with computed gradients"""
        newgrad = newgrad * self.n_tasks  # to match the sum loss
        cnt = 0

        for param in shared_parameters:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[: cnt + 1])
            this_grad = newgrad[beg:en].contiguous().view(param.data.size())
            param.grad = this_grad.data.clone()
            cnt += 1

    def backward(
        self,
        losses: torch.Tensor,
        parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        last_shared_parameters=None,
        representation=None,
        **kwargs
    ):
        """Backward pass with PIVRG weight adjustment"""
        # 从kwargs中提取scores参数，如果不存在则默认为None 【FAIRGRAD特有：从kwargs中提取scores，而不是直接作为参数】
        scores = kwargs.get('scores', None)
        GTG, w = self.get_weighted_loss(losses, shared_parameters, task_specific_parameters, 
                                       last_shared_parameters, representation, scores)
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
        return None, {"GTG": GTG, "weights": w}  # Align with other weight methods


class ConsMTL(WeightMethod):
    """
    ConsMTL (Consensus Multi-Task Learning) method.
    This method introduces an extra loss term to align task-specific representations.
    
    Args:
        n_tasks: Number of tasks
        device: Torch device
        max_norm: Maximum norm for gradient clipping
        lambda_: Regularization parameter for consensus loss
    """
    def __init__(self, n_tasks, device: torch.device, max_norm=1.0, lambda_=1):
        super().__init__(n_tasks, device=device)
        self.max_norm = max_norm
        self.lambda_ = lambda_
        # MTLlib版本中有: print(self.lambda_) 【FAIRGRAD特有：移除了打印语句】

    def get_weighted_loss(
        self,
        losses,
        shared_parameters,
        task_specific_parameters,
        last_shared_parameters,
        representation,
        **kwargs,
    ):
        """
        Calculate weighted loss using ConsMTL algorithm with consensus regularization.
        
        Parameters:
        ----------
        losses : Task losses
        shared_parameters : Shared parameters of the model
        task_specific_parameters : Task-specific parameters
        last_shared_parameters : Last shared layer parameters
        representation : Shared representation (list of task-specific representations)
        
        Returns:
        -------
        GTG matrix and computed weights
        """
        # Compute gradient dimensions
        grad_dims = []

        # Use representation as list of task-specific representations
        #z_list = representation
        # For qm9 and celeba: if representation is not a list, repeat it n_tasks times
        z_list = [representation for _ in range(self.n_tasks)]

        # Initialize gradient storage
        for param in shared_parameters:
            grad_dims.append(param.data.numel())
        grads = torch.Tensor(sum(grad_dims), self.n_tasks).to(self.device)
        
        # Calculate delta_i = ∂l_i/∂z_i for each task
        delta_is = []
        grad_list = []
        for l_i, z_i in zip(losses, z_list):
            delta_i = torch.autograd.grad(l_i, z_i, retain_graph=True, create_graph=True)[0]
            flattened_shared_grad = torch.cat([g.view(-1) for g in delta_i])
            grad_list.append(flattened_shared_grad)
            delta_is.append(delta_i)

        # Compute gradients for shared parameters
        num_tasks = len(losses)  
        params_per_task = len(task_specific_parameters) // num_tasks 

        for i in range(self.n_tasks):
            l_i = losses[i]
            shared_grads = torch.autograd.grad(l_i, shared_parameters, retain_graph=True)
            self.grad2vec(shared_grads, grads, grad_dims, i)

        # Compute gradient inner product matrix
        GG = grads.t().mm(grads).cpu()  # [num_tasks, num_tasks]
        
        # Solve for weights using least squares
        x_start = np.ones(self.n_tasks) / self.n_tasks
        A = GG.data.cpu().numpy()
        
        def objfn(x):
            return np.dot(A, x) - np.power(1 / x, 0.5)

        res = least_squares(objfn, x_start, bounds=(0, np.inf))
        w_cpu = res.x
        ww = torch.Tensor(w_cpu).to(grads.device)
        
        # Compute weighted gradient
        g = (grads * ww.view(1, -1)).sum(1)
        GTG = GG.data.cpu().numpy()

        # Compute consensus delta
        delta = sum([delta_i * w_cpu[i] for i, delta_i in enumerate(delta_is)]).detach()
        
        # Overwrite gradients for shared parameters
        self.overwrite_grad(shared_parameters, g, grad_dims)

        # Apply consensus regularization to task-specific parameters
        for i in range(self.n_tasks):
            l_i = losses[i]
            delta_i = delta_is[i]
            z_i = z_list[i]

            # Get task-specific parameters
            theta_i = task_specific_parameters[i * params_per_task : (i + 1) * params_per_task]

            # Compute extra loss term for consensus regularization
            L_extra_i = -self.lambda_ * torch.dot(delta_i.view(-1), delta.view(-1))

            # Compute gradients for task-specific parameters
            g1 = torch.autograd.grad(l_i, theta_i, retain_graph=True)
            g2 = torch.autograd.grad(L_extra_i, theta_i, retain_graph=True)
            
            # Normalize and apply consensus gradients
            norm1 = torch.norm(torch.cat([torch.flatten(g) for g in g1]))
            for p, g in zip(theta_i, g2):
                p.grad = g
            torch.nn.utils.clip_grad_norm_(theta_i, 0.1 * norm1)
            
            # Combine with original gradients
            for p, g in zip(theta_i, g1):
                p.grad += g

        return GTG, w_cpu

    @staticmethod
    def grad2vec(shared_grads, grads, grad_dims, task):
        """Convert gradients to vector form"""
        grads[:, task].fill_(0.0)
        cnt = 0

        for grad in shared_grads:
            if grad is not None:
                grad_cur = grad.data.detach().clone()
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[: cnt + 1])
                grads[beg:en, task].copy_(grad_cur.data.view(-1))
            cnt += 1

    def overwrite_grad(self, shared_parameters, newgrad, grad_dims):
        """Overwrite model gradients with computed gradients"""
        newgrad = newgrad * self.n_tasks  # to match the sum loss
        cnt = 0

        for param in shared_parameters:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[: cnt + 1])
            this_grad = newgrad[beg:en].contiguous().view(param.data.size())
            param.grad = this_grad.data.clone()
            cnt += 1

    def backward(
        self,
        losses: torch.Tensor,
        parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        last_shared_parameters=None,
        representation=None,
        **kwargs,
    ):
        """Backward pass with ConsMTL weight adjustment"""
        GTG, w = self.get_weighted_loss(losses, shared_parameters, task_specific_parameters, 
                                       last_shared_parameters, representation, **kwargs)
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
        return None, {"GTG": GTG, "weights": w}  # Align with other weight methods


class WeightMethod:
    def __init__(self, n_tasks: int, device: torch.device, max_norm = 1.0):
        super().__init__()
        self.n_tasks = n_tasks
        self.device = device
        self.max_norm = max_norm

    @abstractmethod
    def get_weighted_loss(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ],
        last_shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
        representation: Union[torch.nn.parameter.Parameter, torch.Tensor],
        **kwargs,
    ):
        pass

    def backward(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        last_shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        representation: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, None], Union[dict, None]]:
        """

        Parameters
        ----------
        losses :
        shared_parameters :
        task_specific_parameters :
        last_shared_parameters : parameters of last shared layer/block
        representation : shared representation
        kwargs :

        Returns
        -------
        Loss, extra outputs
        """
        loss, extra_outputs = self.get_weighted_loss(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            last_shared_parameters=last_shared_parameters,
            representation=representation,
            **kwargs,
        )

        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)

        loss.backward()
        return loss, extra_outputs

    def __call__(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        **kwargs,
    ):
        return self.backward(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            **kwargs,
        )

    def parameters(self) -> List[torch.Tensor]:
        """return learnable parameters"""
        return []


class FAMO(WeightMethod):
    """Linear scalarization baseline L = sum_j w_j * l_j where l_j is the loss for task j and w_h"""

    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        gamma: float = 1e-5,
        w_lr: float = 0.025,
        task_weights: Union[List[float], torch.Tensor] = None,
        max_norm: float = 1.0,
    ):
        super().__init__(n_tasks, device=device)
        self.min_losses = torch.zeros(n_tasks).to(device)
        self.w = torch.tensor([0.0] * n_tasks, device=device, requires_grad=True)
        self.w_opt = torch.optim.Adam([self.w], lr=w_lr, weight_decay=gamma)
        self.max_norm = max_norm
    
    def set_min_losses(self, losses):
        self.min_losses = losses

    def get_weighted_loss(self, losses, **kwargs):
        self.prev_loss = losses
        z = F.softmax(self.w, -1)
        D = losses - self.min_losses + 1e-8
        c = (z / D).sum().detach()
        loss = (D.log() * z / c).sum()
        return loss, {"weights": z, "logits": self.w.detach().clone()}

    def update(self, curr_loss):
        delta = (self.prev_loss - self.min_losses + 1e-8).log() - \
                (curr_loss      - self.min_losses + 1e-8).log()
        with torch.enable_grad():
            d = torch.autograd.grad(F.softmax(self.w, -1),
                                    self.w,
                                    grad_outputs=delta.detach())[0]
        self.w_opt.zero_grad()
        self.w.grad = d
        self.w_opt.step()


import torch
import torch.nn.functional as F
from collections import deque

class CorrFAMO(WeightMethod):
    """
    CorrFAMO (Unified Equation Version)
    
    Formula:
        rho_i = Cov(l_i, L_famo) / (Std(l_i) * Std(L_famo))
        L_total = L_famo - lambda * sum(rho_i)
    
    Unified Behavior:
        - Calculates correlation graph at every step.
        - If lambda = 0.0, the gradient from the correlation term is exactly zero,
          mathematically falling back to pure FAMO dynamics.
    """
    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        gamma: float = 1e-5,
        w_lr: float = 0.025,
        lambda_: float = 0.1,
        history_size: int = 10,
        eps: float = 1e-8,
        max_norm: float = 1.0,
    ):
        super().__init__(n_tasks, device=device)
        # ---------- FAMO Base Parameters ----------
        self.min_losses = torch.zeros(n_tasks, device=device)
        self.w = torch.zeros(n_tasks, device=device, requires_grad=True)
        self.w_opt = torch.optim.Adam([self.w], lr=w_lr, weight_decay=gamma)
        
        # ---------- CorrFAMO Parameters ----------
        self.lambda_ = lambda_
        self.eps = eps
        self.max_norm = max_norm
        
        # History buffers (T steps of K tasks)
        self.loss_hist = deque(maxlen=history_size)
        self.weighted_loss_hist = deque(maxlen=history_size)

    def set_min_losses(self, losses):
        self.min_losses = losses.detach()

    # --------------------------------------------------
    # Correlation Calculation (Equation Implementation)
    # --------------------------------------------------
    def _calc_rho(self, current_losses, current_weighted_loss):
        """
        Calculates rho_i based on the provided formula:
        rho_i = cov(Delta l_i, Delta L) / sqrt(var(Delta l_i) * var(Delta L))
        """
        # 1. Handle insufficient history (Cold start)
        # If history is empty, correlation is undefined (or 0).
        # We return a zero tensor with grad attached to inputs to maintain graph connectivity if needed,
        # though effectively it's a constant.
        if len(self.loss_hist) < 2:
            return torch.zeros(self.n_tasks, device=current_losses.device)

        # 2. Prepare Data: History (Detached) + Current (Differentiable)
        # Stack history: [T, K]
        h_losses = torch.stack(list(self.loss_hist), dim=0) 
        # Stack weighted history: [T]
        h_L = torch.stack(list(self.weighted_loss_hist), dim=0)

        # Concatenate current step to form the full window of size T+1
        # all_losses: [T+1, K]
        all_losses = torch.cat([h_losses, current_losses.unsqueeze(0)], dim=0)
        # all_L: [T+1]
        all_L = torch.cat([h_L, current_weighted_loss.unsqueeze(0)], dim=0)

        # 3. Calculate Deltas (Deviations from mean)
        # Delta l_i = l_i - mean(l_i)
        # Delta L   = L   - mean(L)
        # Mean is calculated over the time dimension (dim=0)
        mean_losses = all_losses.mean(dim=0, keepdim=True) # [1, K]
        mean_L = all_L.mean(dim=0, keepdim=True)           # [1]

        delta_li = all_losses - mean_losses # [T+1, K]
        delta_L  = all_L - mean_L           # [T+1]

        # 4. Calculate Variance and Covariance
        # Using biased estimator (denominator N) consistent with Pearson formula
        # var(Delta L)
        var_L = (delta_L ** 2).mean(dim=0)  # Scalar
        # var(Delta l_i)
        var_li = (delta_li ** 2).mean(dim=0) # [K]
        
        # cov(Delta l_i, Delta L)
        # Expand delta_L to [T+1, 1] to broadcast against [T+1, K]
        cov = (delta_li * delta_L.unsqueeze(1)).mean(dim=0) # [K]

        # 5. Calculate Rho (Correlation Coefficient)
        # Add eps to denominator for numerical stability
        denominator = torch.sqrt(var_li * var_L) + self.eps
        rho = cov / denominator # [K]

        return rho

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------
    def get_weighted_loss(self, losses, **kwargs):
        # [FAMO Phase]
        self.prev_loss = losses.detach() # Store for update step
        
        z = F.softmax(self.w, dim=-1)
        D = losses - self.min_losses + self.eps
        c = (z / D).sum().detach()
        famo_loss = (D.log() * z / c).sum()

        # [CorrFAMO Phase - Unified Framework]
        # Always calculate rho, regardless of lambda value.
        rho = self._calc_rho(losses, famo_loss)
        
        # Apply the formula: L_tot = L - lambda * sum(rho)
        # If lambda_ is 0, this subtracts 0, resulting in famo_loss exactly.
        # But the computational graph for rho is still constructed.
        corr_term = rho.sum()
        total_loss = famo_loss - self.lambda_ * corr_term

        return total_loss, {
            "weights": z,
            "logits": self.w.detach().clone(),
            "rho": rho.detach(),
            "corr_term": corr_term.detach()
        }

    # --------------------------------------------------
    # Update (Standard FAMO)
    # --------------------------------------------------
    def update(self, curr_loss):
        if not hasattr(self, "prev_loss"):
            return

        # FAMO Delta Calculation: log(prev - min) - log(curr - min)
        delta = (self.prev_loss - self.min_losses + self.eps).log() - \
                (curr_loss      - self.min_losses + self.eps).log()

        with torch.enable_grad():
            grad_w = torch.autograd.grad(
                F.softmax(self.w, dim=-1),
                self.w,
                grad_outputs=delta.detach(),
            )[0]

        self.w_opt.zero_grad()
        self.w.grad = grad_w
        self.w_opt.step()

        # Update History (Detached)
        # We do this after the step so it doesn't interfere with the current graph,
        # but prepares for the next iteration's correlation calc.
        with torch.no_grad():
            self.loss_hist.append(curr_loss.detach())
            # Re-calculate weighted loss with current weights for history consistency
            current_w_loss = (F.softmax(self.w, -1) * curr_loss).sum()
            self.weighted_loss_hist.append(current_w_loss.detach())


class ConsFAMO(WeightMethod):
    """
    ConsFAMO (Consensus FAMO) method for multi-task learning.
    This method combines FAMO's weighted loss approach for shared parameters
    with ConsMTL's consensus regularization for task-specific parameters.
    
    Args:
        n_tasks: Number of tasks
        device: Torch device
        max_norm: Maximum norm for gradient clipping
        lambda_: Regularization parameter for consensus loss
        gamma: Weight decay for weight optimizer
        w_lr: Learning rate for weight optimizer
    """
    def __init__(self, n_tasks, device: torch.device, max_norm=1.0, lambda_=1, gamma=1e-5, w_lr=0.025):
        super().__init__(n_tasks, device=device)
        self.max_norm = max_norm
        self.lambda_ = lambda_
        self.min_losses = torch.zeros(n_tasks).to(device)
        self.w = torch.tensor([0.0] * n_tasks, device=device, requires_grad=True)
        self.w_opt = torch.optim.Adam([self.w], lr=w_lr, weight_decay=gamma)
        self.prev_loss = None
    
    def set_min_losses(self, losses):
        self.min_losses = losses
    
    def get_weighted_loss(
        self,
        losses,
        shared_parameters,
        task_specific_parameters,
        last_shared_parameters,
        representation,
        **kwargs,
    ):
        """
        Calculate weighted loss using ConsFAMO algorithm.
        
        Parameters:
        ----------
        losses : Task losses
        shared_parameters : Shared parameters of the model
        task_specific_parameters : Task-specific parameters
        last_shared_parameters : Last shared layer parameters
        representation : Shared representation (list of task-specific representations)
        
        Returns:
        -------
        GTG matrix and computed weights
        """
        # Store previous loss for update
        self.prev_loss = losses
        
        # Use representation as list of task-specific representations
        z_list = representation
        # For qm9 and celeba: if representation is not a list, repeat it n_tasks times
        # z_list = [representation for _ in range(self.n_tasks)]
        
        # Compute FAMO weights
        z = F.softmax(self.w, -1)
        D = losses - self.min_losses + 1e-8
        c = (z / D).sum().detach()
        famo_loss = (D.log() * z / c).sum()
        
        # Calculate delta_i = ∂l_i/∂z_i for each task (ConsMTL part)
        delta_is = []
        for l_i, z_i in zip(losses, z_list):
            delta_i = torch.autograd.grad(l_i, z_i, retain_graph=True, create_graph=True)[0]
            delta_is.append(delta_i)
        
        # Compute gradients for shared parameters using FAMO loss
        shared_grads = torch.autograd.grad(famo_loss, shared_parameters, retain_graph=True)
        
        # Overwrite gradients for shared parameters
        for param, grad in zip(shared_parameters, shared_grads):
            param.grad = grad
        
        # Apply consensus regularization to task-specific parameters (ConsMTL part)
        if task_specific_parameters is not None:
            num_tasks = len(losses)
            params_per_task = len(task_specific_parameters) // num_tasks
            
            # Compute consensus delta
            delta = sum([delta_i * z[i] for i, delta_i in enumerate(delta_is)]).detach()
            
            for i in range(self.n_tasks):
                l_i = losses[i]
                delta_i = delta_is[i]
                z_i = z_list[i]
                
                # Get task-specific parameters
                theta_i = task_specific_parameters[i * params_per_task : (i + 1) * params_per_task]
                
                # Compute extra loss term for consensus regularization
                L_extra_i = -self.lambda_ * torch.dot(delta_i.view(-1), delta.view(-1))
                
                # Compute gradients for task-specific parameters
                g1 = torch.autograd.grad(l_i, theta_i, retain_graph=True)
                g2 = torch.autograd.grad(L_extra_i, theta_i, retain_graph=True)
                
                # Normalize and apply consensus gradients
                norm1 = torch.norm(torch.cat([torch.flatten(g) for g in g1]))
                for p, g in zip(theta_i, g2):
                    p.grad = g
                torch.nn.utils.clip_grad_norm_(theta_i, 0.1 * norm1)
                
                # Combine with original gradients
                for p, g in zip(theta_i, g1):
                    p.grad += g
        
        return famo_loss, {"weights": z, "logits": self.w.detach().clone()}
    
    def update(self, curr_loss):
        """
        Update FAMO weights based on loss changes.
        
        Parameters:
        ----------
        curr_loss : Current task losses
        """
        delta = (self.prev_loss - self.min_losses + 1e-8).log() - \
                (curr_loss      - self.min_losses + 1e-8).log()
        with torch.enable_grad():
            d = torch.autograd.grad(F.softmax(self.w, -1),
                                    self.w,
                                    grad_outputs=delta.detach())[0]
        self.w_opt.zero_grad()
        self.w.grad = d
        self.w_opt.step()


class RBLDC(WeightMethod):
    """
    Rate-Balanced LDC (RB-LDC) method for multi-task learning.
    This method combines LDC-MTL's lightweight Router network with FAMO's rate balancing logic.
    It dynamically adjusts task weights based on input features to balance loss decrease rates.
    
    Args:
        n_tasks: Number of tasks
        device: Torch device
        max_norm: Maximum norm for gradient clipping
        router_hidden_dim: Hidden dimension size for the Router network
        feature_dim: Dimension of the input features to the Router
        moving_avg_beta: Beta parameter for moving average of log losses
    """
    def __init__(self, n_tasks, device: torch.device, max_norm=1.0, 
                 router_hidden_dim=64, feature_dim=256, moving_avg_beta=0.9):
        super().__init__(n_tasks, device=device)
        self.max_norm = max_norm
        self.moving_avg_beta = moving_avg_beta
        self.first_step = True
        
        # LDC Component: Lightweight Router Network
        self.router = nn.Sequential(
            nn.Linear(feature_dim, router_hidden_dim),
            nn.ReLU(),
            nn.Linear(router_hidden_dim, n_tasks),
            # Output logits, softmax will be applied in get_weighted_loss
        ).to(device)
        
        # FAMO Component: History tracking for rates
        self.prev_log_losses = torch.zeros(n_tasks).to(device)
        
        # Initialize optimizer for Router parameters
        self.router_optimizer = torch.optim.Adam(self.router.parameters(), lr=0.001)
    
    def get_weighted_loss(
        self,
        losses,
        shared_parameters=None,
        task_specific_parameters=None,
        last_shared_parameters=None,
        representation=None,
        **kwargs,
    ):
        """
        Calculate weighted loss using RB-LDC algorithm.
        
        Parameters:
        ----------
        losses : Task losses
        shared_parameters : Shared parameters of the model
        task_specific_parameters : Task-specific parameters
        last_shared_parameters : Last shared layer parameters
        representation : Shared representation (used as input to Router)
        
        Returns:
        -------
        Weighted loss and computed weights
        """
        # Use representation as input to Router
        # If representation is a list, use the first element
        if isinstance(representation, list):
            router_input = representation[0]
        else:
            router_input = representation
        
        # If representation is a tensor with spatial dimensions, average pool it
        if len(router_input.shape) > 2:
            # Apply global average pooling
            router_input = torch.mean(router_input, dim=tuple(range(2, len(router_input.shape))))
        
        # Router computes logits
        pred_logits = self.router(router_input)
        
        # Apply softmax to get weights
        weights = F.softmax(pred_logits, dim=-1)
        
        # Compute weighted loss (lower level objective)
        weighted_loss = torch.sum(weights * losses)
        
        return weighted_loss, {"weights": weights.detach(), "logits": pred_logits.detach()}
    
    def update_router(self, losses):
        """
        Update Router parameters based on loss decrease rates (FAMO logic).
        
        Parameters:
        ----------
        losses : Current task losses
        
        Returns:
        -------
        Rate discrepancy loss
        """
        # Calculate log losses
        log_losses = torch.log(losses + 1e-8)
        
        # Handle first step
        if self.first_step:
            self.prev_log_losses = log_losses.detach()
            self.first_step = False
            return torch.tensor(0.0).to(self.device)
        
        # Compute actual rates of decrease
        actual_rates = (self.prev_log_losses - log_losses).detach()
        
        # Compute rate differences from mean
        rate_diff = actual_rates - actual_rates.mean()
        
        # Update history with moving average
        self.prev_log_losses = self.moving_avg_beta * self.prev_log_losses + \
                               (1 - self.moving_avg_beta) * log_losses.detach()
        
        # The target gradient for router logits is -rate_diff
        # This will be applied in the backward pass
        return rate_diff
    
    def backward(
        self,
        losses: torch.Tensor,
        parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        last_shared_parameters=None,
        representation=None,
        **kwargs,
    ):
        """
        Backward pass with RB-LDC weight adjustment.
        Implements the two-level optimization:
        1. Update model parameters with weighted loss
        2. Update Router parameters to balance loss decrease rates
        """
        # Get weighted loss and weights
        weighted_loss, extra_outputs = self.get_weighted_loss(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            last_shared_parameters=last_shared_parameters,
            representation=representation,
            **kwargs,
        )
        
        # Zero gradients for Router
        self.router_optimizer.zero_grad()
        
        # If shared_parameters is provided, zero their gradients too
        if shared_parameters is not None:
            for param in shared_parameters:
                if param.grad is not None:
                    param.grad.zero_()
        
        # First backward pass for model parameters
        weighted_loss.backward(retain_graph=True)
        
        # Apply gradient clipping if needed
        if self.max_norm > 0 and shared_parameters is not None:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
        
        # Update Router using FAMO logic
        rate_diff = self.update_router(losses)
        
        # Get the pred_logits from the Router's forward pass
        # We need to re-compute this for the backward pass
        if isinstance(representation, list):
            router_input = representation[0]
        else:
            router_input = representation
        
        if len(router_input.shape) > 2:
            router_input = torch.mean(router_input, dim=tuple(range(2, len(router_input.shape))))
        
        pred_logits = self.router(router_input)
        
        # Apply the target gradient to pred_logits
        # The target gradient is -rate_diff based on FAMO logic
        pred_logits.backward(gradient=-rate_diff, retain_graph=False)
        
        # Update Router parameters
        self.router_optimizer.step()
        
        return weighted_loss, {"weights": extra_outputs["weights"], 
                             "logits": extra_outputs["logits"],
                             "rate_diff": rate_diff}
    
    def parameters(self) -> List[torch.Tensor]:
        """Return learnable parameters (Router parameters)"""
        return list(self.router.parameters())


class NashMTL(WeightMethod):
    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        max_norm: float = 1.0,
        update_weights_every: int = 1,
        optim_niter=20,
    ):
        super(NashMTL, self).__init__(
            n_tasks=n_tasks,
            device=device,
        )

        self.optim_niter = optim_niter
        self.update_weights_every = update_weights_every
        self.max_norm = max_norm

        self.prvs_alpha_param = None
        self.normalization_factor = np.ones((1,))
        self.init_gtg = self.init_gtg = np.eye(self.n_tasks)
        self.step = 0.0
        self.prvs_alpha = np.ones(self.n_tasks, dtype=np.float32)

    def _stop_criteria(self, gtg, alpha_t):
        return (
            (self.alpha_param.value is None)
            or (np.linalg.norm(gtg @ alpha_t - 1 / (alpha_t + 1e-10)) < 1e-3)
            or (
                np.linalg.norm(self.alpha_param.value - self.prvs_alpha_param.value)
                < 1e-6
            )
        )

    def solve_optimization(self, gtg: np.array):
        self.G_param.value = gtg
        self.normalization_factor_param.value = self.normalization_factor

        alpha_t = self.prvs_alpha
        for _ in range(self.optim_niter):
            self.alpha_param.value = alpha_t
            self.prvs_alpha_param.value = alpha_t

            try:
                self.prob.solve(solver=cp.ECOS, warm_start=True, max_iters=100)
            except:
                self.alpha_param.value = self.prvs_alpha_param.value

            if self._stop_criteria(gtg, alpha_t):
                break

            alpha_t = self.alpha_param.value

        if alpha_t is not None:
            self.prvs_alpha = alpha_t

        return self.prvs_alpha

    def _calc_phi_alpha_linearization(self):
        G_prvs_alpha = self.G_param @ self.prvs_alpha_param
        prvs_phi_tag = 1 / self.prvs_alpha_param + (1 / G_prvs_alpha) @ self.G_param
        phi_alpha = prvs_phi_tag @ (self.alpha_param - self.prvs_alpha_param)
        return phi_alpha

    def _init_optim_problem(self):
        self.alpha_param = cp.Variable(shape=(self.n_tasks,), nonneg=True)
        self.prvs_alpha_param = cp.Parameter(
            shape=(self.n_tasks,), value=self.prvs_alpha
        )
        self.G_param = cp.Parameter(
            shape=(self.n_tasks, self.n_tasks), value=self.init_gtg
        )
        self.normalization_factor_param = cp.Parameter(
            shape=(1,), value=np.array([1.0])
        )

        self.phi_alpha = self._calc_phi_alpha_linearization()

        G_alpha = self.G_param @ self.alpha_param
        constraint = []
        for i in range(self.n_tasks):
            constraint.append(
                -cp.log(self.alpha_param[i] * self.normalization_factor_param)
                - cp.log(G_alpha[i])
                <= 0
            )
        obj = cp.Minimize(
            cp.sum(G_alpha) + self.phi_alpha / self.normalization_factor_param
        )
        self.prob = cp.Problem(obj, constraint)

    def get_weighted_loss(
        self,
        losses,
        shared_parameters,
        **kwargs,
    ):
        """

        Parameters
        ----------
        losses :
        shared_parameters : shared parameters
        kwargs :

        Returns
        -------

        """

        extra_outputs = dict()
        if self.step == 0:
            self._init_optim_problem()

        if (self.step % self.update_weights_every) == 0:
            self.step += 1

            grads = {}
            for i, loss in enumerate(losses):
                g = list(
                    torch.autograd.grad(
                        loss,
                        shared_parameters,
                        retain_graph=True,
                    )
                )
                grad = torch.cat([torch.flatten(grad) for grad in g])
                grads[i] = grad

            G = torch.stack(tuple(v for v in grads.values()))
            GTG = torch.mm(G, G.t())

            self.normalization_factor = (
                torch.norm(GTG).detach().cpu().numpy().reshape((1,))
            )
            GTG = GTG / self.normalization_factor.item()
            alpha = self.solve_optimization(GTG.cpu().detach().numpy())
            alpha = torch.from_numpy(alpha)

        else:
            self.step += 1
            alpha = self.prvs_alpha

        weighted_loss = sum([losses[i] * alpha[i] for i in range(len(alpha))])
        extra_outputs["weights"] = alpha
        extra_outputs["GTG"] = GTG.detach().cpu().numpy()
        return weighted_loss, extra_outputs


class LinearScalarization(WeightMethod):
    """Linear scalarization baseline L = sum_j w_j * l_j where l_j is the loss for task j and w_h"""

    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        task_weights: Union[List[float], torch.Tensor] = None,
    ):
        super().__init__(n_tasks, device=device)
        if task_weights is None:
            task_weights = torch.ones((n_tasks,))
        if not isinstance(task_weights, torch.Tensor):
            task_weights = torch.tensor(task_weights)
        assert len(task_weights) == n_tasks
        self.task_weights = task_weights.to(device)

    def get_weighted_loss(self, losses, **kwargs):
        loss = torch.sum(losses * self.task_weights)
        return loss, dict(weights=self.task_weights)


class LSCORR(WeightMethod):
    """
    Linear Scalarization with Correlation Coefficient (LSCORR)
    在LS方法基础上添加了任务相关系数惩罚项，参考CorrFAMO实现
    
    Formula:
        rho_i = Cov(l_i, L_ls) / (Std(l_i) * Std(L_ls))
        L_total = L_ls - lambda * sum(rho_i)
    """
    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        task_weights: Union[List[float], torch.Tensor] = None,
        lambda_: float = 0.1,
        history_size: int = 10,
        eps: float = 1e-8,
        max_norm: float = 1.0,
    ):
        super().__init__(n_tasks, device=device)
        if task_weights is None:
            task_weights = torch.ones((n_tasks,))
        if not isinstance(task_weights, torch.Tensor):
            task_weights = torch.tensor(task_weights)
        assert len(task_weights) == n_tasks
        self.task_weights = task_weights.to(device)
        
        # CorrFAMO Parameters
        self.lambda_ = lambda_
        self.eps = eps
        self.max_norm = max_norm
        
        # History buffers
        self.loss_hist = deque(maxlen=history_size)
        self.weighted_loss_hist = deque(maxlen=history_size)

    def _calc_rho(self, current_losses, current_weighted_loss):
        """
        计算相关系数 rho_i = Cov(l_i, L) / (Std(l_i) * Std(L))
        参考CorrFAMO的实现
        """
        if len(self.loss_hist) < 2:
            return torch.zeros(self.n_tasks, device=current_losses.device)

        h_losses = torch.stack(list(self.loss_hist), dim=0)
        h_L = torch.stack(list(self.weighted_loss_hist), dim=0)

        all_losses = torch.cat([h_losses, current_losses.unsqueeze(0)], dim=0)
        all_L = torch.cat([h_L, current_weighted_loss.unsqueeze(0)], dim=0)

        mean_losses = all_losses.mean(dim=0, keepdim=True)
        mean_L = all_L.mean(dim=0, keepdim=True)

        delta_li = all_losses - mean_losses
        delta_L = all_L - mean_L

        var_L = (delta_L ** 2).mean(dim=0)
        var_li = (delta_li ** 2).mean(dim=0)
        cov = (delta_li * delta_L.unsqueeze(1)).mean(dim=0)

        denominator = torch.sqrt(var_li * var_L) + self.eps
        rho = cov / denominator

        return rho

    def get_weighted_loss(self, losses, **kwargs):
        ls_loss = torch.sum(losses * self.task_weights)
        
        rho = self._calc_rho(losses, ls_loss)
        corr_term = rho.sum()
        
        total_loss = ls_loss - self.lambda_ * corr_term

        self.loss_hist.append(losses.detach())
        self.weighted_loss_hist.append(ls_loss.detach())

        return total_loss, dict(
            weights=self.task_weights,
            rho=rho.detach(),
            corr_term=corr_term.detach()
        )


class ScaleInvariantLinearScalarization(WeightMethod):
    """Linear scalarization baseline L = sum_j w_j * l_j where l_j is the loss for task j and w_h"""

    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        task_weights: Union[List[float], torch.Tensor] = None,
    ):
        super().__init__(n_tasks, device=device)
        if task_weights is None:
            task_weights = torch.ones((n_tasks,))
        if not isinstance(task_weights, torch.Tensor):
            task_weights = torch.tensor(task_weights)
        assert len(task_weights) == n_tasks
        self.task_weights = task_weights.to(device)

    def get_weighted_loss(self, losses, **kwargs):
        loss = torch.sum(torch.log(losses) * self.task_weights)
        return loss, dict(weights=self.task_weights)


class SCALEINVLSCORR(WeightMethod):
    """
    Scale Invariant Linear Scalarization with Correlation Coefficient (SCALEINVLSCORR)
    在ScaleInvariantLS方法基础上添加了任务相关系数惩罚项，参考CorrFAMO实现
    
    Formula:
        rho_i = Cov(l_i, L_scaleinvls) / (Std(l_i) * Std(L_scaleinvls))
        L_total = L_scaleinvls - lambda * sum(rho_i)
    """
    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        task_weights: Union[List[float], torch.Tensor] = None,
        lambda_: float = 0.1,
        history_size: int = 10,
        eps: float = 1e-8,
        max_norm: float = 1.0,
    ):
        super().__init__(n_tasks, device=device)
        if task_weights is None:
            task_weights = torch.ones((n_tasks,))
        if not isinstance(task_weights, torch.Tensor):
            task_weights = torch.tensor(task_weights)
        assert len(task_weights) == n_tasks
        self.task_weights = task_weights.to(device)
        
        # CorrFAMO Parameters
        self.lambda_ = lambda_
        self.eps = eps
        self.max_norm = max_norm
        
        # History buffers
        self.loss_hist = deque(maxlen=history_size)
        self.weighted_loss_hist = deque(maxlen=history_size)

    def _calc_rho(self, current_losses, current_weighted_loss):
        """
        计算相关系数 rho_i = Cov(l_i, L) / (Std(l_i) * Std(L))
        参考CorrFAMO的实现
        """
        if len(self.loss_hist) < 2:
            return torch.zeros(self.n_tasks, device=current_losses.device)

        h_losses = torch.stack(list(self.loss_hist), dim=0)
        h_L = torch.stack(list(self.weighted_loss_hist), dim=0)

        all_losses = torch.cat([h_losses, current_losses.unsqueeze(0)], dim=0)
        all_L = torch.cat([h_L, current_weighted_loss.unsqueeze(0)], dim=0)

        mean_losses = all_losses.mean(dim=0, keepdim=True)
        mean_L = all_L.mean(dim=0, keepdim=True)

        delta_li = all_losses - mean_losses
        delta_L = all_L - mean_L

        var_L = (delta_L ** 2).mean(dim=0)
        var_li = (delta_li ** 2).mean(dim=0)
        cov = (delta_li * delta_L.unsqueeze(1)).mean(dim=0)

        denominator = torch.sqrt(var_li * var_L) + self.eps
        rho = cov / denominator

        return rho

    def get_weighted_loss(self, losses, **kwargs):
        scaleinvls_loss = torch.sum(torch.log(losses) * self.task_weights)
        
        rho = self._calc_rho(losses, scaleinvls_loss)
        corr_term = rho.sum()
        
        total_loss = scaleinvls_loss - self.lambda_ * corr_term

        self.loss_hist.append(losses.detach())
        self.weighted_loss_hist.append(scaleinvls_loss.detach())

        return total_loss, dict(
            weights=self.task_weights,
            rho=rho.detach(),
            corr_term=corr_term.detach()
        )


class MGDA(WeightMethod):
    """Based on the official implementation of: Multi-Task Learning as Multi-Objective Optimization
    Ozan Sener, Vladlen Koltun
    Neural Information Processing Systems (NeurIPS) 2018
    https://github.com/intel-isl/MultiObjectiveOptimization

    """

    def __init__(
        self, n_tasks, device: torch.device, params="shared", normalization="none"
    ):
        super().__init__(n_tasks, device=device)
        self.solver = MinNormSolver()
        assert params in ["shared", "last", "rep"]
        self.params = params
        assert normalization in ["norm", "loss", "loss+", "none"]
        self.normalization = normalization

    @staticmethod
    def _flattening(grad):
        return torch.cat(
            tuple(
                g.reshape(
                    -1,
                )
                for i, g in enumerate(grad)
            ),
            dim=0,
        )

    def get_weighted_loss(
        self,
        losses,
        shared_parameters=None,
        last_shared_parameters=None,
        representation=None,
        **kwargs,
    ):
        """

        Parameters
        ----------
        losses :
        shared_parameters :
        last_shared_parameters :
        representation :
        kwargs :

        Returns
        -------

        """
        # Our code
        grads = {}
        params = dict(
            rep=representation, shared=shared_parameters, last=last_shared_parameters
        )[self.params]
        for i, loss in enumerate(losses):
            g = list(
                torch.autograd.grad(
                    loss,
                    params,
                    retain_graph=True,
                )
            )
            # Normalize all gradients, this is optional and not included in the paper.

            grads[i] = [torch.flatten(grad) for grad in g]

        gn = gradient_normalizers(grads, losses, self.normalization)
        for t in range(self.n_tasks):
            for gr_i in range(len(grads[t])):
                grads[t][gr_i] = grads[t][gr_i] / gn[t]

        sol, min_norm = self.solver.find_min_norm_element(
            [grads[t] for t in range(len(grads))]
        )
        sol = sol * self.n_tasks  # make sure it sums to self.n_tasks
        weighted_loss = sum([losses[i] * sol[i] for i in range(len(sol))])

        return weighted_loss, dict(weights=torch.from_numpy(sol.astype(np.float32)))


class LOG_MGDA(WeightMethod):
    """Based on the official implementation of: Multi-Task Learning as Multi-Objective Optimization
    Ozan Sener, Vladlen Koltun
    Neural Information Processing Systems (NeurIPS) 2018
    https://github.com/intel-isl/MultiObjectiveOptimization

    """

    def __init__(
        self, n_tasks, device: torch.device, params="shared", normalization="none",
        max_norm=1.0,
    ):
        super().__init__(n_tasks, device=device)
        self.solver = MinNormSolver()
        assert params in ["shared", "last", "rep"]
        self.params = params
        assert normalization in ["norm", "loss", "loss+", "none"]
        self.normalization = normalization
        self.max_norm = max_norm

    @staticmethod
    def _flattening(grad):
        return torch.cat(
            tuple(
                g.reshape(
                    -1,
                )
                for i, g in enumerate(grad)
            ),
            dim=0,
        )

    def get_weighted_loss(
        self,
        losses,
        shared_parameters=None,
        last_shared_parameters=None,
        representation=None,
        **kwargs,
    ):
        """

        Parameters
        ----------
        losses :
        shared_parameters :
        last_shared_parameters :
        representation :
        kwargs :

        Returns
        -------

        """
        # Our code
        grads = {}
        params = dict(
            rep=representation, shared=shared_parameters, last=last_shared_parameters
        )[self.params]
        for i, loss in enumerate(losses):
            g = list(
                torch.autograd.grad(
                    (loss + 1e-8).log(),
                    params,
                    retain_graph=True,
                )
            )
            # Normalize all gradients, this is optional and not included in the paper.

            grads[i] = [torch.flatten(grad) for grad in g]

        gn = gradient_normalizers(grads, losses, self.normalization)
        for t in range(self.n_tasks):
            for gr_i in range(len(grads[t])):
                grads[t][gr_i] = grads[t][gr_i] / gn[t]

        sol, min_norm = self.solver.find_min_norm_element(
            [grads[t] for t in range(len(grads))]
        )
        #sol = sol * self.n_tasks  # make sure it sums to self.n_tasks
        c = sum([ sol[i] / (losses[i] + 1e-8).detach() for i in range(len(sol))])
        weighted_loss = sum([(losses[i] + 1e-8).log() * sol[i] / c for i in range(len(sol))])
        return weighted_loss, dict(weights=torch.from_numpy(sol.astype(np.float32)))


class STL(WeightMethod):
    """Single task learning"""

    def __init__(self, n_tasks, device: torch.device, main_task):
        super().__init__(n_tasks, device=device)
        self.main_task = main_task
        self.weights = torch.zeros(n_tasks, device=device)
        self.weights[main_task] = 1.0

    def get_weighted_loss(self, losses: torch.Tensor, **kwargs):
        assert len(losses) == self.n_tasks
        loss = losses[self.main_task]

        return loss, dict(weights=self.weights)


class Uncertainty(WeightMethod):
    """Implementation of `Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics`
    Source: https://github.com/yaringal/multi-task-learning-example/blob/master/multi-task-learning-example-pytorch.ipynb
    """

    def __init__(self, n_tasks, device: torch.device):
        super().__init__(n_tasks, device=device)
        self.logsigma = torch.tensor([0.0] * n_tasks, device=device, requires_grad=True)

    def get_weighted_loss(self, losses: torch.Tensor, **kwargs):
        loss = sum(
            [
                0.5 * (torch.exp(-logs) * loss + logs)
                for loss, logs in zip(losses, self.logsigma)
            ]
        )

        return loss, dict(
            weights=torch.exp(-self.logsigma)
        )  # NOTE: not exactly task weights

    def parameters(self) -> List[torch.Tensor]:
        return [self.logsigma]


class UWCORR(WeightMethod):
    """
    Uncertainty Weighting with Correlation Coefficient (UWCORR)
    在UW方法基础上添加了任务相关系数惩罚项，参考CorrFAMO实现
    
    Formula:
        rho_i = Cov(l_i, L_uw) / (Std(l_i) * Std(L_uw))
        L_total = L_uw - lambda * sum(rho_i)
    """
    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        lambda_: float = 0.1,
        history_size: int = 10,
        eps: float = 1e-8,
        max_norm: float = 1.0,
    ):
        super().__init__(n_tasks, device=device)
        # UW Base Parameters
        self.logsigma = torch.tensor([0.0] * n_tasks, device=device, requires_grad=True)
        
        # CorrFAMO Parameters
        self.lambda_ = lambda_
        self.eps = eps
        self.max_norm = max_norm
        
        # History buffers
        self.loss_hist = deque(maxlen=history_size)
        self.weighted_loss_hist = deque(maxlen=history_size)

    def _calc_rho(self, current_losses, current_weighted_loss):
        """
        计算相关系数 rho_i = Cov(l_i, L) / (Std(l_i) * Std(L))
        参考CorrFAMO的实现
        """
        # 处理历史数据不足的情况
        if len(self.loss_hist) < 2:
            return torch.zeros(self.n_tasks, device=current_losses.device)

        # 准备数据：历史数据 + 当前数据
        h_losses = torch.stack(list(self.loss_hist), dim=0)
        h_L = torch.stack(list(self.weighted_loss_hist), dim=0)

        # 拼接当前步骤
        all_losses = torch.cat([h_losses, current_losses.unsqueeze(0)], dim=0)
        all_L = torch.cat([h_L, current_weighted_loss.unsqueeze(0)], dim=0)

        # 计算偏差（Delta）
        mean_losses = all_losses.mean(dim=0, keepdim=True)
        mean_L = all_L.mean(dim=0, keepdim=True)

        delta_li = all_losses - mean_losses
        delta_L = all_L - mean_L

        # 计算方差和协方差
        var_L = (delta_L ** 2).mean(dim=0)
        var_li = (delta_li ** 2).mean(dim=0)
        cov = (delta_li * delta_L.unsqueeze(1)).mean(dim=0)

        # 计算相关系数
        denominator = torch.sqrt(var_li * var_L) + self.eps
        rho = cov / denominator

        return rho

    def get_weighted_loss(self, losses: torch.Tensor, **kwargs):
        # UW Phase
        uw_loss = sum(
            [
                0.5 * (torch.exp(-logs) * loss + logs)
                for loss, logs in zip(losses, self.logsigma)
            ]
        )

        # CorrFAMO Phase
        rho = self._calc_rho(losses, uw_loss)
        corr_term = rho.sum()
        
        # 应用相关系数惩罚
        total_loss = uw_loss - self.lambda_ * corr_term

        # 更新历史
        self.loss_hist.append(losses.detach())
        self.weighted_loss_hist.append(uw_loss.detach())

        return total_loss, dict(
            weights=torch.exp(-self.logsigma),
            rho=rho.detach(),
            corr_term=corr_term.detach()
        )

    def parameters(self) -> List[torch.Tensor]:
        return [self.logsigma]


class PCGrad(WeightMethod):
    """Modification of: https://github.com/WeiChengTseng/Pytorch-PCGrad/blob/master/pcgrad.py

    @misc{Pytorch-PCGrad,
      author = {Wei-Cheng Tseng},
      title = {WeiChengTseng/Pytorch-PCGrad},
      url = {https://github.com/WeiChengTseng/Pytorch-PCGrad.git},
      year = {2020}
    }

    """

    def __init__(self, n_tasks: int, device: torch.device, reduction="sum"):
        super().__init__(n_tasks, device=device)
        assert reduction in ["mean", "sum"]
        self.reduction = reduction

    def get_weighted_loss(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        **kwargs,
    ):
        raise NotImplementedError

    def _set_pc_grads(self, losses, shared_parameters, task_specific_parameters=None):
        # shared part
        shared_grads = []
        for l in losses:
            shared_grads.append(
                torch.autograd.grad(l, shared_parameters, retain_graph=True)
            )

        if isinstance(shared_parameters, torch.Tensor):
            shared_parameters = [shared_parameters]
        non_conflict_shared_grads = self._project_conflicting(shared_grads)
        for p, g in zip(shared_parameters, non_conflict_shared_grads):
            p.grad = g

        # task specific part
        if task_specific_parameters is not None:
            task_specific_grads = torch.autograd.grad(
                losses.sum(), task_specific_parameters
            )
            if isinstance(task_specific_parameters, torch.Tensor):
                task_specific_parameters = [task_specific_parameters]
            for p, g in zip(task_specific_parameters, task_specific_grads):
                p.grad = g

    def _project_conflicting(self, grads: List[Tuple[torch.Tensor]]):
        pc_grad = copy.deepcopy(grads)
        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j = sum(
                    [
                        torch.dot(torch.flatten(grad_i), torch.flatten(grad_j))
                        for grad_i, grad_j in zip(g_i, g_j)
                    ]
                )
                if g_i_g_j < 0:
                    g_j_norm_square = (
                        torch.norm(torch.cat([torch.flatten(g) for g in g_j])) ** 2
                    )
                    for grad_i, grad_j in zip(g_i, g_j):
                        grad_i -= g_i_g_j * grad_j / g_j_norm_square

        merged_grad = [sum(g) for g in zip(*pc_grad)]
        if self.reduction == "mean":
            merged_grad = [g / self.n_tasks for g in merged_grad]

        return merged_grad

    def backward(
        self,
        losses: torch.Tensor,
        parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        **kwargs,
    ):
        self._set_pc_grads(losses, shared_parameters, task_specific_parameters)
        # make sure the solution for shared params has norm <= self.eps
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
        return None, {}  # NOTE: to align with all other weight methods


class CAGrad(WeightMethod):
    def __init__(self, n_tasks, device: torch.device, c=0.4, max_norm=1.0):
        super().__init__(n_tasks, device=device)
        self.c = c
        self.max_norm = max_norm

    def get_weighted_loss(
        self,
        losses,
        shared_parameters,
        **kwargs,
    ):
        """
        Parameters
        ----------
        losses :
        shared_parameters : shared parameters
        kwargs :
        Returns
        -------
        """
        # NOTE: we allow only shared params for now. Need to see paper for other options.
        grad_dims = []
        for param in shared_parameters:
            grad_dims.append(param.data.numel())
        grads = torch.Tensor(sum(grad_dims), self.n_tasks).to(self.device)

        for i in range(self.n_tasks):
            if i < self.n_tasks:
                losses[i].backward(retain_graph=True)
            else:
                losses[i].backward()
            self.grad2vec(shared_parameters, grads, grad_dims, i)
            # multi_task_model.zero_grad_shared_modules()
            for p in shared_parameters:
                p.grad = None

        g, GTG, w_cpu = self.cagrad(grads, alpha=self.c, rescale=1)
        self.overwrite_grad(shared_parameters, g, grad_dims)
        return GTG, w_cpu

    def cagrad(self, grads, alpha=0.5, rescale=1):
        GG = grads.t().mm(grads).cpu()  # [num_tasks, num_tasks]
        g0_norm = (GG.mean() + 1e-8).sqrt()  # norm of the average gradient

        x_start = np.ones(self.n_tasks) / self.n_tasks
        bnds = tuple((0, 1) for x in x_start)
        cons = {"type": "eq", "fun": lambda x: 1 - sum(x)}
        A = GG.numpy()
        b = x_start.copy()
        c = (alpha * g0_norm + 1e-8).item()

        def objfn(x):
            return (
                x.reshape(1, self.n_tasks).dot(A).dot(b.reshape(self.n_tasks, 1))
                + c
                * np.sqrt(
                    x.reshape(1, self.n_tasks).dot(A).dot(x.reshape(self.n_tasks, 1))
                    + 1e-8
                )
            ).sum()

        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        ww = torch.Tensor(w_cpu).to(grads.device)
        gw = (grads * ww.view(1, -1)).sum(1)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm + 1e-8)
        g = grads.mean(1) + lmbda * gw
        if rescale == 0:
            return g, GG.numpy(), w_cpu
        elif rescale == 1:
            return g / (1 + alpha ** 2), GG.numpy(), w_cpu
        else:
            return g / (1 + alpha), GG.numpy(), w_cpu

    @staticmethod
    def grad2vec(shared_params, grads, grad_dims, task):
        # store the gradients
        grads[:, task].fill_(0.0)
        cnt = 0
        # for mm in m.shared_modules():
        #     for p in mm.parameters():

        for param in shared_params:
            grad = param.grad
            if grad is not None:
                grad_cur = grad.data.detach().clone()
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[: cnt + 1])
                grads[beg:en, task].copy_(grad_cur.data.view(-1))
            cnt += 1

    def overwrite_grad(self, shared_parameters, newgrad, grad_dims):
        newgrad = newgrad * self.n_tasks  # to match the sum loss
        cnt = 0

        # for mm in m.shared_modules():
        #     for param in mm.parameters():
        for param in shared_parameters:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[: cnt + 1])
            this_grad = newgrad[beg:en].contiguous().view(param.data.size())
            param.grad = this_grad.data.clone()
            cnt += 1

    def backward(
        self,
        losses: torch.Tensor,
        parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        **kwargs,
    ):
        GTG, w = self.get_weighted_loss(losses, shared_parameters)
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
        return None, {"GTG": GTG, "weights": w}  # NOTE: to align with all other weight methods


class FairGrad(WeightMethod):
    def __init__(self, n_tasks, device: torch.device, alpha=1.0, max_norm=1.0):
        super().__init__(n_tasks, device=device)
        self.alpha = alpha
        self.max_norm = max_norm

    def get_weighted_loss(
        self,
        losses,
        shared_parameters,
        **kwargs,
    ):
        """
        Parameters
        ----------
        losses :
        shared_parameters : shared parameters
        kwargs :
        Returns
        -------
        """
        # NOTE: we allow only shared params for now. Need to see paper for other options.
        grad_dims = []
        for param in shared_parameters:
            grad_dims.append(param.data.numel())
        grads = torch.Tensor(sum(grad_dims), self.n_tasks).to(self.device)

        for i in range(self.n_tasks):
            if i < self.n_tasks - 1:
                losses[i].backward(retain_graph=True)
            else:
                losses[i].backward()
            self.grad2vec(shared_parameters, grads, grad_dims, i)
            # multi_task_model.zero_grad_shared_modules()
            for p in shared_parameters:
                p.grad = None

        g, GTG, w_cpu = self.fairgrad(grads, alpha=self.alpha)
        self.overwrite_grad(shared_parameters, g, grad_dims)
        return GTG, w_cpu

    def fairgrad(self, grads, alpha=1.0):
        GG = grads.t().mm(grads).cpu()  # [num_tasks, num_tasks]

        x_start = np.ones(self.n_tasks) / self.n_tasks
        A = GG.data.cpu().numpy()

        def objfn(x):
            # return np.power(np.dot(A, x), alpha) - 1 / x
            return np.dot(A, x) - np.power(1 / x, 1 / alpha)

        res = least_squares(objfn, x_start, bounds=(0, np.inf))
        w_cpu = res.x
        ww = torch.Tensor(w_cpu).to(grads.device)
        g = (grads * ww.view(1, -1)).sum(1)
        return g, GG.data.cpu().numpy(), w_cpu

    @staticmethod
    def grad2vec(shared_params, grads, grad_dims, task):
        # store the gradients
        grads[:, task].fill_(0.0)
        cnt = 0
        # for mm in m.shared_modules():
        #     for p in mm.parameters():

        for param in shared_params:
            grad = param.grad
            if grad is not None:
                grad_cur = grad.data.detach().clone()
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[: cnt + 1])
                grads[beg:en, task].copy_(grad_cur.data.view(-1))
            cnt += 1

    def overwrite_grad(self, shared_parameters, newgrad, grad_dims):
        newgrad = newgrad * self.n_tasks  # to match the sum loss
        cnt = 0

        # for mm in m.shared_modules():
        #     for param in mm.parameters():
        for param in shared_parameters:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[: cnt + 1])
            this_grad = newgrad[beg:en].contiguous().view(param.data.size())
            param.grad = this_grad.data.clone()
            cnt += 1

    def backward(
        self,
        losses: torch.Tensor,
        parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        **kwargs,
    ):
        GTG, w = self.get_weighted_loss(losses, shared_parameters)
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
        return None, {"GTG": GTG, "weights": w}  # NOTE: to align with all other weight methods


class GradDrop(WeightMethod):
    def __init__(self, n_tasks, device: torch.device, max_norm=1.0):
        super().__init__(n_tasks, device=device)
        self.max_norm = max_norm

    def get_weighted_loss(
        self,
        losses,
        shared_parameters,
        **kwargs,
    ):
        """
        Parameters
        ----------
        losses :
        shared_parameters : shared parameters
        kwargs :
        Returns
        -------
        """
        # NOTE: we allow only shared params for now. Need to see paper for other options.
        grad_dims = []
        for param in shared_parameters:
            grad_dims.append(param.data.numel())
        grads = torch.Tensor(sum(grad_dims), self.n_tasks).to(self.device)

        for i in range(self.n_tasks):
            if i < self.n_tasks:
                losses[i].backward(retain_graph=True)
            else:
                losses[i].backward()
            self.grad2vec(shared_parameters, grads, grad_dims, i)
            # multi_task_model.zero_grad_shared_modules()
            for p in shared_parameters:
                p.grad = None

        P = 0.5 * (1. + grads.sum(1) / (grads.abs().sum(1)+1e-8))
        U = torch.rand_like(grads[:,0])
        M = P.gt(U).view(-1,1)*grads.gt(0) + P.lt(U).view(-1,1)*grads.lt(0)
        g = (grads * M.float()).mean(1)
        self.overwrite_grad(shared_parameters, g, grad_dims)

    @staticmethod
    def grad2vec(shared_params, grads, grad_dims, task):
        # store the gradients
        grads[:, task].fill_(0.0)
        cnt = 0
        # for mm in m.shared_modules():
        #     for p in mm.parameters():

        for param in shared_params:
            grad = param.grad
            if grad is not None:
                grad_cur = grad.data.detach().clone()
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[: cnt + 1])
                grads[beg:en, task].copy_(grad_cur.data.view(-1))
            cnt += 1

    def overwrite_grad(self, shared_parameters, newgrad, grad_dims):
        newgrad = newgrad * self.n_tasks  # to match the sum loss
        cnt = 0

        # for mm in m.shared_modules():
        #     for param in mm.parameters():
        for param in shared_parameters:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[: cnt + 1])
            this_grad = newgrad[beg:en].contiguous().view(param.data.size())
            param.grad = this_grad.data.clone()
            cnt += 1

    def backward(
        self,
        losses: torch.Tensor,
        parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        **kwargs,
    ):
        #GTG, w = self.get_weighted_loss(losses, shared_parameters)
        self.get_weighted_loss(losses, shared_parameters)
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
        return None, None  # NOTE: to align with all other weight methods


class LOG_CAGrad(WeightMethod):
    def __init__(self, n_tasks, device: torch.device, c=0.4, max_norm=1.0):
        super().__init__(n_tasks, device=device)
        self.max_norm = max_norm
        self.c = c

    def get_weighted_loss(
        self,
        losses,
        shared_parameters,
        **kwargs,
    ):
        """
        Parameters
        ----------
        losses :
        shared_parameters : shared parameters
        kwargs :
        Returns
        -------
        """
        # NOTE: we allow only shared params for now. Need to see paper for other options.
        grad_dims = []
        for param in shared_parameters:
            grad_dims.append(param.data.numel())
        grads = torch.Tensor(sum(grad_dims), self.n_tasks).to(self.device)

        for i in range(self.n_tasks):
            if i < self.n_tasks:
                (losses[i].log()).backward(retain_graph=True)
            else:
                (losses[i].log()).backward()
            self.grad2vec(shared_parameters, grads, grad_dims, i)
            # multi_task_model.zero_grad_shared_modules()
            for p in shared_parameters:
                p.grad = None

        g, GTG, w_cpu = self.cagrad(grads, alpha=self.c, rescale=1)
        self.overwrite_grad(shared_parameters, g, grad_dims)
        #if self.max_norm > 0:
        #    torch.nn.utils.clip_grad_norm_(shared_parameters+task_specific_parameters, self.max_norm)
        return GTG, w_cpu

    def cagrad(self, grads, alpha=0.5, rescale=1):
        GG = grads.t().mm(grads).cpu()  # [num_tasks, num_tasks]
        g0_norm = (GG.mean() + 1e-8).sqrt()  # norm of the average gradient

        x_start = np.ones(self.n_tasks) / self.n_tasks
        bnds = tuple((0, 1) for x in x_start)
        cons = {"type": "eq", "fun": lambda x: 1 - sum(x)}
        A = GG.numpy()
        b = x_start.copy()
        c = (alpha * g0_norm + 1e-8).item()

        def objfn(x):
            return (
                x.reshape(1, self.n_tasks).dot(A).dot(b.reshape(self.n_tasks, 1))
                + c
                * np.sqrt(
                    x.reshape(1, self.n_tasks).dot(A).dot(x.reshape(self.n_tasks, 1))
                    + 1e-8
                )
            ).sum()

        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        ww = torch.Tensor(w_cpu).to(grads.device)
        gw = (grads * ww.view(1, -1)).sum(1)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm + 1e-8)
        g = grads.mean(1) + lmbda * gw
        if rescale == 0:
            return g, GG.numpy(), w_cpu
        elif rescale == 1:
            return g / (1 + alpha ** 2), GG.numpy(), w_cpu
        else:
            return g / (1 + alpha), GG.numpy(), w_cpu

    @staticmethod
    def grad2vec(shared_params, grads, grad_dims, task):
        # store the gradients
        grads[:, task].fill_(0.0)
        cnt = 0
        # for mm in m.shared_modules():
        #     for p in mm.parameters():

        for param in shared_params:
            grad = param.grad
            if grad is not None:
                grad_cur = grad.data.detach().clone()
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[: cnt + 1])
                grads[beg:en, task].copy_(grad_cur.data.view(-1))
            cnt += 1

    def overwrite_grad(self, shared_parameters, newgrad, grad_dims):
        newgrad = newgrad * self.n_tasks  # to match the sum loss
        cnt = 0

        # for mm in m.shared_modules():
        #     for param in mm.parameters():
        for param in shared_parameters:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[: cnt + 1])
            this_grad = newgrad[beg:en].contiguous().view(param.data.size())
            param.grad = this_grad.data.clone()
            cnt += 1

    def backward(
        self,
        losses: torch.Tensor,
        parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        **kwargs,
    ):
        GTG, w = self.get_weighted_loss(losses, shared_parameters)
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
        return None, {"GTG": GTG, "weights": w}  # NOTE: to align with all other weight methods


class RLW(WeightMethod):
    """Random loss weighting: https://arxiv.org/pdf/2111.10603.pdf"""

    def __init__(self, n_tasks, device: torch.device):
        super().__init__(n_tasks, device=device)

    def get_weighted_loss(self, losses: torch.Tensor, **kwargs):
        assert len(losses) == self.n_tasks
        weight = (F.softmax(torch.randn(self.n_tasks), dim=-1)).to(self.device)
        loss = torch.sum(losses * weight)

        return loss, dict(weights=weight)


class RLWCORR(WeightMethod):
    """
    Random Loss Weighting with Correlation Coefficient (RLWCORR)
    在RLW方法基础上添加了任务相关系数惩罚项，参考CorrFAMO实现
    
    Formula:
        rho_i = Cov(l_i, L_rlw) / (Std(l_i) * Std(L_rlw))
        L_total = L_rlw - lambda * sum(rho_i)
    """
    def __init__(
        self,
        n_tasks,
        device: torch.device,
        lambda_: float = 0.1,
        history_size: int = 10,
        eps: float = 1e-8,
        max_norm: float = 1.0,
    ):
        super().__init__(n_tasks, device=device)
        # CorrFAMO Parameters
        self.lambda_ = lambda_
        self.eps = eps
        self.max_norm = max_norm
        
        # History buffers
        self.loss_hist = deque(maxlen=history_size)
        self.weighted_loss_hist = deque(maxlen=history_size)

    def _calc_rho(self, current_losses, current_weighted_loss):
        """
        计算相关系数 rho_i = Cov(l_i, L) / (Std(l_i) * Std(L))
        参考CorrFAMO的实现
        """
        if len(self.loss_hist) < 2:
            return torch.zeros(self.n_tasks, device=current_losses.device)

        h_losses = torch.stack(list(self.loss_hist), dim=0)
        h_L = torch.stack(list(self.weighted_loss_hist), dim=0)

        all_losses = torch.cat([h_losses, current_losses.unsqueeze(0)], dim=0)
        all_L = torch.cat([h_L, current_weighted_loss.unsqueeze(0)], dim=0)

        mean_losses = all_losses.mean(dim=0, keepdim=True)
        mean_L = all_L.mean(dim=0, keepdim=True)

        delta_li = all_losses - mean_losses
        delta_L = all_L - mean_L

        var_L = (delta_L ** 2).mean(dim=0)
        var_li = (delta_li ** 2).mean(dim=0)
        cov = (delta_li * delta_L.unsqueeze(1)).mean(dim=0)

        denominator = torch.sqrt(var_li * var_L) + self.eps
        rho = cov / denominator

        return rho

    def get_weighted_loss(self, losses: torch.Tensor, **kwargs):
        assert len(losses) == self.n_tasks
        weight = (F.softmax(torch.randn(self.n_tasks), dim=-1)).to(self.device)
        rlw_loss = torch.sum(losses * weight)
        
        rho = self._calc_rho(losses, rlw_loss)
        corr_term = rho.sum()
        
        total_loss = rlw_loss - self.lambda_ * corr_term

        self.loss_hist.append(losses.detach())
        self.weighted_loss_hist.append(rlw_loss.detach())

        return total_loss, dict(
            weights=weight,
            rho=rho.detach(),
            corr_term=corr_term.detach()
        )


class IMTLG(WeightMethod):
    """TOWARDS IMPARTIAL MULTI-TASK LEARNING: https://openreview.net/pdf?id=IMPnRXEWpvr"""

    def __init__(self, n_tasks, device: torch.device):
        super().__init__(n_tasks, device=device)

    def get_weighted_loss(
        self,
        losses,
        shared_parameters,
        **kwargs,
    ):
        grads = {}
        norm_grads = {}

        for i, loss in enumerate(losses):
            g = list(
                torch.autograd.grad(
                    loss,
                    shared_parameters,
                    retain_graph=True,
                )
            )
            grad = torch.cat([torch.flatten(grad) for grad in g])
            norm_term = torch.norm(grad)

            grads[i] = grad
            norm_grads[i] = grad / norm_term

        G = torch.stack(tuple(v for v in grads.values()))
        GTG = torch.mm(G, G.t())

        D = (
            G[
                0,
            ]
            - G[
                1:,
            ]
        )

        U = torch.stack(tuple(v for v in norm_grads.values()))
        U = (
            U[
                0,
            ]
            - U[
                1:,
            ]
        )
        first_element = torch.matmul(
            G[
                0,
            ],
            U.t(),
        )
        try:
            second_element = torch.inverse(torch.matmul(D, U.t()))
        except:
            # workaround for cases where matrix is singular
            second_element = torch.inverse(
                torch.eye(self.n_tasks - 1, device=self.device) * 1e-8
                + torch.matmul(D, U.t())
            )

        alpha_ = torch.matmul(first_element, second_element)
        alpha = torch.cat(
            (torch.tensor(1 - alpha_.sum(), device=self.device).unsqueeze(-1), alpha_)
        )

        loss = torch.sum(losses * alpha)
        extra_outputs = {}
        extra_outputs["weights"] = alpha
        extra_outputs["GTG"] = GTG.detach().cpu().numpy()
        return loss, extra_outputs


class LOG_IMTLG(WeightMethod):
    """TOWARDS IMPARTIAL MULTI-TASK LEARNING: https://openreview.net/pdf?id=IMPnRXEWpvr"""

    def __init__(self, n_tasks, device: torch.device):
        super().__init__(n_tasks, device=device)

    def get_weighted_loss(
        self,
        losses,
        shared_parameters,
        **kwargs,
    ):
        grads = {}
        norm_grads = {}

        for i, loss in enumerate(losses):
            g = list(
                torch.autograd.grad(
                    (loss + EPS).log(),
                    shared_parameters,
                    retain_graph=True,
                )
            )
            grad = torch.cat([torch.flatten(grad) for grad in g])
            norm_term = torch.norm(grad)

            grads[i] = grad
            norm_grads[i] = grad / norm_term

        G = torch.stack(tuple(v for v in grads.values()))
        GTG = torch.mm(G, G.t())

        D = (
            G[
                0,
            ]
            - G[
                1:,
            ]
        )

        U = torch.stack(tuple(v for v in norm_grads.values()))
        U = (
            U[
                0,
            ]
            - U[
                1:,
            ]
        )
        first_element = torch.matmul(
            G[
                0,
            ],
            U.t(),
        )
        try:
            second_element = torch.inverse(torch.matmul(D, U.t()))
        except:
            # workaround for cases where matrix is singular
            second_element = torch.inverse(
                torch.eye(self.n_tasks - 1, device=self.device) * 1e-8
                + torch.matmul(D, U.t())
            )

        alpha_ = torch.matmul(first_element, second_element)
        alpha = torch.cat(
            (torch.tensor(1 - alpha_.sum(), device=self.device).unsqueeze(-1), alpha_)
        )

        loss = torch.sum((losses + EPS).log() * alpha)
        extra_outputs = {}
        extra_outputs["weights"] = alpha
        extra_outputs["GTG"] = GTG.detach().cpu().numpy()
        return loss, extra_outputs


class DynamicWeightAverage(WeightMethod):
    """Dynamic Weight Average from `End-to-End Multi-Task Learning with Attention`.
    Modification of: https://github.com/lorenmt/mtan/blob/master/im2im_pred/model_segnet_split.py#L242
    """

    def __init__(
        self, n_tasks, device: torch.device, iteration_window: int = 25, temp=2.0
    ):
        """

        Parameters
        ----------
        n_tasks :
        iteration_window : 'iteration' loss is averaged over the last 'iteration_window' losses
        temp :
        """
        super().__init__(n_tasks, device=device)
        self.iteration_window = iteration_window
        self.temp = temp
        self.running_iterations = 0
        self.costs = np.ones((iteration_window * 2, n_tasks), dtype=np.float32)
        self.weights = np.ones(n_tasks, dtype=np.float32)

    def get_weighted_loss(self, losses, **kwargs):

        cost = losses.detach().cpu().numpy()

        # update costs - fifo
        self.costs[:-1, :] = self.costs[1:, :]
        self.costs[-1, :] = cost

        if self.running_iterations > self.iteration_window:
            ws = self.costs[self.iteration_window :, :].mean(0) / self.costs[
                : self.iteration_window, :
            ].mean(0)
            self.weights = (self.n_tasks * np.exp(ws / self.temp)) / (
                np.exp(ws / self.temp)
            ).sum()

        task_weights = torch.from_numpy(self.weights.astype(np.float32)).to(
            losses.device
        )
        loss = (task_weights * losses).mean()

        self.running_iterations += 1

        return loss, dict(weights=task_weights)


class DWACORR(WeightMethod):
    """
    Dynamic Weight Average with Correlation Coefficient (DWACORR)
    在DWA方法基础上添加了任务相关系数惩罚项，参考CorrFAMO实现
    
    Formula:
        rho_i = Cov(l_i, L_dwa) / (Std(l_i) * Std(L_dwa))
        L_total = L_dwa - lambda * sum(rho_i)
    """
    def __init__(
        self,
        n_tasks,
        device: torch.device,
        iteration_window: int = 25,
        temp=2.0,
        lambda_: float = 0.1,
        history_size: int = 10,
        eps: float = 1e-8,
        max_norm: float = 1.0,
    ):
        super().__init__(n_tasks, device=device)
        self.iteration_window = iteration_window
        self.temp = temp
        self.running_iterations = 0
        self.costs = np.ones((iteration_window * 2, n_tasks), dtype=np.float32)
        self.weights = np.ones(n_tasks, dtype=np.float32)
        
        # CorrFAMO Parameters
        self.lambda_ = lambda_
        self.eps = eps
        self.max_norm = max_norm
        
        # History buffers
        self.loss_hist = deque(maxlen=history_size)
        self.weighted_loss_hist = deque(maxlen=history_size)

    def _calc_rho(self, current_losses, current_weighted_loss):
        """
        计算相关系数 rho_i = Cov(l_i, L) / (Std(l_i) * Std(L))
        参考CorrFAMO的实现
        """
        if len(self.loss_hist) < 2:
            return torch.zeros(self.n_tasks, device=current_losses.device)

        h_losses = torch.stack(list(self.loss_hist), dim=0)
        h_L = torch.stack(list(self.weighted_loss_hist), dim=0)

        all_losses = torch.cat([h_losses, current_losses.unsqueeze(0)], dim=0)
        all_L = torch.cat([h_L, current_weighted_loss.unsqueeze(0)], dim=0)

        mean_losses = all_losses.mean(dim=0, keepdim=True)
        mean_L = all_L.mean(dim=0, keepdim=True)

        delta_li = all_losses - mean_losses
        delta_L = all_L - mean_L

        var_L = (delta_L ** 2).mean(dim=0)
        var_li = (delta_li ** 2).mean(dim=0)
        cov = (delta_li * delta_L.unsqueeze(1)).mean(dim=0)

        denominator = torch.sqrt(var_li * var_L) + self.eps
        rho = cov / denominator

        return rho

    def get_weighted_loss(self, losses, **kwargs):
        cost = losses.detach().cpu().numpy()

        # update costs - fifo
        self.costs[:-1, :] = self.costs[1:, :]
        self.costs[-1, :] = cost

        if self.running_iterations > self.iteration_window:
            ws = self.costs[self.iteration_window :, :].mean(0) / self.costs[
                : self.iteration_window, :
            ].mean(0)
            self.weights = (self.n_tasks * np.exp(ws / self.temp)) / (
                np.exp(ws / self.temp)
            ).sum()

        task_weights = torch.from_numpy(self.weights.astype(np.float32)).to(
            losses.device
        )
        dwa_loss = (task_weights * losses).mean()
        
        rho = self._calc_rho(losses, dwa_loss)
        corr_term = rho.sum()
        
        total_loss = dwa_loss - self.lambda_ * corr_term

        self.loss_hist.append(losses.detach())
        self.weighted_loss_hist.append(dwa_loss.detach())

        self.running_iterations += 1

        return total_loss, dict(
            weights=task_weights,
            rho=rho.detach(),
            corr_term=corr_term.detach()
        )


class WeightMethods:
    def __init__(self, method: str, n_tasks: int, device: torch.device, **kwargs):
        """
        :param method:
        """
        assert method in list(METHODS.keys()), f"unknown method {method}."

        self.method = METHODS[method](n_tasks=n_tasks, device=device, **kwargs)

    def get_weighted_loss(self, losses, **kwargs):
        return self.method.get_weighted_loss(losses, **kwargs)

    def backward(
        self, losses, **kwargs
    ) -> Tuple[Union[torch.Tensor, None], Union[Dict, None]]:
        return self.method.backward(losses, **kwargs)

    def __ceil__(self, losses, **kwargs):
        return self.backward(losses, **kwargs)

    def parameters(self):
        return self.method.parameters()


class FastFairGrad(WeightMethod):
    def __init__(self, n_tasks, device: torch.device, alpha=1.0, max_norm=1.0):
        super().__init__(n_tasks, device=device)
        self.alpha = alpha
        self.max_norm = max_norm
        self.prev_losses = None  # 用于存储上一步的损失值
        
    def get_weighted_loss(
        self,
        losses,
        shared_parameters,
        **kwargs,
    ):
        """
        根据公式 G^T G w = w^{-1/alpha} 直接求解任务权重
        
        Parameters
        ----------
        losses : 当前任务损失
        shared_parameters : 共享参数
        kwargs : 其他参数，包含学习率lr
        
        Returns
        -------
        加权损失, 额外输出
        """
        # 初始化上一步损失（如果是第一次迭代）
        if self.prev_losses is None:
            self.prev_losses = losses.clone().detach()
            # 第一次迭代时，使用均匀权重
            weights = torch.ones(self.n_tasks, device=self.device) / self.n_tasks
            weighted_loss = torch.sum(losses * weights)
            return weighted_loss, {"weights": weights}
        
        # 获取学习率，如果kwargs中没有则使用默认值0.001
        lr = kwargs.get('lr', 0.001)
        
        # 计算损失差并除以学习率作为G^T G w的近似
        # 添加负号因为损失增加通常意味着梯度方向与参数更新相反
        loss_diffs = -(losses - self.prev_losses) / lr
        
        # 直接根据公式求解权重: w = (loss_diffs)^(-alpha)
        # 这里假设G^T G w ≈ loss_diffs，所以 w ≈ (loss_diffs)^(-alpha)
        with torch.no_grad():
            # 添加小值避免除零
            loss_diffs = loss_diffs.clamp(min=1e-8)
            # 根据公式直接计算权重
            weights = torch.pow(loss_diffs, -self.alpha)
            # 归一化权重确保它们的和为1
            weights = weights / weights.sum()
        
        # 计算加权损失
        weighted_loss = torch.sum(losses * weights)
        
        # 更新上一步损失
        self.prev_losses = losses.clone().detach()
        
        return weighted_loss, {"weights": weights}
    
    def backward(
        self,
        losses: torch.Tensor,
        parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        last_shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        representation: Union[torch.nn.parameter.Parameter, torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, None], Union[dict, None]]:
        """
        执行反向传播
        """
        loss, extra_outputs = self.get_weighted_loss(
            losses=losses,
            shared_parameters=shared_parameters,
            **kwargs,
        )

        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)

        loss.backward()
        return loss, extra_outputs


class Router(nn.Module):
    """轻量级路由网络，用于基于共享特征动态生成任务权重"""
    def __init__(self, in_features, n_tasks):
        super(Router, self).__init__()
        # 简单MLP结构 - 初始设置，会在第一次前向传播时根据实际输入维度调整
        self.fc = nn.Linear(in_features, n_tasks)
        self.in_features = in_features
        self.n_tasks = n_tasks
        self.adjusted = False
    
    def forward(self, shared_feat):
        # 计算特征统计量作为路由网络输入
        # 处理shared_feat可能是元组的情况
        if isinstance(shared_feat, tuple):
            # 取元组中的第一个元素作为特征张量
            shared_feat = shared_feat[0]
        
        # 检查是否为张量
        if not isinstance(shared_feat, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor or tuple containing torch.Tensor, got {type(shared_feat).__name__}")
            
        # 处理多维特征
        if shared_feat.dim() > 2:
            feat_stats = torch.mean(shared_feat, dim=[2, 3]) if shared_feat.dim() == 4 else torch.mean(shared_feat, dim=1)
        else:
            feat_stats = shared_feat
        
        # 动态调整线性层以匹配实际输入特征维度
        if not self.adjusted:
            actual_in_dim = feat_stats.size(1)
            if actual_in_dim != self.in_features:
                print(f"Adjusting router input dimension from {self.in_features} to {actual_in_dim}")
                # 创建新的线性层以匹配实际输入维度
                new_fc = nn.Linear(actual_in_dim, self.n_tasks).to(feat_stats.device)
                # 复制旧权重的相关部分（如果可能）
                with torch.no_grad():
                    min_dim = min(self.in_features, actual_in_dim)
                    new_fc.weight[:, :min_dim] = self.fc.weight[:, :min_dim]
                    if min_dim < actual_in_dim:
                        # 初始化额外的权重
                        nn.init.xavier_uniform_(new_fc.weight[:, min_dim:])
                    # 复制偏置
                    new_fc.bias = self.fc.bias
                # 替换线性层
                self.fc = new_fc
                self.in_features = actual_in_dim
                self.adjusted = True
        
        # 输出任务权重，使用softmax确保非负和为1
        return F.softmax(self.fc(feat_stats), dim=-1)


class LDCMTL(WeightMethod):
    """LDC-MTL: Loss Discrepancy Control for Multi-Task Learning
    实现论文中的双层优化结构和损失差异控制机制
    
    参考论文: Loss Discrepancy Control for Multi-Task Learning (Section 4)
    注意: lambda_param需要根据具体任务调整，论文推荐值: CelebA (0.01), Cityscapes (0.1)
    """
    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        lambda_param: float = 0.1,  # 权衡参数λ，根据论文建议调整
        max_norm: float = 1.0,
        feat_dim: int = 512,        # 共享特征维度
        tau_mode: str = "ones",    # 损失差异控制中的τ选择: "ones"或"sigma"
    ):
        super().__init__(n_tasks, device=device)
        self.lambda_param = lambda_param
        self.max_norm = max_norm
        self.tau_mode = tau_mode
        assert tau_mode in ["ones", "sigma"], "tau_mode must be 'ones' or 'sigma'"
        
        # 创建路由网络（核心创新点）
        self.router = Router(feat_dim, n_tasks).to(device)
        
        # 初始化历史损失记录
        self.prev_losses = None
    
    def parameters(self):
        """返回路由网络的参数，使训练器能够将这些参数与主网络参数一起优化
        这符合算法公式中路由网络参数x与主网络参数W使用相同学习率优化的要求
        """
        return self.router.parameters()
    
    def get_weighted_loss(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        task_specific_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        last_shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        representation: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        **kwargs,
    ):
        # 确保representation不为None（用于路由网络）
        if representation is None:
            raise ValueError("LDC-MTL requires representation for router network")
        
        # 通过路由网络动态生成任务权重（核心创新点）
        batch_weights = self.router(representation)
        
        # 根据论文公式(1)和(2)实现
        
        # 下层优化目标：加权和损失 g(W, x) = sum_i σ_i(W) l_i(x)
        # 更精确的实现：先样本内加权，再batch平均
        if losses.dim() == 1:  # [K] - 已对batch平均的损失
            # 对于已平均的损失，使用批次权重的平均值
            weights = torch.mean(batch_weights, dim=0)
            weighted_loss = torch.sum(losses * weights)
        else:  # [B, K] - per-sample损失
            # 样本级加权，更符合论文公式
            weighted_loss = (batch_weights * losses).mean()
        
        # 上层优化目标：任务间损失差异控制 f(W, x^*) = sum_i ||τ_i l_i - τ_{i+1} l_{i+1}||
        # 确定τ值：论文提供两种选项
        if self.tau_mode == "sigma":
            # 选项(i): τ = σ(W)，但detach避免引入二阶梯度
            tau = weights.detach() if losses.dim() == 1 else torch.mean(batch_weights, dim=0).detach()
        else:  # self.tau_mode == "ones"
            # 选项(ii): τ = 1
            tau = torch.ones_like(losses) if losses.dim() == 1 else torch.ones_like(losses[0])
        
        # 计算相邻任务间的损失差异
        discrepancy_loss = 0.0
        for i in range(len(losses) - 1):
            if losses.dim() == 1:  # 已平均的损失
                discrepancy_loss += torch.abs(tau[i] * losses[i] - tau[i+1] * losses[i+1])
            else:  # per-sample损失
                # 对每个样本计算差异，再平均
                sample_diff = torch.abs(tau[i] * losses[:, i] - tau[i+1] * losses[:, i+1])
                discrepancy_loss += sample_diff.mean()
        
        # 总损失：下层损失 + λ * 上层损失差异控制
        total_loss = weighted_loss + self.lambda_param * discrepancy_loss
        
        # 返回平均权重供分析使用
        weights = torch.mean(batch_weights, dim=0)
        
        return total_loss, {"weights": weights, "batch_weights": batch_weights, "tau_mode": self.tau_mode}
    
    def update(self, losses):
        # 路由网络参数已通过反向传播更新，这里只需保存当前损失用于分析
        self.prev_losses = losses.detach()
    
    def backward(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        task_specific_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        last_shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        representation: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, None], Union[dict, None]]:
        # 计算总损失（包含下层和上层目标）
        loss, extra_outputs = self.get_weighted_loss(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            last_shared_parameters=last_shared_parameters,
            representation=representation,
            **kwargs,
        )
        
        # 执行反向传播 - 同时计算主干网络和路由网络的梯度
        # 路由网络参数将通过主网络的优化器更新
        loss.backward()
        
        # 应用梯度裁剪
        if self.max_norm > 0:
            if shared_parameters is not None:
                torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
            # 对路由网络参数也进行梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.router.parameters(), self.max_norm)
        
        # 保存当前损失
        self.update(losses)
        
        return loss, extra_outputs
    
    def parameters(self) -> List[torch.Tensor]:
        """返回路由网络的可学习参数"""
        return self.router.parameters()


class LDCMTLNew1(WeightMethod):
    """LDC-MTL New1: 使用基于统计量的差异度量方法
    
    本类实现了LDC-MTL的改进版本，将原文中的成对差方法替换为基于损失分布的方差度量：
    f(W,x)=Var({τ_i l_i (x)}_i=1^K)=1/K ∑_{i=1}^K (τ_i l_i (x)−ˉl_τ )^2
    其中 ˉl_τ=1/K ∑_i τ_i l_i (x)
    
    优点：
    1. 不依赖任务排序，适用于无天然顺序的任务（如CelebA的40个人脸属性）
    2. 考虑所有任务之间的相互关系，更好地捕捉全局结构
    3. O(K)计算复杂度，与原方法相同
    4. 平滑可微，便于优化
    5. 直接度量"离散程度"，符合"均衡"直觉
    
    参数:
    lambda_param: 权衡参数λ，控制差异度量的权重
    max_norm: 梯度裁剪的最大范数
    feat_dim: 共享特征维度
    tau_mode: τ选择模式: "ones"或"sigma"
    """
    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        lambda_param: float = 0.1,  # 权衡参数λ，根据任务调整
        max_norm: float = 1.0,
        feat_dim: int = 512,        # 共享特征维度
        tau_mode: str = "ones",    # τ选择模式: "ones"或"sigma"
    ):
        super().__init__(n_tasks, device=device)
        self.lambda_param = lambda_param
        self.max_norm = max_norm
        self.tau_mode = tau_mode
        assert tau_mode in ["ones", "sigma"], "tau_mode must be 'ones' or 'sigma'"
        
        # 创建路由网络，用于动态生成任务权重
        self.router = Router(feat_dim, n_tasks).to(device)
        
        # 初始化历史损失记录
        self.prev_losses = None
    
    def parameters(self):
        """返回路由网络的参数，使训练器能够将这些参数与主网络参数一起优化"""
        return self.router.parameters()
    
    def get_weighted_loss(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        task_specific_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        last_shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        representation: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        **kwargs,
    ):
        # 确保representation不为None（用于路由网络）
        if representation is None:
            raise ValueError("LDC-MTL New1 requires representation for router network")
        
        # 通过路由网络动态生成任务权重
        batch_weights = self.router(representation)
        
        # 下层优化目标：加权和损失 g(W, x) = sum_i σ_i(W) l_i(x)
        if losses.dim() == 1:  # [K] - 已对batch平均的损失
            # 对于已平均的损失，使用批次权重的平均值
            weights = torch.mean(batch_weights, dim=0)
            weighted_loss = torch.sum(losses * weights)
        else:  # [B, K] - per-sample损失
            # 样本级加权，更符合论文公式
            weighted_loss = (batch_weights * losses).mean()
        
        # 上层优化目标：基于统计量的差异度量 f(W, x) = Var({τ_i l_i (x)}_i=1^K)
        # 确定τ值
        if self.tau_mode == "sigma":
            # 选项(i): τ = σ(W)，但detach避免引入二阶梯度
            tau = weights.detach() if losses.dim() == 1 else torch.mean(batch_weights, dim=0).detach()
        else:  # self.tau_mode == "ones"
            # 选项(ii): τ = 1
            tau = torch.ones_like(losses) if losses.dim() == 1 else torch.ones_like(losses[0])
        
        # 计算基于方差的差异度量
        if losses.dim() == 1:  # 已平均的损失
            # 计算τ_i * l_i
            tau_l = tau * losses
            # 计算平均损失 ˉl_τ = 1/K ∑_i τ_i l_i
            mean_tau_l = torch.mean(tau_l)
            # 计算方差 Var({τ_i l_i}) = 1/K ∑_i (τ_i l_i - ˉl_τ)^2
            discrepancy_loss = torch.mean((tau_l - mean_tau_l) ** 2)
        else:  # per-sample损失
            # 对每个样本计算差异，再平均
            # 计算τ_i * l_i(x) for each sample
            tau_l = torch.einsum('bk,k->bk', losses, tau)  # [B, K]
            # 计算每个样本的平均损失 ˉl_τ(x) = 1/K ∑_i τ_i l_i(x)
            mean_tau_l = torch.mean(tau_l, dim=1, keepdim=True)  # [B, 1]
            # 计算每个样本的方差，再平均
            sample_variance = torch.mean((tau_l - mean_tau_l) ** 2, dim=1)  # [B]
            discrepancy_loss = torch.mean(sample_variance)  # 标量
        
        # 总损失：下层损失 + λ * 上层损失差异控制
        total_loss = weighted_loss + self.lambda_param * discrepancy_loss
        
        # 返回平均权重供分析使用
        weights = torch.mean(batch_weights, dim=0)
        
        return total_loss, {"weights": weights, "batch_weights": batch_weights, "tau_mode": self.tau_mode}
    
    def update(self, losses):
        # 路由网络参数已通过反向传播更新，这里只需保存当前损失用于分析
        self.prev_losses = losses.detach()
    
    def backward(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        task_specific_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        last_shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        representation: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, None], Union[dict, None]]:
        # 计算总损失（包含下层和上层目标）
        loss, extra_outputs = self.get_weighted_loss(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            last_shared_parameters=last_shared_parameters,
            representation=representation,
            **kwargs,
        )
        
        # 执行反向传播 - 同时计算主干网络和路由网络的梯度
        loss.backward()
        
        # 应用梯度裁剪
        if self.max_norm > 0:
            if shared_parameters is not None:
                torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
            # 对路由网络参数也进行梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.router.parameters(), self.max_norm)
        
        # 保存当前损失
        self.update(losses)
        
        return loss, extra_outputs


class LDCNew2(WeightMethod):
    """LDC-MTL New2: 使用聚类和线性规划优化任务权重
    
    本方法首先按照GO4Align的方式计算gamma，然后对各任务的gamma进行K-means聚类，
    对各组的聚类中心使用线性规划方式计算组权重，最终组合成总损失。
    
    算法流程:
    1. 计算gamma: scale = sum(losses) / losses
    2. 更新adv_probs: adv_probs = adv_probs * exp(-robust_step_size * losses)
    3. 归一化adv_probs
    4. 计算权重: weight = scale * adv_probs
    5. 对权重进行K-means聚类
    6. 对各组聚类中心使用LP计算组权重
    7. 组合最终损失: L = sum(group_weights * tasks_in_group_losses)
    
    参数:
    num_groups: 聚类数量
    robust_step_size: GO4Align中的稳健步长
    max_norm: 梯度裁剪的最大范数
    """
    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        num_groups: int = 2,
        robust_step_size: float = 0.0001,
        max_norm: float = 1.0,
    ):
        super().__init__(n_tasks, device=device)
        self.n_tasks = n_tasks
        self.device = device
        self.max_norm = max_norm
        self.num_groups = num_groups
        self.robust_step_size = robust_step_size
        
        # 初始化adv_probs
        self.adv_probs = torch.ones(n_tasks).to(device) / n_tasks
    
    def solve_group_linear_programming(self, cluster_centers: torch.Tensor) -> np.ndarray:
        """
        使用线性规划求解各组的最优权重
        
        Args:
            cluster_centers: 聚类中心 [G] 其中G是组数
            
        Returns:
            最优组权重 [G]
        """
        G = len(cluster_centers)
        
        # 如果只有一组，直接返回均匀权重
        if G <= 1:
            return np.ones(G) / G
        
        # 构建线性规划问题
        # 变量: [w_1, w_2, ..., w_G, u_1, u_2, ..., u_{G-1}]
        # 总变量数: G + (G-1) = 2G-1
        
        # 目标函数: min ∑_{i=1}^{G-1} u_i
        c = np.zeros(2 * G - 1)
        c[G:2*G-1] = 1.0  # u_i的系数为1
        
        # 等式约束: ∑_{i=1}^{G} w_i = 1
        A_eq = np.zeros((1, 2 * G - 1))
        A_eq[0, :G] = 1.0
        b_eq = np.array([1.0])
        
        # 不等式约束矩阵
        A_ub = []
        b_ub = []
        
        # 约束: u_i ≥ w_i * c_i - w_{i+1} * c_{i+1}
        # 即: -w_i * c_i + w_{i+1} * c_{i+1} + u_i ≥ 0
        for i in range(G-1):
            row = np.zeros(2 * G - 1)
            row[i] = -cluster_centers[i].item()
            row[i+1] = cluster_centers[i+1].item()
            row[G + i] = 1.0  # u_i
            A_ub.append(row)
            b_ub.append(0.0)
        
        # 约束: u_i ≥ -(w_i * c_i - w_{i+1} * c_{i+1})
        # 即: w_i * c_i - w_{i+1} * c_{i+1} + u_i ≥ 0
        for i in range(G-1):
            row = np.zeros(2 * G - 1)
            row[i] = cluster_centers[i].item()
            row[i+1] = -cluster_centers[i+1].item()
            row[G + i] = 1.0  # u_i
            A_ub.append(row)
            b_ub.append(0.0)
        
        # 下界: w_i ≥ 0, u_i ≥ 0
        bounds = [(0, None)] * (2 * G - 1)
        
        # 求解线性规划
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                           bounds=bounds, method='highs')
            
            if result.success:
                # 返回最优组权重
                optimal_weights = result.x[:G]
                return optimal_weights
            else:
                # 如果LP求解失败，使用均匀权重作为fallback
                print(f"Warning: LP求解失败，使用均匀组权重. Status: {result.message}")
                return np.ones(G) / G
                
        except Exception as e:
            print(f"Warning: LP求解异常，使用均匀组权重. Error: {e}")
            return np.ones(G) / G
    
    def update_adv_probs(self, losses: torch.Tensor):
        """
        按GO4ALIGN方式更新adv_probs
        
        Args:
            losses: 当前任务损失 [K]
        """
        # adv_probs = adv_probs * exp(-robust_step_size * losses)
        self.adv_probs = self.adv_probs * torch.exp(-self.robust_step_size * losses)
        
        # 归一化adv_probs
        self.adv_probs = self.adv_probs / torch.sum(self.adv_probs)
    
    def calculate_gamma(self, losses: torch.Tensor, adv_probs: torch.Tensor) -> torch.Tensor:
        """
        按GO4ALIGN方式计算gamma值
        
        Args:
            losses: 任务损失 [K]
            adv_probs: 对抗概率 [K]
            
        Returns:
            gamma值 [K]
        """
        # 计算scale：sum(losses) / losses
        eps = 1e-8
        scale = torch.sum(losses) / (losses + eps)
        
        # 计算gamma：scale * adv_probs
        gamma = scale * adv_probs
        
        return gamma
    
    def get_weighted_loss(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        task_specific_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        last_shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        representation: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        **kwargs,
    ):
        # 确保losses是正确的形状
        if losses.dim() == 0:
            raise ValueError("LDC-New2 requires losses with dimension at least 1")
        elif losses.dim() == 1:
            losses_1d = losses
        else:
            # 如果是多维的，取平均
            losses_1d = losses.mean(dim=0)
        
        K = len(losses_1d)
        
        # 1. 更新adv_probs（按GO4ALIGN方式）
        self.update_adv_probs(losses_1d)
        
        # 2. 计算gamma值（按GO4ALIGN方式）
        gamma_values = self.calculate_gamma(losses_1d, self.adv_probs)
        
        # 3. 对gamma进行kmeans聚类
        gamma_array = gamma_values.detach().cpu().numpy().reshape(-1, 1)
        try:
            # 使用kmeans_pytorch进行聚类
            cluster_ids, cluster_centers = kmeans(
                X=torch.from_numpy(gamma_array), 
                num_clusters=min(self.num_groups, K), 
                distance='euclidean', 
                device=losses.device
            )
        except Exception as e:
            print(f"Warning: K-means聚类失败，使用均匀分组. Error: {e}")
            # 如果聚类失败，将所有任务分配到同一组
            cluster_centers = gamma_values.mean().unsqueeze(0)
            cluster_ids = torch.zeros(K, dtype=torch.long, device=losses.device)
            group_weights = np.ones(1) / 1.0
        else:
            # 4. 对各组聚类中心使用线性规划计算权重
            group_weights = self.solve_group_linear_programming(cluster_centers)
        
        # 5. 构建最终的任务权重
        final_weights = torch.zeros(K).to(losses.device)
        
        if len(cluster_centers) == 1:
            # 如果只有一组，所有任务使用相同的组权重
            final_weights = torch.ones(K).to(losses.device) * group_weights[0]
        else:
            # 为每个任务分配对应的组权重
            for task_idx in range(K):
                cluster_id = cluster_ids[task_idx].item()
                final_weights[task_idx] = group_weights[cluster_id]
        
        # 6. 计算总损失：L = ∑ w_i * l_i
        total_loss = torch.sum(final_weights * losses_1d)
        
        return total_loss, {
            "weights": final_weights,
            "gamma_values": gamma_values,
            "cluster_centers": cluster_centers,
            "group_weights": torch.tensor(group_weights, device=losses.device),
            "cluster_ids": cluster_ids
        }
    
    def backward(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        task_specific_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        last_shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        representation: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, None], Union[dict, None]]:
        # 计算总损失
        loss, extra_outputs = self.get_weighted_loss(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            last_shared_parameters=last_shared_parameters,
            representation=representation,
            **kwargs,
        )
        
        # 执行反向传播
        loss.backward()
        
        # 应用梯度裁剪
        if self.max_norm > 0 and shared_parameters is not None:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
        
        return loss, extra_outputs
    
    def __repr__(self):
        return (f"LDCNew2(n_tasks={self.n_tasks}, num_groups={self.num_groups}, "
                f"robust_step_size={self.robust_step_size}, max_norm={self.max_norm})")


class LDCNew3(WeightMethod):
    """LDC-MTL New3: 与LDCMTL结构一致，使用平滑目标函数和softmax权重
    
    本方法与LDCMTL基本一致，区别在于：
    1. 使用平滑目标函数替代绝对值目标函数
    2. 使用sigma=softmax(omega)替代路由网络
    
    算法流程:
    1. 初始化可学习参数 ω∈R^K (初始化为全1)
    2. 通过softmax计算权重：σ = softmax(ω)
    3. 使用平滑目标函数作为上层优化目标：
       f_ε(σ) = √[∑_{i=1}^{K-1} (σ_i * l_i(x*) - σ_{i+1} * l_{i+1}(x*))^2 + ε]
    4. 双层优化：下层加权和损失 + λ * 上层平滑目标函数
    
    可调参数:
    lambda_param: 权衡参数λ，控制上层平滑目标函数的权重 (默认: 0.1)
    eps: 平滑目标函数的小常数 (默认: 1e-8)
    """
    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        max_norm: float = 1.0,
        eps: float = 1e-8,
        lambda_param: float = 0.1,  # 权衡参数λ，控制上层平滑目标函数的权重
    ):
        """初始化LDC-MTL New3参数
        
        Args:
            n_tasks: 任务数量
            device: 计算设备
            max_norm: 梯度裁剪的最大范数
            eps: 平滑目标函数的小常数
            lambda_param: 权衡参数λ，控制上层平滑目标函数的权重 (默认: 0.1)
        """
        super().__init__(n_tasks, device=device)
        self.n_tasks = n_tasks
        self.device = device
        self.max_norm = max_norm
        self.eps = eps
        self.lambda_param = lambda_param  # 新增lambda参数
        
        # 初始化可学习权重参数 ω∈R^K (初始化为全1)
        self.omega_params = torch.nn.Parameter(
            torch.ones(n_tasks, device=device)  # 初始化为全1
        )
    
    def get_softmax_weights(self) -> torch.Tensor:
        """
        使用softmax计算任务权重：σ = softmax(ω)
        
        Returns:
            任务权重 [K]，和为1
        """
        # 计算softmax权重：σ_i = exp(ω_i) / ∑_j exp(ω_j)
        logits = self.omega_params
        # 数值稳定性：减去最大值
        logits_max = torch.max(logits)
        logits = logits - logits_max
        exp_logits = torch.exp(logits)
        weights = exp_logits / torch.sum(exp_logits)
        return weights
    
    def compute_smooth_objective(self, losses: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        计算平滑目标函数：
        f_ε(σ) = √[∑_{i=1}^{K-1} (σ_i * l_i(x*) - σ_{i+1} * l_{i+1}(x*))^2 + ε]
        
        Args:
            losses: 任务损失 [K]
            weights: 任务权重 [K] (softmax输出)
            
        Returns:
            平滑目标函数值（带平方根）
        """
        # 确保losses是正确的形状
        if losses.dim() == 0:
            raise ValueError("LDC-New3 requires losses with dimension at least 1")
        elif losses.dim() == 1:
            losses_1d = losses
        else:
            # 如果是多维的，取平均
            losses_1d = losses.mean(dim=0)
        
        # 计算加权损失：σ_i * l_i(x*)
        weighted_losses = weights * losses_1d
        
        # 计算相邻任务加权损失的差值平方和
        diff_squared_sum = 0.0
        for i in range(self.n_tasks - 1):
            diff = weighted_losses[i] - weighted_losses[i + 1]
            diff_squared_sum += diff ** 2
        
        # 添加平滑项 ε，然后取平方根
        smooth_objective = torch.sqrt(diff_squared_sum + self.eps)
        return smooth_objective
    
    def get_weighted_loss(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        task_specific_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        last_shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        representation: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        **kwargs,
    ):
        # 确保losses是正确的形状
        if losses.dim() == 0:
            raise ValueError("LDC-New3 requires losses with dimension at least 1")
        elif losses.dim() == 1:
            losses_1d = losses
        else:
            # 如果是多维的，取平均
            losses_1d = losses.mean(dim=0)
        
        # 计算softmax权重 σ = softmax(ω)
        weights = self.get_softmax_weights()
        
        # 根据LDCMTL结构实现双层优化目标
        
        # 下层优化目标：加权和损失 g(W, ω) = sum_i σ_i l_i(x*)
        if losses.dim() == 1:  # [K] - 已对batch平均的损失
            # 对于已平均的损失，使用权重直接计算
            weighted_loss = torch.sum(losses_1d * weights)
        else:  # [B, K] - per-sample损失
            # 样本级加权，更符合LDCMTL的公式
            weighted_loss = (weights * losses_1d).mean()
        
        # 上层优化目标：平滑目标函数 f_ε(σ) = √[∑_{i=1}^{K-1} (σ_i * l_i(x*) - σ_{i+1} * l_{i+1}(x*))^2 + ε]
        smooth_objective = self.compute_smooth_objective(losses_1d, weights)
        
        # 总损失：下层损失 + λ * 上层平滑目标函数（与LDCMTL一致的权衡结构）
        total_loss = weighted_loss + self.lambda_param * smooth_objective
        
        return total_loss, {
            "weights": weights,
            "omega_params": self.omega_params,
            "weighted_loss": weighted_loss,
            "smooth_objective": smooth_objective,
        }
    
    def update(self, losses):
        """保存当前损失用于分析（类似LDCMTL的update方法）"""
        # 这里可以添加损失历史记录逻辑
        pass

    def backward(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        task_specific_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        last_shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        representation: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, None], Union[dict, None]]:
        # 计算总损失（包含下层和上层目标）
        loss, extra_outputs = self.get_weighted_loss(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            last_shared_parameters=last_shared_parameters,
            representation=representation,
            **kwargs,
        )
        
        # 执行反向传播 - 同时计算主干网络和权重参数的梯度
        loss.backward()
        
        # 应用梯度裁剪
        if self.max_norm > 0:
            if shared_parameters is not None:
                torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
            # 对omega参数也进行梯度裁剪
            if self.omega_params.grad is not None:
                torch.nn.utils.clip_grad_norm_([self.omega_params], self.max_norm)
        
        # 保存当前损失
        self.update(losses)
        
        return loss, extra_outputs
    
    def parameters(self) -> List[torch.Tensor]:
        """返回omega参数，供优化器使用"""
        return [self.omega_params]
    
    def __repr__(self):
        return (f"LDCNew3(n_tasks={self.n_tasks}, max_norm={self.max_norm}, "
                f"eps={self.eps}, lambda_param={self.lambda_param})")


class OPFAMO(WeightMethod):
    """Optimistic PFAMO (OPFAMO): 融合乐观梯度下降和协方差惩罚的FAMO改进版本
    
    结合OFAMO的乐观梯度预测和PFAMO的任务冲突惩罚：
    ξ_{t+1} = ξ_t − β[(1+γ2)δ_t − γ2δ_{t-1} + γξ_t + p_t]
    其中：
    - γ2：乐观梯度强度参数
    - p_t：协方差惩罚向量，p_i,t = γ3·∑_{j≠i} min(0,C_ij,t)
    - C_ij,t：任务i和j的损失变化协方差
    - ΔL_i,t：任务i的对数损失变化
    
    优化版本主要改进：
    - 移除迭代计数，简化首次迭代处理（首次迭代等效于OFAMO的首次迭代）
    - 向量化协方差惩罚计算，移除嵌套循环
    - 简化detach操作，减少不必要的克隆
    - 优化梯度处理流程
    
    参数:
    n_tasks: 任务数量
    device: 计算设备
    gamma: 原始FAMO的衰减系数（默认1e-5）
    gamma2: 乐观梯度强度参数（默认0.1）
    gamma3: 惩罚强度超参数（默认0.1）
    w_lr: 权重学习率（默认0.025）
    cov_history_size: 协方差计算的历史缓冲区大小（默认10）
    task_weights: 初始任务权重
    max_norm: 梯度裁剪的最大范数
    """

    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        gamma: float = 1e-5,
        gamma2: float = 0.1,
        gamma3: float = 0.1,
        w_lr: float = 0.025,
        cov_history_size: int = 10,
        task_weights: Union[List[float], torch.Tensor] = None,
        max_norm: float = 1.0,
    ):
        super().__init__(n_tasks, device=device)
        self.min_losses = torch.zeros(n_tasks).to(device)
        self.w = torch.tensor([0.0] * n_tasks, device=device, requires_grad=True)
        self.w_opt = torch.optim.Adam([self.w], lr=w_lr, weight_decay=gamma)
        self.max_norm = max_norm
        
        # OPFAMO特有参数
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        self.cov_history_size = cov_history_size
        
        # 使用张量存储历史数据，提高效率
        self.loss_change_history = torch.zeros(cov_history_size, n_tasks).to(device)
        self.history_index = 0
        self.history_count = 0
        
        # 存储前一次的delta值（OFAMO需要）
        self.delta_prev = torch.zeros(n_tasks).to(device)
        # 存储前一次的损失值
        self.prev_loss = None
    
    def set_min_losses(self, losses):
        self.min_losses = losses

    def get_weighted_loss(self, losses, **kwargs):
        self.prev_loss = losses.detach()
        z = F.softmax(self.w, -1)
        D = losses - self.min_losses + 1e-8
        c = (z / D).sum().detach()
        loss = (D.log() * z / c).sum()
        return loss, {"weights": z, "logits": self.w.detach().clone()}

    def update(self, curr_loss):
        delta_L = (self.prev_loss - self.min_losses+ 1e-8).log() - (curr_loss - self.min_losses+ 1e-8).log()
        
        # 更新损失变化历史
        self._update_loss_change_history(delta_L)
        
        # 计算梯度信号d
        with torch.enable_grad():
            z = F.softmax(self.w, -1)
            d = torch.autograd.grad(z, self.w, grad_outputs=delta_L.detach())[0]
        
        # 计算协方差惩罚项
        penalty = self._compute_covariance_penalty()
        
        # 构建有效的梯度更新
        self.w_opt.zero_grad()
        
        # 使用统一的更新规则，首次迭代时delta_prev为全0向量，等效于OFAMO首次迭代
        effective_grad = (1 + self.gamma2) * d - self.gamma2 * self.delta_prev + penalty
        
        self.w.grad = effective_grad
        self.w_opt.step()
        
        # 保存当前delta以供下次迭代使用
        self.delta_prev = d.detach()
    
    def _update_loss_change_history(self, delta_L):
        """更新损失变化历史缓冲区"""
        # 循环缓冲区
        self.loss_change_history[self.history_index] = delta_L.detach()
        self.history_index = (self.history_index + 1) % self.cov_history_size
        self.history_count = min(self.history_count + 1, self.cov_history_size)
    
    def _compute_covariance_penalty(self):
        """计算基于任务间协方差的惩罚项"""
        if self.history_count < 2:
            return torch.zeros(self.n_tasks, device=self.device)
        
        # 统一切片操作获取历史数据
        # 当history_count等于cov_history_size时，切片会返回整个缓冲区
        history_data = self.loss_change_history[:self.history_count]
        
        # 计算协方差矩阵 (高效向量化实现)
        # 减去均值
        centered_data = history_data - history_data.mean(dim=0, keepdim=True)
        
        # 计算协方差矩阵: C_ij = Cov(ΔL_i, ΔL_j)
        cov_matrix = (centered_data.t() @ centered_data) / (self.history_count - 1)
        
        # 创建掩码矩阵，对角线元素为0（排除自身协方差）
        mask = ~torch.eye(self.n_tasks, dtype=torch.bool, device=self.device)
        
        # 计算所有任务对的负协方差部分
        neg_cov = torch.minimum(cov_matrix, torch.zeros_like(cov_matrix))
        
        # 应用掩码并按行求和，然后乘以gamma3
        penalty = self.gamma3 * (neg_cov * mask).sum(dim=1)
        
        return penalty

class PFAMO(WeightMethod):
    """FAMO with Covariance Penalty (PFAMO): 在FAMO算法基础上添加任务冲突惩罚项的改进版本
    
    通过计算任务间协方差来识别任务冲突，并添加惩罚项来缓解冲突：
    ξ_{t+1} = ξ_t - β(δ_t + γξ_t + p_t)
    其中p_i,t = γ3·Σ_{j≠i} min(0, C_ij,t)，C_ij,t是任务i和j的损失变化协方差
    
    优化版本主要改进：
    - 向量化协方差惩罚计算，移除嵌套循环
    - 简化detach操作，减少不必要的克隆
    - 移除迭代计数，简化状态管理
    - 优化梯度处理流程
    
    参数:
    n_tasks: 任务数量
    device: 计算设备
    gamma: 原始FAMO的衰减系数（默认1e-5）
    w_lr: 权重学习率（默认0.025）
    gamma3: 惩罚强度超参数（默认0.1）
    cov_history_size: 协方差计算的历史缓冲区大小（默认10）
    task_weights: 初始任务权重
    max_norm: 梯度裁剪的最大范数
    """

    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        gamma: float = 1e-5,
        w_lr: float = 0.025,
        gamma3: float = 0.1,
        cov_history_size: int = 10,
        task_weights: Union[List[float], torch.Tensor] = None,
        max_norm: float = 1.0,
    ):
        super().__init__(n_tasks, device=device)
        self.min_losses = torch.zeros(n_tasks).to(device)
        self.w = torch.tensor([0.0] * n_tasks, device=device, requires_grad=True)
        self.w_opt = torch.optim.Adam([self.w], lr=w_lr, weight_decay=gamma)
        self.max_norm = max_norm
        
        # PFAMO参数
        self.gamma3 = gamma3
        self.cov_history_size = cov_history_size
        
        # 使用张量存储历史数据，提高效率
        self.loss_change_history = torch.zeros(cov_history_size, n_tasks).to(device)
        self.history_index = 0
        self.history_count = 0
        
        self.prev_loss = None
    
    def set_min_losses(self, losses):
        self.min_losses = losses

    def get_weighted_loss(self, losses, **kwargs):
        # 简化操作，参考FAMO的处理方式
        self.prev_loss = losses
        z = F.softmax(self.w, -1)
        D = losses - self.min_losses + 1e-8
        c = (z / D).sum().detach()
        loss = (D.log() * z / c).sum()
        return loss, {"weights": z, "logits": self.w.detach().clone()}

    def update(self, curr_loss):
        # 计算对数损失变化 ΔL_i,t = log(ℓ_i,t) - log(ℓ_i,t+1)
        delta_L = (self.prev_loss - self.min_losses + 1e-8).log() - \
                 (curr_loss - self.min_losses + 1e-8).log()
        
        # 更新损失变化历史
        self._update_loss_change_history(delta_L)
        
        # 计算FAMO的梯度信号
        with torch.enable_grad():
            d = torch.autograd.grad(F.softmax(self.w, -1),
                                   self.w,
                                   grad_outputs=delta_L.detach())[0]
        
        # 计算惩罚项 p_i = λ · Σ_{j≠i} min(0, C_ij)
        penalty = self._compute_covariance_penalty()
        
        # 组合梯度: ξ_{t+1} = ξ_t - β(δ_t + γξ_t + p_t)
        # 注意：weight_decay会自动处理γξ_t项
        effective_grad = d + penalty
        
        # 使用优化器更新
        self.w_opt.zero_grad()
        self.w.grad = effective_grad
        self.w_opt.step()
    
    def _update_loss_change_history(self, delta_L):
        """更新损失变化历史缓冲区"""
        # 循环缓冲区
        self.loss_change_history[self.history_index] = delta_L.detach()
        self.history_index = (self.history_index + 1) % self.cov_history_size
        self.history_count = min(self.history_count + 1, self.cov_history_size)
    
    def _compute_covariance_penalty(self):
        """向量化计算基于任务间协方差的惩罚项"""
        if self.history_count < 2:
            return torch.zeros(self.n_tasks, device=self.device)
        
        # 使用统一的切片操作获取有效历史数据
        # 当history_count == cov_history_size时，切片也会返回整个缓冲区
        history_data = self.loss_change_history[:self.history_count]
        
        # 计算协方差矩阵 (高效向量化实现)
        # 减去均值
        centered_data = history_data - history_data.mean(dim=0, keepdim=True)
        
        # 计算协方差矩阵: C_ij = Cov(ΔL_i, ΔL_j)
        cov_matrix = (centered_data.t() @ centered_data) / (self.history_count - 1)
        
        # 向量化计算惩罚项: p_i = γ3 · Σ_{j≠i} min(0, C_ij)
        # 创建对角线掩码，排除对角线元素(i,i)
        mask = ~torch.eye(self.n_tasks, dtype=torch.bool, device=self.device)
        
        # 计算每个任务i的负协方差和
        min_cov = torch.minimum(cov_matrix, torch.tensor(0.0, device=self.device))
        penalty = self.gamma3 * min_cov.masked_select(mask).view(self.n_tasks, -1).sum(dim=1)
        
        return penalty

class OFAMO(WeightMethod):
    """简化版Optimistic FAMO (OFAMO): 在FAMO算法基础上添加乐观梯度项的改进版本
    
    简化实现：
    - 去掉条件逻辑判断和迭代计数
    - 将delta_prev初始化为全0向量，取消特殊的warm-up处理
    - 优化梯度处理，参考FAMO的处理方式
    
    更新规则简化为：ξ_{t+1} = ξ_t - β[(1+γ2)δ_t - γ2δ_{t-1} + γξ_t]
    其中第一次迭代时δ_{t-1}为0向量，等价于原始FAMO
    
    参数:
    n_tasks: 任务数量
    device: 计算设备
    gamma: 原始FAMO的衰减系数（默认1e-5）
    gamma2: 乐观梯度项的系数（默认0.1）
    w_lr: 权重学习率（默认0.025）
    task_weights: 初始任务权重
    max_norm: 梯度裁剪的最大范数
    """

    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        gamma: float = 1e-5,
        gamma2: float = 0.1,
        w_lr: float = 0.025,
        task_weights: Union[List[float], torch.Tensor] = None,
        max_norm: float = 1.0,
    ):
        super().__init__(n_tasks, device=device)
        self.min_losses = torch.zeros(n_tasks).to(device)
        self.w = torch.tensor([0.0] * n_tasks, device=device, requires_grad=True)
        # 使用优化器的weight_decay来处理γξ_t项
        self.w_opt = torch.optim.Adam([self.w], lr=w_lr, weight_decay=gamma)
        self.max_norm = max_norm
        
        # OFAMO参数 - 简化版本
        self.gamma2 = gamma2  # 乐观梯度系数
        self.delta_prev = torch.zeros(n_tasks).to(device)  # 初始化为0，首次迭代时等效于FAMO
    
    def set_min_losses(self, losses):
        self.min_losses = losses

    def get_weighted_loss(self, losses, **kwargs):
        # 参考FAMO的处理方式
        self.prev_loss = losses
        z = F.softmax(self.w, -1)
        D = losses - self.min_losses + 1e-8
        c = (z / D).sum().detach()
        loss = (D.log() * z / c).sum()
        return loss, {"weights": z, "logits": self.w.detach().clone()}

    def update(self, curr_loss):
        # 计算当前梯度信号 δ_t
        delta_t = (self.prev_loss - self.min_losses + 1e-8).log() - \
                 (curr_loss - self.min_losses + 1e-8).log()
        
        # 计算梯度 d = ∂z/∂ξ · δ_t - 参考FAMO的处理方式
        with torch.enable_grad():
            d = torch.autograd.grad(F.softmax(self.w, -1),
                                    self.w,
                                    grad_outputs=delta_t.detach())[0]
        
        # 统一的乐观更新规则，首次迭代时delta_prev为0，等价于原始FAMO更新
        effective_grad = (1 + self.gamma2) * d - self.gamma2 * self.delta_prev
        
        # 使用优化器更新权重
        self.w_opt.zero_grad()
        self.w.grad = effective_grad
        self.w_opt.step()
        
        # 保存当前梯度信号为下一次的delta_prev
        self.delta_prev = d.detach()

class LDCNew4(WeightMethod):
    """
    LDC-New4: 直接对各任务的损失进行线性规划求解最佳权重
    
    与LDCNew2的主要区别：
    - 不使用kmeans聚类算法
    - 直接将原始损失值构建到线性规划中
    - 线性规划优化目标：最小化加权损失的方差，实现任务间的平衡
    
    算法流程:
    1. 直接使用原始损失值作为线性规划输入
    2. 构建线性规划问题：最小化加权损失方差 + L2正则化
    3. 约束条件：权重和为1，权重非负
    4. 求解最优权重
    
    优点:
    - 无需聚类，简化计算流程
    - 直接优化任务平衡
    - 数值稳定，避免聚类失败问题
    """
    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        max_norm: float = 1.0,
    ):
        super().__init__(n_tasks, device=device)
        self.n_tasks = n_tasks
        self.device = device
        self.max_norm = max_norm
    
    def solve_loss_linear_programming(self, losses: torch.Tensor) -> np.ndarray:
        """
        使用与LDCNew2相同的线性规划方法，直接将损失值作为聚类中心
        
        Args:
            losses: 任务损失 [K]
            
        Returns:
            最优权重 [K]
        """
        K = len(losses)
        
        # 如果只有一组任务，直接返回均匀权重
        if K <= 1:
            return np.ones(K) / K if K > 0 else np.array([])
        
        # 将损失值直接作为聚类中心（每个任务一组）
        loss_centers = losses.detach().cpu()
        
        # 构建线性规划问题（与LDCNew2相同）
        # 变量: [w_1, w_2, ..., w_K, u_1, u_2, ..., u_{K-1}]
        # 总变量数: K + (K-1) = 2K-1
        
        # 目标函数: min ∑_{i=1}^{K-1} u_i
        c = np.zeros(2 * K - 1)
        c[K:2*K-1] = 1.0  # u_i的系数为1
        
        # 等式约束: ∑_{i=1}^{K} w_i = 1
        A_eq = np.zeros((1, 2 * K - 1))
        A_eq[0, :K] = 1.0
        b_eq = np.array([1.0])
        
        # 不等式约束矩阵
        A_ub = []
        b_ub = []
        
        # 约束: u_i ≥ w_i * c_i - w_{i+1} * c_{i+1}
        # 即: -w_i * c_i + w_{i+1} * c_{i+1} + u_i ≥ 0
        for i in range(K-1):
            row = np.zeros(2 * K - 1)
            row[i] = -loss_centers[i].item()
            row[i+1] = loss_centers[i+1].item()
            row[K + i] = 1.0  # u_i
            A_ub.append(row)
            b_ub.append(0.0)
        
        # 约束: u_i ≥ -(w_i * c_i - w_{i+1} * c_{i+1})
        # 即: w_i * c_i - w_{i+1} * c_{i+1} + u_i ≥ 0
        for i in range(K-1):
            row = np.zeros(2 * K - 1)
            row[i] = loss_centers[i].item()
            row[i+1] = -loss_centers[i+1].item()
            row[K + i] = 1.0  # u_i
            A_ub.append(row)
            b_ub.append(0.0)
        
        # 下界: w_i ≥ 0, u_i ≥ 0
        bounds = [(0, None)] * (2 * K - 1)
        
        # 求解线性规划
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                           bounds=bounds, method='highs')
            
            if result.success:
                # 返回最优权重
                optimal_weights = result.x[:K]
                return optimal_weights
            else:
                # 如果LP求解失败，使用均匀权重作为fallback
                print(f"Warning: LP求解失败，使用均匀权重. Status: {result.message}")
                return np.ones(K) / K
                
        except Exception as e:
            print(f"Warning: LP求解异常，使用均匀权重. Error: {e}")
            return np.ones(K) / K
    
    def get_weighted_loss(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        task_specific_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        last_shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        representation: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        **kwargs,
    ):
        # 确保losses是正确的形状
        if losses.dim() == 0:
            raise ValueError("LDC-New4 requires losses with dimension at least 1")
        elif losses.dim() == 1:
            losses_1d = losses
        else:
            # 如果是多维的，取平均
            losses_1d = losses.mean(dim=0)
        
        K = len(losses_1d)
        
        # 直接对损失进行线性规划求解
        optimal_weights = self.solve_loss_linear_programming(losses_1d)
        optimal_weights = torch.from_numpy(optimal_weights).to(losses.device)
        
        # 计算总损失：L = ∑ w_i * l_i
        total_loss = torch.sum(optimal_weights * losses_1d)
        
        return total_loss, {
            "weights": optimal_weights,
            "losses": losses_1d,
        }
    
    def backward(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        task_specific_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        last_shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        representation: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, None], Union[dict, None]]:
        # 计算总损失
        total_loss, extra_outputs = self.get_weighted_loss(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            last_shared_parameters=last_shared_parameters,
            representation=representation,
            **kwargs,
        )
        
        # 执行反向传播
        total_loss.backward()
        
        # 应用梯度裁剪
        if self.max_norm > 0:
            if shared_parameters is not None:
                torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
            if task_specific_parameters is not None:
                torch.nn.utils.clip_grad_norm_(task_specific_parameters, self.max_norm)
        
        return total_loss, extra_outputs
    
    def __repr__(self):
        return (f"LDCNew4(n_tasks={self.n_tasks}, max_norm={self.max_norm})")





class AugmentedLDCMTL(WeightMethod):
    """
    AugmentedLDCMTL: Combines LDC-MTL and OpFAMO for multi-task learning.
    
    This method integrates:
    - LDC-MTL: Lightweight Router network with discrepancy control
    - OpFAMO: Covariance-based conflict detection with optimistic gradient descent
    
    The unified loss function is:
    Ltotal = Ldisc + λ·Ltask - γcov·Lcov
    
    where:
    - Ldisc = ∑|σi(W)ℓi(x) - σi+1(W)ℓi+1(x)| (LDC discrepancy term)
    - Ltask = ∑σi(W)ℓi(x) (LDC weighted task term)
    - Lcov = ∑σi(W)·pi,t (OpFAMO covariance enhancement term)
    
    Router parameters W are updated using Optimistic Gradient Descent (OGD).
    
    Args:
        n_tasks: Number of tasks
        device: Torch device
        max_norm: Maximum norm for gradient clipping
        router_hidden_dim: Hidden dimension size for the Router network
        feature_dim: Dimension of the input features to the Router
        lambda_ldc: Weight for LDC task term (λ)
        gamma_cov: Weight for OpFAMO covariance term (γcov)
        buffer_size: Size of history buffer for covariance computation (N)
        ogd_lr: Learning rate for OGD on Router parameters (β)
        ogd_gamma: Momentum coefficient for OGD (γogd)
    """
    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        max_norm: float = 1.0,
        router_hidden_dim: int = 64,
        feature_dim: int = 256,
        lambda_ldc: float = 1.0,
        gamma_cov: float = 0.1,
        buffer_size: int = 100,
        ogd_lr: float = 0.001,
        ogd_gamma: float = 0.9,
    ):
        super().__init__(n_tasks, device=device)
        self.max_norm = max_norm
        self.lambda_ldc = lambda_ldc
        self.gamma_cov = gamma_cov
        self.buffer_size = buffer_size
        self.ogd_lr = ogd_lr
        self.ogd_gamma = ogd_gamma
        
        # LDC Component: Lightweight Router Network
        self.router = nn.Sequential(
            nn.Linear(feature_dim, router_hidden_dim),
            nn.ReLU(),
            nn.Linear(router_hidden_dim, n_tasks),
        ).to(device)
        
        # OpFAMO Component: History buffer for covariance computation
        self.log_loss_buffer = []  # Buffer to store log losses
        self.prev_log_losses = torch.zeros(n_tasks).to(device)
        
        # OGD Component: Previous gradient for momentum
        self.prev_router_grad = None
        
        # Track first step
        self.first_step = True
        
        # Current conflict indicators (detached, constant)
        self.conflict_indicators = torch.zeros(n_tasks).to(device)
    
    def _update_covariance_conflict(self, losses: torch.Tensor):
        """
        Update covariance-based conflict indicators using OpFAMO method.
        
        Args:
            losses: Current task losses [K]
        """
        # Compute current log losses
        current_log_losses = torch.log(losses + EPS)
        
        # Compute log-loss changes: Δℓi,t = log(ℓi,t-1) - log(ℓi,t)
        if not self.first_step:
            delta_log_losses = self.prev_log_losses - current_log_losses
            
            # Add to buffer
            self.log_loss_buffer.append(delta_log_losses.detach().cpu())
            
            # Maintain buffer size
            if len(self.log_loss_buffer) > self.buffer_size:
                self.log_loss_buffer.pop(0)
            
            # Compute covariance matrix if buffer has enough samples
            if len(self.log_loss_buffer) >= 2:
                # Stack buffer into matrix: [buffer_size, K]
                delta_matrix = torch.stack(self.log_loss_buffer)
                
                # Compute covariance matrix: Cij,t = Cov(Δℓi, Δℓj)
                cov_matrix = torch.cov(delta_matrix.T)
                
                # Compute conflict indicators: pi,t = ∑j≠i min(0, Cij,t)
                conflict_indicators = torch.zeros(self.n_tasks)
                for i in range(self.n_tasks):
                    for j in range(self.n_tasks):
                        if i != j:
                            conflict_indicators[i] += min(0.0, cov_matrix[i, j].item())
                
                # Update conflict indicators (detached, constant)
                self.conflict_indicators = conflict_indicators.to(self.device)
        
        # Update previous log losses
        self.prev_log_losses = current_log_losses.detach()
    
    def _compute_total_loss(
        self,
        losses: torch.Tensor,
        router_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute total loss Ltotal = Ldisc + λ·Ltask - γcov·Lcov.
        
        Args:
            losses: Task losses [K]
            router_input: Input features for Router
            
        Returns:
            Total loss
        """
        # Router computes logits
        pred_logits = self.router(router_input)
        
        # Apply softmax to get weights: σ(W) = Softmax(W)
        weights = F.softmax(pred_logits, dim=-1)
        
        # Compute LDC discrepancy term: Ldisc = ∑|σi(W)ℓi(x) - σi+1(W)ℓi+1(x)|
        weighted_losses = weights * losses
        ldisc = torch.sum(torch.abs(weighted_losses[:-1] - weighted_losses[1:]))
        
        # Compute LDC weighted task term: Ltask = ∑σi(W)ℓi(x)
        ltask = torch.sum(weighted_losses)
        
        # Compute OpFAMO covariance enhancement term: Lcov = ∑σi(W)·pi,t
        # Note: conflict_indicators is detached (constant)
        lcov = torch.sum(weights * self.conflict_indicators.detach())
        
        # Total loss: Ltotal = Ldisc + λ·Ltask - γcov·Lcov
        ltotal = ldisc + self.lambda_ldc * ltask - self.gamma_cov * lcov
        
        return ltotal
    
    def get_weighted_loss(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        task_specific_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        last_shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        representation: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        **kwargs,
    ):
        """
        Calculate weighted loss using AugmentedLDCMTL algorithm.
        
        Parameters:
        ----------
        losses : Task losses [K]
        shared_parameters : Shared parameters of the model
        task_specific_parameters : Task-specific parameters
        last_shared_parameters : Last shared layer parameters
        representation : Shared representation (used as input to Router)
        
        Returns:
        -------
        Total loss and extra outputs
        """
        # Ensure losses is 1D
        if losses.dim() == 0:
            losses = losses.unsqueeze(0)
        elif losses.dim() > 1:
            losses = losses.mean(dim=0)
        
        # Prepare router input from representation
        if isinstance(representation, list):
            router_input = representation[0]
        else:
            router_input = representation
        
        # If representation has spatial dimensions, apply global average pooling
        if len(router_input.shape) > 2:
            router_input = torch.mean(router_input, dim=tuple(range(2, len(router_input.shape))))
        
        # Update covariance-based conflict indicators
        self._update_covariance_conflict(losses)
        
        # Compute total loss
        total_loss = self._compute_total_loss(losses, router_input)
        
        # Get current weights for logging
        pred_logits = self.router(router_input)
        weights = F.softmax(pred_logits, dim=-1)
        
        return total_loss, {
            "weights": weights,
            "losses": losses,
            "conflict_indicators": self.conflict_indicators,
        }
    
    def backward(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        task_specific_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        last_shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        representation: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, None], Union[dict, None]]:
        """
        Backward pass with OGD update for Router parameters.
        
        Parameters:
        ----------
        losses : Task losses
        shared_parameters : Shared parameters of the model
        task_specific_parameters : Task-specific parameters
        last_shared_parameters : Last shared layer parameters
        representation : Shared representation
        
        Returns:
        -------
        Total loss and extra outputs
        """
        # Compute total loss
        total_loss, extra_outputs = self.get_weighted_loss(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            last_shared_parameters=last_shared_parameters,
            representation=representation,
            **kwargs,
        )
        
        # Backward pass
        total_loss.backward()
        
        # Apply gradient clipping to model parameters
        if self.max_norm > 0:
            if shared_parameters is not None:
                torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
            if task_specific_parameters is not None:
                torch.nn.utils.clip_grad_norm_(task_specific_parameters, self.max_norm)
        
        # OGD update for Router parameters
        # Wt+1 = Wt - β(∇WLtotalt + γogd(∇WLtotalt - ∇WLtotalt-1))
        current_router_grad = []
        for param in self.router.parameters():
            if param.grad is not None:
                current_router_grad.append(param.grad.clone())
            else:
                current_router_grad.append(torch.zeros_like(param))
        
        if not self.first_step and self.prev_router_grad is not None:
            # Apply OGD with momentum
            for i, param in enumerate(self.router.parameters()):
                if param.grad is not None:
                    # OGD update: Wt+1 = Wt - β(∇WLtotalt + γogd(∇WLtotalt - ∇WLtotalt-1))
                    momentum_term = self.ogd_gamma * (current_router_grad[i] - self.prev_router_grad[i])
                    ogd_update = self.ogd_lr * (current_router_grad[i] + momentum_term)
                    param.data -= ogd_update
        else:
            # First step: standard gradient descent
            for i, param in enumerate(self.router.parameters()):
                if param.grad is not None:
                    param.data -= self.ogd_lr * current_router_grad[i]
        
        # Store current gradients for next step
        self.prev_router_grad = current_router_grad
        
        # Clear Router gradients after manual update
        for param in self.router.parameters():
            if param.grad is not None:
                param.grad = None
        
        # Update first step flag
        self.first_step = False
        
        return total_loss, extra_outputs
    
    def __repr__(self):
        return (f"AugmentedLDCMTL(n_tasks={self.n_tasks}, max_norm={self.max_norm}, "
                f"lambda_ldc={self.lambda_ldc}, gamma_cov={self.gamma_cov}, "
                f"buffer_size={self.buffer_size}, ogd_lr={self.ogd_lr}, "
                f"ogd_gamma={self.ogd_gamma})")


class GRAPE(WeightMethod):
    """
    GRAPE (Gradient-Aware Pareto Estimation)
    
    A multi-task learning optimization method based on GO4Align framework,
    but with indicator replaced by EMA of log loss change rate.
    
    Key Features:
    - Two-level optimization framework (Lower-level: indicator update & K-means, Upper-level: weighted loss)
    - Indicator: EMA of log loss change rate for scale-invariant task similarity
    - K-means clustering on indicators for group-based weighting
    - Reversed softmax on cluster centers for group weights
    
    Args:
        n_tasks: Number of tasks M
        K: Number of groups, default=2
        alpha: EMA smoothing coefficient, default=0.1
        tau: Softmax temperature, default=1.0
    """
    
    def __init__(self, n_tasks: int, device: torch.device, K: int = 2, 
                 alpha: float = 0.1, tau: float = 1.0, max_norm: float = 1.0):
        super().__init__(n_tasks, device=device)
        self.K = K
        self.alpha = alpha
        self.tau = tau
        self.max_norm = max_norm
        
        # Numerical stability constant
        self.eps = 1e-8
        
        # State tensors (manually managed on correct device)
        # Previous losses for computing log loss change rate
        self.prev_losses = torch.zeros(n_tasks, device=device)
        # Indicators for task similarity (EMA of log loss change rate)
        self.indicators = torch.zeros(n_tasks, device=device)
        # Flag to handle first iteration (no prev_losses available)
        self.is_first_iteration = True
    
    def update_prev_losses(self, losses: torch.Tensor):
        """
        Save current losses as prev_losses for next iteration.
        
        Args:
            losses: shape (M,), detached
        """
        self.prev_losses.copy_(losses.detach())
        self.is_first_iteration = False
    
    def get_weighted_loss(self, losses: torch.Tensor, **kwargs):
        """
        Calculate weighted loss using GRAPE two-level optimization.
        
        Args:
            losses: shape (M,), with gradient, for computing weighted loss
        
        Returns:
            Weighted total loss for backpropagation
        """
        losses_with_grad = losses
        losses_detached = losses.detach()
        
        with torch.no_grad():
            # Lower-level optimization steps
            
            # Step 1: Update indicator (EMA of log loss change rate)
            if self.is_first_iteration:
                # First iteration: initialize indicators to zero
                self.indicators.zero_()
            else:
                # Compute log loss change rate
                # Formula: gamma_t^i = (1 - alpha) * gamma_{t-1}^i + alpha * (log ell_{i,t-1} - log ell_{i,t})
                log_prev_losses = torch.log(self.prev_losses + self.eps)
                log_curr_losses = torch.log(losses_detached + self.eps)
                log_loss_change = log_prev_losses - log_curr_losses
                
                # EMA update
                self.indicators = (1 - self.alpha) * self.indicators + self.alpha * log_loss_change
            
            # Step 2: K-means clustering on indicators
            if self.is_first_iteration:
                # First iteration: uniform weights (no clustering)
                task_weights = torch.ones(self.n_tasks, device=self.device) / self.n_tasks
            else:
                # Reshape indicators for K-means: (M, 1)
                indicators_reshaped = self.indicators.unsqueeze(1)
                
                # K-means clustering
                try:
                    from kmeans_pytorch import kmeans
                    cluster_ids, cluster_centers = kmeans(
                        X=indicators_reshaped,
                        num_clusters=self.K,
                        distance='euclidean',
                        device=self.device,
                        verbose=False
                    )
                except ImportError:
                    # Fallback: use simple assignment if kmeans_pytorch not available
                    cluster_ids = torch.zeros(self.n_tasks, dtype=torch.long, device=self.device)
                    cluster_centers = torch.zeros(self.K, device=self.device)
                
                # Squeeze cluster_centers: (K, 1) -> (K,)
                cluster_centers = cluster_centers.squeeze(1)
                
                # Move cluster_ids to the correct device
                cluster_ids = cluster_ids.to(self.device)
                
                # Build assignment matrix G_t: shape (K, M)
                G_t = torch.zeros(self.K, self.n_tasks, device=self.device)
                for k in range(self.K):
                    G_t[k, cluster_ids == k] = 1
                
                # Step 3: Compute group weights via reversed softmax on cluster centers
                # Formula: omega = Softmax(-c / tau)
                # where c = [c^1, c^2, ..., c^K] are cluster centers
                # First apply Z-score normalization to cluster centers
                mean_c = cluster_centers.mean()
                std_c = cluster_centers.std() + self.eps
                normalized_centers = (cluster_centers - mean_c) / std_c
                omega = torch.softmax(-normalized_centers / self.tau, dim=0)
                
                # Step 4: Compute task weights from group weights
                # lambda_i = sum_k omega^k * G_t[k, i]
                task_weights = torch.mm(omega.unsqueeze(0), G_t).squeeze(0)
        
        # Upper-level optimization: compute weighted loss
        weighted_loss = (task_weights * losses_with_grad).sum()
        
        return weighted_loss, {"weights": task_weights, "indicators": self.indicators}
    
    def __repr__(self):
        return (f"GRAPE(n_tasks={self.n_tasks}, K={self.K}, "
                f"alpha={self.alpha}, tau={self.tau})")


METHODS = dict(
    stl=STL,
    ls=LinearScalarization,
    lscorr=LSCORR,
    uw=Uncertainty,
    uwcorr=UWCORR,
    scaleinvls=ScaleInvariantLinearScalarization,
    scaleinvlscorr=SCALEINVLSCORR,
    rlw=RLW,
    rlwcorr=RLWCORR,
    dwa=DynamicWeightAverage,
    dwacorr=DWACORR,

    pcgrad=PCGrad,
    mgda=MGDA,
    graddrop=GradDrop,
    log_mgda=LOG_MGDA,
    cagrad=CAGrad,
    log_cagrad=LOG_CAGrad,
    imtl=IMTLG,
    log_imtl=LOG_IMTLG,
    nashmtl=NashMTL,
    famo=FAMO,
    famo_lbfgs=FamoLBFGS,
    ofamo=OFAMO,
    pfamo=PFAMO,
    opfamo=OPFAMO,
    corrfamo=CorrFAMO,
    fairgrad=FairGrad,
    fast_fairgrad=FastFairGrad,
    ldcmtl=LDCMTL,
    ldcnew1=LDCMTLNew1,
    ldcnew2=LDCNew2,
    ldcnew3=LDCNew3,
    ldcnew4=LDCNew4,
    pivrg=PIVRG,
    consmtl=ConsMTL,
    cons_city=ConsMTLCity,
    consfamo=ConsFAMO,
    conscityfamo=ConsCityFAMO,
    rbldc=RBLDC,
    augmented_ldcmtl=AugmentedLDCMTL,
    # GO4Align methods
    go4align=GO4ALIGN,
    go4aligncov=GO4ALIGNCOV,
    go4aligncorr=GO4ALIGNCORR,
    # GRAPE method
    grape=GRAPE,
    # group=GROUP,
    # group_random=GROUP_RANDOM,
    # group_sklearn_spectral_clustering_cluster_qr=GROUP_sklearn_spectral_clustering_cluster_qr,
    # group_sklearn_spectral_clustering_discretize=GROUP_sklearn_spectral_clustering_discretize,
    # group_sklearn_spectral_clustering_kmeans=GROUP_sklearn_spectral_clustering_kmeans,
)
