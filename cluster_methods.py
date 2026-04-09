import copy
import random
from abc import abstractmethod
from typing import Dict, List, Tuple, Union

import cvxpy as cp
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize

from kmeans_pytorch import kmeans # training on gpus
# from sklearn.cluster import KMeans, SpectralClustering # training on cpu
# assign labels after the Laplacian embedding.
# The cluster_qr method [5] directly extract clusters from eigenvectors in spectral clustering.
# Simple, direct, and efficient multi-way spectral clustering, 2019 Anil Damle, Victor Minden, Lexing Ying

# from .sdp_kmeans import sdp_kmeans, connected_components, spectral_embedding
# https://github.com/simonsfoundation/sdp_kmeans.git
# Mariano Tepper, Anirvan Sengupta, Dmitri Chklovskii, The surprising secret identity of the semidefinite relaxation of K-means: manifold learning, 2017


class WeightMethod:
    def __init__(self, n_tasks: int, device: torch.device):
        super().__init__()
        self.n_tasks = n_tasks
        self.device = device

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

class GO4ALIGN(WeightMethod):
    def __init__(
            self,
            n_tasks: int,
            device: torch.device,
            num_groups: int,
            task_weights: Union[List[float], torch.Tensor] = None,
            robust_step_size: float = 0.0001,
            max_norm: float = 1.0,
    ):
        super().__init__(n_tasks, device=device)
        self.n_tasks = n_tasks
        self.device = device
        if task_weights is None:
            task_weights = torch.ones((n_tasks,))
        if not isinstance(task_weights, torch.Tensor):
            task_weights = torch.tensor(task_weights)
        assert len(task_weights) == n_tasks
        self.task_weights = task_weights.to(device)

        self.adv_probs = torch.ones(n_tasks).to(device) / n_tasks
        self.robust_step_size = robust_step_size
        self.num_groups = num_groups

    def get_weighted_loss(self, losses, **kwargs):
        adjusted_loss = losses.detach()
        
        # 添加数值稳定性保护
        adjusted_loss = torch.clamp(adjusted_loss, min=1e-8, max=1e8)
        
        scale = adjusted_loss.sum() / adjusted_loss
        # 防止scale出现过大值
        scale = torch.clamp(scale, min=1e-8, max=1e8)
        
        self.adv_probs = self.adv_probs * torch.exp(- self.robust_step_size * adjusted_loss)
        # 防止adv_probs出现零或无穷
        self.adv_probs = torch.clamp(self.adv_probs, min=1e-8, max=1e8)
        self.adv_probs = self.adv_probs / (self.adv_probs.sum() + 1e-8)
        
        weight = scale * self.adv_probs
        # 防止weight出现零或无穷
        weight = torch.clamp(weight, min=1e-8, max=1e8)

        # 新增：权重归一化处理，确保数值范围在[0,1]之间
        weight_min = weight.min()
        weight_max = weight.max()
        if weight_max - weight_min > 1e-8:  # 避免除零
            weight = (weight - weight_min) / (weight_max - weight_min)
        else:
            weight = torch.ones_like(weight) / self.n_tasks

        weight = weight.unsqueeze(1)

        if self.num_groups >=2:
            cluster_ids_x, cluster_centers = kmeans(X=weight, num_clusters=self.num_groups, distance='euclidean', device=self.device)
            mask = torch.zeros(self.n_tasks, self.num_groups).to(self.device)
            cluster_ids = cluster_ids_x.unsqueeze(1).to(self.device)
            cluster_centers = cluster_centers.to(self.device)
            mask.scatter_(1, cluster_ids, 1)
            kmeans_weight = torch.mm(mask,cluster_centers).squeeze(1)

        elif self.num_groups == 1:
            kmeans_weight = torch.ones(self.n_tasks).to(self.device)
            kmeans_weight = kmeans_weight * torch.mean(weight)

        loss = torch.sum(losses * kmeans_weight)
        return loss, dict(weights=torch.cat([kmeans_weight]))


class GROUP(WeightMethod):
    def __init__(
            self,
            n_tasks: int,
            device: torch.device,
            task_weights: Union[List[float], torch.Tensor] = None,
            robust_step_size: float = 0.0001,
    ):
        super().__init__(n_tasks, device=device)
        self.n_tasks = n_tasks
        if task_weights is None:
            task_weights = torch.ones((n_tasks,))
        if not isinstance(task_weights, torch.Tensor):
            task_weights = torch.tensor(task_weights)
        assert len(task_weights) == n_tasks
        self.task_weights = task_weights.to(device)

    def get_weighted_loss(self, losses, **kwargs):
        loss = torch.sum(losses * self.task_weights) * self.n_tasks
        return loss, dict(weights=torch.cat([self.task_weights]))

class GROUP_RANDOM(WeightMethod):
    def __init__(
            self,
            n_tasks: int,
            device: torch.device,
            task_weights: Union[List[float], torch.Tensor] = None,
            robust_step_size: float = 0.0001,
    ):
        super().__init__(n_tasks, device=device)
        self.n_tasks = n_tasks
        if task_weights is None:
            task_weights = torch.ones((n_tasks,))
        if not isinstance(task_weights, torch.Tensor):
            task_weights = torch.tensor(task_weights)
        assert len(task_weights) == n_tasks
        self.task_weights = task_weights.to(device)

    def get_weighted_loss(self, losses, **kwargs):
        idx = torch.randperm(self.task_weights.shape[0])
        weights = self.task_weights[idx]
        loss = torch.sum(losses * weights) * self.n_tasks
        return loss, dict(weights=torch.cat([weights]))

class GROUP_sklearn_spectral_clustering_cluster_qr(WeightMethod):
    def __init__(
            self,
            n_tasks: int,
            device: torch.device,
            num_groups: int,
            task_weights: Union[List[float], torch.Tensor] = None,
            robust_step_size: float = 0.0001,
    ):
        super().__init__(n_tasks, device=device)
        self.n_tasks = n_tasks
        self.device = device
        self.max_norm = max_norm
        if task_weights is None:
            task_weights = torch.ones((n_tasks,))
        if not isinstance(task_weights, torch.Tensor):
            task_weights = torch.tensor(task_weights)
        assert len(task_weights) == n_tasks
        self.task_weights = task_weights.to(device)

        self.adv_probs = torch.ones(n_tasks).to(device) / n_tasks
        self.robust_step_size = robust_step_size
        self.num_groups = num_groups

    def get_weighted_loss(self, losses, **kwargs):
        adjusted_loss = losses.detach()
        scale = adjusted_loss.sum() / adjusted_loss
        self.adv_probs = self.adv_probs * torch.exp(- self.robust_step_size * adjusted_loss)
        self.adv_probs = self.adv_probs / (self.adv_probs.sum())
        weight = scale * self.adv_probs

        weight = weight.unsqueeze(1)

        if self.num_groups >=2:
            # pdb.set_trace()
            results = SpectralClustering(n_clusters=self.num_groups,
                                         assign_labels='cluster_qr',
                                         affinity='linear',
                                         random_state=0).fit(weight.cpu())
            cluster_ids = results.labels_
            cluster_ids = torch.tensor(cluster_ids).unsqueeze(1).to(self.device)

            assignment_matrix = torch.zeros(self.n_tasks, self.num_groups).to(self.device)
            cluster_ids = cluster_ids.to(torch.int64)
            assignment_matrix.scatter_(1, cluster_ids, 1)
            assignment_matrix = assignment_matrix.to(torch.float64)

            cluster_centers = (assignment_matrix * weight).sum(dim=0) / assignment_matrix.sum(0)
            cluster_centers = cluster_centers.unsqueeze(1)

            group_weight = torch.mm(assignment_matrix, cluster_centers).squeeze(1) # 3*2, 2*1


        elif self.num_groups == 1:
            group_weight = torch.ones(self.n_tasks).to(self.device)
            group_weight = group_weight * torch.mean(weight)

        loss = torch.sum(losses * group_weight)
        return loss, dict(weights=torch.cat([group_weight]))

class GROUP_sklearn_spectral_clustering_discretize(WeightMethod):
    def __init__(
            self,
            n_tasks: int,
            device: torch.device,
            num_groups: int,
            task_weights: Union[List[float], torch.Tensor] = None,
            robust_step_size: float = 0.0001,
    ):
        super().__init__(n_tasks, device=device)
        self.n_tasks = n_tasks
        self.device = device
        if task_weights is None:
            task_weights = torch.ones((n_tasks,))
        if not isinstance(task_weights, torch.Tensor):
            task_weights = torch.tensor(task_weights)
        assert len(task_weights) == n_tasks
        self.task_weights = task_weights.to(device)

        self.adv_probs = torch.ones(n_tasks).to(device) / n_tasks
        self.robust_step_size = robust_step_size
        self.num_groups = num_groups

    def get_weighted_loss(self, losses, **kwargs):
        adjusted_loss = losses.detach()
        scale = adjusted_loss.sum() / adjusted_loss
        self.adv_probs = self.adv_probs * torch.exp(- self.robust_step_size * adjusted_loss)
        self.adv_probs = self.adv_probs / (self.adv_probs.sum())
        weight = scale * self.adv_probs

        weight = weight.unsqueeze(1)

        if self.num_groups >=2:
            # pdb.set_trace()
            results = SpectralClustering(n_clusters=self.num_groups,
                                         assign_labels='discretize',
                                         affinity='linear',
                                         random_state=0).fit(weight.cpu())
            cluster_ids = results.labels_
            cluster_ids = torch.tensor(cluster_ids).unsqueeze(1).to(self.device)

            assignment_matrix = torch.zeros(self.n_tasks, self.num_groups).to(self.device)
            cluster_ids = cluster_ids.to(torch.int64)
            assignment_matrix.scatter_(1, cluster_ids, 1)
            assignment_matrix = assignment_matrix.to(torch.float64)

            cluster_centers = (assignment_matrix * weight).sum(dim=0) / assignment_matrix.sum(0)
            cluster_centers = cluster_centers.unsqueeze(1)

            group_weight = torch.mm(assignment_matrix, cluster_centers).squeeze(1) # 3*2, 2*1


        elif self.num_groups == 1:
            group_weight = torch.ones(self.n_tasks).to(self.device)
            group_weight = group_weight * torch.mean(weight)

        loss = torch.sum(losses * group_weight)
        return loss, dict(weights=torch.cat([group_weight]))

class GROUP_sklearn_spectral_clustering_kmeans(WeightMethod):
    def __init__(
            self,
            n_tasks: int,
            device: torch.device,
            num_groups: int,
            task_weights: Union[List[float], torch.Tensor] = None,
            robust_step_size: float = 0.0001,
    ):
        super().__init__(n_tasks, device=device)
        self.n_tasks = n_tasks
        self.device = device
        if task_weights is None:
            task_weights = torch.ones((n_tasks,))
        if not isinstance(task_weights, torch.Tensor):
            task_weights = torch.tensor(task_weights)
        assert len(task_weights) == n_tasks
        self.task_weights = task_weights.to(device)

        self.adv_probs = torch.ones(n_tasks).to(device) / n_tasks
        self.robust_step_size = robust_step_size
        self.num_groups = num_groups

    def get_weighted_loss(self, losses, **kwargs):
        adjusted_loss = losses.detach()
        scale = adjusted_loss.sum() / adjusted_loss
        self.adv_probs = self.adv_probs * torch.exp(- self.robust_step_size * adjusted_loss)
        self.adv_probs = self.adv_probs / (self.adv_probs.sum())
        weight = scale * self.adv_probs

        weight = weight.unsqueeze(1)

        if self.num_groups >=2:
            # pdb.set_trace()
            results = SpectralClustering(n_clusters=self.num_groups,
                                         assign_labels='kmeans',
                                         affinity='linear',
                                         random_state=0).fit(weight.cpu())
            cluster_ids = results.labels_
            cluster_ids = torch.tensor(cluster_ids).unsqueeze(1).to(self.device)

            assignment_matrix = torch.zeros(self.n_tasks, self.num_groups).to(self.device)
            cluster_ids = cluster_ids.to(torch.int64)
            assignment_matrix.scatter_(1, cluster_ids, 1)
            assignment_matrix = assignment_matrix.to(torch.float64)

            cluster_centers = (assignment_matrix * weight).sum(dim=0) / assignment_matrix.sum(0)
            cluster_centers = cluster_centers.unsqueeze(1)

            group_weight = torch.mm(assignment_matrix, cluster_centers).squeeze(1) # 3*2, 2*1


        elif self.num_groups == 1:
            group_weight = torch.ones(self.n_tasks).to(self.device)
            group_weight = group_weight * torch.mean(weight)

        loss = torch.sum(losses * group_weight)
        return loss, dict(weights=torch.cat([group_weight]))

class GROUP_sdp_clustering(WeightMethod):
    def __init__(
            self,
            n_tasks: int,
            device: torch.device,
            num_groups: int,
            task_weights: Union[List[float], torch.Tensor] = None,
            robust_step_size: float = 0.0001,
    ):
        super().__init__(n_tasks, device=device)
        self.n_tasks = n_tasks
        self.device = device
        if task_weights is None:
            task_weights = torch.ones((n_tasks,))
        if not isinstance(task_weights, torch.Tensor):
            task_weights = torch.tensor(task_weights)
        assert len(task_weights) == n_tasks
        self.task_weights = task_weights.to(device)

        self.adv_probs = torch.ones(n_tasks).to(device) / n_tasks
        self.robust_step_size = robust_step_size
        self.num_groups = num_groups

    def get_weighted_loss(self, losses, **kwargs):
        adjusted_loss = losses.detach()
        scale = adjusted_loss.sum() / adjusted_loss
        self.adv_probs = self.adv_probs * torch.exp(- self.robust_step_size * adjusted_loss)
        self.adv_probs = self.adv_probs / (self.adv_probs.sum())
        weight = scale * self.adv_probs

        weight = weight.unsqueeze(1)

        if self.num_groups >=2:
            D, Q = sdp_kmeans(X=weight.cpu().numpy(), n_clusters=self.num_groups, method='cvx')
            assignment_matrix_init = connected_components(Q)
            assignment_matrix = torch.tensor(assignment_matrix_init).to(self.device)
            assignment_matrix = assignment_matrix.transpose(0,1).to(torch.float64)

            cluster_centers = (assignment_matrix * weight).sum(dim=0) / assignment_matrix.sum(0)
            cluster_centers = cluster_centers.unsqueeze(1)

            group_weight = torch.mm(assignment_matrix, cluster_centers).squeeze(1) # 3*2, 2*1

        elif self.num_groups == 1:
            group_weight = torch.ones(self.n_tasks).to(self.device)
            group_weight = group_weight * torch.mean(weight)

        loss = torch.sum(losses * group_weight)
        return loss, dict(weights=torch.cat([group_weight]))

class GO4ALIGNCOV(WeightMethod):
    """
    GO4ALIGN with Covariance Penalty (GO4ALIGN-Cov)
    在GO4ALIGN基础上添加了任务协方差惩罚项
    """
    def __init__(
            self,
            n_tasks: int,
            device: torch.device,
            num_groups: int,
            task_weights: Union[List[float], torch.Tensor] = None,
            robust_step_size: float = 0.0001,
            max_norm: float = 1.0,
            # --- 新增参数 ---
            gamma3: float = 1.0,        # 惩罚强度
            cov_history_size: int = 10  # 协方差历史窗口
    ):
        super().__init__(n_tasks, device=device)
        self.n_tasks = n_tasks
        self.device = device
        self.max_norm = max_norm
        if task_weights is None:
            task_weights = torch.ones((n_tasks,))
        if not isinstance(task_weights, torch.Tensor):
            task_weights = torch.tensor(task_weights)
        assert len(task_weights) == n_tasks
        self.task_weights = task_weights.to(device)

        self.adv_probs = torch.ones(n_tasks).to(device) / n_tasks
        self.robust_step_size = robust_step_size
        self.num_groups = num_groups
        # 协方差惩罚相关参数
        self.gamma3 = gamma3
        self.cov_history_size = cov_history_size
        
        # 历史缓冲区
        self.loss_change_history = torch.zeros(cov_history_size, n_tasks).to(device)
        self.history_ptr = 0
        self.history_len = 0
        
        self.prev_loss = None

    def _update_history(self, curr_loss):
        """计算 Log-Delta 并更新历史缓冲区"""
        if self.prev_loss is None:
            self.prev_loss = curr_loss.detach()
            return

        # 计算对数损失变化量: log(L_prev) - log(L_curr)
        # 加上 1e-8 防止 log(0)
        delta_L = (self.prev_loss + 1e-8).log() - (curr_loss.detach() + 1e-8).log()
        
        # 更新循环缓冲区
        self.loss_change_history[self.history_ptr] = delta_L.detach()
        self.history_ptr = (self.history_ptr + 1) % self.cov_history_size
        self.history_len = min(self.history_len + 1, self.cov_history_size)
        
        self.prev_loss = curr_loss.detach()

    def _compute_covariance_penalty(self):
        """向量化计算协方差惩罚项"""
        if self.history_len < 2:
            return torch.zeros(self.n_tasks, device=self.device)
        
        # 1. 获取有效历史数据 [N, M]
        valid_history = self.loss_change_history[:self.history_len]
        
        # 2. 计算中心化数据
        centered = valid_history - valid_history.mean(dim=0, keepdim=True)
        
        # 3. 计算协方差矩阵 [M, M]
        # 注意：分母是 N-1 (无偏估计)
        cov_matrix = (centered.T @ centered) / (self.history_len - 1)
        
        # 4. 计算惩罚项
        # 创建对角线掩码 (排除任务自身的方差)
        mask = ~torch.eye(self.n_tasks, dtype=torch.bool, device=self.device)
        # 只取负协方差 (冲突)
        min_cov = torch.minimum(cov_matrix, torch.tensor(0.0, device=self.device))
        
        # sum_{j!=i} min(0, Cov_ij)
        penalty = self.gamma3 * (min_cov * mask.float()).sum(dim=1)
        
        return penalty

    def get_weighted_loss(self, losses, **kwargs):
        adjusted_loss = losses.detach()
        
        # --- 步骤 1: 更新历史并计算惩罚 ---
        self._update_history(adjusted_loss)
        penalty = self._compute_covariance_penalty()
        
        # --- 步骤 2: 计算基础指标 ---
        scale = adjusted_loss.sum() / adjusted_loss
        self.adv_probs = self.adv_probs * torch.exp(- self.robust_step_size * adjusted_loss)
        self.adv_probs = self.adv_probs / (self.adv_probs.sum())
        
        # 组合得到基础特征权重，应用协方差惩罚
        weight = scale * self.adv_probs * torch.exp(penalty)

        # --- 步骤 3: K-Means 聚类平滑 ---
        weight = weight.unsqueeze(1)
        if self.num_groups >=2:
            cluster_ids_x, cluster_centers = kmeans(X=weight, num_clusters=self.num_groups, distance='euclidean', device=self.device)
            mask = torch.zeros(self.n_tasks, self.num_groups).to(self.device)
            cluster_ids = cluster_ids_x.unsqueeze(1).to(self.device)
            cluster_centers = cluster_centers.to(self.device)
            mask.scatter_(1, cluster_ids, 1)
            kmeans_weight = torch.mm(mask,cluster_centers).squeeze(1)

        elif self.num_groups == 1:
            kmeans_weight = torch.ones(self.n_tasks).to(self.device)
            kmeans_weight = kmeans_weight * torch.mean(weight)

        # 计算最终加权 Loss
        loss = torch.sum(losses * kmeans_weight)
        return loss, dict(weights=torch.cat([kmeans_weight]))