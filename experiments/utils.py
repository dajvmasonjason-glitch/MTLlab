import torch
import numpy as np
from typing import List, Dict, Union, Tuple


def list_of_float(values: str) -> List[float]:
    """
    Convert a string of comma-separated numbers into a list of floats
    
    Args:
        values: String containing comma-separated numbers
    
    Returns:
        List of floats extracted from the input string
    """
    return [float(v) for v in values.split(",")]


def get_task_weight(task_weights: Union[List[float], torch.Tensor, None], n_tasks: int, device: torch.device) -> torch.Tensor:
    """
    Get task weights for multi-task learning
    
    Args:
        task_weights: Task weights provided by user or None
        n_tasks: Number of tasks
        device: Device to move the tensor to
    
    Returns:
        Task weights as a tensor
    """
    if task_weights is None:
        task_weights = torch.ones((n_tasks,)).to(device)
    elif isinstance(task_weights, list):
        task_weights = torch.tensor(task_weights, device=device)
    elif isinstance(task_weights, np.ndarray):
        task_weights = torch.from_numpy(task_weights).to(device)
    
    assert len(task_weights) == n_tasks, f"Task weights length {len(task_weights)} does not match number of tasks {n_tasks}"
    return task_weights


def get_robust_step_size(robust_step_size: float) -> float:
    """
    Get robust step size for adversarial optimization
    
    Args:
        robust_step_size: User-provided robust step size
    
    Returns:
        Validated robust step size
    """
    if robust_step_size is None:
        return 0.0001
    return robust_step_size


def check_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Ensure tensor is on the correct device
    
    Args:
        tensor: Input tensor
        device: Target device
    
    Returns:
        Tensor moved to the target device
    """
    if tensor.device != device:
        return tensor.to(device)
    return tensor


def normalize_weights(weights: torch.Tensor) -> torch.Tensor:
    """
    Normalize weights to sum to 1
    
    Args:
        weights: Input weights
    
    Returns:
        Normalized weights
    """
    weight_sum = weights.sum()
    if weight_sum > 0:
        return weights / weight_sum
    return weights


def compute_task_adjusted_weights(losses: torch.Tensor, robust_step_size: float, adv_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute task-adjusted weights using adversarial probabilities
    
    Args:
        losses: Task losses
        robust_step_size: Step size for adversarial probability update
        adv_probs: Current adversarial probabilities
    
    Returns:
        Tuple of (adjusted weights, updated adversarial probabilities)
    """
    adjusted_loss = losses.detach()
    scale = adjusted_loss.sum() / adjusted_loss
    adv_probs = adv_probs * torch.exp(-robust_step_size * adjusted_loss)
    adv_probs = normalize_weights(adv_probs)
    weight = scale * adv_probs
    return weight, adv_probs


def create_assignment_matrix(cluster_ids: torch.Tensor, n_tasks: int, num_groups: int, device: torch.device) -> torch.Tensor:
    """
    Create assignment matrix from cluster IDs
    
    Args:
        cluster_ids: Cluster IDs for each task
        n_tasks: Number of tasks
        num_groups: Number of clusters/groups
        device: Device to create the matrix on
    
    Returns:
        Assignment matrix where each row corresponds to a task and each column to a cluster
    """
    assignment_matrix = torch.zeros(n_tasks, num_groups).to(device)
    cluster_ids = cluster_ids.to(torch.int64)
    assignment_matrix.scatter_(1, cluster_ids, 1)
    return assignment_matrix.to(torch.float64)


def compute_group_weights(assignment_matrix: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Compute group weights based on assignment matrix and task weights
    
    Args:
        assignment_matrix: Task-to-cluster assignment matrix
        weight: Task weights
    
    Returns:
        Group weights for each task
    """
    cluster_centers = (assignment_matrix * weight).sum(dim=0) / assignment_matrix.sum(0)
    cluster_centers = cluster_centers.unsqueeze(1)
    group_weight = torch.mm(assignment_matrix, cluster_centers).squeeze(1)
    return group_weight
