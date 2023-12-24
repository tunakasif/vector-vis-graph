import torch


def project_onto_vector(
    from_tensor: torch.Tensor,
    onto_vector: torch.Tensor,
) -> torch.Tensor:
    return torch.linalg.vecdot(from_tensor, onto_vector) / onto_vector.norm()


def project_onto_tensor(
    from_tensor: torch.Tensor,
    onto_tensor: torch.Tensor,
) -> torch.Tensor:
    batched = torch.vmap(project_onto_vector, in_dims=(None, 0))
    return batched(from_tensor, onto_tensor)
