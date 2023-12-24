from typing import Optional

import torch

from vector_vis_graph.utils import project_onto_tensor


def calculate_weight(
    vector_a: torch.Tensor,
    vector_b: torch.Tensor,
    time_a: torch.Tensor,
    time_b: torch.Tensor,
    weighted: bool = False,
) -> torch.Tensor:
    if weighted:
        inner = torch.dot(vector_a, vector_b) / (vector_a.norm() * vector_b.norm())
        return inner / torch.abs(time_b - time_a)
    else:
        return torch.tensor(1.0)


def _natural_vvg_loop(
    multivariate_tensor: torch.Tensor,
    timeline: torch.Tensor,
    projections: torch.Tensor,
    weighted: bool,
    device: torch.device,
) -> torch.Tensor:
    time_length = timeline.size(0)
    vvg_adjacency = torch.zeros(time_length, time_length, device=device)
    for a in range(time_length - 1):
        t_a = timeline[a]
        x_aa = projections[a, a]

        for b in range(a + 1, time_length):
            x_ab = projections[a, b]
            t_b = timeline[b]

            x_acs = projections[a, a + 1 : b]
            t_cs = timeline[a + 1 : b]

            lhs = torch.div(x_acs - x_ab, t_b - t_cs)
            rhs = torch.div(x_aa - x_ab, t_b - t_a)

            if torch.all(lhs < rhs):
                vvg_adjacency[a, b] = calculate_weight(
                    multivariate_tensor[a],
                    multivariate_tensor[b],
                    t_a,
                    t_b,
                    weighted=weighted,
                )

    return vvg_adjacency


def natural_vvg(
    multivariate_tensor: torch.Tensor,
    *,
    timeline: Optional[torch.Tensor] = None,
    weighted: bool = False,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if device is None:
        device = multivariate_tensor.device

    if timeline is None:
        timeline = torch.arange(multivariate_tensor.size(0), device=device)
    elif len(timeline.shape) != 1:
        raise ValueError(f"timeline must be a 1D tensor, got {timeline.shape}")
    elif multivariate_tensor.size(0) != timeline.size(0):
        raise ValueError(
            "multivariate_tensor and timeline must have the same length, "
            f"got {multivariate_tensor.size(0)} and {timeline.size(0)}"
        )

    projections = project_onto_tensor(multivariate_tensor, multivariate_tensor)
    return _natural_vvg_loop(
        multivariate_tensor,
        timeline,
        projections,
        weighted,
        device,
    )
