from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from trajdata.utils import arr_utils


class SimStatistic:
    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, scene_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()


class VelocityHistogram(SimStatistic):
    def __init__(self, bins: List[int]) -> None:
        super().__init__("vel_hist")
        self.bins = bins

    def __call__(self, scene_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        velocities: np.ndarray = np.linalg.norm(scene_df[["vx", "vy"]], axis=1)

        return np.histogram(velocities, bins=self.bins)


class LongitudinalAccHistogram(SimStatistic):
    def __init__(self, bins: List[int]) -> None:
        super().__init__("lon_acc_hist")
        self.bins = bins

    def __call__(self, scene_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        accels: np.ndarray = np.linalg.norm(scene_df[["ax", "ay"]], axis=1)
        lon_accels: np.ndarray = accels * np.cos(scene_df["heading"])

        return np.histogram(lon_accels, bins=self.bins)


class LateralAccHistogram(SimStatistic):
    def __init__(self, bins: List[int]) -> None:
        super().__init__("lat_acc_hist")
        self.bins = bins

    def __call__(self, scene_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        accels: np.ndarray = np.linalg.norm(scene_df[["ax", "ay"]], axis=1)
        lat_accels: np.ndarray = accels * np.sin(scene_df["heading"])

        return np.histogram(lat_accels, bins=self.bins)


class JerkHistogram(SimStatistic):
    def __init__(self, bins: List[int], dt: float) -> None:
        super().__init__("jerk_hist")
        self.bins = bins
        self.dt = dt

    def __call__(self, scene_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        accels: np.ndarray = np.linalg.norm(scene_df[["ax", "ay"]], axis=1)
        jerk: np.ndarray = (
            arr_utils.agent_aware_diff(accels, scene_df.index.get_level_values(0))
            / self.dt
        )

        return np.histogram(jerk, bins=self.bins)


def calc_stats(
    positions: Tensor, heading: Tensor, dt: float, bins: Dict[str, Tensor], disable_control_on_stationary=False
) -> Dict[str, Tensor]:
    """Calculate scene statistics for a simulated scene.

    Args:
        positions (Tensor): N x T x 2 tensor of agent positions (in world coordinates).
        heading (Tensor): N x T x 1 tensor of agent headings (in world coordinates).
        dt (float): The data's delta timestep.
        bins (Dict[str, Tensor]): A mapping from statistic name to a Tensor of bin edges.

    Returns:
        Dict[str, Tensor]: A mapping of value names to histograms.
    """

    velocity: Tensor = (
        torch.diff(
            positions,
            dim=1,
            prepend=positions[:, [0]] - (positions[:, [1]] - positions[:, [0]]),
        )
        / dt
    )

    velocity_norm: Tensor = torch.linalg.vector_norm(velocity, dim=-1)
    # replace nan with 0
    positions = torch.where(torch.isnan(positions), torch.zeros_like(positions), positions)
    # consider those valid timesteps
    valid_mask = positions.sum(-1) != 0
    # consider agents with at least 2 timesteps
    valid_mask2 = positions[:, 1].sum(-1) != 0
    valid_mask = valid_mask & valid_mask2.unsqueeze(-1)

    if disable_control_on_stationary:
        if disable_control_on_stationary == 'current_speed':
            moving_mask = velocity_norm[:, 0] > 5e-1
            valid_mask = valid_mask & moving_mask.unsqueeze(-1)
        elif disable_control_on_stationary == 'any_speed':
            moving = velocity_norm > 5e-1
            moving_mask = moving.any(-1)
            valid_mask = valid_mask & moving_mask.unsqueeze(-1)
        # TBD: support on_lane
        else:
            raise NotImplementedError()
    # print('velocity_norm', velocity_norm.shape, velocity_norm)
    # print('valid_mask', valid_mask.shape, valid_mask)    

    accel: Tensor = (
        torch.diff(
            velocity,
            dim=1,
            prepend=velocity[:, [0]] - (velocity[:, [1]] - velocity[:, [0]]),
        )
        / dt
    )
    accel_norm: Tensor = torch.linalg.vector_norm(accel, dim=-1)

    lon_acc: Tensor = accel_norm * torch.cos(heading.squeeze(-1))
    lat_acc: Tensor = accel_norm * torch.sin(heading.squeeze(-1))

    jerk: Tensor = (
        torch.diff(
            accel_norm,
            dim=1,
            prepend=accel_norm[:, [0]] - (accel_norm[:, [1]] - accel_norm[:, [0]]),
        )
        / dt
    )

    return {
        "velocity": torch.histogram(velocity_norm[valid_mask], bins["velocity"]),
        "lon_accel": torch.histogram(lon_acc[valid_mask], bins["lon_accel"]),
        "lat_accel": torch.histogram(lat_acc[valid_mask], bins["lat_accel"]),
        "jerk": torch.histogram(jerk[valid_mask], bins["jerk"]),
    }
