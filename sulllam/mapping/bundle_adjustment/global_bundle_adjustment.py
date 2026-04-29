from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import theseus as th

from sulllam.mapping.bundle_adjustment.base_bundle_adjustment import BaseBundleAdjustment
from sulllam.mapping.bundle_adjustment.local_bundle_adjustment import (
    _reproj_error_opt_cam,
    _reproj_error_fixed_cam,
)


@dataclass
class GlobalBundleAdjustmentConfig:
    huber_radius: float = 2.0
    max_iterations: int = 15
    step_size: float = 0.5
    abs_err_tolerance: float = 1e-4
    rel_err_tolerance: float = 1e-4
    max_observations: int = 50_000
    max_mean_error_per_obs: float = 50.0


class GlobalBundleAdjustment(BaseBundleAdjustment):
    def __init__(self, config: GlobalBundleAdjustmentConfig | None = None) -> None:
        self.config = config or GlobalBundleAdjustmentConfig()

    def run(self, mapper, K: np.ndarray) -> None:
        cfg = self.config
        print(f"\n[GBA] --- Starting Global Bundle Adjustment ---")
        print(f"[GBA] Total keyframes: {len(mapper.keyframes)}")

        if len(mapper.keyframes) < 2:
            print("[GBA] Not enough keyframes. Skipping.")
            return

        all_obs = mapper.pointmap.observations[: mapper.pointmap.num_observations]
        if len(all_obs) == 0:
            print("[GBA] No observations. Skipping.")
            return

        # Optionally subsample observations to keep runtime tractable
        if len(all_obs) > cfg.max_observations:
            idx = np.random.choice(len(all_obs), cfg.max_observations, replace=False)
            all_obs = all_obs[idx]
            print(f"[GBA] Subsampled to {cfg.max_observations} observations")

        all_kf_ids = {kf.idx for kf in mapper.keyframes}
        ba_obs = all_obs[np.isin(all_obs[:, 1], list(all_kf_ids))]
        local_point_ids = np.unique(ba_obs[:, 0]).astype(int)

        if len(local_point_ids) == 0:
            print("[GBA] No 3D points. Skipping.")
            return

        print(f"[GBA] {len(mapper.keyframes)} keyframes, {len(local_point_ids)} points, "
              f"{len(ba_obs)} observations")

        K_tensor = torch.tensor([[K[0, 0], K[1, 1], K[0, 2], K[1, 2]]], dtype=torch.float64)
        K_var = th.Vector(tensor=K_tensor, name="K_shared_gba")

        objective = th.Objective(dtype=torch.float64)
        weight = th.ScaleCostWeight(torch.tensor(1.0, dtype=torch.float64))
        log_loss_radius = th.Vector(
            tensor=torch.tensor([[np.log(cfg.huber_radius)]], dtype=torch.float64),
            name="log_loss_radius_gba",
        )

        se3_vars: dict[int, th.SE3] = {}
        for kf in mapper.keyframes:
            t = torch.from_numpy(kf.pose[:3, :]).unsqueeze(0).double()
            se3_vars[kf.idx] = th.SE3(tensor=t, name=f"gba_cam_{kf.idx}")

        pt_vars: dict[int, th.Point3] = {}
        for pt_id in local_point_ids:
            t = torch.from_numpy(mapper.pointmap.points_3d[pt_id]).unsqueeze(0).double()
            pt_vars[pt_id] = th.Point3(tensor=t, name=f"gba_pt_{pt_id}")

        anchor_id = mapper.keyframes[0].idx
        print(f"[GBA] Anchored keyframe: {anchor_id}")

        fixed_edges = opt_edges = 0
        for o in ba_obs:
            pt_id = int(o[0])
            kf_id = int(o[1])
            if kf_id not in se3_vars or pt_id not in pt_vars:
                continue

            uv_obs = torch.tensor([[o[2], o[3]]], dtype=torch.float64)
            uv_var = th.Vector(tensor=uv_obs, name=f"gba_uv_{pt_id}_{kf_id}")
            cam_var = se3_vars[kf_id]
            pt_var = pt_vars[pt_id]

            if kf_id == anchor_id:
                cost_fn = th.AutoDiffCostFunction(
                    optim_vars=[pt_var],
                    err_fn=_reproj_error_fixed_cam,
                    dim=2,
                    aux_vars=[cam_var, K_var, uv_var],
                    cost_weight=weight,
                    name=f"gba_cost_{pt_id}_{kf_id}",
                )
                fixed_edges += 1
            else:
                cost_fn = th.AutoDiffCostFunction(
                    optim_vars=[cam_var, pt_var],
                    err_fn=_reproj_error_opt_cam,
                    dim=2,
                    aux_vars=[K_var, uv_var],
                    cost_weight=weight,
                    name=f"gba_cost_{pt_id}_{kf_id}",
                )
                opt_edges += 1

            objective.add(th.RobustCostFunction(
                cost_function=cost_fn,
                loss_cls=th.HuberLoss,
                log_loss_radius=log_loss_radius,
                name=f"gba_robust_{pt_id}_{kf_id}",
            ))

        print(f"[GBA] Graph: {fixed_edges} fixed-cam edges, {opt_edges} opt-cam edges")

        objective.update()
        initial_error = objective.error_metric().sum().item()
        print(f"[GBA] Initial error: {initial_error:.4f}")
        mean_err_per_obs = initial_error / max(len(ba_obs), 1)
        if mean_err_per_obs > cfg.max_mean_error_per_obs:
            print(f"[GBA] Mean error per obs {mean_err_per_obs:.1f} exceeds threshold "
                  f"{cfg.max_mean_error_per_obs:.1f}. Skipping to avoid divergence.")
            print(f"[GBA] --- Global Bundle Adjustment Complete ---\n")
            return

        optimizer = th.LevenbergMarquardt(
            objective,
            max_iterations=cfg.max_iterations,
            step_size=cfg.step_size,
            abs_err_tolerance=cfg.abs_err_tolerance,
            rel_err_tolerance=cfg.rel_err_tolerance,
            linearization_cls=th.SparseLinearization,
            linear_solver_cls=th.CholmodSparseSolver,
            vectorize=True,
        )
        info = optimizer.optimize()

        final_error = objective.error_metric().sum().item()
        print(f"[GBA] Status      : {info.status[0]}")
        print(f"[GBA] Iterations  : {info.converged_iter[0].item() + 1}")
        print(f"[GBA] Final error : {final_error:.4f}  (Δ {initial_error - final_error:.4f})")

        if final_error > initial_error:
            print(f"[GBA] Error increased — discarding results.")
            print(f"[GBA] --- Global Bundle Adjustment Complete ---\n")
            return

        for kf_id, cam_var in se3_vars.items():
            if kf_id == anchor_id:
                continue
            kf = next(k for k in mapper.keyframes if k.idx == kf_id)
            pose_4x4 = np.eye(4)
            pose_4x4[:3, :] = cam_var.tensor.detach().cpu().numpy()[0]
            kf.pose = pose_4x4

        for pt_id, pt_var in pt_vars.items():
            mapper.pointmap.points_3d[pt_id] = pt_var.tensor.detach().cpu().numpy()[0]

        print(f"[GBA] --- Global Bundle Adjustment Complete ---\n")
