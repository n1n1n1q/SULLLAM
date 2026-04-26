from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import theseus as th

from sulllam.mapping.bundle_adjustment.base_bundle_adjustment import BaseBundleAdjustment


@dataclass
class LocalBundleAdjustmentConfig:
    window_size: int = 5
    huber_radius: float = 2.0
    max_iterations: int = 10
    abs_err_tolerance: float = 1e-4
    rel_err_tolerance: float = 1e-4


def _reproj_error_opt_cam(optim_vars, aux_vars):
    pose, point = optim_vars
    K_vec, uv = aux_vars

    pt_cam = pose.transform_from(point)
    K_tensor = K_vec.tensor
    uv_tensor = uv.tensor

    fx, fy, cx, cy = K_tensor[:, 0], K_tensor[:, 1], K_tensor[:, 2], K_tensor[:, 3]

    z = pt_cam[:, 2] + 1e-8
    u = fx * (pt_cam[:, 0] / z) + cx
    v = fy * (pt_cam[:, 1] / z) + cy

    return torch.stack([u, v], dim=1) - uv_tensor


def _reproj_error_fixed_cam(optim_vars, aux_vars):
    (point,) = optim_vars
    pose, K_vec, uv = aux_vars

    pt_cam = pose.transform_from(point)
    K_tensor = K_vec.tensor
    uv_tensor = uv.tensor

    fx, fy, cx, cy = K_tensor[:, 0], K_tensor[:, 1], K_tensor[:, 2], K_tensor[:, 3]

    z = pt_cam[:, 2] + 1e-8
    u = fx * (pt_cam[:, 0] / z) + cx
    v = fy * (pt_cam[:, 1] / z) + cy

    return torch.stack([u, v], dim=1) - uv_tensor


class LocalBundleAdjustment(BaseBundleAdjustment):
    def __init__(self, config: LocalBundleAdjustmentConfig | None = None):
        self.config = config or LocalBundleAdjustmentConfig()

    def run(self, mapper, K: np.ndarray) -> None:
        cfg = self.config
        print(f"\n[BA] --- Starting Local Bundle Adjustment ---")
        print(f"[BA] Total keyframes: {len(mapper.keyframes)}")

        if len(mapper.keyframes) < 2:
            print("[BA] Not enough keyframes. Skipping.")
            return

        local_kfs = mapper.keyframes[-cfg.window_size:]
        local_kf_ids = {kf.idx for kf in local_kfs}
        print(f"[BA] Local keyframe IDs: {sorted(local_kf_ids)}")

        all_obs = mapper.pointmap.observations[: mapper.pointmap.num_observations]
        if len(all_obs) == 0:
            print("[BA] No observations found. Skipping.")
            return

        ba_obs = all_obs[np.isin(all_obs[:, 1], list(local_kf_ids))]
        local_point_ids = np.unique(ba_obs[:, 0]).astype(int)

        if len(local_point_ids) == 0:
            print("[BA] No 3D points in the local window. Skipping.")
            return

        print(f"[BA] {len(local_point_ids)} points, {len(ba_obs)} observations")

        K_tensor = torch.tensor([[K[0, 0], K[1, 1], K[0, 2], K[1, 2]]], dtype=torch.float64)
        K_var = th.Vector(tensor=K_tensor, name="K_shared")

        objective = th.Objective(dtype=torch.float64)
        weight = th.ScaleCostWeight(torch.tensor(1.0, dtype=torch.float64))
        log_loss_radius = th.Vector(
            tensor=torch.tensor([[np.log(cfg.huber_radius)]], dtype=torch.float64),
            name="log_loss_radius",
        )

        se3_vars: dict[int, th.SE3] = {}
        for kf in local_kfs:
            pose_tensor = torch.from_numpy(kf.pose[:3, :]).unsqueeze(0).double()
            se3_vars[kf.idx] = th.SE3(tensor=pose_tensor, name=f"cam_{kf.idx}")

        pt_vars: dict[int, th.Point3] = {}
        for pt_id in local_point_ids:
            pt_tensor = torch.from_numpy(mapper.pointmap.points_3d[pt_id]).unsqueeze(0).double()
            pt_vars[pt_id] = th.Point3(tensor=pt_tensor, name=f"pt_{pt_id}")

        fixed_kf_id = local_kfs[0].idx
        print(f"[BA] Anchored (fixed) keyframe: {fixed_kf_id}")

        fixed_edges = opt_edges = 0
        for o in ba_obs:
            pt_id = int(o[0])
            kf_id = int(o[1])
            uv_obs = torch.tensor([[o[2], o[3]]], dtype=torch.float64)
            uv_var = th.Vector(tensor=uv_obs, name=f"uv_{pt_id}_{kf_id}")
            cam_var = se3_vars[kf_id]
            pt_var = pt_vars[pt_id]

            if kf_id == fixed_kf_id:
                cost_fn = th.AutoDiffCostFunction(
                    optim_vars=[pt_var],
                    err_fn=_reproj_error_fixed_cam,
                    dim=2,
                    aux_vars=[cam_var, K_var, uv_var],
                    cost_weight=weight,
                    name=f"cost_{pt_id}_{kf_id}",
                )
                fixed_edges += 1
            else:
                cost_fn = th.AutoDiffCostFunction(
                    optim_vars=[cam_var, pt_var],
                    err_fn=_reproj_error_opt_cam,
                    dim=2,
                    aux_vars=[K_var, uv_var],
                    cost_weight=weight,
                    name=f"cost_{pt_id}_{kf_id}",
                )
                opt_edges += 1

            objective.add(th.RobustCostFunction(
                cost_function=cost_fn,
                loss_cls=th.HuberLoss,
                log_loss_radius=log_loss_radius,
                name=f"robust_{pt_id}_{kf_id}",
            ))

        print(f"[BA] Graph: {fixed_edges} fixed-cam edges, {opt_edges} opt-cam edges")

        objective.update()
        initial_error = objective.error_metric().sum().item()
        print(f"[BA] Initial error: {initial_error:.4f}")

        optimizer = th.LevenbergMarquardt(
            objective,
            max_iterations=cfg.max_iterations,
            step_size=1.0,
            abs_err_tolerance=cfg.abs_err_tolerance,
            rel_err_tolerance=cfg.rel_err_tolerance,
            linearization_cls=th.SparseLinearization,
            linear_solver_cls=th.CholmodSparseSolver,
            vectorize=True,
        )
        info = optimizer.optimize()

        final_error = objective.error_metric().sum().item()
        print(f"[BA] Status      : {info.status[0]}")
        print(f"[BA] Iterations  : {info.converged_iter[0].item() + 1}")
        print(f"[BA] Final error : {final_error:.4f}  (Δ {initial_error - final_error:.4f})")

        for kf_id, cam_var in se3_vars.items():
            if kf_id == fixed_kf_id:
                continue
            pose_4x4 = np.eye(4)
            pose_4x4[:3, :] = cam_var.tensor.detach().cpu().numpy()[0]
            kf = next(k for k in mapper.keyframes if k.idx == kf_id)
            kf.pose = pose_4x4

        for pt_id, pt_var in pt_vars.items():
            mapper.pointmap.points_3d[pt_id] = pt_var.tensor.detach().cpu().numpy()[0]

        print(f"[BA] --- Local Bundle Adjustment Complete ---\n")
