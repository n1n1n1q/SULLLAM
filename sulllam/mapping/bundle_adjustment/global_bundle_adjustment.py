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
    # Per-point post-optimization rollback radius, expressed as a multiple
    # of the Huber radius. A point whose maximum reprojection residual in
    # any observer exceeds this threshold reverts to its pre-optimization
    # 3D position. Catches optimizer "improvements" that flip points behind
    # cameras or send them to infinity.
    rollback_radius_huber: float = 5.0
    # If a larger fraction of points than this is rolled back, we treat the
    # whole optimization as untrustworthy and revert all variables.
    max_rollback_fraction: float = 0.4
    # Cheirality epsilon: drop observations / roll back points where the
    # depth in any observer is below this (in camera-frame z).
    min_depth: float = 1e-3


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

        # Pre-filter dead points (merged duplicates) and observations that
        # fail cheirality at the *current* state. A point can have ended up
        # behind a camera after PGO; including those in BA gives the
        # optimizer free reign to "improve" by flipping signs, which it
        # exploits.
        if len(ba_obs) > 0:
            pre_n = len(ba_obs)
            keep = np.zeros(len(ba_obs), dtype=bool)
            kf_pose_by_id = {kf.idx: kf.pose for kf in mapper.keyframes}
            for j in range(len(ba_obs)):
                pt_id = int(ba_obs[j, 0])
                kf_id = int(ba_obs[j, 1])
                if mapper.pointmap.is_dead(pt_id):
                    continue
                pose = kf_pose_by_id.get(kf_id)
                if pose is None:
                    continue
                R = pose[:3, :3]
                t = pose[:3, 3]
                pt = mapper.pointmap.points_3d[pt_id]
                z = float((R @ pt + t)[2])
                if z > cfg.min_depth:
                    keep[j] = True
            ba_obs = ba_obs[keep]
            dropped = pre_n - len(ba_obs)
            if dropped > 0:
                print(f"[GBA] Pre-filtered {dropped} observations "
                      f"(dead points + cheirality failures)")

        local_point_ids = np.unique(ba_obs[:, 0]).astype(int) if len(ba_obs) else np.array([], dtype=int)

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
        for o_idx, o in enumerate(ba_obs):
            pt_id = int(o[0])
            kf_id = int(o[1])
            if kf_id not in se3_vars or pt_id not in pt_vars:
                continue

            # Observation index makes names unique even when multiple
            # keypoints in the same keyframe map to the same point (which
            # can happen after cross-LC duplicate fusion).
            uname = f"{pt_id}_{kf_id}_{o_idx}"

            uv_obs = torch.tensor([[o[2], o[3]]], dtype=torch.float64)
            uv_var = th.Vector(tensor=uv_obs, name=f"gba_uv_{uname}")
            cam_var = se3_vars[kf_id]
            pt_var = pt_vars[pt_id]

            if kf_id == anchor_id:
                cost_fn = th.AutoDiffCostFunction(
                    optim_vars=[pt_var],
                    err_fn=_reproj_error_fixed_cam,
                    dim=2,
                    aux_vars=[cam_var, K_var, uv_var],
                    cost_weight=weight,
                    name=f"gba_cost_{uname}",
                )
                fixed_edges += 1
            else:
                cost_fn = th.AutoDiffCostFunction(
                    optim_vars=[cam_var, pt_var],
                    err_fn=_reproj_error_opt_cam,
                    dim=2,
                    aux_vars=[K_var, uv_var],
                    cost_weight=weight,
                    name=f"gba_cost_{uname}",
                )
                opt_edges += 1

            objective.add(th.RobustCostFunction(
                cost_function=cost_fn,
                loss_cls=th.HuberLoss,
                log_loss_radius=log_loss_radius,
                name=f"gba_robust_{uname}",
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
        try:
            info = optimizer.optimize()
        except Exception as exc:
            print(f"[GBA] LM optimize raised {type(exc).__name__}: {exc}")
            print(f"[GBA] --- Global Bundle Adjustment Complete ---\n")
            return

        final_error = objective.error_metric().sum().item()
        converged_iter = info.converged_iter[0].item()
        if converged_iter < 0:
            iter_str = f"max ({cfg.max_iterations})"
        else:
            iter_str = str(converged_iter + 1)
        print(f"[GBA] Status      : {info.status[0]}")
        print(f"[GBA] Iterations  : {iter_str}")
        print(f"[GBA] Final error : {final_error:.4f}  (Δ {initial_error - final_error:.4f})")

        # See LBA — only discard on non-finite or non-improving final state.
        if not np.isfinite(final_error) or final_error >= initial_error:
            print(f"[GBA] Error increased or is non-finite — discarding results.")
            print(f"[GBA] --- Global Bundle Adjustment Complete ---\n")
            return

        # Snapshot pre-optimization state so we can selectively roll back.
        pre_poses = {kf_id: mapper.keyframe_by_idx(kf_id).pose.copy()
                     for kf_id in se3_vars
                     if mapper.keyframe_by_idx(kf_id) is not None}
        pre_points = {pt_id: mapper.pointmap.points_3d[pt_id].copy()
                      for pt_id in pt_vars}

        # Apply the optimization results.
        for kf_id, cam_var in se3_vars.items():
            if kf_id == anchor_id:
                continue
            kf = mapper.keyframe_by_idx(kf_id)
            if kf is None:
                continue
            pose_4x4 = np.eye(4)
            pose_4x4[:3, :] = cam_var.tensor.detach().cpu().numpy()[0]
            kf.pose = pose_4x4

        for pt_id, pt_var in pt_vars.items():
            mapper.pointmap.points_3d[pt_id] = pt_var.tensor.detach().cpu().numpy()[0]

        # Per-point rollback: drop a point's update if it ends up with a
        # blown-up reprojection residual or behind a camera in any observer.
        rollback_radius = cfg.rollback_radius_huber * cfg.huber_radius
        pt_to_obs: dict[int, list[tuple[int, np.ndarray]]] = {}
        for o in ba_obs:
            pt_id = int(o[0])
            kf_id = int(o[1])
            uv = np.array([o[2], o[3]], dtype=np.float64)
            pt_to_obs.setdefault(pt_id, []).append((kf_id, uv))

        rolled_back = 0
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        for pt_id in list(pt_vars.keys()):
            new_pt = mapper.pointmap.points_3d[pt_id]
            obs_list = pt_to_obs.get(pt_id, [])
            bad = False
            for kf_id, uv_obs in obs_list:
                kf = mapper.keyframe_by_idx(kf_id)
                if kf is None:
                    continue
                pt_cam = kf.R @ new_pt + kf.t
                z = pt_cam[2]
                if z <= cfg.min_depth:
                    bad = True
                    break
                u = fx * pt_cam[0] / z + cx
                v = fy * pt_cam[1] / z + cy
                residual = float(np.hypot(u - uv_obs[0], v - uv_obs[1]))
                if residual > rollback_radius:
                    bad = True
                    break
            if bad:
                mapper.pointmap.points_3d[pt_id] = pre_points[pt_id]
                rolled_back += 1

        rollback_fraction = rolled_back / max(len(pt_vars), 1)
        print(f"[GBA] Rolled back {rolled_back}/{len(pt_vars)} points "
              f"({rollback_fraction:.1%}) over {rollback_radius:.1f}px residual / cheirality.")

        if rollback_fraction > cfg.max_rollback_fraction:
            print(f"[GBA] Rollback fraction exceeds {cfg.max_rollback_fraction:.0%} — "
                  f"reverting all variables.")
            for kf_id, pose in pre_poses.items():
                kf = mapper.keyframe_by_idx(kf_id)
                if kf is not None:
                    kf.pose = pose
            for pt_id, p in pre_points.items():
                mapper.pointmap.points_3d[pt_id] = p

        print(f"[GBA] --- Global Bundle Adjustment Complete ---\n")
