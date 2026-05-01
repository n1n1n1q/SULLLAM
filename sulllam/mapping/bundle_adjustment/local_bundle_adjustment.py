from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import theseus as th
import cv2

from sulllam.mapping.bundle_adjustment.base_bundle_adjustment import BaseBundleAdjustment
from sulllam.preprocessing.entropy import image_entropy, map_entropy_to_alpha


@dataclass
class LocalBundleAdjustmentConfig:
    window_size: int = 20
    huber_radius: float = 2.0
    max_iterations: int = 10
    abs_err_tolerance: float = 1e-4
    rel_err_tolerance: float = 1e-4
    use_match_confidence: bool = False
    confidence_gamma: float = 1.0
    use_adaptive_barron: bool = False
    barron_alpha_min: float = -2.0
    barron_alpha_max: float = 2.0
    num_fixed_keyframes: int = 2  # how many oldest KFs in window to fix (>=2 locks scale gauge)


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

        all_obs = mapper.pointmap.observations[: mapper.pointmap.num_observations]
        if len(all_obs) == 0:
            print("[BA] No observations found. Skipping.")
            return

        ba_obs = all_obs[np.isin(all_obs[:, 1], list(local_kf_ids))]
        observed_kf_ids = set(np.unique(ba_obs[:, 1]).astype(int).tolist())
        local_kfs = [kf for kf in local_kfs if kf.idx in observed_kf_ids]
        local_kf_ids = {kf.idx for kf in local_kfs}
        print(f"[BA] Local keyframe IDs (with observations): {sorted(local_kf_ids)}")
        local_point_ids = np.unique(ba_obs[:, 0]).astype(int)

        if len(local_point_ids) == 0:
            print("[BA] No 3D points in the local window. Skipping.")
            return

        # --- Pull in observations from keyframes OUTSIDE the window that see our local points.
        # These external cameras will be fixed (not optimized) but their reprojection edges
        # constrain the local 3D points so they stay consistent with the rest of the map.
        external_obs_mask = (
            np.isin(all_obs[:, 0], local_point_ids)
            & ~np.isin(all_obs[:, 1], list(local_kf_ids))
        )
        external_obs = all_obs[external_obs_mask]
        external_kf_ids = set(np.unique(external_obs[:, 1]).astype(int).tolist())
        print(f"[BA] {len(local_point_ids)} points, {len(ba_obs)} window obs, "
              f"{len(external_obs)} external obs from {len(external_kf_ids)} fixed external KFs")

        # --- Decide which keyframes are FIXED vs OPTIMIZED.
        # We fix the N oldest keyframes in the window (default N=2) so that SE(3) gauge
        # AND scale are both locked. With only one fixed pose, monocular BA has a free
        # scale direction and the Hessian is rank-deficient.
        local_kfs_sorted = sorted(local_kfs, key=lambda k: k.idx)
        n_fixed = max(2, cfg.num_fixed_keyframes)
        n_fixed = min(n_fixed, len(local_kfs_sorted))  # don't fix more than we have
        fixed_kf_ids_in_window = {kf.idx for kf in local_kfs_sorted[:n_fixed]}
        print(f"[BA] Anchored (fixed) window keyframes: {sorted(fixed_kf_ids_in_window)}")

        # --- Match-confidence weights (unchanged) ---
        kf_scores = {}
        if cfg.use_match_confidence:
            for kf in local_kfs:
                if kf.match_scores is not None and len(kf.match_scores) > 0:
                    kf_scores[kf.idx] = kf.match_scores
            if kf_scores:
                print(f"[BA] Using match confidence weights from {len(kf_scores)} keyframes")

        # --- Adaptive Barron (unchanged) ---
        kf_entropies = {}
        alpha_global = 0.0
        if cfg.use_adaptive_barron:
            for kf in local_kfs:
                if hasattr(kf, 'image') and kf.image is not None:
                    gray = cv2.cvtColor(kf.image, cv2.COLOR_BGR2GRAY) if len(kf.image.shape) == 3 else kf.image
                    ent = image_entropy(gray)
                    kf_entropies[kf.idx] = ent
                else:
                    kf_entropies[kf.idx] = 4.0
            if kf_entropies:
                mean_entropy = np.mean(list(kf_entropies.values()))
                alpha_global = map_entropy_to_alpha(mean_entropy, cfg.barron_alpha_min, cfg.barron_alpha_max)
                print(f"[BA] Adaptive Barron: entropy {mean_entropy:.3f} → alpha {alpha_global:.3f}")

        # --- Theseus setup ---
        K_tensor = torch.tensor([[K[0, 0], K[1, 1], K[0, 2], K[1, 2]]], dtype=torch.float64)
        K_var = th.Vector(tensor=K_tensor, name="K_shared")

        objective = th.Objective(dtype=torch.float64)
        log_loss_radius = th.Vector(
            tensor=torch.tensor([[np.log(cfg.huber_radius)]], dtype=torch.float64),
            name="log_loss_radius",
        )

        # SE3 variables for the WINDOW keyframes (some optimized, some fixed)
        se3_vars: dict[int, th.SE3] = {}
        for kf in local_kfs:
            pose_tensor = torch.from_numpy(kf.pose[:3, :]).unsqueeze(0).double()
            se3_vars[kf.idx] = th.SE3(tensor=pose_tensor, name=f"cam_{kf.idx}")

        # SE3 variables for EXTERNAL keyframes — these will be used only as aux_vars
        # (i.e. fixed) in the cost functions, so they will not be optimized regardless.
        kf_by_id = {kf.idx: kf for kf in mapper.keyframes}
        external_se3_vars: dict[int, th.SE3] = {}
        for ext_id in external_kf_ids:
            kf = kf_by_id.get(ext_id)
            if kf is None:
                continue
            pose_tensor = torch.from_numpy(kf.pose[:3, :]).unsqueeze(0).double()
            external_se3_vars[ext_id] = th.SE3(tensor=pose_tensor, name=f"cam_ext_{ext_id}")

        # 3D point variables (always optimized)
        pt_vars: dict[int, th.Point3] = {}
        for pt_id in local_point_ids:
            pt_tensor = torch.from_numpy(mapper.pointmap.points_3d[pt_id]).unsqueeze(0).double()
            pt_vars[pt_id] = th.Point3(tensor=pt_tensor, name=f"pt_{pt_id}")

        # --- Build the factor graph ---
        fixed_edges = opt_edges = ext_edges = 0

        # Window observations
        for o in ba_obs:
            pt_id = int(o[0])
            kf_id = int(o[1])
            uv_obs = torch.tensor([[o[2], o[3]]], dtype=torch.float64)
            uv_var = th.Vector(tensor=uv_obs, name=f"uv_{pt_id}_{kf_id}")
            cam_var = se3_vars[kf_id]
            pt_var = pt_vars[pt_id]

            weight_scalar = 1.0
            if cfg.use_match_confidence and kf_id in kf_scores:
                scores = kf_scores[kf_id]
                if len(scores) > 0:
                    avg_score = float(np.mean(scores))
                    weight_scalar = avg_score ** cfg.confidence_gamma

            edge_weight = th.ScaleCostWeight(torch.tensor(weight_scalar, dtype=torch.float64))

            if kf_id in fixed_kf_ids_in_window:
                # Camera is fixed — only point is optimized
                cost_fn = th.AutoDiffCostFunction(
                    optim_vars=[pt_var],
                    err_fn=_reproj_error_fixed_cam,
                    dim=2,
                    aux_vars=[cam_var, K_var, uv_var],
                    cost_weight=edge_weight,
                    name=f"cost_{pt_id}_{kf_id}",
                )
                fixed_edges += 1
            else:
                # Camera and point both optimized
                cost_fn = th.AutoDiffCostFunction(
                    optim_vars=[cam_var, pt_var],
                    err_fn=_reproj_error_opt_cam,
                    dim=2,
                    aux_vars=[K_var, uv_var],
                    cost_weight=edge_weight,
                    name=f"cost_{pt_id}_{kf_id}",
                )
                opt_edges += 1

            objective.add(th.RobustCostFunction(
                cost_function=cost_fn,
                loss_cls=th.HuberLoss,
                log_loss_radius=log_loss_radius,
                name=f"robust_{pt_id}_{kf_id}",
            ))

        # External observations — point optimized, external camera fixed (as aux_var)
        for o in external_obs:
            pt_id = int(o[0])
            kf_id = int(o[1])
            if kf_id not in external_se3_vars:
                continue
            uv_obs = torch.tensor([[o[2], o[3]]], dtype=torch.float64)
            uv_var = th.Vector(tensor=uv_obs, name=f"uv_ext_{pt_id}_{kf_id}")
            cam_var = external_se3_vars[kf_id]
            pt_var = pt_vars[pt_id]

            edge_weight = th.ScaleCostWeight(torch.tensor(1.0, dtype=torch.float64))

            cost_fn = th.AutoDiffCostFunction(
                optim_vars=[pt_var],
                err_fn=_reproj_error_fixed_cam,
                dim=2,
                aux_vars=[cam_var, K_var, uv_var],
                cost_weight=edge_weight,
                name=f"cost_ext_{pt_id}_{kf_id}",
            )
            ext_edges += 1

            objective.add(th.RobustCostFunction(
                cost_function=cost_fn,
                loss_cls=th.HuberLoss,
                log_loss_radius=log_loss_radius,
                name=f"robust_ext_{pt_id}_{kf_id}",
            ))

        print(f"[BA] Graph: {fixed_edges} fixed-cam edges (window), "
              f"{opt_edges} opt-cam edges, {ext_edges} external fixed-cam edges")

        objective.update()
        initial_error = objective.error_metric().sum().item()
        print(f"[BA] Initial error: {initial_error:.4f}")

        optimizer = th.LevenbergMarquardt(
            objective,
            max_iterations=cfg.max_iterations,
            # step_size=1.0,
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

        # --- Write back optimized poses (skip ALL fixed window keyframes) ---
        for kf_id, cam_var in se3_vars.items():
            if kf_id in fixed_kf_ids_in_window:
                continue
            pose_4x4 = np.eye(4)
            pose_4x4[:3, :] = cam_var.tensor.detach().cpu().numpy()[0]
            kf = next(k for k in mapper.keyframes if k.idx == kf_id)
            kf.pose = pose_4x4

        # --- Write back optimized points ---
        for pt_id, pt_var in pt_vars.items():
            mapper.pointmap.points_3d[pt_id] = pt_var.tensor.detach().cpu().numpy()[0]

        print(f"[BA] --- Local Bundle Adjustment Complete ---\n")