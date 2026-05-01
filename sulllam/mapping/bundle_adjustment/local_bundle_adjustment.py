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
    # Huber radius (pixels). Reprojection errors above this enter Huber's
    # linear regime, where the gradient saturates. Setting it too small
    # (e.g. 2 px) on a pipeline whose pre-BA reprojection errors are
    # routinely 10-30 px stalls LM — it can't find a step that lowers
    # the saturated cost. 5 px gives the optimizer enough quadratic basin
    # to find a useful direction while still robustifying against outliers.
    huber_radius: float = 5.0
    max_iterations: int = 15
    abs_err_tolerance: float = 1e-4
    rel_err_tolerance: float = 1e-4
    # LM step-size scaling. ``1.0`` (the Theseus default) takes the full
    # Gauss-Newton step on each accepted iteration, which on this problem
    # tends to overshoot the basin and gets every step rejected. ``0.2``
    # is conservative enough that the first few iterations actually move
    # things in the right direction.
    step_size: float = 0.2
    use_match_confidence: bool = False
    confidence_gamma: float = 1.0
    use_adaptive_barron: bool = False
    barron_alpha_min: float = -2.0
    barron_alpha_max: float = 2.0


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

        # Pre-filter: skip dead points and observations that fail cheirality
        # in the *current* state. After PGO with rigid corrections, multi-
        # view points can land behind some of their observers; including
        # those produces near-singular Jacobians that crash the Cholesky
        # solver in LM.
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
                R_p = pose[:3, :3]
                t_p = pose[:3, 3]
                pt = mapper.pointmap.points_3d[pt_id]
                if not np.all(np.isfinite(pt)):
                    continue
                z = float((R_p @ pt + t_p)[2])
                if z > 1e-3:
                    keep[j] = True
            ba_obs = ba_obs[keep]
            dropped = pre_n - len(ba_obs)
            if dropped > 0:
                print(f"[BA] Pre-filtered {dropped} observations "
                      f"(dead points + cheirality failures)")

        local_point_ids = np.unique(ba_obs[:, 0]).astype(int) if len(ba_obs) else np.array([], dtype=int)

        if len(local_point_ids) == 0:
            print("[BA] No 3D points in the local window. Skipping.")
            return

        print(f"[BA] {len(local_point_ids)} points, {len(ba_obs)} observations")

        kf_scores = {}
        if cfg.use_match_confidence:
            for kf in local_kfs:
                if kf.match_scores is not None and len(kf.match_scores) > 0:
                    kf_scores[kf.idx] = kf.match_scores
            if kf_scores:
                print(f"[BA] Using match confidence weights from {len(kf_scores)} keyframes")

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
        for o_idx, o in enumerate(ba_obs):
            pt_id = int(o[0])
            kf_id = int(o[1])
            # Observation index makes names unique even when multiple
            # keypoints in the same keyframe map to the same point (after
            # cross-LC duplicate fusion).
            uname = f"{pt_id}_{kf_id}_{o_idx}"
            uv_obs = torch.tensor([[o[2], o[3]]], dtype=torch.float64)
            uv_var = th.Vector(tensor=uv_obs, name=f"uv_{uname}")
            cam_var = se3_vars[kf_id]
            pt_var = pt_vars[pt_id]

            weight_scalar = 1.0
            if cfg.use_match_confidence and kf_id in kf_scores:
                scores = kf_scores[kf_id]
                if len(scores) > 0:
                    avg_score = float(np.mean(scores))
                    weight_scalar = avg_score ** cfg.confidence_gamma

            edge_weight = th.ScaleCostWeight(torch.tensor(weight_scalar, dtype=torch.float64))

            if kf_id == fixed_kf_id:
                cost_fn = th.AutoDiffCostFunction(
                    optim_vars=[pt_var],
                    err_fn=_reproj_error_fixed_cam,
                    dim=2,
                    aux_vars=[cam_var, K_var, uv_var],
                    cost_weight=edge_weight,
                    name=f"cost_{uname}",
                )
                fixed_edges += 1
            else:
                cost_fn = th.AutoDiffCostFunction(
                    optim_vars=[cam_var, pt_var],
                    err_fn=_reproj_error_opt_cam,
                    dim=2,
                    aux_vars=[K_var, uv_var],
                    cost_weight=edge_weight,
                    name=f"cost_{uname}",
                )
                opt_edges += 1

            objective.add(th.RobustCostFunction(
                cost_function=cost_fn,
                loss_cls=th.HuberLoss,
                log_loss_radius=log_loss_radius,
                name=f"robust_{uname}",
            ))

        print(f"[BA] Graph: {fixed_edges} fixed-cam edges, {opt_edges} opt-cam edges")

        objective.update()
        initial_error = objective.error_metric().sum().item()
        print(f"[BA] Initial error: {initial_error:.4f}")

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
            # Cholmod's not-positive-definite error and similar numerical
            # failures shouldn't take the whole pipeline down — the map
            # state is still whatever it was before this BA call.
            print(f"[BA] LM optimize raised {type(exc).__name__}: {exc}")
            print(f"[BA] --- Local Bundle Adjustment Complete ---\n")
            return

        final_error = objective.error_metric().sum().item()
        converged_iter = info.converged_iter[0].item()
        # Theseus reports ``converged_iter = -1`` when LM ran the full
        # iteration budget without hitting abs/rel tolerance. The "+1"
        # convention then prints "Iterations: 0", which is wrong and
        # misleading — LM actually executed ``max_iterations`` steps.
        if converged_iter < 0:
            iter_str = f"max ({cfg.max_iterations})"
        else:
            iter_str = str(converged_iter + 1)
        print(f"[BA] Status      : {info.status[0]}")
        print(f"[BA] Iterations  : {iter_str}")
        print(f"[BA] Final error : {final_error:.4f}  (Δ {initial_error - final_error:.4f})")

        # Discard when the optimizer left us in a worse or non-finite state.
        # Theseus's ``converged_iter`` is ``-1`` whenever the absolute/relative
        # tolerance wasn't reached, *even when LM accepted many steps*, so we
        # don't gate on it directly — the error-metric comparison is a
        # truer signal of whether to commit.
        if not np.isfinite(final_error) or final_error >= initial_error:
            print(f"[BA] Error increased or is non-finite — discarding results.")
            print(f"[BA] --- Local Bundle Adjustment Complete ---\n")
            return

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
