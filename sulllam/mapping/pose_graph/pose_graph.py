from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import theseus as th


@dataclass
class PoseGraphEdge:
    from_id: int
    to_id: int
    relative_pose: np.ndarray
    information: np.ndarray = field(default_factory=lambda: np.eye(6, dtype=np.float64))
    is_loop_closure: bool = False
    # Rotation-only edges constrain only the orientation block of the
    # residual. Used for loop-closure edges (monocular scale ambiguity) and
    # for odometry edges where the camera was nearly stationary translation-
    # wise (pure rotation), making the essential-matrix t direction noise.
    rotation_only: bool = False


class PoseGraph:
    def __init__(self) -> None:
        self._edges: list[PoseGraphEdge] = []

    def add_odometry_edge(
        self,
        from_id: int,
        to_id: int,
        relative_pose: np.ndarray,
        information: np.ndarray | None = None,
        rotation_only: bool = False,
    ) -> None:
        info = information if information is not None else np.eye(6, dtype=np.float64)
        self._edges.append(PoseGraphEdge(
            from_id, to_id, relative_pose, info,
            is_loop_closure=False, rotation_only=rotation_only,
        ))

    def add_loop_closure_edge(
        self,
        from_id: int,
        to_id: int,
        relative_pose: np.ndarray,
        information: np.ndarray | None = None,
    ) -> None:
        info = information if information is not None else np.eye(6, dtype=np.float64) * 4.0
        # Loop-closure edges are intrinsically rotation-only because their
        # translation comes from a unit-norm essential-matrix decomposition.
        self._edges.append(PoseGraphEdge(
            from_id, to_id, relative_pose, info,
            is_loop_closure=True, rotation_only=True,
        ))

    @property
    def edges(self) -> list[PoseGraphEdge]:
        return self._edges

    @property
    def loop_closure_edges(self) -> list[PoseGraphEdge]:
        return [e for e in self._edges if e.is_loop_closure]

    def has_loop_closures(self) -> bool:
        return any(e.is_loop_closure for e in self._edges)


def _pose_graph_error(optim_vars, aux_vars):
    pose_from, pose_to = optim_vars
    (rel_pose,) = aux_vars
    predicted = pose_to.inverse().compose(rel_pose).compose(pose_from)
    return predicted.log_map()


def _pose_graph_error_fixed_from(optim_vars, aux_vars):
    (pose_to,) = optim_vars
    pose_from, rel_pose = aux_vars
    predicted = pose_to.inverse().compose(rel_pose).compose(pose_from)
    return predicted.log_map()


def _pose_graph_error_fixed_to(optim_vars, aux_vars):
    (pose_from,) = optim_vars
    pose_to, rel_pose = aux_vars
    predicted = pose_to.inverse().compose(rel_pose).compose(pose_from)
    return predicted.log_map()


def _pose_graph_rot_error(optim_vars, aux_vars):
    pose_from, pose_to = optim_vars
    (rel_pose,) = aux_vars
    predicted = pose_to.inverse().compose(rel_pose).compose(pose_from)
    return predicted.log_map()[..., 3:]


def _pose_graph_rot_error_fixed_from(optim_vars, aux_vars):
    (pose_to,) = optim_vars
    pose_from, rel_pose = aux_vars
    predicted = pose_to.inverse().compose(rel_pose).compose(pose_from)
    return predicted.log_map()[..., 3:]


def _pose_graph_rot_error_fixed_to(optim_vars, aux_vars):
    (pose_from,) = optim_vars
    pose_to, rel_pose = aux_vars
    predicted = pose_to.inverse().compose(rel_pose).compose(pose_from)
    return predicted.log_map()[..., 3:]


@dataclass
class PoseGraphOptimizerConfig:
    max_iterations: int = 20
    abs_err_tolerance: float = 1e-5
    rel_err_tolerance: float = 1e-5
    step_size: float = 1.0


class PoseGraphOptimizer:
    def __init__(self, config: PoseGraphOptimizerConfig | None = None) -> None:
        self.config = config or PoseGraphOptimizerConfig()

    def optimize(self, mapper, pose_graph: PoseGraph) -> None:
        cfg = self.config
        print(f"\n[PG] --- Starting Pose Graph Optimization ---")

        if len(mapper.keyframes) < 2:
            print("[PG] Not enough keyframes. Skipping.")
            return

        if not pose_graph.has_loop_closures():
            print("[PG] No loop closures. Skipping.")
            return

        node_ids: set[int] = set()
        for edge in pose_graph.edges:
            node_ids.add(edge.from_id)
            node_ids.add(edge.to_id)

        kf_map = {kf.idx: kf for kf in mapper.keyframes if kf.idx in node_ids}
        if len(kf_map) < 2:
            print("[PG] Insufficient keyframes in pose graph. Skipping.")
            return

        print(f"[PG] Nodes: {len(kf_map)}, Edges: {len(pose_graph.edges)} "
              f"({len(pose_graph.loop_closure_edges)} loop closures)")

        se3_vars: dict[int, th.SE3] = {}
        for kf_id, kf in kf_map.items():
            t = torch.from_numpy(kf.pose[:3, :]).unsqueeze(0).double()
            se3_vars[kf_id] = th.SE3(tensor=t, name=f"pg_cam_{kf_id}")

        anchor_id = min(kf_map.keys())
        print(f"[PG] Anchored keyframe: {anchor_id}")

        objective = th.Objective(dtype=torch.float64)

        for i, edge in enumerate(pose_graph.edges):
            if edge.from_id not in se3_vars or edge.to_id not in se3_vars:
                continue

            rel_t = torch.from_numpy(edge.relative_pose[:3, :]).unsqueeze(0).double()
            rel_var = th.SE3(tensor=rel_t, name=f"pg_rel_{i}")

            pose_from = se3_vars[edge.from_id]
            pose_to = se3_vars[edge.to_id]

            # Rotation-only edges (all loop closures, plus odometry edges
            # flagged as pure rotation) drop the translation block of the
            # residual. Translation-bearing odometry edges keep the full
            # 6-DoF residual.
            if edge.rotation_only:
                err_dim = 3
                err_full = _pose_graph_rot_error
                err_fixed_from = _pose_graph_rot_error_fixed_from
                err_fixed_to = _pose_graph_rot_error_fixed_to
                # Use the rotation block of the information matrix.
                info_block = edge.information[3:, 3:]
                diag = np.maximum(np.diag(info_block).astype(np.float64), 1e-9)
            else:
                err_dim = 6
                err_full = _pose_graph_error
                err_fixed_from = _pose_graph_error_fixed_from
                err_fixed_to = _pose_graph_error_fixed_to
                diag = np.maximum(np.diag(edge.information).astype(np.float64), 1e-9)

            # DiagonalCostWeight multiplies the residual element-wise, so the
            # squared cost picks up diag(weight)**2 per dimension. For a
            # Mahalanobis cost r^T diag(lambda_i) r the matching per-dim
            # weight is sqrt(lambda_i). This lets us downweight translation
            # vs rotation independently on odometry edges (the monocular
            # translation magnitude is uncertain, the rotation isn't).
            sqrt_diag = np.sqrt(diag)
            edge_weight = th.DiagonalCostWeight(
                torch.tensor(sqrt_diag, dtype=torch.float64).unsqueeze(0)
            )

            if edge.from_id == anchor_id:
                cost = th.AutoDiffCostFunction(
                    optim_vars=[pose_to],
                    err_fn=err_fixed_from,
                    dim=err_dim,
                    aux_vars=[pose_from, rel_var],
                    cost_weight=edge_weight,
                    name=f"pg_edge_{i}",
                )
            elif edge.to_id == anchor_id:
                cost = th.AutoDiffCostFunction(
                    optim_vars=[pose_from],
                    err_fn=err_fixed_to,
                    dim=err_dim,
                    aux_vars=[pose_to, rel_var],
                    cost_weight=edge_weight,
                    name=f"pg_edge_{i}",
                )
            else:
                cost = th.AutoDiffCostFunction(
                    optim_vars=[pose_from, pose_to],
                    err_fn=err_full,
                    dim=err_dim,
                    aux_vars=[rel_var],
                    cost_weight=edge_weight,
                    name=f"pg_edge_{i}",
                )
            objective.add(cost)

        if objective.size_cost_functions() == 0:
            print("[PG] No valid edges for optimization. Skipping.")
            return

        objective.update()
        initial_error = objective.error_metric().sum().item()
        print(f"[PG] Initial error: {initial_error:.4f}")

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
        converged_iter = info.converged_iter[0].item()
        if converged_iter < 0:
            iter_str = f"max ({cfg.max_iterations})"
        else:
            iter_str = str(converged_iter + 1)
        print(f"[PG] Status      : {info.status[0]}")
        print(f"[PG] Iterations  : {iter_str}")
        print(f"[PG] Final error : {final_error:.4f}  (delta {initial_error - final_error:.4f})")

        # Per-keyframe correction transform that takes a world point seen by
        # the OLD pose to its consistent location under the NEW pose:
        #     p_new = R_corr @ p_old + t_corr
        # with R_corr = R_wc_new^T @ R_wc_old and t_corr = R_wc_new^T @ (t_wc_old - t_wc_new).
        # The anchor keyframe's correction is identity by definition.
        pose_corrections: dict[int, tuple[np.ndarray, np.ndarray]] = {
            anchor_id: (np.eye(3), np.zeros(3))
        }
        for kf_id, cam_var in se3_vars.items():
            if kf_id == anchor_id:
                continue
            kf = kf_map[kf_id]
            old_pose = kf.pose.copy()
            new_3x4 = cam_var.tensor.detach().cpu().numpy()[0]
            new_pose = np.eye(4)
            new_pose[:3, :] = new_3x4

            R_old, t_old = old_pose[:3, :3], old_pose[:3, 3]
            R_new, t_new = new_pose[:3, :3], new_pose[:3, 3]
            R_cw_new = R_new.T
            t_cw_new = -R_new.T @ t_new
            R_corr = R_cw_new @ R_old
            t_corr = R_cw_new @ t_old + t_cw_new
            pose_corrections[kf_id] = (R_corr, t_corr)

        # Anchor each 3D point explicitly to the keyframe in which it was
        # triangulated. With each map point now potentially shared across
        # many keyframes (post-LC fusion / association), there is no longer
        # a single "earliest observer with a correction" — but there *is*
        # always exactly one triangulating keyframe per point, recorded at
        # creation time.
        applied = 0
        skipped_no_tri = 0
        for pt_id in mapper.pointmap.alive_point_ids():
            tri_kf_id = int(mapper.pointmap.triangulating_kf[pt_id])
            if tri_kf_id < 0:
                skipped_no_tri += 1
                continue
            if tri_kf_id not in pose_corrections:
                continue
            R_corr, t_corr = pose_corrections[tri_kf_id]
            p = mapper.pointmap.points_3d[pt_id]
            mapper.pointmap.points_3d[pt_id] = R_corr @ p + t_corr
            applied += 1

        for kf_id, cam_var in se3_vars.items():
            if kf_id == anchor_id:
                continue
            kf = kf_map[kf_id]
            new_3x4 = cam_var.tensor.detach().cpu().numpy()[0]
            new_pose = np.eye(4)
            new_pose[:3, :] = new_3x4
            kf.pose = new_pose

        print(f"[PG] Corrected {applied} 3D points "
              f"(anchored to triangulating keyframe, "
              f"{len(pose_corrections) - 1} keyframes updated, "
              f"{skipped_no_tri} points skipped — no triangulating KF recorded)")
        print(f"[PG] --- Pose Graph Optimization Complete ---\n")
