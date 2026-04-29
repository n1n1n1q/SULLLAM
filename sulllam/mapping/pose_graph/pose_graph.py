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


class PoseGraph:
    def __init__(self) -> None:
        self._edges: list[PoseGraphEdge] = []

    def add_odometry_edge(
        self,
        from_id: int,
        to_id: int,
        relative_pose: np.ndarray,
        information: np.ndarray | None = None,
    ) -> None:
        info = information if information is not None else np.eye(6, dtype=np.float64)
        self._edges.append(PoseGraphEdge(from_id, to_id, relative_pose, info, is_loop_closure=False))

    def add_loop_closure_edge(
        self,
        from_id: int,
        to_id: int,
        relative_pose: np.ndarray,
        information: np.ndarray | None = None,
    ) -> None:
        info = information if information is not None else np.eye(6, dtype=np.float64) * 4.0
        self._edges.append(PoseGraphEdge(from_id, to_id, relative_pose, info, is_loop_closure=True))

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
        weight = th.ScaleCostWeight(torch.tensor(1.0, dtype=torch.float64))

        for i, edge in enumerate(pose_graph.edges):
            if edge.from_id not in se3_vars or edge.to_id not in se3_vars:
                continue

            rel_t = torch.from_numpy(edge.relative_pose[:3, :]).unsqueeze(0).double()
            rel_var = th.SE3(tensor=rel_t, name=f"pg_rel_{i}")

            pose_from = se3_vars[edge.from_id]
            pose_to = se3_vars[edge.to_id]

            info_scalar = float(np.trace(edge.information) / 6.0)
            edge_weight = th.ScaleCostWeight(
                torch.tensor(info_scalar, dtype=torch.float64)
            )

            if edge.from_id == anchor_id:
                cost = th.AutoDiffCostFunction(
                    optim_vars=[pose_to],
                    err_fn=_pose_graph_error_fixed_from,
                    dim=6,
                    aux_vars=[pose_from, rel_var],
                    cost_weight=edge_weight,
                    name=f"pg_edge_{i}",
                )
            elif edge.to_id == anchor_id:
                cost = th.AutoDiffCostFunction(
                    optim_vars=[pose_from],
                    err_fn=_pose_graph_error_fixed_to,
                    dim=6,
                    aux_vars=[pose_to, rel_var],
                    cost_weight=edge_weight,
                    name=f"pg_edge_{i}",
                )
            else:
                cost = th.AutoDiffCostFunction(
                    optim_vars=[pose_from, pose_to],
                    err_fn=_pose_graph_error,
                    dim=6,
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
        print(f"[PG] Status      : {info.status[0]}")
        print(f"[PG] Iterations  : {info.converged_iter[0].item() + 1}")
        print(f"[PG] Final error : {final_error:.4f}  (delta {initial_error - final_error:.4f})")

        pose_corrections: dict[int, tuple[np.ndarray, np.ndarray]] = {}
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
            R_wc_new = R_new.T
            t_wc_new = -R_new.T @ t_new
            R_corr = R_wc_new @ R_old
            t_corr = R_wc_new @ t_old + t_wc_new
            pose_corrections[kf_id] = (R_corr, t_corr)

        corrected_points: set[int] = set()
        pt_correction_sum: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}

        for kf_id, (R_corr, t_corr) in pose_corrections.items():
            obs_ids = mapper.pointmap._kf_to_obs.get(kf_id, [])
            for obs_id in obs_ids:
                pt_id = int(mapper.pointmap.observations[obs_id, 0])
                if pt_id not in pt_correction_sum:
                    pt_correction_sum[pt_id] = (np.zeros((3, 3)), np.zeros(3), 0)
                R_acc, t_acc, count = pt_correction_sum[pt_id]
                pt_correction_sum[pt_id] = (R_acc + R_corr, t_acc + t_corr, count + 1)

        for pt_id, (R_sum, t_sum, count) in pt_correction_sum.items():
            R_avg = R_sum / count
            t_avg = t_sum / count
            p = mapper.pointmap.points_3d[pt_id]
            mapper.pointmap.points_3d[pt_id] = R_avg @ p + t_avg
            corrected_points.add(pt_id)

        for kf_id, cam_var in se3_vars.items():
            if kf_id == anchor_id:
                continue
            kf = kf_map[kf_id]
            new_3x4 = cam_var.tensor.detach().cpu().numpy()[0]
            new_pose = np.eye(4)
            new_pose[:3, :] = new_3x4
            kf.pose = new_pose

        print(f"[PG] Corrected {len(corrected_points)} 3D points (averaged across {len(pose_corrections)} keyframes)")
        print(f"[PG] --- Pose Graph Optimization Complete ---\n")
