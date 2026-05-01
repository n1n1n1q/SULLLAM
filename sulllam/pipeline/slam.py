from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation

from sulllam.localization.extraction.superpoint import SuperPointFeatureExtractor
from sulllam.localization.global_descriptor import compute_global_descriptor
from sulllam.localization.matching.lightglue import LightGlueMatcher
from sulllam.mapping.map import Keyframe, Mapper
from sulllam.mapping.pose_graph import PoseGraph
from sulllam.pipeline.config import SLAMConfig
from sulllam.pipeline.triangulator import Triangulator


def _Rt_to_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T


@dataclass
class KeyframeSelectionConfig:
    # Hard floor on frames between consecutive KFs. Avoids back-to-back KFs
    # when the camera is briefly stationary or detector noise spikes.
    min_frames_since_kf: int = 4
    # Force a new KF after this many frames even if everything looks well
    # tracked. Bounds drift between KF-anchored bundle updates.
    max_frames_since_kf: int = 30
    # Promote when the fraction of last-KF map points still seen falls below
    # this. Equivalently: the new frame contains a lot of un-tracked content.
    max_tracked_ratio: float = 0.85
    # Don't trust `tracked_ratio` until the last KF has at least this many
    # observed map points (otherwise the ratio is dominated by noise early
    # in the sequence).
    min_map_points_for_ratio: int = 30
    # Force a new KF if inlier matches against the last KF drop below this
    # (tracking is degrading; we need a closer reference).
    min_inlier_matches: int = 80


@dataclass
class OdometryScaleConfig:
    """Information weighting and pure-rotation handling for odometry edges.

    `recoverPose` returns ``|t| = 1`` regardless of physical motion, so
    odometry translations carry direction information but no metric
    magnitude — and during pure rotation the *direction* itself is noise
    because the essential matrix is degenerate. Two mitigations live here:

    1. The translation block of every odometry edge's information matrix
       is downweighted (``translation_info_scale``), so PGO can absorb
       loop-closure rotations by nudging translations instead of rotating
       the whole tail of the trajectory rigidly around stale points.
    2. When the median angular parallax of inlier matches drops below
       ``pure_rotation_parallax_deg`` we treat the edge as rotation-only:
       ``t_rel`` is zeroed in the absolute pose composition, no new map
       points are triangulated, and the pose-graph edge ignores its
       translation residual entirely.
    """
    enable: bool = True
    # Information matrix relative weight on the translation block of an
    # odometry edge (rotation block stays at 1.0). Smaller = PGO has more
    # freedom to move camera centres along the translation axis.
    translation_info_scale: float = 0.1
    # Median angular parallax (degrees) below which the edge is treated
    # as pure rotation. The right value is dataset-dependent: it has to
    # be smaller than the typical parallax produced by *real* camera
    # translation in the sequence. For our slowly-translating, far-field
    # sequences (typical parallax 0.4-1.0°) we use 0.3° — this catches
    # genuine spin-in-place segments without false-flagging real motion.
    pure_rotation_parallax_deg: float = 0.3
    # Need at least this many inlier matches to trust the parallax median.
    # Below this we conservatively treat the edge as rotation-bearing.
    min_inliers_for_parallax: int = 30


@dataclass
class TrackingFailureConfig:
    """Detection + handling for catastrophic tracking loss between KFs.

    On a sharp rotation (or any view change too aggressive for the matcher),
    the pool of inlier correspondences between current frame and last KF
    can collapse to a handful — at which point ``cv.recoverPose`` returns
    an essentially random ``(R_rel, t_rel)``. Committing that into the pose
    chain produces the classic "trajectory spike at the turn" — multiple
    consecutive bad KFs each adding ~1 unit of wrong-direction translation,
    that loop closures + rotation-only PGO can only partially heal.

    When tracking is lost we instead:
    - hold the global pose and last-KF reference fixed,
    - record the current frame at the last good camera position (so the
      trajectory has a flat segment, not a spike),
    - skip new KF promotion and triangulation entirely.
    Tracking automatically recovers on the next frame whose match against
    the (still-current) last KF passes the gates.

    If too many consecutive frames are lost we have to accept *something*
    or matching will only get worse; ``max_consecutive_lost`` bounds that
    grace period and forces a normal KF promotion afterwards (best-effort
    pose), still tagged so triangulation doesn't seed garbage.
    """
    enable: bool = True
    # Below this many RANSAC inliers, the recoverPose ``R_rel``/``t_rel``
    # are not trustworthy enough to advance the pose chain. Sized to be
    # comfortably above the 8-point essential-matrix minimum (8) and the
    # parallax-gate floor (``min_inliers_for_parallax = 30``), so that a
    # frame which is too noisy to even *decide* pure-rotation also fails
    # this gate.
    min_inliers: int = 30
    # Track-ratio floor (tracked / last-KF map points). Independently of
    # raw inlier count, a frame that retains <5% of last KF's mapped
    # features after RANSAC is observing a substantially different scene
    # — typical of a fast turn. Only enforced once the last KF actually
    # has enough mapped points for the ratio to be meaningful.
    min_track_ratio: float = 0.05
    min_map_points_for_ratio: int = 30
    # After this many consecutive lost frames we give up and accept the
    # current (best-effort) recoverPose result as a regular KF. Otherwise
    # the matcher would keep comparing against an ever-staler last KF.
    max_consecutive_lost: int = 8


class SLAMPipeline:
    def __init__(self, config: SLAMConfig):
        self.config = config
        self.mapper = Mapper()
        self.triangulator = Triangulator(
            K=config.K,
            max_reproj_error=config.max_reproj_error,
            max_depth=config.max_depth,
            max_points=config.max_points,
        )
        self.pose_graph = PoseGraph()
        self.kf_selection = config.kf_selection or KeyframeSelectionConfig()
        self.odom_scale = getattr(config, "odom_scale", None) or OdometryScaleConfig()
        self.tracking_failure = (
            getattr(config, "tracking_failure", None) or TrackingFailureConfig()
        )

        self._R_global = np.eye(3)
        self._t_global = np.zeros(3)
        self._current_match_scores: np.ndarray = np.array([])
        self._frame_idx = 0
        self._last_pgo_kf: int = -10**9
        self._last_kf_image: np.ndarray | None = None
        # Counts consecutive frames where matching against the last KF
        # collapsed below trust thresholds. Reset on every successful
        # frame; bounded by ``TrackingFailureConfig.max_consecutive_lost``.
        self._consecutive_lost: int = 0

        # Per-frame "tracked" record. Each entry is (ref_kf_idx, R_rel, t_rel)
        # so the absolute pose of frame `i` is always derivable as
        # ``T_rel ∘ T_ref_kf``. After any pose update (PGO, GBA, LBA) the
        # full per-frame trajectory is recomputed by re-evaluating these
        # records against the updated KF poses.
        self._frame_records: list[tuple[int, np.ndarray, np.ndarray]] = []
        self.trajectory: list[np.ndarray] = []

        config.clouds_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _camera_pos(R: np.ndarray, t: np.ndarray) -> np.ndarray:
        return -R.T @ t

    def _record_frame(
        self,
        ref_kf_idx: int,
        R_rel: np.ndarray,
        t_rel: np.ndarray,
        camera_pos: np.ndarray,
    ) -> None:
        """Append a per-frame record + matching trajectory point in lockstep.

        Both lists must have the same length so a rebuild can re-index them
        consistently.
        """
        self._frame_records.append((ref_kf_idx, R_rel.copy(), t_rel.copy()))
        self.trajectory.append(camera_pos)

    def _update_last_record(
        self,
        ref_kf_idx: int,
        R_rel: np.ndarray,
        t_rel: np.ndarray,
        camera_pos: np.ndarray,
    ) -> None:
        if not self._frame_records:
            self._record_frame(ref_kf_idx, R_rel, t_rel, camera_pos)
            return
        self._frame_records[-1] = (ref_kf_idx, R_rel.copy(), t_rel.copy())
        self.trajectory[-1] = camera_pos

    def _rebuild_trajectory(self) -> None:
        """Recompute every per-frame trajectory entry from the current KF poses.

        Without this, every PGO/GBA correction creates a discontinuity at
        the newest frame while everything before it stays at its old
        position — looks like the trajectory "jumps" on every loop closure.
        """
        new_traj: list[np.ndarray] = []
        last_pos = np.zeros(3)
        for ref_idx, R_rel, t_rel in self._frame_records:
            ref_kf = self.mapper.keyframe_by_idx(ref_idx)
            if ref_kf is None:
                new_traj.append(last_pos)
                continue
            R = R_rel @ ref_kf.R
            t = R_rel @ ref_kf.t + t_rel
            pos = self._camera_pos(R, t)
            new_traj.append(pos)
            last_pos = pos
        self.trajectory = new_traj
        if self.mapper.current_keyframe is not None:
            self._R_global = self.mapper.current_keyframe.R.copy()
            self._t_global = self.mapper.current_keyframe.t.copy()

    def _initialize(self, image: np.ndarray) -> None:
        kps, descs = self.config.extractor.extract(image)
        feats = None
        if isinstance(self.config.extractor, SuperPointFeatureExtractor):
            feats = self.config.extractor.extract_tensors(image)

        initial_kf = Keyframe(
            idx=0,
            keypoints=kps,
            descriptors=descs,
            pose=_Rt_to_T(self._R_global, self._t_global),
            global_descriptor=compute_global_descriptor(descs),
        )
        initial_kf.feats = feats
        self.mapper.add_keyframe(initial_kf)
        self._last_kf_image = image.copy()
        self._record_frame(
            ref_kf_idx=0,
            R_rel=np.eye(3),
            t_rel=np.zeros(3),
            camera_pos=np.zeros(3),
        )
        self._frame_idx = 1

    def _match_against_last_kf(
        self, last_kf: Keyframe, curr_descs, curr_feats: dict | None
    ) -> tuple[list[cv.DMatch], np.ndarray]:
        cfg = self.config
        if (
            isinstance(cfg.matcher, LightGlueMatcher)
            and curr_feats is not None
            and last_kf.feats is not None
        ):
            return cfg.matcher.match_tensors(last_kf.feats, curr_feats)
        return cfg.matcher.match(last_kf.descriptors, curr_descs)

    def _should_promote_to_keyframe(
        self,
        *,
        tracked: int,
        total_map_points: int,
        inlier_count: int,
        frames_since_kf: int,
    ) -> tuple[bool, str]:
        kfc = self.kf_selection
        if frames_since_kf < kfc.min_frames_since_kf:
            return False, "min-gap"
        if frames_since_kf >= kfc.max_frames_since_kf:
            return True, "max-gap"
        if inlier_count < kfc.min_inlier_matches:
            return True, f"low-inliers({inlier_count})"
        # Bootstrap phase: the last KF doesn't have enough map points yet,
        # so the tracking-ratio gate is unreliable. Promote at the floor
        # cadence to keep seeding the map.
        if total_map_points < kfc.min_map_points_for_ratio:
            return True, f"bootstrap(map={total_map_points})"
        ratio = tracked / max(total_map_points, 1)
        if ratio < kfc.max_tracked_ratio:
            return True, f"track-ratio({ratio:.2f})"
        return False, "tracking-ok"

    def _process_frame(self, image: np.ndarray) -> dict:
        cfg = self.config
        i = self._frame_idx

        curr_kps, curr_descs = cfg.extractor.extract(image)
        curr_feats: dict | None = None
        if isinstance(cfg.extractor, SuperPointFeatureExtractor):
            curr_feats = cfg.extractor.extract_tensors(image)

        last_kf = self.mapper.current_keyframe
        # `_initialize` always seeds a KF before `_process_frame` is called.
        assert last_kf is not None, "last_kf must exist"

        matches, scores = self._match_against_last_kf(last_kf, curr_descs, curr_feats)
        self._current_match_scores = scores

        if len(matches) < 8:
            print(f"[SLAM] Frame {i}: too few matches vs last KF "
                  f"({len(matches)}); holding pose")
            self._consecutive_lost += 1
            self._record_lost_frame(last_kf)
            self._frame_idx += 1
            return {"matches": matches, "prev_keypoints": last_kf.keypoints,
                    "curr_keypoints": curr_kps, "image": image,
                    "is_keyframe": False, "tracking_lost": True}

        prev_pts = np.array(
            [last_kf.keypoints[m.queryIdx].pt for m in matches]
        ).reshape(-1, 1, 2)
        curr_pts = np.array(
            [curr_kps[m.trainIdx].pt for m in matches]
        ).reshape(-1, 1, 2)

        estimate = cfg.pose_estimator.estimate(prev_pts, curr_pts)
        R_rel, t_rel = estimate["R"], estimate["t"].reshape(-1)
        inliers_mask = estimate["inliers_mask"]
        inliers = inliers_mask.ravel() == 1
        inlier_count = int(inliers.sum())

        # Tracking-failure gate. When the inlier pool collapses (sharp
        # turn, motion blur, occlusion) ``recoverPose`` still returns a
        # pose, but ``R_rel`` is noisy and the unit-norm ``t_rel`` direction
        # is essentially random. Committing those into the pose chain
        # spawns multi-unit spikes (one bad direction × N bad KFs) that
        # loop-closure PGO can only partially heal. Detect that case
        # here and freeze pose / skip KF promotion until matching
        # recovers against the (still-current) last KF.
        last_kf_kp_to_pt = self.mapper.pointmap.kf_kp_to_pt(last_kf.idx)
        total_map_points = len(last_kf_kp_to_pt)
        tracked = self._count_tracked_inliers(
            matches, inliers, last_kf_kp_to_pt
        )
        tf_cfg = self.tracking_failure
        track_ratio_critical = (
            tf_cfg.enable
            and total_map_points >= tf_cfg.min_map_points_for_ratio
            and (tracked / max(total_map_points, 1)) < tf_cfg.min_track_ratio
        )
        inliers_critical = tf_cfg.enable and inlier_count < tf_cfg.min_inliers
        tracking_lost = inliers_critical or track_ratio_critical

        if tracking_lost and self._consecutive_lost < tf_cfg.max_consecutive_lost:
            self._consecutive_lost += 1
            print(
                f"[SLAM] Frame {i}: tracking lost "
                f"(inliers={inlier_count}, tracked={tracked}/{total_map_points}, "
                f"streak={self._consecutive_lost}/{tf_cfg.max_consecutive_lost}); "
                f"holding pose"
            )
            self._record_lost_frame(last_kf)
            self._frame_idx += 1
            return {
                "matches": matches,
                "prev_keypoints": last_kf.keypoints,
                "curr_keypoints": curr_kps,
                "image": image,
                "is_keyframe": False,
                "tracking_lost": True,
            }

        forced_accept = False
        if tracking_lost:
            print(
                f"[SLAM] Frame {i}: tracking lost streak hit "
                f"max_consecutive_lost={tf_cfg.max_consecutive_lost}; "
                f"force-accepting current pose estimate "
                f"(rotation-only, no triangulation)"
            )
            forced_accept = True

        self._consecutive_lost = 0

        # Pure-rotation gate. When the camera barely translates between
        # last_kf and now (think turning in place), the essential matrix
        # is degenerate and `cv.recoverPose` still returns a unit-norm
        # ``t`` whose direction is essentially noise. Adding that into the
        # absolute pose chain produces the "tail sticking off the loop"
        # drift you can see when most edges are rotational. We detect this
        # via the median angular parallax of inlier matches and zero
        # ``t_rel`` (and skip triangulation / mark a rotation-only edge)
        # whenever it falls below threshold.
        #
        # On a forced-accept (tracking-lost streak gave up) we treat the
        # edge as rotation-only too: the unit-norm ``t_rel`` direction was
        # decided by ~30 inliers at most, so committing it as full 6-DoF
        # would seed the same spike we were trying to avoid. Rotation
        # block is the more reliable half — let PGO sort the rest out.
        rotation_only = False
        parallax_deg = 0.0
        sc_cfg = self.odom_scale
        if forced_accept:
            rotation_only = True
            t_rel = np.zeros(3)
        elif inlier_count >= sc_cfg.min_inliers_for_parallax:
            parallax_deg = self._median_parallax_deg(
                prev_pts, curr_pts, inliers, R_rel
            )
            if parallax_deg < sc_cfg.pure_rotation_parallax_deg:
                rotation_only = True
                t_rel = np.zeros(3)

        # Compose pose against the LAST KEYFRAME, not the previous frame.
        # The recoverPose convention `(R_rel, t_rel)` maps last-KF camera
        # coords to current-frame camera coords, so:
        #     R_world->curr = R_rel @ R_world->kf
        #     t_world->curr = R_rel @ t_world->kf + t_rel
        # `t_rel` is unit-norm (monocular essential-matrix decomposition)
        # unless the pure-rotation gate above zeroed it.
        R_world_to_cam = R_rel @ last_kf.R
        t_world_to_cam = R_rel @ last_kf.t + t_rel

        self._R_global = R_world_to_cam
        self._t_global = t_world_to_cam

        camera_pos = self._camera_pos(R_world_to_cam, t_world_to_cam)
        self._record_frame(
            ref_kf_idx=last_kf.idx,
            R_rel=R_rel,
            t_rel=t_rel,
            camera_pos=camera_pos,
        )

        # Decide whether this frame should be a keyframe. Tracked / inlier
        # counts and the last-KF point map were already computed above for
        # the tracking-failure gate; reuse them here.
        frames_since_kf = i - last_kf.idx

        is_kf, reason = self._should_promote_to_keyframe(
            tracked=tracked,
            total_map_points=total_map_points,
            inlier_count=inlier_count,
            frames_since_kf=frames_since_kf,
        )

        frame_info = {
            "matches": matches,
            "prev_keypoints": last_kf.keypoints,
            "curr_keypoints": curr_kps,
            "image": image,
            "is_keyframe": is_kf,
        }

        if not is_kf:
            self._frame_idx += 1
            return frame_info

        print(f"[SLAM] Frame {i}: promoting to keyframe (reason={reason}, "
              f"tracked={tracked}/{total_map_points}, inliers={inlier_count})")

        # Promote to keyframe.
        global_desc = compute_global_descriptor(curr_descs)
        curr_kf = Keyframe(
            idx=i,
            keypoints=curr_kps,
            descriptors=curr_descs,
            pose=_Rt_to_T(R_world_to_cam, t_world_to_cam),
            match_scores=self._current_match_scores,
            global_descriptor=global_desc,
        )
        curr_kf.feats = curr_feats
        self.mapper.add_keyframe(curr_kf)

        # Backfill: the very first KF was created in `_initialize` before
        # this code path existed, so make sure its global descriptor is set.
        if last_kf.global_descriptor is None:
            last_kf.global_descriptor = compute_global_descriptor(last_kf.descriptors)

        # 1) Associate already-mapped points: every inlier match whose
        #    last-KF keypoint is already an observer of a map point becomes
        #    an extra observation of that same map point on the new KF.
        #    This is what gives loop closures something to bundle later.
        for j, m in enumerate(matches):
            if not inliers[j]:
                continue
            kp_idx_prev = m.queryIdx
            kp_idx_curr = m.trainIdx
            if kp_idx_prev not in last_kf_kp_to_pt:
                continue
            pt_id = last_kf_kp_to_pt[kp_idx_prev]
            if self.mapper.pointmap.is_dead(pt_id):
                continue
            uv = np.array(curr_kps[kp_idx_curr].pt, dtype=np.float64)
            self.mapper.pointmap.add_observation(
                pt_id, curr_kf.idx, uv, kp_idx=kp_idx_curr
            )

        # 2) Triangulate brand-new map points only from inliers whose
        #    last-KF keypoint isn't already a map point. Avoids creating
        #    duplicate landmarks for already-tracked features. We skip
        #    triangulation entirely on pure-rotation edges — with zero
        #    baseline triangulation produces points at infinity and would
        #    poison the map with garbage landmarks.
        candidates: list[dict] = []
        new_idxs = [
            j for j, m in enumerate(matches)
            if inliers[j] and m.queryIdx not in last_kf_kp_to_pt
        ]
        if new_idxs and not rotation_only:
            prev_inlier_pts = np.array(
                [last_kf.keypoints[matches[j].queryIdx].pt for j in new_idxs],
                dtype=np.float64,
            )
            curr_inlier_pts = np.array(
                [curr_kps[matches[j].trainIdx].pt for j in new_idxs],
                dtype=np.float64,
            )
            kp_indices_prev = [matches[j].queryIdx for j in new_idxs]
            kp_indices_curr = [matches[j].trainIdx for j in new_idxs]

            image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            candidates = self.triangulator.triangulate(
                R_prev=last_kf.R,
                t_prev=last_kf.t,
                R_curr=curr_kf.R,
                t_curr=curr_kf.t,
                pts1=prev_inlier_pts,
                pts2=curr_inlier_pts,
                image_rgb=image_rgb,
                kp_indices_prev=kp_indices_prev,
                kp_indices_curr=kp_indices_curr,
            )

        # 3) Add candidates to the map.
        #
        # NOTE on monocular scale (root-cause #3):
        # We deliberately do NOT rescale t_rel or the new points to a
        # depth-derived "scene scale" here. Doing so creates an
        # inconsistency between edges: existing map points live at the
        # accumulated (unit-baseline) scale of their birth edge, while a
        # rescaled curr_kf.pose lives in a different scale. The
        # data-association step above re-observes those existing points in
        # curr_kf, and the moment curr_kf's centre is rescaled, all those
        # already-added observations have ~kilo-pixel reprojection error
        # against unmoved 3D points and LBA blows up. The scale-handling
        # we *do* apply lives in the information matrix below: translation
        # is downweighted so PGO can absorb the mismatch instead of
        # rotating-the-chain-rigidly to satisfy a magnitude it can't trust.
        for cand in candidates:
            pt_id = self.mapper.pointmap.add_point(
                cand["pt3d"],
                color=cand["color"],
                triangulating_kf_id=last_kf.idx,
            )
            self.mapper.pointmap.add_observation(
                pt_id,
                last_kf.idx,
                cand["uv_prev"],
                kp_idx=cand.get("kp_idx_prev", -1),
            )
            self.mapper.pointmap.add_observation(
                pt_id,
                curr_kf.idx,
                cand["uv_curr"],
                kp_idx=cand.get("kp_idx_curr", -1),
            )

        # 4) Build the odometry edge. On a translating edge t_rel stays
        #    unit-norm and the translation block is downweighted in the
        #    information matrix. On a pure-rotation edge t_rel is zero and
        #    the edge is flagged rotation-only so PGO ignores its
        #    translation residual entirely.
        rel_pose = _Rt_to_T(R_rel, t_rel)
        odom_info = self._odometry_information()
        self.pose_graph.add_odometry_edge(
            last_kf.idx,
            curr_kf.idx,
            rel_pose,
            information=odom_info,
            rotation_only=rotation_only,
        )
        if rotation_only:
            print(f"[SLAM] Frame {i}: pure-rotation edge "
                  f"(parallax={parallax_deg:.2f}°) → rotation-only PG edge, "
                  f"no new triangulations")

        # 6) Update this frame's record so the trajectory point is now
        #    anchored to the new KF (identity rel-pose) instead of the
        #    previous KF + scaled translation. This keeps the rebuild logic
        #    trivial: each KF's frame is `kf.pose ∘ I = kf.pose`.
        camera_pos = self._camera_pos(self._R_global, self._t_global)
        self._update_last_record(
            ref_kf_idx=curr_kf.idx,
            R_rel=np.eye(3),
            t_rel=np.zeros(3),
            camera_pos=camera_pos,
        )

        # Local BA: cadence is now in *keyframes*, not raw frames.
        kf_count = len(self.mapper.keyframes)
        if kf_count >= cfg.ba_min_frames and kf_count % max(cfg.ba_frequency, 1) == 0:
            cfg.bundle_adjustment.run(self.mapper, cfg.K)
            self._rebuild_trajectory()

        # Loop closure detection on KF cadence.
        loop_closed = False
        if kf_count % max(cfg.lc_frequency, 1) == 0:
            lc_candidates = cfg.loop_closure_detector.detect(
                curr_kf, self.mapper, cfg.K, matcher_obj=cfg.matcher
            )
            for lc in lc_candidates:
                self.pose_graph.add_loop_closure_edge(
                    from_id=lc["match_kf"].idx,
                    to_id=lc["query_kf"].idx,
                    relative_pose=lc["relative_pose"],
                    information=lc.get("information"),
                )
                # Cross-loop-closure data association: share map points
                # between the two ends of the loop so PGO+GBA actually has
                # something connecting the trajectory's start and tail.
                self._associate_across_loop(lc["match_kf"], lc["query_kf"])
                loop_closed = True

        # Throttle PGO/GBA: even if the LC detector keeps firing on the same
        # revisit, don't re-run PGO unless enough new keyframes have been
        # added since the last optimisation. Cadence is in keyframes now.
        run_pgo = loop_closed and (
            curr_kf.idx - self._last_pgo_kf >= cfg.lc_frequency
        )

        if run_pgo:
            cfg.pose_graph_optimizer.optimize(self.mapper, self.pose_graph)
            self._last_pgo_kf = curr_kf.idx
            self._rebuild_trajectory()

        if run_pgo and kf_count >= cfg.gba_min_frames:
            cfg.global_bundle_adjustment.run(self.mapper, cfg.K)
            self._rebuild_trajectory()

        self._R_global = self.mapper.current_keyframe.R.copy()
        self._t_global = self.mapper.current_keyframe.t.copy()
        self._last_kf_image = image.copy()

        # Final sync of this frame's trajectory entry against the post-BA/
        # post-PGO/post-GBA KF pose.
        camera_pos = self._camera_pos(self._R_global, self._t_global)
        self._update_last_record(
            ref_kf_idx=curr_kf.idx,
            R_rel=np.eye(3),
            t_rel=np.zeros(3),
            camera_pos=camera_pos,
        )

        self.mapper.pointmap.save_pointcloud(cfg.clouds_dir / f"{i}.ply")

        self._frame_idx += 1

        return frame_info

    def _count_tracked_inliers(
        self,
        matches: list[cv.DMatch],
        inliers: np.ndarray,
        last_kf_kp_to_pt: dict[int, int],
    ) -> int:
        """How many RANSAC inliers re-observe an existing map point.

        Computed once per frame (in ``_process_frame``) and reused for the
        tracking-failure gate, the keyframe-promotion decision, and the
        data-association loop, since each step needs the same answer.
        """
        if not last_kf_kp_to_pt:
            return 0
        n = 0
        for j, m in enumerate(matches):
            if not inliers[j]:
                continue
            pt_id = last_kf_kp_to_pt.get(m.queryIdx)
            if pt_id is None:
                continue
            if not self.mapper.pointmap.is_dead(pt_id):
                n += 1
        return n

    def _record_lost_frame(self, last_kf: Keyframe) -> None:
        """Record a frame where matching collapsed without advancing pose.

        Trajectory length must equal the number of processed frames so
        that downstream consumers (publishing, .npy export) stay in
        lockstep with the input sequence. Re-emit the most recent record
        relative to ``last_kf`` so a subsequent ``_rebuild_trajectory`` —
        e.g. after a loop-closure PGO — places this frame at the same
        spot as the previous good frame, producing a flat segment instead
        of an erroneous spike at the camera turn.
        """
        if self._frame_records:
            ref_idx, R_rel, t_rel = self._frame_records[-1]
            if ref_idx == last_kf.idx:
                R = R_rel @ last_kf.R
                t = R_rel @ last_kf.t + t_rel
                pos = self._camera_pos(R, t)
                self._record_frame(ref_idx, R_rel, t_rel, pos)
                return
        self._record_frame(
            ref_kf_idx=last_kf.idx,
            R_rel=np.eye(3),
            t_rel=np.zeros(3),
            camera_pos=self._camera_pos(last_kf.R, last_kf.t),
        )

    def _odometry_information(self) -> np.ndarray:
        """Information matrix for an odometry edge.

        Translation block is downweighted because monocular ``t`` only has
        a magnitude estimate from triangulated depth, not a true metric
        scale. Rotation block stays at unit weight.
        """
        sc_cfg = self.odom_scale
        info = np.eye(6, dtype=np.float64)
        info[:3, :3] *= max(sc_cfg.translation_info_scale, 1e-6)
        return info

    def _median_parallax_deg(
        self,
        prev_pts: np.ndarray,
        curr_pts: np.ndarray,
        inliers: np.ndarray,
        R_rel: np.ndarray,
    ) -> float:
        """Median angle (degrees) between matched rays after de-rotation.

        For a pair of matched image points, back-project each to a unit
        ray in its own camera frame, rotate the curr-frame ray back into
        last_kf's frame using ``R_rel.T`` (R_rel maps last_kf -> curr) and
        compare directions. Pure rotation makes the two rays coincide
        (angle ≈ 0) regardless of the depth of the world point; any
        non-trivial baseline produces a non-zero angle proportional to
        ``baseline / depth``. The median over inlier matches is a robust
        proxy for "is there real translation in this edge?"
        """
        if not inliers.any():
            return 0.0
        K = self.config.K
        try:
            K_inv = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            return 0.0
        p1 = prev_pts.reshape(-1, 2)[inliers]
        p2 = curr_pts.reshape(-1, 2)[inliers]
        if len(p1) < 4:
            return 0.0
        h1 = np.concatenate([p1, np.ones((len(p1), 1))], axis=1)
        h2 = np.concatenate([p2, np.ones((len(p2), 1))], axis=1)
        d1 = (K_inv @ h1.T).T
        d2 = (K_inv @ h2.T).T
        n1 = np.linalg.norm(d1, axis=1, keepdims=True)
        n2 = np.linalg.norm(d2, axis=1, keepdims=True)
        n1[n1 < 1e-12] = 1.0
        n2[n2 < 1e-12] = 1.0
        d1 /= n1
        d2 /= n2
        # R_rel maps last_kf coords -> curr coords; transpose maps back.
        d2_in_prev = (R_rel.T @ d2.T).T
        cos_ang = np.clip(np.sum(d1 * d2_in_prev, axis=1), -1.0, 1.0)
        angles = np.degrees(np.arccos(cos_ang))
        return float(np.median(angles))

    def _associate_across_loop(self, match_kf: Keyframe, query_kf: Keyframe) -> None:
        """Share map points between the two ends of a detected loop closure.

        Re-matches the two keyframes' SuperPoint features with LightGlue and:
        - if both keypoints already have a map point: fuse them (older wins),
        - if only one side has a map point: add an observation of that point
          to the keyframe that doesn't see it yet.

        Without this step every triangulated point sees only its two adjacent
        keyframes, so global BA has no constraints linking the loop ends.
        """
        cfg = self.config
        # We need both raw feature dicts to use the LightGlue tensor path.
        # Fall back to descriptor-based matching when not available.
        if (
            isinstance(cfg.matcher, LightGlueMatcher)
            and match_kf.feats is not None
            and query_kf.feats is not None
        ):
            matches, _ = cfg.matcher.match_tensors(match_kf.feats, query_kf.feats)
        else:
            matches, _ = cfg.matcher.match(match_kf.descriptors, query_kf.descriptors)
        if not matches:
            return

        match_kp_to_pt = self.mapper.pointmap.kf_kp_to_pt(match_kf.idx)
        query_kp_to_pt = self.mapper.pointmap.kf_kp_to_pt(query_kf.idx)

        added_obs = 0
        fused = 0
        for m in matches:
            kp_match = m.queryIdx
            kp_query = m.trainIdx
            pt_in_match = match_kp_to_pt.get(kp_match)
            pt_in_query = query_kp_to_pt.get(kp_query)
            if pt_in_match is None and pt_in_query is None:
                continue

            if pt_in_match is not None and pt_in_query is not None:
                if pt_in_match == pt_in_query:
                    continue
                # Survivor: the older (lower id) point — it's been refined
                # by more passes of BA already.
                survivor = min(pt_in_match, pt_in_query)
                dead = max(pt_in_match, pt_in_query)
                if self.mapper.pointmap.is_dead(survivor) or self.mapper.pointmap.is_dead(dead):
                    continue
                self.mapper.pointmap.merge_points(survivor, dead)
                fused += 1
                continue

            if pt_in_match is not None and pt_in_query is None:
                if self.mapper.pointmap.is_dead(pt_in_match):
                    continue
                uv = np.array(query_kf.keypoints[kp_query].pt, dtype=np.float64)
                self.mapper.pointmap.add_observation(
                    pt_in_match, query_kf.idx, uv, kp_idx=kp_query
                )
                added_obs += 1
            elif pt_in_query is not None and pt_in_match is None:
                if self.mapper.pointmap.is_dead(pt_in_query):
                    continue
                uv = np.array(match_kf.keypoints[kp_match].pt, dtype=np.float64)
                self.mapper.pointmap.add_observation(
                    pt_in_query, match_kf.idx, uv, kp_idx=kp_match
                )
                added_obs += 1

        if added_obs or fused:
            print(f"[LC-DA] kf {match_kf.idx} ↔ kf {query_kf.idx}: "
                  f"+{added_obs} observations, {fused} duplicates fused")

    def run(self, images: list[np.ndarray], ros_publisher=None) -> np.ndarray:
        self._initialize(images[0])

        for i, image in enumerate(images[1:], start=1):
            frame_info = self._process_frame(image)

            if ros_publisher is not None:
                self._publish(ros_publisher, frame_info, i)

        return np.array(self.trajectory)

    def _publish(self, ros_publisher, frame_info: dict, i: int) -> None:
        kf = self.mapper.current_keyframe
        if kf is None:
            return

        translations, orientations = [], []
        for k in self.mapper.keyframes:
            translations.append(-k.R.T @ k.t)
            orientations.append(Rotation.from_matrix(k.R.T).as_quat())

        image = frame_info["image"]
        matches = frame_info["matches"]
        curr_kps = frame_info["curr_keypoints"]

        last_kf_image = self._last_kf_image if self._last_kf_image is not None else image
        try:
            pair = cv.hconcat([last_kf_image, image])
        except cv.error:
            pair = image
        try:
            match_img = cv.drawMatches(
                last_kf_image,
                kf.keypoints,
                image,
                curr_kps,
                matches,
                None,
            )
        except cv.error:
            match_img = image

        ros_publisher.publish_trajectory(translations, orientations)
        ros_publisher.publish_current_pair(pair)
        ros_publisher.publish_current_matches(match_img)
        ros_publisher.publish_pointcloud(
            self.mapper.pointmap.points_3d[: self.mapper.pointmap.num_points],
            self.mapper.pointmap.point_colors[: self.mapper.pointmap.num_points],
        )
