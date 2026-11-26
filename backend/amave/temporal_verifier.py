import cv2
from typing import List, Dict, Any, Tuple
import numpy as np

class TemporalVerifier:
    def __init__(self, min_persistence_frames: int = 2, motion_disruption_threshold: float = 0.15):
        self.min_persistence_frames = min_persistence_frames
        self.motion_disruption_threshold = motion_disruption_threshold

    @staticmethod
    def is_ego_border_box(box: List[float], img_w: int, img_h: int) -> bool:
        """
        Filters out camera ego-vehicle mounting artifacts (e.g. dashcam vehicle hood,
        A-pillar, side-mirror) affixed to extreme screen borders.
        """
        x1, y1, x2, y2 = box
        h = y2 - y1
        touches_bottom = (y2 >= img_h - 12)
        touches_left = (x1 <= 15)
        touches_right = (x2 >= img_w - 15)
        if touches_bottom and (touches_left or touches_right) and (h > img_h * 0.35):
            return True
        return False

    def evaluate_temporal_persistence(
        self,
        temporal_history: List[Dict[str, Any]],
        img_shape: Tuple[int, int] = (1080, 1920)
    ) -> Dict[str, Any]:
        """
        Stage 2: Evaluates temporal persistence and kinematic trajectory dynamics across candidate buffer.
        Tracks candidate continuity, deceleration, and post-impact stationary hazard persistence.
        """
        if not temporal_history or len(temporal_history) < 2:
            return {
                "score": 0.0,
                "consecutive_count": 0,
                "persistence_ratio": 0.0,
                "passed": False,
                "has_deceleration": False,
                "is_stationary_post_impact": False
            }

        img_h, img_w = img_shape[:2]
        consecutive = 0
        max_consecutive = 0
        total_detections = 0
        centroids = []

        for frame in temporal_history:
            if frame.get("has_candidate", False):
                raw_dets = frame.get("detections", [])
                v_dets = [d for d in raw_dets if not self.is_ego_border_box(d.get("box", [0, 0, 0, 0]), img_w, img_h)]
                if v_dets:
                    consecutive += 1
                    total_detections += 1
                    max_consecutive = max(max_consecutive, consecutive)
                    best_b = max(v_dets, key=lambda d: d.get("confidence", 0.0))["box"]
                    centroids.append(np.array([(best_b[0] + best_b[2]) / 2.0, (best_b[1] + best_b[3]) / 2.0]))
                else:
                    consecutive = 0
            else:
                consecutive = 0

        window_len = len(temporal_history)
        persistence_ratio = total_detections / window_len if window_len > 0 else 0.0
        passed = (max_consecutive >= self.min_persistence_frames) or (
            persistence_ratio >= 0.40 and window_len >= 5 and total_detections >= self.min_persistence_frames
        )

        # Kinematic deceleration & post-impact stationary hazard analysis
        has_deceleration = False
        is_stationary_post_impact = False
        if len(centroids) >= 3:
            disps = [float(np.linalg.norm(centroids[i] - centroids[i - 1])) for i in range(1, len(centroids))]
            recent_disp = float(np.mean(disps[-2:]))
            if len(disps) >= 4:
                early_disp = float(np.mean(disps[:2]))
                if early_disp > 6.0 and recent_disp < 4.5:
                    has_deceleration = True
            if recent_disp < 5.0 and persistence_ratio >= 0.50:
                is_stationary_post_impact = True

        return {
            "score": round(persistence_ratio, 4),
            "consecutive_count": max_consecutive,
            "persistence_ratio": round(persistence_ratio, 4),
            "passed": passed,
            "has_deceleration": has_deceleration,
            "is_stationary_post_impact": is_stationary_post_impact
        }

    def evaluate_motion_consistency(
        self,
        frame_sequence: List[np.ndarray],
        target_boxes: List[List[float]],
        temporal_kinematics: Dict[str, Any] = None,
        max_detection_conf: float = 0.50
    ) -> float:
        """
        Stage 3: Computes motion consistency score using Farneback Optical Flow with local background ring estimation.
        Evaluates dynamic collision impact disruption or anomalous post-impact stoppage inside candidate regions.
        Distinguishes laminar forward highway flow (directional coherence >= 0.82) from turbulent collision kinetics
        (directional coherence < 0.75, chaotic vector dispersion) and post-collision stoppage.
        """
        if len(frame_sequence) < 2 or not target_boxes:
            return 0.35

        f1 = frame_sequence[-2]
        f2 = frame_sequence[-1]

        h_orig, w_orig = f1.shape[:2]
        h_proc, w_proc = 240, 320
        f1_small = cv2.resize(f1, (w_proc, h_proc))
        f2_small = cv2.resize(f2, (w_proc, h_proc))

        gray1 = cv2.cvtColor(f1_small, cv2.COLOR_BGR2GRAY) if len(f1_small.shape) == 3 else f1_small
        gray2 = cv2.cvtColor(f2_small, cv2.COLOR_BGR2GRAY) if len(f2_small.shape) == 3 else f2_small

        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=0.5, levels=2, winsize=11,
            iterations=2, poly_n=5, poly_sigma=1.1, flags=0
        )
        u = flow[..., 0]
        v = flow[..., 1]

        sx = w_proc / max(1.0, float(w_orig))
        sy = h_proc / max(1.0, float(h_orig))

        kinematics = temporal_kinematics or {}
        has_deceleration = kinematics.get("has_deceleration", False)
        is_stationary_post_impact = kinematics.get("is_stationary_post_impact", False)

        box_scores = []
        for box in target_boxes:
            if self.is_ego_border_box(box, w_orig, h_orig):
                continue

            x1, y1 = max(0, int(box[0] * sx)), max(0, int(box[1] * sy))
            x2, y2 = min(w_proc, int(box[2] * sx)), min(h_proc, int(box[3] * sy))
            if x2 <= x1 or y2 <= y1:
                continue

            bw, bh = x2 - x1, y2 - y1
            # Local background context ring (expanded 25% around candidate box)
            pad_x = int(bw * 0.25)
            pad_y = int(bh * 0.25)
            rx1, ry1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            rx2, ry2 = min(w_proc, x2 + pad_x), min(h_proc, y2 + pad_y)

            ring_u = u[ry1:ry2, rx1:rx2]
            ring_v = v[ry1:ry2, rx1:rx2]

            # Mask out candidate box itself from background ring
            mask = np.ones((ry2 - ry1, rx2 - rx1), dtype=bool)
            bx1, by1 = x1 - rx1, y1 - ry1
            bx2, by2 = bx1 + bw, by1 + bh
            mask[by1:by2, bx1:bx2] = False

            if np.any(mask):
                bg_u = float(np.median(ring_u[mask]))
                bg_v = float(np.median(ring_v[mask]))
            else:
                bg_u = float(np.median(u))
                bg_v = float(np.median(v))

            # Relative motion of vehicle relative to immediate roadway background
            b_u = u[y1:y2, x1:x2] - bg_u
            b_v = v[y1:y2, x1:x2] - bg_v
            b_mag = np.sqrt(b_u**2 + b_v**2)

            mean_mag = float(np.mean(b_mag))
            std_mag = float(np.std(b_mag))
            angles = np.arctan2(b_v, b_u)
            angle_var = float(np.var(angles))
            local_discontinuity = float(np.linalg.norm([np.mean(u[y1:y2, x1:x2]) - bg_u, np.mean(v[y1:y2, x1:x2]) - bg_v]))

            # Directional vector coherence (ratio of net vector magnitude to mean magnitude)
            net_mag = float(np.linalg.norm([np.mean(b_u), np.mean(b_v)]))
            coherence = net_mag / (mean_mag + 1e-6)

            # Classify motion kinetics
            if (coherence < 0.75 and std_mag > 0.65 and mean_mag > 0.40) or \
               (angle_var > 1.10 and std_mag > 0.65 and mean_mag > 0.40):
                score = 0.70  # Chaotic turbulent collision kinetics
            elif local_discontinuity > 0.80 and (has_deceleration or is_stationary_post_impact):
                score = 0.65  # Post-impact stoppage with background motion discontinuity
            elif (has_deceleration or is_stationary_post_impact) and max_detection_conf >= 0.35:
                score = 0.55  # Confirmed post-impact deceleration/stationary hazard
            elif coherence >= 0.82 or (mean_mag < 0.35 and local_discontinuity < 0.50):
                score = 0.20  # Smooth laminar vehicular flow (unidirectional rigid motion)
            else:
                score = 0.35

            box_scores.append(score)

        if not box_scores:
            return 0.20

        return round(float(np.mean(box_scores)), 4)

    def evaluate_vehicle_interaction(
        self,
        detections: List[Dict[str, Any]],
        img_shape: Tuple[int, int] = (1080, 1920)
    ) -> float:
        """
        Stage 4: Evaluates vehicle interaction / collision evidence.
        Applies NMS deduplication, checks pairwise collision overlap IoU,
        and scores normal adjacent vehicles with low non-contact score.
        """
        if not detections:
            return 0.0

        img_h, img_w = img_shape[:2]
        valid_dets = [d for d in detections if not self.is_ego_border_box(d.get("box", [0, 0, 0, 0]), img_w, img_h)]
        if not valid_dets:
            return 0.15

        # Step A: Deduplicate candidate proposals
        dedup_dets = self._deduplicate_boxes(valid_dets, iou_thresh=0.40)

        if len(dedup_dets) >= 2:
            max_iou = 0.0
            for i in range(len(dedup_dets)):
                for j in range(i + 1, len(dedup_dets)):
                    b1 = dedup_dets[i]["box"]
                    b2 = dedup_dets[j]["box"]
                    iou = self._compute_iou(b1, b2)
                    max_iou = max(max_iou, iou)

            if max_iou > 0.08:
                # Direct physical collision contact with spatial overlap
                return round(min(1.0, 0.60 + max_iou * 0.40), 4)
            else:
                # Multiple vehicles in vicinity but without collision overlap
                return 0.20
        else:
            # Single candidate detection box: check aspect ratio for rollover anomaly
            box = dedup_dets[0]["box"]
            w = abs(box[2] - box[0])
            h = abs(box[3] - box[1])
            aspect_ratio = w / (h + 1e-6)
            is_edge = (box[0] <= 15 or box[1] <= 15 or box[2] >= img_w - 15 or box[3] >= img_h - 15)
            if not is_edge and (aspect_ratio < 0.25 or aspect_ratio > 4.5):
                return 0.60  # Confirmed isolated rollover / severe structural deformity
            else:
                return 0.15  # Isolated vehicle outline (no interaction)

    def _deduplicate_boxes(self, detections: List[Dict[str, Any]], iou_thresh: float = 0.40) -> List[Dict[str, Any]]:
        """Suppresses redundant duplicate bounding boxes on the same object."""
        if len(detections) <= 1:
            return detections

        sorted_dets = sorted(detections, key=lambda d: d.get("confidence", 0.0), reverse=True)
        keep = []
        for det in sorted_dets:
            overlap = False
            for kept in keep:
                if self._compute_iou(det["box"], kept["box"]) > iou_thresh:
                    overlap = True
                    break
            if not overlap:
                keep.append(det)
        return keep

    @staticmethod
    def _compute_iou(b1: List[float], b2: List[float]) -> float:
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])

        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area1 = abs(b1[2] - b1[0]) * abs(b1[3] - b1[1])
        area2 = abs(b2[2] - b2[0]) * abs(b2[3] - b2[1])
        union = area1 + area2 - inter + 1e-7
        return float(inter / union)
