"""Reference video search with mandatory SBERT/Mahalanobis confirmation.

Geometry proposes and tracks candidates; semantic distance authorizes detections.
SBERT runs in a short-lived worker so it does not share memory with the VLM.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np


class InstanceLocator:
    def __init__(self, reference: np.ndarray, min_inliers: int = 8):
        self.reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        self.height, self.width = self.reference.shape
        self.outline = np.float32(
            [
                [0, 0],
                [self.width - 1, 0],
                [self.width - 1, self.height - 1],
                [0, self.height - 1],
            ]
        )
        # A tightly cropped single-object reference permits foreground extraction.
        # Fall back to its rectangle if segmentation is degenerate.
        if min(self.height, self.width) >= 20:
            labels = np.zeros(self.reference.shape, np.uint8)
            cv2.grabCut(
                reference,
                labels,
                (2, 2, self.width - 4, self.height - 4),
                np.zeros((1, 65), np.float64),
                np.zeros((1, 65), np.float64),
                5,
                cv2.GC_INIT_WITH_RECT,
            )
            foreground = np.uint8((labels == 1) | (labels == 3)) * 255
            contours, _ = cv2.findContours(
                foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                hull = cv2.convexHull(max(contours, key=cv2.contourArea))
                fraction = cv2.contourArea(hull) / (self.width * self.height)
                if 0.2 <= fraction <= 0.9:
                    self.outline = hull[:, 0].astype(np.float32)
        self.sift = cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.02)
        self.keys, self.descriptors = self.sift.detectAndCompute(self.reference, None)
        if self.descriptors is None or len(self.keys) < min_inliers:
            raise ValueError(
                "Reference has too few distinctive details for geometric matching"
            )
        self.min_inliers = min_inliers
        self.matrix = None
        self.previous = None
        self.points = None
        self.last_score = 0.0

    def polygon(self, matrix):
        return cv2.transform(self.outline[None], matrix[:2])[0]

    def support_polygon(self, matrix):
        corners = np.float32(
            [[[0, 0], [self.width, 0], [self.width, self.height], [0, self.height]]]
        )
        return cv2.transform(corners, matrix[:2])[0]

    def appearance(self, gray, matrix):
        patch = cv2.warpAffine(
            gray,
            matrix[:2],
            (self.width, self.height),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        )
        visible = cv2.warpAffine(
            np.ones(gray.shape, np.uint8),
            matrix[:2],
            (self.width, self.height),
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        )
        visible = cv2.erode(visible, np.ones((5, 5), np.uint8)).astype(bool)
        if visible.mean() < 0.5:
            return 0.0
        a = cv2.GaussianBlur(self.reference, (5, 5), 0)[visible].astype(float)
        b = cv2.GaussianBlur(patch, (5, 5), 0)[visible].astype(float)
        a -= a.mean()
        b -= b.mean()
        return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9))

    def valid(self, gray, matrix):
        if matrix is None or not np.isfinite(matrix).all():
            return False
        polygon = self.polygon(matrix)
        h, w = gray.shape
        area = abs(cv2.contourArea(polygon))
        if not 100 <= area <= 0.25 * w * h:
            return False
        self.last_score = self.appearance(gray, matrix)
        return self.last_score >= 0.45

    def detect(self, gray):
        keys, descriptors = self.sift.detectAndCompute(gray, None)
        if descriptors is None or len(keys) < 2:
            return None
        pairs = cv2.BFMatcher().knnMatch(self.descriptors, descriptors, k=2)
        matches = [
            pair[0]
            for pair in pairs
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
        ]
        # Several reference descriptors must not count as independent matches
        # to the same video feature.
        unique = {}
        for match in sorted(matches, key=lambda m: m.distance):
            unique.setdefault(match.trainIdx, match)
        matches = list(unique.values())
        if len(matches) < self.min_inliers:
            return None
        source = np.float32([self.keys[m.queryIdx].pt for m in matches])
        target = np.float32([keys[m.trainIdx].pt for m in matches])
        affine, mask = cv2.estimateAffinePartial2D(
            source, target, method=cv2.RANSAC, ransacReprojThreshold=3
        )
        if affine is None or mask is None:
            return None
        inliers = mask.ravel().astype(bool)
        if inliers.sum() < self.min_inliers or inliers.mean() < 0.5:
            return None
        spread = cv2.contourArea(cv2.convexHull(source[inliers]))
        if spread < 0.04 * self.width * self.height:
            return None
        matrix = np.vstack([affine, [0, 0, 1]])
        return matrix if self.valid(gray, matrix) else None

    def follow(self, gray):
        if self.points is None or len(self.points) < self.min_inliers:
            return None
        forward, status, _ = cv2.calcOpticalFlowPyrLK(
            self.previous, gray, self.points, None, winSize=(21, 21), maxLevel=3
        )
        if forward is None:
            return None
        backward, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.previous, forward, None, winSize=(21, 21), maxLevel=3
        )
        if backward is None:
            return None
        good = (
            (status.ravel() == 1)
            & (back_status.ravel() == 1)
            & (np.linalg.norm(self.points - backward, axis=2).ravel() < 1.5)
        )
        if good.sum() < self.min_inliers:
            return None
        affine, mask = cv2.estimateAffinePartial2D(
            self.points[good], forward[good], method=cv2.RANSAC, ransacReprojThreshold=2
        )
        if affine is None or mask is None or mask.sum() < self.min_inliers:
            return None
        matrix = np.vstack([affine, [0, 0, 1]]) @ self.matrix
        return matrix if self.valid(gray, matrix) else None

    def update(self, frame, *, allow_detection=True):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        matrix = self.follow(gray) if self.matrix is not None else None
        source = "tracked"
        if matrix is None and allow_detection:
            matrix = self.detect(gray)
            source = "reference_match"
        self.matrix = matrix
        self.previous = gray
        self.points = None
        if matrix is None:
            return None
        polygon = self.polygon(matrix)
        mask = np.zeros(gray.shape, np.uint8)
        cv2.fillConvexPoly(mask, self.support_polygon(matrix).astype(np.int32), 255)
        self.points = cv2.goodFeaturesToTrack(gray, 100, 0.01, 3, mask=mask)
        return {
            "polygon": polygon.tolist(),
            "appearance_ncc": self.last_score,
            "source": source,
        }


def run(
    reference,
    video,
    out,
    *,
    ref_box=None,
    width=1920,
    verifier=None,
    verify_every=10,
    baseline=False,
):
    if verifier is None and not baseline:
        raise ValueError(
            "A SBERT/Mahalanobis verifier is required; baseline must be explicitly enabled"
        )
    if not math.isfinite(verify_every) or verify_every < 0:
        raise ValueError("verify_every must be a finite non-negative number")
    started = time.perf_counter()
    reference = Path(reference)
    video = Path(video)
    ref = cv2.imread(str(reference))
    if ref is None:
        raise ValueError(f"Cannot read reference: {reference}")
    if ref_box:
        x1, y1, x2, y2 = ref_box
        if not (0 <= x1 < x2 <= ref.shape[1] and 0 <= y1 < y2 <= ref.shape[0]):
            raise ValueError("Reference box is outside the image")
        ref = ref[y1:y2, x1:x2]
    locator = InstanceLocator(ref)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video}")
    writer = None
    rows = []
    checks = []
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video has no valid frame rate")
        source_w, source_h = int(cap.get(3)), int(cap.get(4))
        scale = min(1.0, width / source_w)
        size = (
            max(2, int(source_w * scale) // 2 * 2),
            max(2, int(source_h * scale) // 2 * 2),
        )
        out = Path(out)
        out.mkdir(parents=True, exist_ok=False)
        cv2.imwrite(str(out / "reference.jpg"), ref)
        if verifier is not None:
            print(
                "Preparing reference semantic profile..."
                if not baseline
                else "Preparing baseline verifier...",
                flush=True,
            )
            verifier = verifier(ref)
            if not baseline:
                (out / "semantic_profile.json").write_text(
                    json.dumps(verifier.metadata, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        writer = cv2.VideoWriter(
            str(out / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
        )
        if not writer.isOpened():
            raise RuntimeError("Cannot open MP4 video writer")
        index = 0
        last_verified = -math.inf
        last_check_id = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            result = locator.update(
                frame, allow_detection=index % max(1, round(fps)) == 0
            )
            if result is None:
                last_check_id = None
            if (
                result is not None
                and verifier is not None
                and (
                    result["source"] == "reference_match"
                    or (
                        verify_every > 0 and index - last_verified >= fps * verify_every
                    )
                )
            ):
                polygon = np.asarray(result["polygon"])
                x1, y1 = np.maximum(0, np.floor(polygon.min(axis=0))).astype(int)
                x2, y2 = np.minimum(size, np.ceil(polygon.max(axis=0))).astype(int)
                check_started = time.perf_counter()
                decision = verifier.verify(frame[y1:y2, x1:x2])
                if not baseline:
                    distance, threshold = (
                        decision.get("distance"),
                        decision.get("threshold"),
                    )
                    unscored_rejection = (
                        distance is None
                        and decision.get("accepted") is False
                        and decision.get("reason") == "unusable_description"
                    )
                    if (
                        (
                            not unscored_rejection
                            and (
                                not isinstance(distance, (float, int))
                                or not math.isfinite(distance)
                                or distance < 0
                            )
                        )
                        or not isinstance(threshold, (float, int))
                        or not math.isfinite(threshold)
                        or threshold < 0
                    ):
                        raise ValueError(
                            "Semantic verifier did not produce a valid Mahalanobis calculation"
                        )
                    # Enforce the actual inequality here too: no visual score or
                    # VLM yes/no answer can override a failed semantic distance.
                    decision["accepted"] = (
                        not unscored_rejection
                        and decision.get("accepted") is True
                        and distance <= threshold
                    )
                    decision["match"] = "yes" if decision["accepted"] else "no"
                checks.append(
                    {
                        "frame": index,
                        "elapsed_sec": time.perf_counter() - check_started,
                        **decision,
                    }
                )
                last_check_id = len(checks) - 1
                if not baseline:
                    distance_label = (
                        "unavailable" if distance is None else f"{distance:.4f}"
                    )
                    print(
                        f"frame {index}: Mahalanobis={distance_label}, threshold={threshold:.4f}, accepted={decision['accepted']}",
                        flush=True,
                    )
                last_verified = index
                if decision["match"] != "yes":
                    locator.matrix = None
                    result = None
                    last_check_id = None
            if result is not None and not baseline and last_check_id is None:
                # A tracker is never allowed to create an unconfirmed detection.
                locator.matrix = None
                result = None
            row = {"frame": index, "time_sec": index / fps, "found": result is not None}
            if result:
                polygon = np.asarray(result["polygon"])
                lo = np.maximum(0, np.floor(polygon.min(axis=0))).astype(int)
                hi = np.minimum(size, np.ceil(polygon.max(axis=0))).astype(int)
                row.update(result)
                if not baseline:
                    confirmation_frame = checks[last_check_id]["frame"]
                    row.update(
                        semantic_check_id=last_check_id,
                        semantic_source_frame=confirmation_frame,
                        semantic_age_sec=(index - confirmation_frame) / fps,
                        semantic_state="confirmed"
                        if index == confirmation_frame
                        else "propagated",
                    )
                row["bbox_xyxy"] = [int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])]
                cv2.rectangle(frame, tuple(lo), tuple(hi), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    "Reference target",
                    (int(lo[0]), max(18, int(lo[1]) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
            writer.write(frame)
            rows.append(row)
            if index % round(fps * 4) == 0:
                cv2.imwrite(str(out / f"frame_{index:06d}.jpg"), frame)
                print(
                    f"frame {index}: {'found' if result else 'not found'}", flush=True
                )
            index += 1
        if not rows:
            raise ValueError("Video contains no decodable frames")
        payload = {
            "method": (
                "SIFT/RANSAC proposals -> independent VLM descriptions -> SBERT -> "
                "Mahalanobis acceptance -> optical flow / GrabCut"
            )
            if not baseline
            else "geometric_baseline",
            "mandatory_mahalanobis": not baseline,
            "reference": str(reference.resolve()),
            "reference_box": ref_box,
            "reference_outline": locator.outline.tolist(),
            "video": str(video.resolve()),
            "source_size": [source_w, source_h],
            "output_size": list(size),
            "bbox_coordinates": "output pixels",
            "fps": fps,
            "processed_frames": len(rows),
            "found_frames": sum(r["found"] for r in rows),
            "elapsed_sec": time.perf_counter() - started,
            "vlm_model": verifier.model if verifier else None,
            "verify_every_sec": verify_every,
            "vlm_checks": checks if baseline else [],
            "semantic_checks": checks if not baseline else [],
            "semantic_profile": verifier.metadata if not baseline else None,
            "frames": rows,
            "limitations": "No audio. At least half of the reference region must be visible. "
            "Appearance score is not identity probability. No ground-truth accuracy measured.",
        }
        (out / "results.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if not baseline:
            with (out / "semantic_distances.csv").open(
                "w", newline="", encoding="utf-8"
            ) as output:
                columns = [
                    "frame",
                    "time_sec",
                    "distance",
                    "threshold",
                    "accepted",
                    "caption",
                ]
                table = csv.DictWriter(output, fieldnames=columns)
                table.writeheader()
                for check in checks:
                    table.writerow(
                        {
                            "frame": check["frame"],
                            "time_sec": check["frame"] / fps,
                            "distance": check["distance"],
                            "threshold": check["threshold"],
                            "accepted": check["accepted"],
                            "caption": check.get("candidate_description", {}).get(
                                "caption", ""
                            ),
                        }
                    )
        return payload
    finally:
        cap.release()
        if writer is not None:
            writer.release()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True, help="New output directory")
    parser.add_argument(
        "--ref-box", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2")
    )
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--vlm", choices=["ollama", "none"], default="ollama")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Explicit comparison mode without semantic confirmation",
    )
    parser.add_argument("--model", default="qwen3.5:0.8b")
    parser.add_argument(
        "--host", default="http://127.0.0.1:11434", help="Ollama server URL"
    )
    parser.add_argument(
        "--verify-every",
        type=float,
        default=10,
        help="Periodic VLM check interval in video seconds; 0 checks acquisitions only",
    )
    parser.add_argument("--sbert-model", default="all-MiniLM-L6-v2")
    parser.add_argument(
        "--scene-context",
        default="",
        help="Optional acquisition context, not target identity",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=6.0,
        help="Mandatory Mahalanobis acceptance threshold",
    )
    parser.add_argument("--prior-variance", type=float, default=0.01)
    parser.add_argument(
        "--covariance", help="Optional .npy array of pre-estimated diagonal variances"
    )
    args = parser.parse_args(argv)
    if args.width < 320:
        parser.error("--width must be at least 320")
    cv2.setNumThreads(4)
    if args.vlm == "none" and not args.baseline:
        parser.error(
            "--vlm none requires --baseline; normal runs must use SBERT/Mahalanobis"
        )
    verifier = None
    if not args.baseline:
        from moid.semantic import SemanticVerifier

        verifier = lambda ref: SemanticVerifier(
            ref,
            model=args.model,
            host=args.host,
            sbert_model=args.sbert_model,
            threshold=args.tau,
            variance=args.prior_variance,
            covariance_path=args.covariance,
            scene_context=args.scene_context,
        )
    result = run(
        args.reference,
        args.video,
        args.out,
        ref_box=args.ref_box,
        width=args.width,
        verifier=verifier,
        verify_every=args.verify_every,
        baseline=args.baseline,
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in {"frames", "semantic_profile", "reference_outline"}
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
