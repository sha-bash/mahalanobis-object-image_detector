"""Audit a saved semantic run and optionally score single-target frame annotations."""

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--annotations", type=Path)
    args = parser.parse_args()
    result = json.loads((args.run / "results.json").read_text(encoding="utf-8"))
    assert result["mandatory_mahalanobis"] is True, "This is not a semantic run"
    rows, checks = result["frames"], result["semantic_checks"]
    assert [r["frame"] for r in rows] == list(range(len(rows)))
    for row in rows:
        if row["found"]:
            check = checks[row["semantic_check_id"]]
            assert (
                check["accepted"] is True
                and 0 <= check["distance"] <= check["threshold"]
            )
            assert row["semantic_source_frame"] == check["frame"] <= row["frame"]
    capture = cv2.VideoCapture(str(args.run / "annotated.mp4"))
    assert capture.isOpened(), "Cannot open exported video"
    fps = capture.get(cv2.CAP_PROP_FPS)
    decoded = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            assert list(image.shape[1::-1]) == result["output_size"]
            decoded += 1
    finally:
        capture.release()
    assert decoded == len(rows) == result["processed_frames"]
    assert abs(fps - result["fps"]) < 0.001
    metrics = {
        "decoded_frames": decoded,
        "fps": fps,
        "found_frames": sum(r["found"] for r in rows),
        "semantic_gate_audit_passed": True,
    }
    if args.annotations:
        gt = json.loads(args.annotations.read_text(encoding="utf-8"))
        images = {image["id"]: image for image in gt["images"]}
        seen, scores = set(), []
        for annotation in gt["annotations"]:
            image = images[annotation["image_id"]]
            match = re.fullmatch(r"frame_(\d+)", Path(image["file_name"]).stem)
            if not match:
                raise ValueError("Annotations must use frame_N image filenames")
            index = int(match[1])
            if index in seen:
                raise ValueError("Supply annotations for a single target per frame")
            seen.add(index)
            row = rows[index]
            scale = np.array(result["output_size"]) / [image["width"], image["height"]]
            x, y, w, h = annotation["bbox"]
            target = np.array([x, y, x + w, y + h]) * np.tile(scale, 2)
            score = 0.0
            if row["found"]:
                predicted = np.array(row["bbox_xyxy"])
                overlap = np.maximum(
                    0,
                    np.minimum(target[2:], predicted[2:])
                    - np.maximum(target[:2], predicted[:2]),
                ).prod()
                union = (
                    (target[2:] - target[:2]).prod()
                    + (predicted[2:] - predicted[:2]).prod()
                    - overlap
                )
                score = float(overlap / union) if union > 0 else 0.0
            scores.append(score)
        if not scores:
            raise ValueError("No target annotations")
        metrics.update(
            annotated_frames=len(scores),
            mean_iou=float(np.mean(scores)),
            frames_iou_at_least_05=sum(s >= 0.5 for s in scores),
            note="Only annotated frames scored; unannotated frames are not negatives. "
            "This does not establish generalization to independent videos.",
        )
    (args.run / "validation.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
