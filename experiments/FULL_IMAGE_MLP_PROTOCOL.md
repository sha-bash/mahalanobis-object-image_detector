# Frozen full-image follow-up, before any new detection inference

Use all 21 validation/test clips from MLP_PROTOCOL.md, two source frames per clip
(positions 40,60 of first selected ID). Query is that ID's position-0 crop. Whole
frames, no GT candidate boxes provided to any detector. All categories retained.
YOLOv8n local pretrained weights, CPU four threads, image size 640, conf .25,
NMS IoU .45, max 20 proposals, no class/vehicle filter. This is a COCO-pretrained
proposal baseline with limited coverage of novel categories, not a universal detector.

Run SIFT/affine RANSAC using InstanceLocator defaults; no initial video box.
YOLO candidates are matched to reference using frozen DINO cosine or shared
visual MLP cosine (seed 2027). Top-1 candidate per query, deterministic tie by
proposal index. Apply already frozen GT-pair validation thresholds to these
distances: a transfer evaluation, not new calibration on test. For semantic
ablation, verify the SAME selected candidate using text MLP full Mahalanobis
and its already frozen threshold. Accepted iff both visual and text accept.
This gate cannot increase recall; test whether it removes wrong detections and
quantify sacrificed correct detections. Scores for all proposals are saved.

VLM sees only query/candidate crops independently; missing descriptions imply
refusal and are logged. Caption only the union of two selected proposals per frame,
at most 84 new crops plus 21 queries already cached. Models execute sequentially.
No resampling, no reference replacement, no dropping unsupported categories.

Metrics: TP iff issued box IoU>=.5; wrong box FP+FN, refusal FN. Mean IoU over all
target-present frames includes zeros; wrong-box nonzero IoU still included.
Wrong-instance acceptance additionally requires IoU>=.5 to another annotated ID.
Per-video and aggregate precision/recall, FP/frame and IoU. No target-absence FPR
because these selected frames contain target. Two frames per clip do not evaluate
tracking loss/reacquisition; use previous video results separately, not as new MLP
tracking evidence. Bootstrap unit video. Latency separate proposal/embedding/VLM.
