# Frozen expansion protocol, 2026-10-11

Registered before generating new captions, training projectors, or computing new
validation/test scores. Existing car/airplane/boat experiments are development only.
This is an instance-search study, not a reproduction of category-level one-shot GMOT.

## Questions and hypotheses

Does independent textual verification reduce false acceptance of other instances
at a validation-calibrated recall budget? Does the author's learned text projector
transfer to unseen identities? Does covariance help beyond cosine on the SAME
features? Does a visual Siamese/triplet head help? Does text add to that head?

Primary comparison: visual cosine versus equal-weight visual/text MLP Mahalanobis
fusion, one reference. H0: paired sequence-macro FPR reduction <= 0; H1: > 0.
Success additionally requires test recall not below the baseline reference recall;
report the measured recall difference and interval, not a post-hoc tolerance.
Failure to reject H0 does not prove equivalence. Other comparisons exploratory.

## Data and independence

Official GMOT-40 images/trajectory labels, CC BY-NC-SA 4.0, retained locally.
New categories: ball, balloon, bird, fish, insect, person, stock. For each category,
clip 0=train, 1=validation, 2 and 3=test. No swapping based on results. Each clip:
first four sorted track IDs with >=61 boxes of width and height >=20 pixels among
frames 0..299. Use eligible positions 0,10,20 as references (K=1 or K=3), positions
40,60 as candidates. Inventory found >=4 eligible IDs in every clip BEFORE scoring.
Thus 28 videos, 112 within-video identities, 560 crops. Different trajectories in
one video are not independent observations. Cross-video physical identity cannot
be guaranteed by this dataset; report this limitation. Same-scene reference only.
No augmentation counts as independent data. No separate-capture generalization claim.
This expansion covers seven categories, not all objects or buildings.

Train: 7 clips/28 IDs/140 crops; validation: 7/28/140; test: 14/56/280.
Train IDs include all five selected crops for metric learning. Validation/test
references are never training examples. Covariance, MLP and score scaling use train only.

## Series and fixed methods

GT-crop identity experiment (no detection IoU claims): SIFT ratio-test similarity;
frozen DINOv2-small cosine; text Qwen3.5:0.8b -> MiniLM cosine; text raw diagonal
and Ledoit-Wolf Mahalanobis; exact upstream MLP architecture applied to text with
cosine/diagonal/Ledoit-Wolf; shared visual MLP (Siamese/triplet) with cosine and
Ledoit-Wolf; equal-weight fusion of visual and text Mahalanobis distances after
division by train negative-pair medians. No tuning fusion weights on test.

Author's projector from mahalanobis-concept-drift commit
fb3bccd1e4a0768fcfb15817fb4fcb809f2af4ad: 384->256->128->64, BatchNorm/GELU,
dropout .1, triplet margin 1, Adam lr .001, 20 epochs, seeds 2027/2028/2029.
Balanced batches contain 4 identities and all 5 examples each. Triplets select
same-ID positives and other-ID negatives; no validation early stopping. Fixed
training before test. Primary seed 2027; others quantify optimization sensitivity,
not independent datasets. Freeze encoders. Shared weights are used on both sides.
All Mahalanobis covariances use symmetric train within-ID pair differences; use
Ledoit-Wolf plus 1e-4 I or 50% diagonal shrinkage plus 1e-4. Normalize vectors first.
Do not estimate target covariance from one reference or conflate classes with modalities.

Baseline recall reference: arithmetic mean of validation sequence-macro recalls
of SIFT and frozen DINO cosine, each at its own maximum macro-F1 threshold (ties:
lower macro-FPR, then smaller threshold). Compute separately for K=1 and K=3.
Then ALL methods, including baselines, are calibrated on validation to minimize
macro-FPR subject to macro-recall >= that reference (ties: higher recall, smaller
threshold). Baseline membership and rules fixed here before validation. Freeze
thresholds before opening test scores. Report infeasibility and test recall drift.

## Metrics and statistical unit

Each query is compared with all 8 same-video candidate crops: 2 positive, 6
different-instance negative pairs. K-reference score=min reference distance.
Report per-video and macro precision, recall, F1, FPR, confusion counts, ROC/PR
curves and AUC; pair-level negatives mean verified DIFFERENT INSTANCE crops,
not empty video frames. Top-1 identity accuracy separate, fractional credit for ties.
95% paired bootstrap over videos (10000 draws, seed 2027), paired exact sign test;
also show category-level paired effects (only 7 categories). No frame-iid tests.
Three MLP seeds, not three independent replicates of the dataset.

## Full-image and video follow-up

Fixed full-image proposal baselines: pretrained YOLOv8n (CPU) + frozen visual
matching if dependencies work; SIFT/RANSAC. Detectors output classes, so require
reference matching to count as instance baselines. Record unsupported categories.
Shared proposals are necessary to isolate verifier contribution. GT crops are an
oracle diagnostic and must never be reported as end-to-end detection accuracy.
Full-image sample frames: selected candidate source frames of the first chosen ID
per validation/test clip. Query uses that ID's first reference; wrong-ID boxes
count FP and target miss FN. TP requires IoU>=.5 with target; mean IoU includes
zero for refusal. Gaps in annotations are unknown, not verified target absence.
Any dependency/data failure is reported; no invented external benchmark scores.
Initial-box Siamese trackers, if run, use a distinct protocol; a shared learned
visual head is explicitly described as matching, not a published full tracker.

## Compute and artifacts

CPU, four threads, batch <=8 for image encoder. VLM, DINO and MiniLM loaded in
separate stages, checkpoint captions per crop and features before training.
Record code/protocol/data/model hashes, seeds, versions, elapsed time and peak
process-tree RSS where available. Warm distance timing is separate from caption
and feature cost; no cross-cache speed claims. Russian plot labels.
Save all completed stages, test frozen parameters, raw predictions and failures.
