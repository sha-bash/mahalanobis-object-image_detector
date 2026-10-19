# GMOT-40: preregistered external pilot, 2026-10-02

Frozen before image inspection or model inference. This extends the earlier debug
study; it is not a claim to evaluate the official GMOT task.

Source: https://spritea.github.io/GMOT40/download.html and official GitHub release
v0.1. Authors: Bai et al., CVPR 2021. License: CC BY-NC-SA 4.0 per official
repository https://github.com/Spritea/GMOT40. Keep downloaded pixels/labels out of Git.

## Selection and input

Use all four `car-0` through `car-3` clips, never select by model score. For each,
choose the lowest numeric target ID labelled in frame 0 with width >= 40 and
height >= 20 pixels and at least 30 labelled frames in the first 300 frames.
If no target qualifies, report that clip as ineligible; do not substitute another
category. Use its frame-0 bounding-box crop as the single reference. Algorithms
receive no initial video box. Evaluate source frames 1 through min(299,last),
excluding the reference frame. This is a same-capture reference protocol, not
cross-camera or independent-capture re-identification.

All selected clips are an external fixed-configuration test of the existing
method. No training, covariance fitting or threshold optimization on these clips.
Unknown cross-clip identity/capture overlap must be disclosed; four filenames do
not guarantee four independent captures. No confirmatory p-values from four clips.

## Comparisons and decisions

Paired geometry versus geometry + current VLM/SBERT gate. Seed 2027, tau=6,
variance=.01, qwen3.5:0.8b, pinned local all-MiniLM-L6-v2, original image resolution
(runner cap 1920). Use neutral scene context (empty string) for both new tests:
the debug runner's aerial-drone hint must not be applied to unrelated footage.
All other method settings remain fixed. Fresh VLM captions per run; save all
descriptions, failures, predictions, hashes, versions and times. Report algorithm
initialization failures, never drop them as successful experiments.

IoU >= .5 defines successful localization. Every labelled target frame enters
recall and mean IoU, including misses. Wrong localization is FP + FN. Identify
wrong-instance acceptance by overlap >= .5 with a different annotated ID.
Unlabelled target frames are IGNORE for primary evaluation, not proven physical
absence: full occlusion and annotation omissions cannot be distinguished safely.
Also report annotation-absence outputs descriptively, never as verified absent FPR.
Aggregate equal-weight per clip as primary; pooled counts secondary.

Evidence of useful filtering requires fewer wrong-instance acceptances, recall
>= .70 and recall reduction <= .05 versus geometry; otherwise report no useful
improvement under this pilot criterion. With zero wrong-instance baseline errors,
the reduction hypothesis is untested, not supported. This criterion is a new
pilot extension, not retroactively used to choose earlier configurations.

## Separate semantic-module diagnostic

For each selected target choose two positive crops: eligible labelled frames
(width >=40, height >=20, source frame >=30, <=299) nearest the 1/3 and 2/3
positions of that eligible frame list, removing duplicates. At each selected
frame choose the lowest other numeric ID meeting the same size restriction.
If none exists, record unavailable; do not search for a favourable negative.
Positive and negative candidates come from the SAME frame when possible.
Compare accepting every supplied candidate to the fixed semantic gate, tau=6.
This oracle-crop diagnostic bypasses localization and cannot establish full-pipeline
improvement. Negative objects are same-category instances, not necessarily visually
near-identical. Report TPR/FPR and balanced accuracy at candidate level, per clip
and pooled. Threshold curves are exploratory replay only, without retuning.

Compute limits: four clips x at most 299 frames x two methods; up to 16 candidate
checks plus four references. Additional data or model changes require a new study.
