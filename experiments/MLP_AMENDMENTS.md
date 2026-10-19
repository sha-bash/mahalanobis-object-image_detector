# Execution notes, 2026-10-11

Before scoring or training, caption generation was checkpointed and resumed with
keep_alive=5m instead of 0. The original runtime unloaded the VLM after every
image to permit alternating with SBERT. This experiment has separate stages,
so retaining only the VLM during its stage avoids repeated model loads. Prompt,
temperature, context, token budget, images, cache keys and evaluation stay fixed.
Existing descriptions are retained; cold/resumed runtime sessions are recorded.
This is a compute optimization, not selection based on recognition results.

Implementation detail fixed before training: identity-balanced triplets are
forwarded jointly through the upstream MLP so all three branches share BatchNorm
statistics. The upstream training script forwards three branches separately.
The actual architecture is unchanged; this training adaptation is disclosed.

Before any feature evaluation/training, several missing descriptions were observed
(insufficient visible attributes, no valid cache). They are retained as permanent
text refusals, encoded only with an inert placeholder to keep row alignment;
excluded from text training/covariance. Any pair lacking a valid reference/candidate
description receives sentinel distance1e30, never included in the threshold grid.
If the recall budget is infeasible, report infeasible and maximum attainable recall;
do not discard the method or silently relax the budget. Visual methods retain all
images. This failure policy was added after observing description failures, before
seeing classifier scores, and is explicitly an execution amendment.

Statistical clarification before training/test scores: the exact conditional sign
test addresses H0: P(delta>0) <= P(delta<0), discarding exact ties. This directional
null must not be relabeled a test of the mean. The bootstrap estimates the mean
FPR difference. Practical success still requires positive mean FPR reduction and
the frozen reference recall; this is separate from statistical significance.
PR curves exclude the unachievable threshold that accepts permanent refusals.
Reported AP integrates only achievable recall increments (missing descriptions
contribute zero). Conventional rank ROC AUC in raw CSV treats refusals as lowest
ties; the plotted operational ROC is truncated to achievable decisions.
