# Reproduce the public-data pilot

Use existing `.venv` (ML) and `.audit-venv` (data preparation/monitoring/plots),
or recreate them using the pinned requirements in `REPRODUCE.md`. Run from the
repository root in PowerShell. Data and run directories are ignored by Git.
Do not overwrite previous run directories: rename the local run root in the
scripts for a repeat, or use another checkout with its own ignored directories.

The GMOT-40 authors release the data under CC BY-NC-SA 4.0. Source and citation:
https://spritea.github.io/GMOT40/download.html ; Bai et al., GMOT-40, CVPR 2021.
These commands download only the authors' official release, not third-party mirrors.

```powershell
New-Item -ItemType Directory -Force experiments/local/gmot40
curl.exe -L --fail --retry 2 https://github.com/Spritea/GMOT40/releases/download/v0.1/track_label.zip -o experiments/local/gmot40/track_label.zip
curl.exe -L --fail --retry 2 https://github.com/Spritea/GMOT40/releases/download/v0.1/GenericMOT_JPEG_Sequence.zip -o experiments/local/gmot40/GenericMOT_JPEG_Sequence.zip
.audit-venv/Scripts/python.exe scripts/prepare_gmot40.py
```

Verify archive hashes against `evidence/gmot40/inventory.json`. The preparation
script refuses to overwrite `prepared/`, converts JPEGs to lossless FFV1 AVI,
and saves reference/panel crops and explicit annotations. The video processing
clock is set to 30 fps because the archive does not include per-clip timing.
This affects periodic rechecks and does not establish the source capture fps.
Frame 0 is excluded from videos evaluated by the algorithms. All others up to
source frame 299 are processed; shorter clips use every remaining frame.

The exact local SBERT snapshot used here is shown below; adjust only its directory
prefix on another machine, retaining revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
Ollama weights remain `qwen3.5:0.8b`; compare digests in the saved journal before
calling another run the same model. No cloud inference credentials are required.

```powershell
$gmotSbert = 'C:/Users/sha-b/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
foreach ($gmotClip in @('car-0','car-1','car-2','car-3')) {
  .audit-venv/Scripts/python.exe scripts/run_matrix.py --python .venv/Scripts/python.exe --out "experiments/runs/gmot40/$gmotClip" --variants geometry semantic --manifest experiments/gmot40_manifest.json --sequence $gmotClip --fresh-captions --managed-server --sbert-model $gmotSbert
}
.venv/Scripts/python.exe scripts/probe_gmot40.py --out experiments/runs/gmot40_probe --sbert-model $gmotSbert
.audit-venv/Scripts/python.exe scripts/summarize_gmot40.py
.venv/Scripts/python.exe -m pytest -q
```

`run_matrix.py` records each nonzero exit and proceeds to the next fixed variant.
Initialization failure is a result, not permission to pick an easier reference.
The semantic probe operates on the frozen `panel.json` independently of geometric
initialization, saving every candidate score and VLM caption. Its threshold curves
are diagnostic replay, not online runs at multiple thresholds. Do not optimize on
this test panel. Online wrong-instance counts and oracle candidate error rates
are different units and must not be combined.
