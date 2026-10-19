"""Keep a local, hash-checked pre-test checkpoint without modifying Git."""
import hashlib
import json
from pathlib import Path


def freeze_inputs(directory: Path, names: list[str]) -> Path:
    """Create once; refuse to silently replace choices after calibration."""
    payload = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
               for name in names}
    checkpoint = directory / 'pretest_checkpoint.json'
    if checkpoint.exists():
        if json.loads(checkpoint.read_text(encoding='utf-8')) != payload:
            raise RuntimeError('Frozen inputs changed. Use a separate experiment directory; '
                               'do not overwrite choices after viewing test results.')
    else:
        with checkpoint.open('x', encoding='utf-8') as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
    return checkpoint
