"""Safe CPU persistence/inference for the author's BatchNorm/GELU projector."""
import json
from pathlib import Path

import numpy as np
import torch

from mcd.projection.projector import DeepMahalanobisProjector


def save_projector(model, config_path):
    path=Path(config_path)
    path.parent.mkdir(parents=True,exist_ok=True)
    weights=path.with_suffix('.pt')
    torch.save(model.state_dict(),weights)
    path.write_text(json.dumps(dict(schema=1,architecture=model.architecture_dict(),weights=weights.name),indent=2),encoding='utf-8')


def load_projector(config_path):
    path=Path(config_path)
    cfg=json.loads(path.read_text(encoding='utf-8'))
    if cfg.get('schema')!=1:
        raise ValueError('Unsupported learned projector schema')
    weights=(path.parent/cfg['weights']).resolve()
    if weights.parent!=path.parent.resolve():
        raise ValueError('Weights must be alongside projector config')
    model=DeepMahalanobisProjector.from_architecture_dict(cfg['architecture'])
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
    return model.eval()


def project(model,features,batch_size=128):
    x=np.asarray(features,dtype=np.float32)
    if x.ndim!=2 or not np.isfinite(x).all() or batch_size<1:
        raise ValueError('Invalid projector input')
    model.eval()
    if not len(x):
        return np.empty((0,model.architecture_dict()['output_dim']))
    with torch.inference_mode():
        return np.vstack([model(torch.from_numpy(x[i:i+batch_size])).numpy() for i in range(0,len(x),batch_size)]).astype(float)
