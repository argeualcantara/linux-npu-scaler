import logging

import torch

log = logging.getLogger(__name__)


def _remap_suxrobgm_keys(state_dict: dict) -> dict:
    """
    Translates suxrobGM/fsrcnn layer names to our layer names.

    suxrobGM/fsrcnn uses:          We use:
        feature.N.*                    feature_extraction.N.*
        shrink.N.*                     shrinking.N.*
        map.N.*                        mapping.N.*
        expand.N.*                     expanding.N.*
        deconv.*                       (skipped — different upsampler)
    """
    prefix_map = {
        'feature.': 'feature_extraction.',
        'shrink.':  'shrinking.',
        'map.':     'mapping.',
        'expand.':  'expanding.',
    }
    remapped = {}
    skipped  = []
    for key, val in state_dict.items():
        new_key = None
        for src, dst in prefix_map.items():
            if key.startswith(src):
                new_key = dst + key[len(src):]
                break
        if new_key is not None:
            remapped[new_key] = val
        else:
            skipped.append(key)

    if skipped:
        log.warning(f"Skipped (incompatible upsampler — will fine-tune from random init): {skipped}")
    return remapped


def load_pretrained(model, path: str, device):
    """
    Loads weights from a pretrained checkpoint into the model.

    Supported formats:
      1. Our own checkpoints          — 'model_state_dict' key
      2. suxrobGM/fsrcnn checkpoints  — 'model' key, different layer names
      3. Generic PyTorch Lightning    — 'state_dict' key
      4. Raw state_dict               — bare OrderedDict
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
            log.info(f"Checkpoint format: our checkpoint (epoch={ckpt.get('epoch','?')})")
        elif 'model' in ckpt:
            raw = ckpt['model']
            log.info(f"Checkpoint format: suxrobGM/fsrcnn checkpoint (epoch={ckpt.get('epoch','?')})")
            log.info("Remapping layer names...")
            state_dict = _remap_suxrobgm_keys(raw)
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
            log.info(f"Checkpoint format: generic state_dict checkpoint")
        else:
            state_dict = ckpt
            log.info(f"Checkpoint format: raw state_dict")
    else:
        state_dict = ckpt
        log.info(f"Checkpoint format: raw tensor checkpoint")

    model_state = model.state_dict()
    loaded, skipped_shape, skipped_missing = 0, [], []

    for key, val in state_dict.items():
        if key not in model_state:
            skipped_missing.append(key)
        elif model_state[key].shape != val.shape:
            skipped_shape.append(
                f"{key}: ckpt={tuple(val.shape)} model={tuple(model_state[key].shape)}"
            )
        else:
            model_state[key] = val
            loaded += 1

    model.load_state_dict(model_state)

    if skipped_shape:
        log.warning(f"Skipped shape mismatch (random init): {len(skipped_shape)} tensors")
        for s in skipped_shape:
            log.warning(f"  {s}")
    if skipped_missing:
        log.warning(f"Skipped (not in model): {skipped_missing}")

    log.info(f"Successfully loaded {loaded}/{len(state_dict)} weight tensors.")
    if loaded == 0:
        log.warning("No weights transferred — model will train from scratch.")
    return model
