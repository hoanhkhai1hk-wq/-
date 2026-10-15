"""Strict checkpoint loading for the FTW v1 CC-BY model.

Loads only the 634 ``model.*`` tensors into a fresh smp.Unet with
``strict=True``. The single non-model entry ``criterion.weight`` (training
loss weight) is explicitly excluded and recorded. Any missing key, extra key
or shape mismatch fails hard: no random init, no ignored keys, no other model.
"""

import torch
import segmentation_models_pytorch as smp

EXPECTED_HYPER = {
    "model": "unet",
    "backbone": "efficientnet-b3",
    "in_channels": 8,
    "num_classes": 3,
}
EXCLUDED_KEYS = ["criterion.weight"]
PREFIX = "model."


def load_model(checkpoint_path):
    """Return (model, hyper_parameters, load_report) with strict verification."""
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = ck.get("state_dict")
    if state is None:
        raise ValueError("checkpoint has no state_dict")
    hp = ck.get("hyper_parameters", {})

    for key, expected in EXPECTED_HYPER.items():
        actual = hp.get(key)
        if actual != expected:
            raise ValueError(
                f"checkpoint hyper_parameters mismatch: {key}={actual!r}, expected {expected!r}"
            )

    excluded = sorted(k for k in state if not k.startswith(PREFIX))
    if excluded != EXCLUDED_KEYS:
        raise ValueError(
            f"unexpected non-model checkpoint entries {excluded!r}, expected {EXCLUDED_KEYS!r}"
        )

    model = smp.Unet(
        encoder_name=hp["backbone"],
        encoder_weights=None,
        in_channels=hp["in_channels"],
        classes=hp["num_classes"],
    )
    target = model.state_dict()

    filtered = {}
    for key in state:
        if key.startswith(PREFIX):
            short = key[len(PREFIX):]
            if short not in target:
                raise ValueError(f"checkpoint key not in fresh SMP model: {key}")
            if tuple(state[key].shape) != tuple(target[short].shape):
                raise ValueError(
                    f"shape mismatch for {key}: checkpoint {tuple(state[key].shape)} "
                    f"vs model {tuple(target[short].shape)}"
                )
            filtered[short] = state[key]

    missing = sorted(set(target) - set(filtered))
    if missing:
        raise ValueError(f"model keys missing from checkpoint: {missing[:5]}... ({len(missing)} total)")

    model.load_state_dict(filtered, strict=True)
    model.eval()

    report = {
        "checkpoint_state_entries": len(state),
        "model_tensors_loaded": len(filtered),
        "excluded_training_keys": EXCLUDED_KEYS,
        "strict": True,
        "hyper_parameters": {k: hp.get(k) for k in EXPECTED_HYPER},
    }
    return model, hp, report
