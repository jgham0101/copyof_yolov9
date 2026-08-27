from __future__ import annotations
import torch

def select_decoded_prediction(prediction, nc: int = 80):
    if torch.is_tensor(prediction):
        decoded = prediction
    elif isinstance(prediction, (list, tuple)):
        if len(prediction) < 1:
            raise RuntimeError("Empty Proposed output.")
        decoded = prediction[0]
        if not torch.is_tensor(decoded):
            raise TypeError(
                "Proposed output index 0 must be decoded Tensor, "
                f"got {type(decoded)}"
            )
    else:
        raise TypeError(f"Unsupported Proposed output: {type(prediction)}")

    if decoded.ndim != 3:
        raise RuntimeError(
            f"Decoded Tensor must be 3D, got {tuple(decoded.shape)}"
        )

    expected = 4 + int(nc)
    if decoded.shape[1] != expected and decoded.shape[2] != expected:
        raise RuntimeError(
            f"Expected {expected} detection channels, got {tuple(decoded.shape)}"
        )
    return decoded
