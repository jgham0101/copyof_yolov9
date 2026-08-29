from __future__ import annotations
import torch
def select_decoded_prediction(prediction, nc: int = 80):
    if torch.is_tensor(prediction):
        decoded=prediction
    elif isinstance(prediction,(list,tuple)):
        if len(prediction)<1:
            raise RuntimeError("Empty Proposed inference output")
        decoded=prediction[0]
        if not torch.is_tensor(decoded):
            raise TypeError(f"Proposed index0 must be Tensor, got {type(decoded)}")
    else:
        raise TypeError(f"Unsupported output: {type(prediction)}")
    if decoded.ndim!=3:
        raise RuntimeError(f"Decoded prediction must be 3D: {tuple(decoded.shape)}")
    expected=4+int(nc)
    if decoded.shape[1]!=expected and decoded.shape[2]!=expected:
        raise RuntimeError(f"Decoded output lacks {expected} channels: {tuple(decoded.shape)}")
    return decoded
