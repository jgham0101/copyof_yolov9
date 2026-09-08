import torch
def select_decoded_prediction(prediction,nc=80):
    if torch.is_tensor(prediction):
        x=prediction
    elif isinstance(prediction,(list,tuple)) and len(prediction)>=1 and torch.is_tensor(prediction[0]):
        x=prediction[0]
    else:
        raise TypeError(type(prediction))
    if x.ndim!=3:raise RuntimeError(tuple(x.shape))
    expected=4+int(nc)
    if x.shape[1]!=expected and x.shape[2]!=expected:
        raise RuntimeError(tuple(x.shape))
    return x
