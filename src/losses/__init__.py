from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss

losses = {
    'cross_entropy': CrossEntropyLoss,
    'bce_with_logits': BCEWithLogitsLoss,
    'mse': MSELoss,
}